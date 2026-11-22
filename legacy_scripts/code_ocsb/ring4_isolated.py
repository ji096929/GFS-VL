"""第四环：把不同纯度的 B 作为前景原型送入 COSeg 相关性网络，
测它对查询集的直接贡献（不恢复 C，不训练）。

两种接口：
  onlyB  fg 原型 100 个全部来自 B
  S+B    S 前景点集与 B 点集合并后，统一 FPS + 聚类出 100 个原型

Tier：
  T1  q>=0.6 全量
  T2a 策略 C r<=0.8m（每 block 内以最高置信超点为核心）
  T2b Top-5
  T3  B ∩ 真值前景（理论上界，非可用方法）

完全自洽：U 的前景概率由同一个模型、同一套 support 原型在 U 上跑出，
超点编号、世界坐标、backbone 输出点全部在同一份预处理里产生，不依赖缓存编号。
原型提取复刻 COSeg.getMutiplePrototypes（FPS 种子 -> 特征空间聚类均值 ->
空簇用种子 -> 点数不足补零）。
同时输出 B 与查询点云的空间重合率，用于判断 onlyB 的增益是否来自覆盖查询物体。
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as Fs

sys.path.insert(0, '/data01/jh/code/COSeg')
sys.path.insert(0, '/data01/jh/code')
from lib.pointops2.functions import pointops  # noqa: E402
from util.data_util import voxelize  # noqa: E402
import diag_bc as D  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
NAMES_PATH = '/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
PER_CLASS = 30
K = 100


def prepare_block(name, cfg):
    """复现 data_prepare_seg，额外返回保留点索引与原始世界坐标。"""
    r = np.load(os.path.join(DATA, name + '.npy'))
    coord = r[:, 0:3].astype(np.float32).copy()
    feat = r[:, 3:6].astype(np.float32)
    label = r[:, 6].astype(np.int64)
    seg = r[:, 7].astype(np.int64)
    idx = np.arange(len(coord))
    if cfg.voxel_size:
        coord -= np.min(coord, 0)
        uq = voxelize(coord, cfg.voxel_size)
        idx = idx[uq]
        coord, feat, label, seg = coord[uq], feat[uq], label[uq], seg[uq]
    if cfg.voxel_max and label.shape[0] > cfg.voxel_max:
        init = label.shape[0] // 2
        crop = np.argsort(np.sum(np.square(coord - coord[init]), 1))[:cfg.voxel_max]
        idx = idx[crop]
        coord, feat, label, seg = coord[crop], feat[crop], label[crop], seg[crop]
    cmin, cmax = np.min(coord, 0), np.max(coord, 0)
    coord = coord - (cmin + cmax) / 2.0
    return {
        'coord': torch.FloatTensor(coord),
        'feat': torch.FloatTensor(feat) / 255.0,
        'label': torch.LongTensor(label),
        'seg': torch.LongTensor(seg),
        'world': r[idx, 0:3].astype(np.float64),
    }


def get_prototypes(coord, feat, k, feat_dim, dev):
    """复刻 COSeg.getMutiplePrototypes：coord 上 FPS，特征空间聚类均值。"""
    n = len(feat)
    if n == 0:
        return torch.zeros((k, feat_dim), device=dev)
    if n <= k:  # 原生：补零特征
        return torch.cat([feat, torch.zeros((k - n, feat_dim), device=dev)], 0)
    idx = pointops.furthestsampling(
        coord.contiguous().float(),
        torch.tensor([n], dtype=torch.int32, device=dev),
        torch.tensor([k], dtype=torch.int32, device=dev)).long()
    seeds = feat[idx]
    d = torch.linalg.norm(feat[:, None, :] - seeds[None, :, :], dim=2)
    a = torch.argmin(d, dim=1)
    cnt = torch.bincount(a, minlength=k)
    sums = torch.zeros((k, feat_dim), device=dev)
    sums.index_add_(0, a, feat)
    out = sums / cnt.clamp(min=1).unsqueeze(1).float()
    empty = cnt == 0
    if bool(empty.any()):
        out[empty] = seeds[empty]
    return out


@torch.no_grad()
def feats_all(model, s_x, s_off, s_bin, q_x, q_off, q_map,
              u_x, u_off, u_zero, dev):
    """每个 episode 只跑一次的三个 getFeatures。"""
    s_fd, s_cd, _, s_yl, _, _ = model.getFeatures(
        s_x.to(dev), s_off.to(dev), s_bin.to(dev), None)
    q_fd, q_cd, q_ol, _, _, _ = model.getFeatures(
        q_x.to(dev), q_off.to(dev), q_map.to(dev), None)
    u_fd, u_cd, u_ol, _, _, _ = model.getFeatures(
        u_x.to(dev), u_off.to(dev), u_zero.to(dev), None)
    return {'s_fd': s_fd, 's_cd': s_cd, 's_yl': s_yl,
            'q_fd': q_fd, 'q_cd': q_cd, 'q_ol': q_ol,
            'q_xyz': q_x[:, :3].to(dev).contiguous(), 'q_off': q_off.to(dev),
            'u_fd': u_fd, 'u_cd': u_cd, 'u_ol': u_ol,
            'u_xyz': u_x[:, :3].to(dev).contiguous(), 'u_off': u_off.to(dev)}


@torch.no_grad()
def predict(model, feats, fg_proto, bg_proto):
    """复刻 COSeg.forward 单 query 后半段。"""
    q_fd, q_cd = feats['q_fd'], feats['q_cd']
    sparse = torch.cat([bg_proto, fg_proto], 0)  # [2K, C]

    base_sim = Fs.cosine_similarity(
        q_fd[:, None, :], model.base_prototypes[None, :, :], dim=2)
    base_g = base_sim.max(dim=1, keepdim=True)[0]

    corr = Fs.cosine_similarity(q_fd[:, None, :], sparse[None, :, :], dim=2)
    corr = model.lin1(
        corr.view(corr.shape[0], 2, -1)).permute(2, 1, 0).unsqueeze(0)
    for layer in model.agglayers:
        corr = layer(corr, base_g)
    corr = corr.squeeze(0).permute(2, 1, 0).contiguous()
    corr = model.class_reduce(corr).squeeze(1)

    coord = q_cd.contiguous()
    batch = torch.zeros(corr.shape[0], dtype=torch.int64, device=coord.device)
    radius = 2.5 * model.args.grid_size * 2.0
    off = torch.tensor([coord.shape[0]], dtype=torch.int32, device=coord.device)
    nbr = pointops.ball_query(radius, model.args.max_num_neighbors,
                              coord, coord, off, off)[0].long()
    corr = model.kpconv(corr, coord, batch, nbr.clone())
    out = model.cls(corr)
    pred = pointops.interpolation(
        q_cd, feats['q_xyz'], out.contiguous(),
        feats['q_ol'], feats['q_off']).transpose(0, 1).unsqueeze(0)
    return pred[0]  # [2, Nq]


def nn_index(src_xyz, dst_xyz):
    """dst 每个点在 src 中的最近邻索引。"""
    _, idx = cKDTree(src_xyz).query(dst_xyz)
    return idx


def select_B(mode, q_sp, ctr, sp_bid, sp_fg, n_u):
    """返回 B 的超点掩码。q_sp/sp_fg 都是超点级。"""
    B = q_sp >= 0.6
    if not B.any():
        return B
    if mode == 'T1':
        return B
    if mode == 'T2b':
        idx = np.argsort(-q_sp)
        keep = np.zeros(n_u, dtype=bool)
        keep[idx[:5]] = True
        return B & keep
    if mode == 'T2a':
        keep = np.zeros(n_u, dtype=bool)
        for b in np.unique(sp_bid[B]):
            idx = np.where(B & (sp_bid == b))[0]
            if len(idx) == 0:
                continue
            core = idx[np.argmax(q_sp[idx])]
            dist = np.linalg.norm(ctr[idx] - ctr[core], axis=1)
            keep[idx[dist <= 0.8]] = True
        return B & keep
    if mode == 'T3':
        return B & (sp_fg > 0.5)
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=PER_CLASS)
    ap.add_argument('--out', default=os.path.join(CACHE, 'ring4.json'))
    args = ap.parse_args()

    names = [l.strip() for l in open(NAMES_PATH)]
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))
    model, cfg = D.build_model()
    dev = torch.device('cuda:0')
    feat_dim = model.feat_dim

    base_variants = ['S', 'onlyB_T1', 'onlyB_T2a', 'onlyB_T2b', 'onlyB_T3',
                     'SB_T1', 'SB_T2a', 'SB_T2b', 'SB_T3']
    fuse_specs = [('T1', 0.2), ('T1', 0.3), ('T2b', 0.2), ('T2b', 0.3),
                  ('T3', 0.2), ('T3', 0.3)]
    fnames = ['FUSE_%s_l%.1f' % (t, l) for t, l in fuse_specs]
    variants = base_variants + fnames

    rows = []
    for c in NEW_IDS:
        eps = [m for m in metas if m['class'] == c][:args.per_class]
        cname = names[c]
        ious = {v: [] for v in variants}
        n_bsp = {v: [] for v in variants}
        ov = {'b5': [], 'qcov': [], 'pos5': [], 'neg5': []}
        for m in eps:
            s = prepare_block(m['support'], cfg)
            q = prepare_block(m['query'], cfg)
            us = [prepare_block(nm, cfg) for nm in m['u']]

            u_coord = torch.cat([u['coord'] for u in us], 0)
            u_feat = torch.cat([u['feat'] for u in us], 0)
            u_lab = torch.cat([u['label'] for u in us], 0)
            seg_cat = torch.cat(
                [u['seg'] + i * 10 ** 7 for i, u in enumerate(us)], 0)
            _, u_seg = torch.unique(seg_cat, return_inverse=True)
            u_seg_np = u_seg.numpy()
            n_u = int(u_seg_np.max()) + 1
            u_bid = np.concatenate(
                [np.full(len(u['coord']), i, np.int64)
                 for i, u in enumerate(us)])
            u_world = np.concatenate([u['world'] for u in us])

            cnt = np.maximum(np.bincount(u_seg_np, minlength=n_u), 1)
            sp_bid = np.zeros(n_u, dtype=np.int64)
            sp_bid[u_seg_np] = u_bid
            uc = u_coord.numpy().astype(np.float64)
            ctr = np.stack([np.bincount(u_seg_np, weights=uc[:, k],
                                        minlength=n_u) for k in range(3)], 1)
            ctr = ctr / cnt[:, None]

            s_bin = (s['label'] == c).long()
            q_map = (q['label'] == c).long()
            s_x = torch.cat([s['coord'], s['feat']], 1)
            q_x = torch.cat([q['coord'], q['feat']], 1)
            u_x = torch.cat([u_coord, u_feat], 1)
            s_off = torch.tensor([len(s_x)], dtype=torch.int32)
            q_off = torch.tensor([len(q_x)], dtype=torch.int32)
            u_off = torch.tensor([len(u_x)], dtype=torch.int32)
            u_zero = torch.zeros(len(u_x), dtype=torch.long)
            feats = feats_all(model, s_x, s_off, s_bin, q_x, q_off, q_map,
                              u_x, u_off, u_zero, dev)

            s_fg = feats['s_yl'].bool()
            s_bg = ~s_fg
            bg_proto = get_prototypes(feats['s_cd'][s_bg], feats['s_fd'][s_bg],
                                      K, feat_dim, dev)
            s_fg_proto = get_prototypes(
                feats['s_cd'][s_fg], feats['s_fd'][s_fg], K, feat_dim, dev)

            # U 的前景概率：与原生 forward 完全等价（S 原型 -> U 特征）
            u_feats = {'q_fd': feats['u_fd'], 'q_cd': feats['u_cd'],
                       'q_ol': feats['u_ol'], 'q_xyz': feats['u_xyz'],
                       'q_off': feats['u_off']}
            u_pred = predict(model, u_feats, s_fg_proto, bg_proto)
            p_u = Fs.softmax(u_pred, dim=0)[1].cpu().numpy()
            q_sp = np.bincount(u_seg_np, weights=p_u, minlength=n_u) / cnt
            fg_sp = np.bincount(
                u_seg_np, weights=(u_lab == c).numpy().astype(np.float64),
                minlength=n_u) / cnt

            # backbone 输出点 -> 中心化点级的最近邻（用于挑 B 的 downsampled 点）
            u_nn = nn_index(u_coord.numpy(), feats['u_cd'].cpu().numpy())
            seg_at_ds = u_seg_np[u_nn]

            y_full = (q['label'] == c).numpy().astype(np.int64)
            qx_world = q['world']

            # 重合率（T1 口径）
            B_all = select_B('T1', q_sp, ctr, sp_bid, fg_sp, n_u)
            if B_all.any():
                b_mask = np.isin(u_seg_np, np.where(B_all)[0])
                b_world = u_world[b_mask]
                d_bq, _ = cKDTree(qx_world).query(b_world)
                ov['b5'].append(float((d_bq <= 0.05).mean()))
                d_qb, _ = cKDTree(b_world).query(qx_world)
                ov['qcov'].append(float((d_qb <= 0.05).mean()))
                pos = b_mask & (fg_sp[u_seg_np] > 0.5)
                neg = b_mask & ~(fg_sp[u_seg_np] > 0.5)
                if pos.any():
                    dp, _ = cKDTree(qx_world).query(u_world[pos])
                    ov['pos5'].append(float((dp <= 0.05).mean()))
                if neg.any():
                    dn, _ = cKDTree(qx_world).query(u_world[neg])
                    ov['neg5'].append(float((dn <= 0.05).mean()))

            margins = {}
            for v in base_variants:
                if v == 'S':
                    fg_proto = s_fg_proto
                    n_bsp[v].append(0)
                else:
                    tier = v.split('_')[1]
                    Bm = select_B(tier, q_sp, ctr, sp_bid, fg_sp, n_u)
                    sel = np.isin(seg_at_ds, np.where(Bm)[0])
                    n_bsp[v].append(int(sel.sum()))
                    if sel.sum() == 0:
                        b_c = b_f = None
                    else:
                        b_c = feats['u_cd'][torch.from_numpy(sel).to(dev)]
                        b_f = feats['u_fd'][torch.from_numpy(sel).to(dev)]
                    if v.startswith('onlyB'):
                        fg_proto = get_prototypes(
                            b_c, b_f, K, feat_dim, dev) if b_c is not None \
                            else torch.zeros((K, feat_dim), device=dev)
                    else:
                        if b_c is None:
                            pc, pf = feats['s_cd'][s_fg], feats['s_fd'][s_fg]
                        else:
                            pc = torch.cat([feats['s_cd'][s_fg], b_c], 0)
                            pf = torch.cat([feats['s_fd'][s_fg], b_f], 0)
                        fg_proto = get_prototypes(pc, pf, K, feat_dim, dev)
                pred = predict(model, feats, fg_proto, bg_proto)
                margins[v] = (pred[1] - pred[0]).cpu().numpy()

            # 双路 logit 融合：(1-λ)*S + λ*onlyB，纯算术，不增加前向
            for tier, lam in fuse_specs:
                margins['FUSE_%s_l%.1f' % (tier, lam)] = (
                    (1.0 - lam) * margins['S'] +
                    lam * margins['onlyB_' + tier])
            for v in variants:
                margin = margins[v]
                p = margin > 0
                tp = int((p & (y_full > 0)).sum())
                un = int((p | (y_full > 0)).sum())
                ious[v].append(tp / max(un, 1))
                if v in fnames:
                    n_bsp[v].append(n_bsp['onlyB_' + v.split('_')[1]][-1])
        print('%-12s done' % cname, flush=True)
        rows.append({
            'class': cname, 'n': len(eps),
            'iou': {v: float(np.mean(x)) for v, x in ious.items()},
            'n_B': {v: float(np.mean(x)) if x else 0 for v, x in n_bsp.items()},
            'ov': {k: (float(np.mean(x)) if x else 0.0) for k, x in ov.items()},
        })

    json.dump(rows, open(args.out, 'w'), indent=1)

    order = variants
    print('\n%-12s %7s %8s' % ('class', 'n_B', 'B@5cm') +
          ''.join('%11s' % v for v in order))
    for r in rows:
        print('%-12s %7.0f %8.3f'
              % (r['class'], r['n_B']['onlyB_T1'], r['ov']['b5']) +
              ''.join('%11.4f' % r['iou'][v] for v in order))
    print('%-12s %7s %8.3f' % ('MEAN', '', np.mean([r['ov']['b5'] for r in rows])) +
          ''.join('%11.4f' % np.mean([r['iou'][v] for r in rows]) for v in order))
    print('\nΔ vs S:')
    for v in order[1:]:
        ds = [r['iou'][v] - r['iou']['S'] for r in rows]
        print('%-11s %+7.4f  (min %+.4f  max %+.4f)'
              % (v, np.mean(ds), min(ds), max(ds)))
    print('\nB 与查询点云重合率（T1）:')
    print('%-12s %8s %8s %8s %8s' % ('class', 'B@5cm', 'Qcov', 'Bpos@5', 'Bneg@5'))
    for r in rows:
        o = r['ov']
        print('%-12s %8.3f %8.3f %8.3f %8.3f'
              % (r['class'], o['b5'], o['qcov'], o['pos5'], o['neg5']))
    print('->', args.out)


if __name__ == '__main__':
    main()
