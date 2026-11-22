"""弱标注支持集痛点摸底（零训练，val_560，仅 S->Q，不引入 U）。

五种支持集掩码（均由支持集真实标签自动生成）：
  full      逐点真值（上界，等同原生 COSeg）
  tight     真实前景点 AABB 内全部点视为前景
  rough     AABB 六面各外扩 5cm 内全部点视为前景
  3click    真实前景点 FPS 3 个，特征放回补齐到 100 个前景原型
  1click    距真实前景质心最近的 1 个点，同样补齐

另测朴素同源清洗（tight/rough）：以盒外点特征均值为参照，
剔除盒内与参照余弦相似度高于盒内 q 分位的点后再建原型。

输出：污染率、漏出率、逐类 mIoU、Arith、官方 Cumul、清洗恢复量。
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
import diag_bc as D  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
NAMES_PATH = '/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
K = 100
VARIANTS = ['full', 'tight', 'rough', '3click', '1click',
            'tight_c50', 'tight_c70', 'rough_c50', 'rough_c70']


def load_block(name, cfg):
    d = np.load(os.path.join(DATA, name + '.npy'))
    return D.data_prepare_seg(d[:, 0:3], d[:, 3:6], d[:, 6], d[:, 7],
                              cfg.voxel_size, cfg.voxel_max)


def get_prototypes(coord, feat, k, feat_dim, dev):
    """复刻 COSeg.getMutiplePrototypes：coord FPS 种子 + 特征空间聚类均值。"""
    n = len(feat)
    if n == 0:
        return torch.zeros((k, feat_dim), device=dev)
    if n <= k:
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


def pad_prototypes(feat, k):
    """有放回补齐到 k 个。"""
    n = len(feat)
    if n == 0:
        return None
    if n >= k:
        return feat[:k]
    rep = (k // n) + 1
    return feat.repeat(rep, 1)[:k]


@torch.no_grad()
def feats_all(model, s_x, s_off, s_bin, q_x, q_off, q_map, dev):
    s_fd, s_cd, _, s_yl, _, _ = model.getFeatures(
        s_x.to(dev), s_off.to(dev), s_bin.to(dev), None)
    q_fd, q_cd, q_ol, _, _, _ = model.getFeatures(
        q_x.to(dev), q_off.to(dev), q_map.to(dev), None)
    return {'s_fd': s_fd, 's_cd': s_cd, 's_yl': s_yl,
            'q_fd': q_fd, 'q_cd': q_cd, 'q_ol': q_ol,
            'q_xyz': q_x[:, :3].to(dev).contiguous(), 'q_off': q_off.to(dev)}


@torch.no_grad()
def predict(model, feats, fg_proto, bg_proto):
    q_fd, q_cd = feats['q_fd'], feats['q_cd']
    sparse = torch.cat([bg_proto, fg_proto], 0)
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
    return pred[0]


def fps_np(xyz, k):
    n = len(xyz)
    if n == 0:
        return np.zeros(0, np.int64)
    k = min(k, n)
    sel = np.zeros(k, np.int64)
    d2 = np.sum((xyz - xyz[0]) ** 2, 1)
    for i in range(1, k):
        sel[i] = int(np.argmax(d2))
        d2 = np.minimum(d2, np.sum((xyz - xyz[sel[i]]) ** 2, 1))
    return sel


def build_masks(coord_o, fg_o, cd_ds):
    """返回 {name: mask_ds}，另附 tight/rough 的原始点级盒信息。"""
    p = coord_o[fg_o]
    masks, boxes = {}, {}
    if len(p) == 0:
        return masks, boxes
    lo, hi = p.min(0), p.max(0)
    masks['tight'] = ((cd_ds >= lo) & (cd_ds <= hi)).all(1)
    lo_r, hi_r = lo - 0.05, hi + 0.05
    masks['rough'] = ((cd_ds >= lo_r) & (cd_ds <= hi_r)).all(1)
    boxes['tight'] = (lo, hi)
    boxes['rough'] = (lo_r, hi_r)

    nn = cKDTree(cd_ds).query(p[fps_np(p, 3)])[1]
    m3 = np.zeros(len(cd_ds), bool)
    m3[np.unique(nn)] = True
    masks['3click'] = m3

    nn1 = cKDTree(cd_ds).query(p.mean(0)[None])[1]
    m1 = np.zeros(len(cd_ds), bool)
    m1[nn1] = True
    masks['1click'] = m1
    return masks, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per-class', type=int, default=56)
    ap.add_argument('--out', default=os.path.join(CACHE, 'weak_support.json'))
    ap.add_argument('--classes', default='')
    args = ap.parse_args()

    names = [l.strip() for l in open(NAMES_PATH)]
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))
    model, cfg = D.build_model()
    dev = torch.device('cuda:0')
    feat_dim = model.feat_dim
    class_ids = NEW_IDS
    if args.classes:
        want = set(args.classes.split(','))
        class_ids = [c for c in NEW_IDS if names[c] in want]

    rows = []
    for c in class_ids:
        class_tp = {v: 0 for v in VARIANTS}
        class_un = {v: 0 for v in VARIANTS}
        eps = [m for m in metas if m['class'] == c][:args.n_per_class]
        acc = {v: [] for v in VARIANTS}
        ct, cr, lk = [], [], []
        nbox_t, nbox_r = [], []
        for m in eps:
            s = load_block(m['support'], cfg)
            q = load_block(m['query'], cfg)
            coord_o = s[0].numpy()
            fg_o = (s[2].numpy() == c)
            if fg_o.sum() == 0:
                continue

            s_bin = (s[2] == c).long()
            q_map = (q[2] == c).long()
            s_x = torch.cat([s[0], s[1]], 1)
            q_x = torch.cat([q[0], q[1]], 1)
            s_off = torch.tensor([len(s_x)], dtype=torch.int32)
            q_off = torch.tensor([len(q_x)], dtype=torch.int32)
            feats = feats_all(model, s_x, s_off, s_bin, q_x, q_off, q_map, dev)

            cd = feats['s_cd'].cpu().numpy()
            s_yl = feats['s_yl'].bool()
            masks, boxes = build_masks(coord_o, fg_o, cd)
            masks['full'] = s_yl.cpu().numpy()

            # 污染率 / 漏出率（原始点级）
            for key, (lo, hi) in boxes.items():
                in_raw = ((coord_o >= lo) & (coord_o <= hi)).all(1)
                nb = int(in_raw.sum())
                nbg = int((in_raw & ~fg_o).sum())
                rate = nbg / max(nb, 1)
                if key == 'tight':
                    ct.append(rate); nbox_t.append(nb)
                else:
                    cr.append(rate); nbox_r.append(nb)
            lo, hi = boxes['tight']
            in_t = ((coord_o >= lo) & (coord_o <= hi)).all(1)
            lk.append(float((~in_t & fg_o).sum()) / max(int(fg_o.sum()), 1))

            y_full = (q[2] == c).numpy().astype(np.int64)
            bg_proto = get_prototypes(feats['s_cd'][~s_yl], feats['s_fd'][~s_yl],
                                      K, feat_dim, dev)
            refs = {}
            for key in ('tight', 'rough'):
                msk = masks[key]
                if (~msk).sum() > 10:
                    refs[key] = Fs.normalize(
                        feats['s_fd'][~torch.from_numpy(msk).to(dev)].mean(
                            0, keepdim=True), dim=1)

            for v in VARIANTS:
                base = v.split('_c')[0]
                msk = masks[base]
                mt = torch.from_numpy(msk).to(dev)
                if '_c' in v and base in refs:
                    qv = float(v.split('_c')[1]) / 100.0
                    sim = (Fs.normalize(feats['s_fd'], dim=1) * refs[base]).sum(1)
                    thr = torch.quantile(sim[mt], qv)
                    mt = mt & (sim <= thr)
                if v in ('3click', '1click'):
                    fg_proto = pad_prototypes(feats['s_fd'][mt], K)
                    if fg_proto is None:
                        fg_proto = torch.zeros((K, feat_dim), device=dev)
                else:
                    fg_proto = get_prototypes(
                        feats['s_cd'][mt], feats['s_fd'][mt], K, feat_dim, dev)
                pred = predict(model, feats, fg_proto, bg_proto)
                p = (pred[1] - pred[0]).cpu().numpy() > 0
                tp = int((p & (y_full > 0)).sum())
                un = int((p | (y_full > 0)).sum())
                acc[v].append(tp / max(un, 1))
                class_tp[v] += tp
                class_un[v] += un
        rows.append({
            'class': names[c], 'n': len(eps),
            'iou': {v: float(np.mean(x)) for v, x in acc.items()},
            'cumul': {v: float(class_tp[v] / max(class_un[v], 1))
                      for v in VARIANTS},
            'contam_tight': float(np.mean(ct)) if ct else 0,
            'contam_rough': float(np.mean(cr)) if cr else 0,
            'leak': float(np.mean(lk)) if lk else 0,
            'nbox_tight': float(np.mean(nbox_t)) if nbox_t else 0,
            'nbox_rough': float(np.mean(nbox_r)) if nbox_r else 0,
        })
        print('%-12s n=%d  tight=%.4f rough=%.4f 3c=%.4f 1c=%.4f  contam=%.3f'
              % (names[c], len(eps), rows[-1]['iou']['tight'],
                 rows[-1]['iou']['rough'], rows[-1]['iou']['3click'],
                 rows[-1]['iou']['1click'], rows[-1]['contam_tight']),
              flush=True)

    json.dump(rows, open(args.out, 'w'), indent=1)

    print('\n=== 弱标注噪声 ===')
    print('%-12s %11s %11s %8s %9s %9s'
          % ('class', 'contam_tight', 'contam_rough', 'leak', 'nbox_t', 'nbox_r'))
    for r in rows:
        print('%-12s %11.3f %11.3f %8.4f %9.0f %9.0f'
              % (r['class'], r['contam_tight'], r['contam_rough'], r['leak'],
                 r['nbox_tight'], r['nbox_rough']))

    print('\n=== Arith mIoU ===')
    print('%-12s' % 'class' + ''.join('%10s' % v for v in VARIANTS))
    for r in rows:
        print('%-12s' % r['class'] + ''.join('%10.4f' % r['iou'][v] for v in VARIANTS))
    print('%-12s' % 'Arith' +
          ''.join('%10.4f' % np.mean([r['iou'][v] for r in rows]) for v in VARIANTS))

    print('\n=== 官方 Cumul (sum TP / sum Union) ===')
    print('%-12s %9s %9s' % ('variant', 'tp', 'union'))
    for v in VARIANTS:
        print('%-12s %9d %9d   Cumul=%.4f'
              % (v, g_tp[v], g_un[v], g_tp[v] / max(g_un[v], 1)))

    print('\n=== Δ vs full（pp，Arith）===')
    print('%-12s' % 'class' + ''.join('%10s' % v for v in VARIANTS[1:]))
    for r in rows:
        print('%-12s' % r['class'] +
              ''.join('%+10.2f' % (100 * (r['iou'][v] - r['iou']['full']))
                      for v in VARIANTS[1:]))
    base = np.mean([r['iou']['full'] for r in rows])
    print('%-12s' % 'MEAN' +
          ''.join('%+10.2f' % (100 * (np.mean([r['iou'][v] for r in rows]) - base))
                  for v in VARIANTS[1:]))
    print('->', args.out)


if __name__ == '__main__':
    main()
