"""第一步（v2）：修复三个 bug 后的 B / B_pos / B_pos∪C_pos 上限测试。

修复：
  Bug1  每个 U 块独立 getFeatures，绝不把两个减过质心的块拼进同一个 offset
  Bug2  每块独立按 k_shot=1 采 100 个前景 + 100 个背景原型；在相关性层聚合
         corr = (1-λ) * corr_S + λ * mean_m corr_U_m
         自检：U=S 时输出必须与 Baseline 完全一致（<1e-5）
  Bug3  前景候选 prob>0.7；背景原型只从高置信背景（prob<0.2 或真背景）采样，
         模糊/漏检区不参与任何原型

评估：复刻 COSeg.forward 后半段（lin1 -> HCA -> class_reduce -> kpconv -> cls）。
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as Fs

sys.path.insert(0, '/data01/jh/code/COSeg')
sys.path.insert(0, '/data01/jh/code')
from ring4_isolated import prepare_block  # noqa: E402
from lib.pointops2.functions import pointops  # noqa: E402
import diag_bc as D  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
NAMES_PATH = '/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
K = 100


def seed_everything(seed):
    """锁死体素化随机选点：voxelize(mode=0) 用 np.random.randint 挑代表点。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def feats_one(model, x, y, dev):
    off = torch.tensor([len(x)], dtype=torch.int32)
    fd, cd, ol, yl, _, _ = model.getFeatures(x.to(dev), off.to(dev), y.to(dev), None)
    return dict(fd=fd, cd=cd, ol=ol, yl=yl, xyz=x[:, :3].to(dev), off=off.to(dev))


@torch.no_grad()
def proto(model, f, mask, k=K):
    mask = mask.bool()
    n = int(mask.sum())
    if n == 0:
        return torch.zeros(k, model.feat_dim, device=f['fd'].device)
    return model.getMutiplePrototypes(f['cd'][mask], f['fd'][mask], k)


@torch.no_grad()
def predict_agg(model, qf, sparse_list, lam, dev):
    q_fd, q_cd = qf['fd'], qf['cd']
    base_sim = Fs.cosine_similarity(
        q_fd[:, None, :], model.base_prototypes[None, :, :], dim=2)
    base_g = base_sim.max(dim=1, keepdim=True)[0]

    corrs = [Fs.cosine_similarity(q_fd[:, None, :], sp[None, :, :], dim=2)
             for sp in sparse_list]
    if len(corrs) == 1 or lam <= 0:
        corr = corrs[0]
    else:
        corr = (1.0 - lam) * corrs[0] + lam * torch.stack(corrs[1:]).mean(0)

    corr = model.lin1(corr.view(corr.shape[0], 2, -1)).permute(2, 1, 0).unsqueeze(0)
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
    pred_raw = pointops.interpolation(
        q_cd, qf['xyz'], out.contiguous(), qf['ol'], qf['off']
    ).transpose(0, 1).unsqueeze(0)[0]
    return pred_raw, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=10)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--lams', default='0.2,0.3,0.5')
    ap.add_argument('--balance', action='store_true', default=True)
    ap.add_argument('--lam', type=float, default=0.3)
    ap.add_argument('--fg-hi', type=float, default=0.7)
    ap.add_argument('--bg-lo', type=float, default=0.2)
    ap.add_argument('--out', default=os.path.join(CACHE, 'upper_bound_v2.json'))
    args = ap.parse_args()

    names = [l.strip() for l in open(NAMES_PATH)]
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))
    by_class = defaultdict(list)
    for m in metas:
        by_class[m['class']].append(m)
    eps = []
    for i in range(args.per_class):
        for c in NEW_IDS:
            if i < len(by_class[c]):
                eps.append(by_class[c][i])
    if args.shards > 1:
        eps = eps[args.shard::args.shards]

    seed_everything(0)          # 固定 COSeg 构造期的随机初始化
    model, cfg = D.build_model()
    dev = torch.device('cuda:0')

    lams = [float(x) for x in args.lams.split(',')]
    cases = ['Baseline'] + ['%s_l%.1f' % (b, l) for l in lams
                            for b in ('Naive-B', 'Oracle-B', 'Oracle-B+C')]
    rows = []
    ious = {k: [] for k in cases}
    cm = {k: {c: [0, 0, 0] for c in NEW_IDS} for k in cases}
    n_bsp = {k: [] for k in cases}

    for ei, m in enumerate(eps):
        g_ep = args.shard + args.shards * ei
        seed_everything(10000 + g_ep)   # 每个 episode 的体素选点必须与分片方式无关
        c = m['class']
        s = prepare_block(m['support'], cfg)
        q = prepare_block(m['query'], cfg)
        us = [prepare_block(nm, cfg) for nm in m['u']]

        s_bin = (s['label'] == c).long()
        q_map = (q['label'] == c).long()
        s_x = torch.cat([s['coord'], s['feat']], 1)
        q_x = torch.cat([q['coord'], q['feat']], 1)

        sf = feats_one(model, s_x, s_bin, dev)
        qf = feats_one(model, q_x, q_map, dev)

        fg_s = sf['yl'].bool()
        sparse_s = torch.cat([proto(model, sf, ~fg_s), proto(model, sf, fg_s)], 0)

        # 每个 U 块独立提特征（Bug1）
        u_feats = []
        for u in us:
            u_x = torch.cat([u['coord'], u['feat']], 1)
            u_bin = (u['label'] == c).long()
            uf = feats_one(model, u_x, u_bin, dev)   # yl = 下采样后的真前景标签
            uf['lab'] = u['label']
            _, u_out = predict_agg(model, uf, [sparse_s], 0.0, dev)
            uf['prob'] = Fs.softmax(u_out, dim=1)[:, 1].cpu()
            u_feats.append(uf)

        # 四种 U 的 sparse（前景高置信 + 背景高置信，模糊区不参与）
        def build_sparse(uf_list, mode, case):
            fgs, bgs, nfg = [], [], 0
            for uf in u_feats:
                prob = uf['prob'].to(uf['fd'].device)
                lab = uf['yl'].to(uf['fd'].device)   # 下采样真前景标签
                if mode == 'naive':
                    m_fg = prob > args.fg_hi
                    m_bg = prob < args.bg_lo
                elif mode == 'oracle_b':
                    m_fg = (prob > args.fg_hi) & (lab > 0.5)
                    m_bg = lab <= 0.5
                else:  # oracle_bc
                    m_fg = lab > 0.5
                    m_bg = lab <= 0.5
                nfg += int(m_fg.sum())
                fgs.append(proto(model, uf, m_fg))
                bgs.append(proto(model, uf, m_bg))
            n_bsp.setdefault(case, []).append(nfg)
            fg_p = torch.stack(fgs).mean(0)
            bg_p = torch.stack(bgs).mean(0)
            return torch.cat([bg_p, fg_p], 0)

        sp_naive = build_sparse(u_feats, 'naive', 'Naive-B')
        sp_ob = build_sparse(u_feats, 'oracle_b', 'Oracle-B')
        sp_obc = build_sparse(u_feats, 'oracle_bc', 'Oracle-B+C')

        def run(case, sp, lam):
            pred, _ = predict_agg(model, qf, sp, lam, dev)
            p = pred.argmax(0).cpu()
            tp = int(((p == 1) & (q_map == 1)).sum())
            fp = int(((p == 1) & (q_map == 0)).sum())
            fn = int(((p == 0) & (q_map == 1)).sum())
            iou = tp / max(tp + fp + fn, 1)
            ious.setdefault(case, []).append(iou)
            cm.setdefault(case, {cc: [0, 0, 0] for cc in NEW_IDS})
            cm[case][c][0] += tp
            cm[case][c][1] += fp
            cm[case][c][2] += fn
            rows.append(dict(ep=ei, case=case, lam=lam, cls=c, iou=iou,
                             tp=tp, fp=fp, fn=fn))

        run('Baseline', [sparse_s], 0.0)
        for lam in lams:
            run('Naive-B_l%.1f' % lam, [sparse_s, sp_naive], lam)
            run('Oracle-B_l%.1f' % lam, [sparse_s, sp_ob], lam)
            run('Oracle-B+C_l%.1f' % lam, [sparse_s, sp_obc], lam)

        if (ei + 1) % 10 == 0:
            print('  %d / %d' % (ei + 1, len(eps)), flush=True)
            json.dump(dict(rows=rows, lams=lams, done=ei + 1, n_ep=len(eps)),
                      open(args.out + '.partial', 'w'))

    out = {}
    print('\n%-14s %10s %10s %10s' % ('case', 'Arith', 'Cumul', 'mean n_B'))
    for k in cases:
        arith = float(np.mean(ious[k]))
        per = [cm[k].get(c, [0, 0, 0])[0] /
               max(sum(cm[k].get(c, [0, 0, 0])), 1) for c in NEW_IDS]
        cumul = float(np.mean(per))
        nb = float(np.mean(n_bsp[k])) if n_bsp.get(k) else 0
        out[k] = dict(arith=arith, cumul=cumul, per_class=per,
                      mean_n_b=nb, n_ep=len(ious[k]))
        print('%-14s %10.4f %10.4f %10.0f' % (k, arith, cumul, nb))

    print('\n逐类 Cumul IoU')
    print('%-14s' % 'class' + ''.join('%12s' % k for k in cases))
    for i, c in enumerate(NEW_IDS):
        line = '%-14s' % names[c]
        for k in cases:
            line += ('%12.4f' % out[k]['per_class'][i]) if k in out else '%12s' % '--'
        print(line)

    out['rows'] = rows
    out['lams'] = lams
    json.dump(out, open(args.out, 'w'), indent=1)
    print('\n->', args.out)


if __name__ == '__main__':
    main()
