"""验证三个根因：FPS 离群放大、反向完整管道校验、以及采样策略。

只读缓存，不改主管道。输出四组对比：

1. FPS 种子精度 vs 全体候选精度
   `getMutiplePrototypes` 用 3D 坐标做 FPS 选种子，追求空间最远；
   如果假阳性散落在房间四角，种子会被离群点占满。
   对比 `p0>0.7` 全部点的精度与 FPS 选中的 100 个种子的精度。

2. 超点级过滤 + 概率加权采样 的精度
   先按 `.segs.json` 超点聚合 p0 得到 q_c(R)，只保留高置信超点，
   再按 q_c 加权采样。对比同一批 B 点。

3. 反向完整管道校验 `proto_B -> S`
   用 U 块的原型作为参考集，反向预测支持集 S（100% 真值），
   得到 rev_iou / rev_fg（S 真前景上的平均激活）/ eta_rev
   （S 真背景上的 relu(logit_fg - logit_bg)）。用这些分数给 B 排序。

4. 块级相关性：块的 B 纯度 vs 反向校验分数
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as Fs
from sklearn.metrics import roc_auc_score

from ocsb_core import (CACHE, NEW_IDS, build_episodes, build_proto,
                       episode_tensors, feats_one, iou_counts, load_model,
                       predict_agg, prepare_block, seed_everything,
                       seg_downsample)
from lib.pointops2.functions import pointops


def fps_seed_idx(coord, k):
    """复现 getMutiplePrototypes 里的 FPS 选种，返回种子在输入里的下标。"""
    off = torch.tensor([coord.shape[0]], dtype=torch.int32, device=coord.device)
    return pointops.furthestsampling(
        coord.contiguous(), off,
        torch.cuda.IntTensor([k])).long()


def seg_stats(seg_d, p0):
    """按超点聚合 p0，返回每个点的超点均值 q_c(R)。"""
    sg = seg_d.detach().cpu().numpy()
    uniq, inv = np.unique(sg, return_inverse=True)
    p = p0.detach().cpu().numpy()
    cnt = np.bincount(inv).astype(np.float64)
    s1 = np.bincount(inv, weights=p)
    mean = s1 / np.maximum(cnt, 1)
    return (torch.from_numpy(mean[inv]).to(p0.device).float(),
            torch.from_numpy(cnt[inv]).to(p0.device).float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=10)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--fg-hi', type=float, default=0.7)
    ap.add_argument('--bg-lo', type=float, default=0.2)
    ap.add_argument('--q-hi', type=float, default=0.6, help='超点级前景置信阈')
    ap.add_argument('--rev-lam', type=float, default=1.0, help='反向校验的融合强度')
    ap.add_argument('--out', default=os.path.join(CACHE, 'purity_fix.json'))
    args = ap.parse_args()

    eps = build_episodes(args.per_class, args.shard, args.shards)
    model, cfg = load_model()
    dev = torch.device('cuda:0')

    rec = []   # 每个 U 块一行
    pt = dict(fps=[], sp=[], allb=[], rev=[])
    pt_lab = []

    for ei, m in enumerate(eps):
        g_ep = args.shard + args.shards * ei
        seed_everything(10000 + g_ep)
        c = m['class']
        s = prepare_block(m['support'], cfg)
        us = [prepare_block(nm, cfg) for nm in m['u']]
        s_x, s_bin = episode_tensors(s, c)
        sf = feats_one(model, s_x, s_bin, dev)
        fg_s = sf['yl'].bool()
        sparse_s = torch.cat([build_proto(model, sf, ~fg_s),
                              build_proto(model, sf, fg_s)], 0)

        for u in us:
            u_x, u_bin = episode_tensors(u, c)
            uf = feats_one(model, u_x, u_bin, dev)
            uf['seg_d'] = seg_downsample(uf['xyz'], u['seg'].to(dev), uf['cd'])
            _, out0 = predict_agg(model, uf, [sparse_s], 0.0, dev)
            p0 = Fs.softmax(out0, dim=1)[:, 1]
            b_mask = p0 > args.fg_hi
            nb = int(b_mask.sum())
            if nb < 10:
                continue
            lab_b = uf['yl'][b_mask]
            base_prec = float(lab_b.float().mean())

            # --- 1. FPS 种子精度 ---
            cd_b, fd_b = uf['cd'][b_mask], uf['fd'][b_mask]
            k = min(100, nb)
            try:
                fidx = fps_seed_idx(cd_b, k)
                fps_prec = float(lab_b[fidx].float().mean())
            except Exception:
                fps_prec = float('nan')

            # --- 2. 超点过滤 + 概率加权采样 ---
            q_seg, seg_cnt = seg_stats(uf['seg_d'], p0)
            keep = b_mask & (q_seg > args.q_hi) & (seg_cnt >= 20)
            if int(keep.sum()) >= k:
                w = q_seg[keep]
                pick = torch.nonzero(keep).squeeze(1)[
                    torch.multinomial(w / w.sum(), k, replacement=False)]
                sp_prec = float(uf['yl'][pick].float().mean())
                sp_n = int(keep.sum())
            else:
                sp_prec, sp_n = float('nan'), int(keep.sum())

            # --- 3. 反向完整管道校验：proto_B -> S ---
            b_sparse = torch.cat([build_proto(model, uf, p0 < args.bg_lo),
                                  build_proto(model, uf, b_mask)], 0)
            pred_s, _ = predict_agg(model, sf, [b_sparse], args.rev_lam, dev)
            _, _, _, rev_iou = iou_counts(pred_s, s_bin)
            logit = pred_s
            pS = Fs.softmax(logit, dim=0)[1]
            fg_true = s_bin > 0
            rev_fg = float(pS[fg_true].mean()) if fg_true.any() else 0.0
            marg = Fs.relu(logit[1] - logit[0])
            eta_rev = float(marg[~fg_true].mean()) if (~fg_true).any() else 0.0

            rec.append(dict(ep=ei, cls=c, n_b=nb, base_prec=base_prec,
                            fps_prec=fps_prec, sp_prec=sp_prec, sp_n=sp_n,
                            rev_iou=rev_iou, rev_fg=rev_fg, eta_rev=eta_rev,
                            rev_v=rev_iou - eta_rev))
            pt['allb'].append(base_prec)
            pt['fps'].append(fps_prec)
            pt['sp'].append(sp_prec)
            pt['rev'].append(rev_iou)

        if (ei + 1) % 10 == 0:
            print('  %d / %d' % (ei + 1, len(eps)), flush=True)

    r = {k: np.array([x[k] for x in rec], dtype=np.float64)
         for k in ('n_b', 'base_prec', 'fps_prec', 'sp_prec', 'rev_iou',
                   'rev_fg', 'eta_rev', 'rev_v')}
    n = len(rec)
    print('\n=== 1. 采样策略对原型纯度的影响（%d 个 U 块） ===' % n)
    print('%-22s %9s %9s' % ('策略', '均值', '中位数'))
    for name, key in (('全体 p0>0.7 点', 'base_prec'),
                      ('FPS 选中的 100 个种子', 'fps_prec'),
                      ('超点过滤+加权采样', 'sp_prec')):
        v = r[key][~np.isnan(r[key])]
        print('%-22s %9.4f %9.4f  (n=%d)' % (
            name, v.mean(), np.median(v), len(v)))

    print('\n=== 2. 反向校验 `proto_B -> S` 的分布 ===')
    for k in ('rev_iou', 'rev_fg', 'eta_rev', 'rev_v'):
        print('  %-9s mean=%.4f  median=%.4f  p10=%.4f  p90=%.4f' % (
            k, r[k].mean(), np.median(r[k]),
            np.percentile(r[k], 10), np.percentile(r[k], 90)))

    print('\n=== 3. 反向校验能否区分高纯度块 ===')
    hi = r['base_prec'] >= 0.7
    lo = r['base_prec'] <= 0.3
    sel = hi | lo
    if sel.sum() >= 10 and hi.sum() > 0 and lo.sum() > 0:
        for k in ('rev_iou', 'rev_fg', 'rev_v'):
            try:
                auc = roc_auc_score(hi[sel].astype(int), r[k][sel])
            except ValueError:
                auc = float('nan')
            print('  %-9s AUC=%.4f  (高纯度块 %d / 低纯度块 %d)  高均值=%.4f 低均值=%.4f' % (
                k, auc, hi.sum(), lo.sum(),
                r[k][hi].mean(), r[k][lo].mean()))
    else:
        print('  高/低纯度块太少：%d / %d' % (hi.sum(), lo.sum()))

    print('\n=== 4. 块级相关性（Spearman） ===')
    from scipy.stats import spearmanr
    for k in ('rev_iou', 'rev_fg', 'rev_v'):
        rho, p = spearmanr(r[k], r['base_prec'])
        print('  %-9s rho=%+.4f  p=%.2e' % (k, rho, p))

    json.dump(dict(rec=rec, summary={k: r[k].tolist() for k in r}),
              open(args.out, 'w'), indent=1)
    print('\n->', args.out)


if __name__ == '__main__':
    main()
