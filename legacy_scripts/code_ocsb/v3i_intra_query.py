"""查询集内自桥接（U=Q）：在查询集自己身上挑 B_Q / C_Q 构造补充原型。

与跨房间版本的唯一区别：把 `uf` 换成 `qf`。

必须警惕的风险：自匹配
--------------------
`sparse_q` 的原型来自查询集自己的点，然后拿去预测同一批点。被选进 B_Q 的点
与自己的簇中心相似度接近 1，会被强烈推成前景。所以这可能是"锐化已有预测"
而不是"恢复遗漏"。

`build_proto` 返回簇均值（不是原始点），所以是软自匹配，但程度仍需实测。

三路输出
--------
pred_s      仅 S 原型
pred_q_l03  S + Q 自桥接原型，λ=0.3
pred_q_l05  λ=0.5（更强，用来放大自匹配效应）
另存 pred_q_l00_... 无意义，故省略。
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as Fs

from scipy.spatial import cKDTree

from ocsb_core import (CACHE, K, build_episodes, build_proto, episode_tensors,
                       feats_one, iou_counts, load_model, predict_agg,
                       prepare_block, seed_everything, seg_downsample)
from train_bridge_scorer import BridgeScorer
from v3h_build_cache import load_scorer, pick_protos


def pick_random(model, uf, p0, args, dev):
    """对照组：同样的池子、同样的比例、同样的槽位，但随机选点。

    用来区分"scorer 选点质量"和"原型来自查询集自己"这两个因素。
    """
    n_c = int(round(K * args.c_ratio))
    b_pool = p0 > args.fg_hi
    b_idx = torch.nonzero(b_pool).squeeze(1)
    b_sel = torch.zeros(p0.shape, dtype=torch.bool, device=dev)
    if b_idx.numel() >= 5:
        n_keep = min(max(int(round(b_idx.numel() * args.keep_ratio)), 10),
                     b_idx.numel())
        b_sel[b_idx[torch.randperm(b_idx.numel(), device=dev)[:n_keep]]] = True
    else:
        b_sel[b_idx] = True

    c_pool = (p0 > args.c_lo) & (p0 <= args.tau_b)
    c_idx = torch.nonzero(c_pool).squeeze(1)
    c_sel = torch.zeros(p0.shape, dtype=torch.bool, device=dev)
    if c_idx.numel() >= 3 and b_sel.any():
        cd = uf['cd']
        dist, _ = cKDTree(cd[b_sel].cpu().numpy()).query(
            cd[c_idx].cpu().numpy(), k=1)
        cand = c_idx[torch.from_numpy(dist < args.near_r).to(dev)]
        if cand.numel():
            take = min(n_c, cand.numel())
            c_sel[cand[torch.randperm(cand.numel(), device=dev)[:take]]] = True

    lab = uf['yl']
    st = dict(b_prec=float(lab[b_sel].float().mean()) if b_sel.any() else 0.0,
              c_prec=float(lab[c_sel].float().mean()) if c_sel.any() else 0.0,
              n_b=int(b_sel.sum()), n_c=int(c_sel.sum()))
    return b_sel, c_sel, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=56)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--fg-hi', type=float, default=0.7)
    ap.add_argument('--bg-lo', type=float, default=0.2)
    ap.add_argument('--c-lo', type=float, default=0.3)
    ap.add_argument('--tau-b', type=float, default=0.7)
    ap.add_argument('--keep-ratio', type=float, default=0.35)
    ap.add_argument('--c-ratio', type=float, default=0.15)
    ap.add_argument('--near-r', type=float, default=0.15)
    ap.add_argument('--pick-mode', default='score', choices=['score', 'random'],
                    help='score=BridgeScorer 排序；random=同池随机（对照）')
    ap.add_argument('--out', default=os.path.join(CACHE, 'v3i'))
    args = ap.parse_args()

    eps = build_episodes(args.per_class, args.shard, args.shards)
    half = len(build_episodes(args.per_class)) // 2
    model, cfg = load_model()
    dev = torch.device('cuda:0')
    sco_a = load_scorer(os.path.join(CACHE, 'scorer_trainedB.pt'), dev)
    sco_b = load_scorer(os.path.join(CACHE, 'scorer_trainedA.pt'), dev)

    recs = []
    for ei, m in enumerate(eps):
        g_ep = args.shard + args.shards * ei
        seed_everything(10000 + g_ep)
        scorer = sco_a if g_ep < half else sco_b
        c = m['class']
        s = prepare_block(m['support'], cfg)
        q = prepare_block(m['query'], cfg)
        s_x, s_bin = episode_tensors(s, c)
        q_x, q_map = episode_tensors(q, c)
        sf = feats_one(model, s_x, s_bin, dev)
        qf = feats_one(model, q_x, q_map, dev)
        qf['seg_d'] = seg_downsample(qf['xyz'], q['seg'].to(dev), qf['cd'])
        fg_s = sf['yl'].bool()
        s_fg = build_proto(model, sf, fg_s)
        s_bg = build_proto(model, sf, ~fg_s)
        sparse_s = torch.cat([s_bg, s_fg], 0)

        pred_s, out_s = predict_agg(model, qf, [sparse_s], 0.0, dev)
        p0_q = Fs.softmax(out_s, dim=1)[:, 1]

        # rev_fg 必须按训练时的定义算：该块 B 原型在 S 真前景上的平均激活。
        # 直接传 0 会让 z-score 后的分布和训练时不一致。
        b_sp_q = torch.cat([build_proto(model, qf, p0_q < args.bg_lo),
                            build_proto(model, qf, p0_q > args.fg_hi)], 0)
        ps_q, _ = predict_agg(model, sf, [b_sp_q], 1.0, dev)
        ps_q = Fs.softmax(ps_q, dim=0)[1]
        rev_q = float(ps_q[s_bin > 0].mean()) if (s_bin > 0).any() else 0.0

        # 在查询集自己身上挑 B_Q / C_Q
        if args.pick_mode == 'random':
            b_sel, c_sel, st = pick_random(model, qf, p0_q, args, dev)
        else:
            b_sel, c_sel, w_full, st = pick_protos(
                model, qf, p0_q, s_fg, s_bg, rev_q, scorer, args, dev)

        n_c = int(round(K * args.c_ratio))
        n_b = K - n_c
        bg_p = build_proto(model, qf, p0_q < args.bg_lo)
        b_p = build_proto(model, qf, b_sel, n_b)
        c_p = build_proto(model, qf, c_sel, n_c)
        sparse_q = torch.cat([bg_p, b_p, c_p], 0)

        pred_l03, _ = predict_agg(model, qf, [sparse_s, sparse_q], 0.3, dev)
        pred_l05, _ = predict_agg(model, qf, [sparse_s, sparse_q], 0.5, dev)

        rec = dict(g_ep=g_ep, cls=c, b_prec=st['b_prec'], c_prec=st['c_prec'],
                   n_b=st['n_b'], n_c=st['n_c'], rev_q=rev_q)
        for key, pred in (('pred_s', pred_s), ('pred_q_l03', pred_l03),
                          ('pred_q_l05', pred_l05)):
            tp, fp, fn, iou = iou_counts(pred, q_map)
            rec[key] = iou
            rec[key + '_tp'] = tp
            rec[key + '_fp'] = fp
            rec[key + '_fn'] = fn
        recs.append(rec)

        if (ei + 1) % 20 == 0:
            print('  %d / %d' % (ei + 1, len(eps)), flush=True)

    torch.save(recs, args.out + '_s%d.pt' % args.shard)
    print('B_Q 精度 %.4f  C_Q 精度 %.4f  n_b %.0f n_c %.0f  (%d eps)' % (
        np.mean([r['b_prec'] for r in recs]),
        np.mean([r['c_prec'] for r in recs]),
        np.mean([r['n_b'] for r in recs]),
        np.mean([r['n_c'] for r in recs]), len(recs)))
    print('-> %s_s%d.pt' % (args.out, args.shard))


if __name__ == '__main__':
    main()
