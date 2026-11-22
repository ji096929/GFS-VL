"""跨块互验证：把"考官"换成没参与选出候选点的独立块。

为什么需要
----------
`proto_B -> S` 与 `S -> B` 用的是同一组余弦矩阵，只是聚合方向不同，
信息高度重叠（实测 AUC 0.626 vs 0.624），无法打破对称。

三个独立考官
------------
1. U1 <-> U2：候选点由 S 在 U1 上选出，用**另一个无标签块 U2** 的
   高置信前景/背景原型去考。真前景是跨房间共有的目标，单房间偶发的
   连体假阳性在 U2 上难以同时匹配。
2. Um <-> Q：用查询块 Q 自身在 S 下的高置信前景/背景原型去考。
3. Leave-One-Support-Out：用**同类别另一个 episode 的支持块 S2**（100%
   真值，且没参与选出 B）去考。这一条在 1-shot 协议下用了额外标注，
   只作为"独立考官能达到什么水平"的上限验证。

每条都同时给特征级（余弦差）和完整管道级（predict_agg 反向）两个版本。
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as Fs
from sklearn.metrics import roc_auc_score

from ocsb_core import (CACHE, build_episodes, build_proto, episode_tensors,
                       feats_one, iou_counts, load_model, predict_agg,
                       prepare_block, seed_everything)


def max_cos(f, protos):
    """f: (N,C) -> (N,) 每个点对原型集的最大余弦。"""
    if protos.shape[0] == 0:
        return torch.zeros(f.shape[0], device=f.device)
    return Fs.cosine_similarity(f[:, None, :], protos[None, :, :], dim=2).max(1)[0]


def p0_of(model, f, sparse_s, dev):
    _, out = predict_agg(model, f, [sparse_s], 0.0, dev)
    return Fs.softmax(out, dim=1)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=10)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--fg-hi', type=float, default=0.7)
    ap.add_argument('--bg-lo', type=float, default=0.2)
    ap.add_argument('--q-fg', type=float, default=0.8)
    ap.add_argument('--q-bg', type=float, default=0.1)
    ap.add_argument('--out', default=os.path.join(CACHE, 'cross_block.json'))
    args = ap.parse_args()

    eps = build_episodes(args.per_class, args.shard, args.shards)
    by_class = defaultdict(list)
    for i, m in enumerate(eps):
        by_class[m['class']].append(i)
    model, cfg = load_model()
    dev = torch.device('cuda:0')

    sig = defaultdict(list)
    labels = []

    for ei, m in enumerate(eps):
        g_ep = args.shard + args.shards * ei
        seed_everything(10000 + g_ep)
        c = m['class']
        s = prepare_block(m['support'], cfg)
        q = prepare_block(m['query'], cfg)
        us = [prepare_block(nm, cfg) for nm in m['u']]
        s_x, s_bin = episode_tensors(s, c)
        q_x, q_map = episode_tensors(q, c)
        sf = feats_one(model, s_x, s_bin, dev)
        qf = feats_one(model, q_x, q_map, dev)
        fg_s = sf['yl'].bool()
        sparse_s = torch.cat([build_proto(model, sf, ~fg_s),
                              build_proto(model, sf, fg_s)], 0)

        # 查询块的高置信前景/背景原型（考官 2）
        pq = p0_of(model, qf, sparse_s, dev)
        q_fg_p = build_proto(model, qf, pq > args.q_fg)
        q_bg_p = build_proto(model, qf, pq < args.q_bg)

        # 同类别另一个 episode 的支持块（考官 3）
        s2 = None
        others = [j for j in by_class[c] if j != ei]
        if others:
            j = others[ei % len(others)]
            s2 = prepare_block(eps[j]['support'], cfg)
            s2_x, s2_bin = episode_tensors(s2, c)
            sf2 = feats_one(model, s2_x, s2_bin, dev)
            fg2 = sf2['yl'].bool()
            s2_fg_p = build_proto(model, sf2, fg2)
            s2_bg_p = build_proto(model, sf2, ~fg2)

        uf_list, p0_list = [], []
        for u in us:
            u_x, u_bin = episode_tensors(u, c)
            uf = feats_one(model, u_x, u_bin, dev)
            uf_list.append(uf)
            p0_list.append(p0_of(model, uf, sparse_s, dev))

        for k, uf in enumerate(uf_list):
            b_mask = p0_list[k] > args.fg_hi
            if int(b_mask.sum()) < 10:
                continue
            f_b = uf['fd'][b_mask]
            lab = uf['yl'][b_mask].float()

            # --- 考官 1：另一个无标签块 ---
            j2 = 1 - k if len(uf_list) > 1 else None
            if j2 is not None:
                other_fg = build_proto(model, uf_list[j2], p0_list[j2] > args.fg_hi)
                other_bg = build_proto(model, uf_list[j2], p0_list[j2] < args.bg_lo)
                s_u2 = max_cos(f_b, other_fg) - max_cos(f_b, other_bg)
            else:
                s_u2 = torch.zeros_like(lab)

            # --- 考官 2：查询块 ---
            s_q = max_cos(f_b, q_fg_p) - max_cos(f_b, q_bg_p)

            # --- 考官 3：独立支持块（真值） ---
            if s2 is not None:
                s_ho = max_cos(f_b, s2_fg_p) - max_cos(f_b, s2_bg_p)
                b_sparse = torch.cat([build_proto(model, uf, p0_list[k] < args.bg_lo),
                                      build_proto(model, uf, b_mask)], 0)
                pred_s2, _ = predict_agg(model, sf2, [b_sparse], 1.0, dev)
                _, _, _, rev_ho = iou_counts(pred_s2, s2_bin)
            else:
                s_ho = torch.zeros_like(lab)
                rev_ho = 0.0

            sig['u2'].append(s_u2.detach().cpu())
            sig['q'].append(s_q.detach().cpu())
            sig['holdout'].append(s_ho.detach().cpu())
            sig['rev_holdout'].append(torch.full_like(lab.cpu(), float(rev_ho)))
            sig['p0'].append(p0_list[k][b_mask].detach().cpu())
            labels.append(lab.detach().cpu())

        if (ei + 1) % 10 == 0:
            print('  %d / %d' % (ei + 1, len(eps)), flush=True)

    cat = {k: torch.cat(v).numpy() for k, v in sig.items()}
    y = torch.cat(labels).numpy()
    print('\nB 点总数 %d  基准精度 %.4f' % (len(y), y.mean()))
    print('\n%-14s %8s %11s %11s %11s' % (
        '考官', 'AUC', 'Top50%', 'Top30%', 'Top10%'))
    for k, name in (('p0', 'p0（同源基线）'),
                    ('u2', 'U1<->U2'),
                    ('q', 'Um<->Q'),
                    ('holdout', 'LOO 支持块（特征）'),
                    ('rev_holdout', 'LOO 支持块（管道）')):
        v = cat[k]
        try:
            auc = float(roc_auc_score(y, v))
        except ValueError:
            auc = float('nan')
        line = []
        for frac in (0.5, 0.3, 0.1):
            kk = max(int(len(v) * frac), 1)
            idx = np.argsort(-v)[:kk]
            line.append(float(y[idx].mean()))
        print('%-14s %8.4f %11.4f %11.4f %11.4f' % (name, auc, *line))

    out = {k: cat[k].tolist() for k in cat}
    out['y'] = y.tolist()
    out['base_prec'] = float(y.mean())
    json.dump(out, open(args.out, 'w'))
    print('\n->', args.out)


if __name__ == '__main__':
    main()
