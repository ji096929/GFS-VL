"""OCSB 第一步：外部池 U 的四种用法能带来多少上限。

四行对照，全部零训练，只改送进相关性网络的原型来源：

    Baseline      仅 support 的 1-shot 原型
    Naive-B       support + U 中「预测为前景」的点（含假阳性，最贴近可用方法）
    Oracle-B      support + U 中「预测为前景 且 真值是前景」的点（纯桥梁 B_pos）
    Oracle-B+C    support + U 中「真值是前景」的全部点（桥梁 + 遗漏区 C_pos）

每行再扫 λ（U 原型的融合权重），得到完整的性能-λ 曲线。
λ=0 时 Naive-B / Oracle-* 三行严格退化成 Baseline，可作为一种自检。

评估口径
--------
- Arith : 每个 episode 先算 IoU 再平均（逐集等权）
- Cumul : 先把 10 个类的 TP/FP/FN 累加，再算每类 IoU 后平均（类间等权，
          与 COSeg 官方报告口径一致）
- 所有 Δ 都是同 episode 配对差，t 值是配对 t 统计量

复现性
------
每个 episode 前用 `10000 + g_ep` 重置种子。`voxelize(mode=0)` 会随机挑
体素代表点，不锁种子的话跨进程 Baseline 能差 18pp（详见 ocsb_core 的说明）。
g_ep 用全局编号，所以结果与分片方式无关。
"""
import argparse
import json
import os

import torch
import torch.nn.functional as Fs

from ocsb_core import (CACHE, NEW_IDS, build_episodes, build_proto,
                       build_u_sparse, episode_tensors, feats_one, iou_counts,
                       load_class_names, load_model, predict_agg,
                       prepare_block, seed_everything)

CASES_U = [('naive', 'Naive-B'), ('oracle_b', 'Oracle-B'),
           ('oracle_bc', 'Oracle-B+C')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=10, help='每类取前 N 个 episode')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--lams', default='0.2,0.3,0.5', help='逗号分隔的 λ 网格')
    ap.add_argument('--fg-hi', type=float, default=0.7, help='前景候选的置信阈')
    ap.add_argument('--bg-lo', type=float, default=0.2, help='高置信背景的置信阈')
    ap.add_argument('--out', default=os.path.join(CACHE, 'upper_bound.json'))
    ap.add_argument('--min-gt', type=float, default=0.0,
                    help='只保留 query 目标占比 >= 该值的 episode；'
                         '官方 get_class2scans 的阈值是 0.05')
    args = ap.parse_args()

    names = load_class_names()
    # 无论是否过滤，都用完整的 539 集列表来定位 episode，
    # 这样 g_ep 仍是原始全局编号，seed_everything(10000+g_ep) 与分片方式无关。
    eps_all = build_episodes(args.per_class)
    if args.min_gt > 0:
        gt_cache = os.path.join(CACHE, 'val_query_gt.json')

        def _key(m):
            # 同一个 block 会被不同类的 episode 复用（539 集里有 60 个重复），
            # 只用 block 名会让跨类占比互相覆盖，必须带上 class。
            return '%s|%d' % (m['query'], m['class'])

        gtr = json.load(open(gt_cache)) if os.path.exists(gt_cache) else {}
        missing = [m for m in eps_all if _key(m) not in gtr]
        if missing:
            import numpy as np
            from ring4_isolated import DATA as BLK
            for m in missing:
                lab = np.asarray(np.load(
                    os.path.join(BLK, m['query'] + '.npy'),
                    mmap_mode='r')[:, 6]).astype(int)
                gtr[_key(m)] = float((lab == m['class']).mean())
            json.dump(gtr, open(gt_cache, 'w'))
        keep = [i for i, m in enumerate(eps_all) if gtr[_key(m)] >= args.min_gt]
    else:
        keep = list(range(len(eps_all)))
    todo = keep[args.shard::args.shards]
    print('episodes %d / %d  (per_class=%d  min_gt=%.2f  shard=%d/%d)' % (
        len(todo), len(eps_all), args.per_class, args.min_gt,
        args.shard, args.shards), flush=True)

    model, cfg = load_model()
    dev = torch.device('cuda:0')

    lams = [float(x) for x in args.lams.split(',')]
    cases = ['Baseline'] + ['%s_l%.1f' % (b, l) for l in lams for _, b in CASES_U]
    rows = []
    ious = {k: [] for k in cases}
    cm = {k: {c: [0, 0, 0] for c in NEW_IDS} for k in cases}   # 每类的 TP/FP/FN
    n_bsp = {k: [] for k in cases}                             # 每集进了多少前景点

    for ei, g_ep in enumerate(todo):
        m = eps_all[g_ep]
        seed_everything(10000 + g_ep)

        c = m['class']
        s = prepare_block(m['support'], cfg)
        q = prepare_block(m['query'], cfg)
        us = [prepare_block(nm, cfg) for nm in m['u']]

        s_x, s_bin = episode_tensors(s, c)
        q_x, q_map = episode_tensors(q, c)

        # ---- support 侧：固定的 1-shot 原型（背景 K 个 + 前景 K 个）----
        sf = feats_one(model, s_x, s_bin, dev)
        qf = feats_one(model, q_x, q_map, dev)
        fg_s = sf['yl'].bool()
        sparse_s = torch.cat([build_proto(model, sf, ~fg_s),
                              build_proto(model, sf, fg_s)], 0)

        # ---- U 侧：每个块独立过一遍，用 S 原型拿到初始前景概率 ----
        # 每块单独 prepare/提特征，绝不能把多个块减质心后拼成一个 offset
        u_feats = []
        for u in us:
            u_x, u_bin = episode_tensors(u, c)
            uf = feats_one(model, u_x, u_bin, dev)
            _, u_out = predict_agg(model, uf, [sparse_s], 0.0, dev)
            uf['prob'] = Fs.softmax(u_out, dim=1)[:, 1].cpu()   # 降采样分辨率
            u_feats.append(uf)

        # ---- 三种 U 口径的原型 ----
        # 注意 n_bsp 的 key 是「带 λ 的 case 名」，所以每个 U 口径的前景点数
        # 要记到它展开后的每一行上，不能直接写 n_bsp['Naive-B']。
        sparse_u, nfg_of = {}, {}
        for mode, base in CASES_U:
            sparse_u[base], nfg_of[base] = build_u_sparse(
                model, u_feats, mode, args.fg_hi, args.bg_lo)

        def run(case, sparse_list, lam, nfg=0):
            """跑一行并记录逐集 IoU 与逐类 TP/FP/FN。"""
            pred, _ = predict_agg(model, qf, sparse_list, lam, dev)
            tp, fp, fn, iou = iou_counts(pred, q_map)
            ious[case].append(iou)
            n_bsp[case].append(nfg)
            cm[case][c][0] += tp
            cm[case][c][1] += fp
            cm[case][c][2] += fn
            rows.append(dict(ep=g_ep, case=case, lam=lam, cls=c,
                             iou=iou, tp=tp, fp=fp, fn=fn))

        run('Baseline', [sparse_s], 0.0)
        for lam in lams:
            for _, b in CASES_U:
                run('%s_l%.1f' % (b, lam), [sparse_s, sparse_u[b]], lam,
                    nfg_of[b])

        # 每 10 个 episode 落一次盘，进程被杀也不至于全丢
        if (ei + 1) % 10 == 0:
            print('  %d / %d' % (ei + 1, len(todo)), flush=True)
            json.dump(dict(rows=rows, lams=lams, done=ei + 1, n_ep=len(todo)),
                      open(args.out + '.partial', 'w'))

    # ---- 汇总 ----
    out = {}
    print('\n%-14s %10s %10s %10s' % ('case', 'Arith', 'Cumul', 'mean n_B'))
    for k in cases:
        arith = float(sum(ious[k]) / max(len(ious[k]), 1))
        per = [cm[k][c][0] / max(sum(cm[k][c]), 1) for c in NEW_IDS]
        cumul = float(sum(per) / len(per))
        nb = float(sum(n_bsp[k]) / len(n_bsp[k])) if n_bsp[k] else 0
        out[k] = dict(arith=arith, cumul=cumul, per_class=per,
                      mean_n_b=nb, n_ep=len(ious[k]))
        print('%-14s %10.4f %10.4f %10.0f' % (k, arith, cumul, nb))

    print('\n逐类 Cumul IoU')
    print('%-14s' % 'class' + ''.join('%14s' % k for k in cases))
    for i, c in enumerate(NEW_IDS):
        print('%-14s' % names[c] + ''.join(
            '%14.4f' % out[k]['per_class'][i] for k in cases))

    out['rows'] = rows
    out['lams'] = lams
    json.dump(out, open(args.out, 'w'), indent=1)
    print('\n->', args.out)


if __name__ == '__main__':
    main()
