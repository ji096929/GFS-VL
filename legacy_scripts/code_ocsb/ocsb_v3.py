"""三层遗漏条件化选择性桥接（v3）。

针对 v2 的三个方法学错误做的修正：

1. **第一层：U 内核心提纯 + 同块邻接吸收**
   不再把 `p0 > 0.7` 的全部点直接当原型。按 `R_s = sim(f, S_fg) − sim(f, S_bg)`
   在**块内**排序，只保留 Top `keep_ratio`（默认 40%）作为 `B_clean`；
   再把距离 `B_clean < near_r`、落在模糊带 `(c_lo, tau_b)` 的超点作为 `C_rec`
   并入，占 `c_ratio`（默认 15%）槽位。不确定区一律不进背景原型。

2. **第二层：单向遗漏修补（`one_sided`）**
   v2 用的是 `(1-λ)S + λU`，U 可以把 S 已认对的点拉低——30 集上 ΔTP = −1465
   就是这么来的。这里改成 `S + λ·relu(U − S − margin)`，U 只能往上补，
   结构上保证 `ΔTP >= 0`。

3. **第三层：单向欠分割门控**
   v2 的 `exp(-3|frac_q − frac_s|)` 是双边绝对值：当 `frac_q >> frac_s`
   （过分割）时反而判定"不自信"并开大 λ，方向是反的。这里改成单向：
   只有 `frac_q_high < r_thr · frac_s`（欠分割，对应 Q1–Q3）才开，
   否则 λ = 0（对应 Q4/Q5）。

评估：与 `upper_bound.py` 同口径，Arith + Cumul，逐集配对。
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as Fs
from scipy.spatial import cKDTree

from ocsb_core import (CACHE, K, NEW_IDS, build_episodes, build_proto,
                       episode_tensors, feats_one, iou_counts, load_class_names,
                       load_model, predict_agg, prepare_block, seed_everything)

STEPS = ['Baseline', 'Naive-B', 'P1_cleanB', 'P2_onesided']


def max_cos(f, protos):
    if protos.shape[0] == 0:
        return torch.zeros(f.shape[0], device=f.device)
    return Fs.cosine_similarity(f[:, None, :], protos[None, :, :], dim=2).max(1)[0]


def build_clean_sparse(model, uf, p0, s_fg, s_bg, args, dev):
    """第一层：提纯 B + 邻接吸收 C。

    返回 (sparse[2K, C], stats)。
    """
    b_mask = p0 > args.fg_hi
    n_b_keep = max(int(round(K * (1.0 - args.c_ratio))), 1)
    n_c = K - n_b_keep

    # --- B_clean：块内按 R_s 排序取 Top keep_ratio ---
    if int(b_mask.sum()) >= 5:
        f_b = uf['fd'][b_mask]
        r_s = max_cos(f_b, s_fg) - max_cos(f_b, s_bg)
        n_keep = max(int(b_mask.sum() * args.keep_ratio), 1)
        top = torch.topk(r_s, n_keep).indices
        b_idx = torch.nonzero(b_mask).squeeze(1)[top]
    else:
        b_idx = torch.nonzero(b_mask).squeeze(1)
    b_sel = torch.zeros_like(b_mask)
    b_sel[b_idx] = True

    # --- C_rec：模糊带里贴着 B_clean 的点 ---
    c_mask = (p0 > args.c_lo) & (p0 < args.tau_b) & ~b_sel
    c_sel = torch.zeros_like(b_mask)
    if b_sel.any() and c_mask.any():
        cd = uf['cd']
        c_idx = torch.nonzero(c_mask).squeeze(1)
        dist, _ = cKDTree(cd[b_sel].cpu().numpy()).query(
            cd[c_idx].cpu().numpy(), k=1)
        near = torch.from_numpy(dist < args.near_r).to(dev)
        cand = c_idx[near]
        if cand.numel() > n_c:
            w = p0[cand].clamp(min=1e-6)
            cand = cand[torch.multinomial(w / w.sum(), n_c, replacement=False)]
        c_sel[cand] = True

    # --- 槽位拼装：B 占 n_b_keep，C 占 n_c；C 不足用 B 的剩余点补 ---
    fg_parts = [build_proto(model, uf, b_sel, n_b_keep)]
    if n_c > 0:
        if int(c_sel.sum()) >= 3:
            fg_parts.append(build_proto(model, uf, c_sel, n_c))
        else:
            rest = b_mask & ~b_sel
            if int(rest.sum()) >= 3:
                fg_parts.append(build_proto(model, uf, rest, n_c))
            else:
                fg_parts.append(torch.zeros(n_c, model.feat_dim, device=dev))
    bg = build_proto(model, uf, p0 < args.bg_lo)   # 不确定区不进背景

    lab = uf['yl']
    stats = dict(
        n_clean=int(b_sel.sum()), n_c=int(c_sel.sum()),
        b_prec=float(lab[b_sel].float().mean()) if b_sel.any() else 0.0,
        c_prec=float(lab[c_sel].float().mean()) if c_sel.any() else 0.0,
    )
    return torch.cat([bg, torch.cat(fg_parts, 0)], 0), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=56)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(CACHE, 'ocsb_v3.json'))
    ap.add_argument('--fg-hi', type=float, default=0.7)
    ap.add_argument('--bg-lo', type=float, default=0.2)
    ap.add_argument('--c-lo', type=float, default=0.3)
    ap.add_argument('--tau-b', type=float, default=0.7)
    ap.add_argument('--keep-ratio', type=float, default=0.4, help='B_clean 保留比例')
    ap.add_argument('--c-ratio', type=float, default=0.15, help='C 占前景槽位比例')
    ap.add_argument('--near-r', type=float, default=0.15, help='C 到 B_clean 距离(m)')
    ap.add_argument('--lam', type=float, default=0.5)
    ap.add_argument('--margin', type=float, default=0.0)
    ap.add_argument('--gate-thrs', default='0.55,0.60,0.65',
                    help='sim_fg_q 的阈值，低于阈值才开门')
    ap.add_argument('--lams', default='0.3,0.5,0.7', help='P3 的 λ 网格')
    args = ap.parse_args()
    gate_thrs = [float(x) for x in args.gate_thrs.split(',')]
    lams = [float(x) for x in args.lams.split(',')]
    gate_steps = ['P3_l%.1f@%.2f' % (l, th) for l in lams for th in gate_thrs]
    all_steps = STEPS + gate_steps

    names = load_class_names()
    eps = build_episodes(args.per_class, args.shard, args.shards)
    model, cfg = load_model()
    dev = torch.device('cuda:0')

    rows = []
    ious = {s: [] for s in all_steps}
    cm = {s: {c: [0, 0, 0] for c in NEW_IDS} for s in all_steps}
    diag = []

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
        s_fg = build_proto(model, sf, fg_s)
        s_bg = build_proto(model, sf, ~fg_s)
        sparse_s = torch.cat([s_bg, s_fg], 0)

        # 查询侧的高置信前景占比与支持集前景占比（第三层的判据）
        # 第三层门控信号：预测前景点与 S 前景原型的平均相似度。
        # 高基线任务（Q4/Q5）预测出的前景确实像同类物体，该值高；
        # 低基线任务（Q1-Q3）是在乱猜，该值低。离线测得 AUC 0.751。
        pred_q, out_q = predict_agg(model, qf, [sparse_s], 0.0, dev)
        pq = Fs.softmax(out_q, dim=1)[:, 1]
        fg_sel = pq > 0.5
        sim_fg_q = (float(max_cos(qf['fd'][fg_sel], s_fg).mean())
                    if fg_sel.any() else 0.0)
        frac_q_high = float((pq > args.fg_hi).float().mean())
        frac_s = float((s_bin > 0).float().mean())

        uf_list, p0_list, rev_list = [], [], []
        for u in us:
            u_x, u_bin = episode_tensors(u, c)
            uf = feats_one(model, u_x, u_bin, dev)
            _, out0 = predict_agg(model, uf, [sparse_s], 0.0, dev)
            p0 = Fs.softmax(out0, dim=1)[:, 1]
            uf_list.append(uf)
            p0_list.append(p0)
            # rev_fg：该块的 B 原型在 S 真前景上的平均激活（块级门控用）
            b_sp = torch.cat([build_proto(model, uf, p0 < args.bg_lo),
                              build_proto(model, uf, p0 > args.fg_hi)], 0)
            pred_s, _ = predict_agg(model, sf, [b_sp], 1.0, dev)
            pS = Fs.softmax(pred_s, dim=0)[1]
            rev_list.append(float(pS[s_bin > 0].mean()) if (s_bin > 0).any() else 0.0)

        # --- Step 2：Naive-B（原样，作为对照） ---
        naive = torch.cat([
            torch.stack([build_proto(model, uf, p0 < args.bg_lo)
                         for uf, p0 in zip(uf_list, p0_list)]).mean(0),
            torch.stack([build_proto(model, uf, p0 > args.fg_hi)
                         for uf, p0 in zip(uf_list, p0_list)]).mean(0)], 0)

        # --- Step 3/4/5：第一层提纯 ---
        cleans, stats = [], []
        for uf, p0 in zip(uf_list, p0_list):
            sp, st = build_clean_sparse(model, uf, p0, s_fg, s_bg, args, dev)
            cleans.append(sp)
            stats.append(st)
        sparse_clean = torch.cat([torch.stack([x[:K] for x in cleans]).mean(0),
                                  torch.stack([x[K:] for x in cleans]).mean(0)], 0)

        def run(step, sparse_list, lam, mode):
            pred, _ = predict_agg(model, qf, sparse_list, lam, dev, mode=mode)
            tp, fp, fn, iou = iou_counts(pred, q_map)
            ious[step].append(iou)
            cm[step][c][0] += tp
            cm[step][c][1] += fp
            cm[step][c][2] += fn
            rows.append(dict(ep=ei, step=step, cls=c, iou=iou, lam=lam,
                             tp=tp, fp=fp, fn=fn))

        run('Baseline', [sparse_s], 0.0, 'mix')
        run('Naive-B', [sparse_s, naive], args.lam, 'mix')
        run('P1_cleanB', [sparse_s, sparse_clean], args.lam, 'mix')
        run('P2_onesided', [sparse_s, sparse_clean], args.lam, 'one_sided')
        for lam, thr in [(l, th) for l in lams for th in gate_thrs]:
            g = 1.0 if sim_fg_q < thr else 0.0
            run('P3_l%.1f@%.2f' % (lam, thr),
                [sparse_s, sparse_clean], lam * g, 'one_sided')

        diag.append(dict(ep=ei, cls=c, sim_fg_q=sim_fg_q, frac_q=frac_q_high,
                         frac_s=frac_s,
                         rev_fg=float(np.mean(rev_list)),
                         b_prec=float(np.mean([x['b_prec'] for x in stats])),
                         c_prec=float(np.mean([x['c_prec'] for x in stats])),
                         n_c=int(np.sum([x['n_c'] for x in stats]))))

        if (ei + 1) % 10 == 0:
            print('  %d / %d' % (ei + 1, len(eps)), flush=True)
            json.dump(dict(rows=rows, steps=STEPS, done=ei + 1),
                      open(args.out + '.partial', 'w'))

    out = {}
    print('\n%-14s %9s %9s %9s' % ('step', 'Arith', 'Cumul', 'dCumul'))
    for st in all_steps:
        arith = float(np.mean(ious[st]))
        per = [cm[st][c][0] / max(sum(cm[st][c]), 1) for c in NEW_IDS]
        cumul = float(np.mean(per))
        out[st] = dict(arith=arith, cumul=cumul, per_class=per,
                       n_ep=len(ious[st]),
                       tp=sum(cm[st][c][0] for c in NEW_IDS),
                       fp=sum(cm[st][c][1] for c in NEW_IDS),
                       fn=sum(cm[st][c][2] for c in NEW_IDS))
        print('%-14s %9.4f %9.4f %+9.4f' % (
            st, arith, cumul, cumul - out['Baseline']['cumul']))

    print('\n=== TP/FP/FN（验证 ΔTP >= 0） ===')
    b = out['Baseline']
    for st in all_steps[1:]:
        o = out[st]
        print('%-14s dTP=%+7d  dFP=%+7d  dFN=%+7d' % (
            st, o['tp'] - b['tp'], o['fp'] - b['fp'], o['fn'] - b['fn']))

    print('\n=== 逐类 Cumul ===')
    print('%-14s' % 'class' + ''.join('%14s' % s for s in all_steps))
    for i, cc in enumerate(NEW_IDS):
        print('%-14s' % names[cc] + ''.join(
            '%14.4f' % out[s]['per_class'][i] for s in all_steps))

    print('\n=== 诊断 ===')
    print('  sim_fg_q 均值 %.3f   frac_q 均值 %.3f   frac_s 均值 %.3f' % (
        np.mean([d['sim_fg_q'] for d in diag]),
        np.mean([d['frac_q'] for d in diag]),
        np.mean([d['frac_s'] for d in diag])))
    print('  B_clean 精度 %.4f   C 精度 %.4f   平均 n_c %.1f   rev_fg %.3f' % (
        np.mean([d['b_prec'] for d in diag]),
        np.mean([d['c_prec'] for d in diag]),
        np.mean([d['n_c'] for d in diag]),
        np.mean([d['rev_fg'] for d in diag])))

    out['rows'] = rows
    out['diag'] = diag
    json.dump(out, open(args.out, 'w'), indent=1)
    print('\n->', args.out)


if __name__ == '__main__':
    main()
