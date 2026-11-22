"""第二步：OASB + CSR 的零训练实现（五步消融）。

对应 `3D OCSB原始idea.md` 的式 (7)(8)(9)(10)(12)(13)(15)。

五步
----
Step 1  Baseline          仅 S 原型（bg K + fg K）
Step 2  Naive-B           S + U 中 prob > fg_hi 的全部点（现有可用方法）
Step 3  +OASB             S + [B 槽位 75% + 边缘恢复 C 槽位 25%]，Δ>0 即收
Step 4  +CSR              C 改用 V=Δ−η 过滤，B 按式(13) 的 w_c(B) 加权/剔除
Step 5  +λ_adap           λ 随 S 的置信度与 U 的净贡献动态变化

两段式结构
----------
`analyze_blocks()` 把所有需要的前向一次跑完（B 原型、bridged logits、
Δ、η），返回逐块的分析结果；`assemble()` 只负责按不同口径挑点、采原型。
Step 3 和 Step 4 共用同一份分析，差异只在过滤条件，不重复跑前向。

与原始公式的对应与简化
----------------------
- 超点级 → 点级：`getFeatures` 返回降采样点，而 `prepare_block` 的 `seg`
  只在原始分辨率。用 KDTree 把 seg 映射到降采样点，一个超点内的降采样
  点共享同一个 seg id；超点只用于分组加权，预测仍是点级。
- 单类别（1-way）：式(12) 的 argmax 多类竞争退化为二分类，P̄(c|C) 取
  前景 softmax 概率。
- Δ_c(B,C) 用完整前向的 logit 差：式(7) 定义的是"引入 B 后的类别得分
  变化"，只有过完 lin1/HCA/kpconv/cls 的 logit 才反映预测变化。多跑一次
  前向，代价可接受。
- 负对照 N_c：式(8) 要求来自真实支持点云，这里同时用支持集真实背景 D_S
  和 U 内高置信背景 D_U。D_S 是严格版本，D_U 是补充。

C 为什么要"贴近 B"
------------------
OASB 恢复的是边缘漏检：与已识别核心区相邻、被阈值切掉的前景。距离约束
r 一方面避免把远处无关区域拉进来，另一方面避免把整片背景当遗漏区。
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

STEPS = ['Baseline', 'Naive-B', 'OASB', 'CSR', 'OASB+CSR+adap']


def seg_downsample(xyz, seg, cd):
    """把原始分辨率的超点 id 映射到降采样点：取最近原始点的 seg。"""
    if xyz.shape[0] == cd.shape[0]:
        return seg
    tree = cKDTree(xyz.detach().cpu().numpy())
    _, idx = tree.query(cd.detach().cpu().numpy())
    return torch.from_numpy(np.asarray(seg.cpu().numpy())[idx]).to(cd.device)


def weighted_pick(scores, mask, n):
    """在 mask 内按 scores 加权无放回抽至多 n 个索引。"""
    idx = torch.nonzero(mask).squeeze(1)
    if idx.numel() <= n:
        return idx
    w = scores[idx].clamp(min=1e-6)
    return idx[torch.multinomial(w / w.sum(), n, replacement=False)]


def analyze_blocks(model, u_feats, sparse_s, args, dev):
    """对每个 U 块做一次完整分析，返回逐块结果供 assemble 复用。

    每块的字段：
      p0      初始前景概率（S 原型）
      b_mask  prob > fg_hi 的桥梁核心
      d_mask  高置信背景，负对照
      delta   Δ_c(B,C) 逐点（式 7）
      eta     风险敏感负对照项（式 8）
      keep3   OASB 的 C 掩码：未识别 + 贴近 B + Δ>0
      keep4   CSR 的 C 掩码：未识别 + 贴近 B + V>0
      w_c     恢复可靠性（式 12）
      w_b     桥梁权重（式 13）
    """
    infos = []
    for uf in u_feats:
        _, out0 = predict_agg(model, uf, [sparse_s], 0.0, dev)
        p0 = Fs.softmax(out0, dim=1)[:, 1]
        b_mask = p0 > args.fg_hi
        d_mask = p0 < args.bg_lo
        # C 只取"模糊带"：高置信背景不算遗漏，低置信背景也不是漏检
        c_mask = (p0 > args.c_lo) & (p0 < args.tau_b)

        # B 必须构成完整的一路参考集（bg K + fg K），融合才与式(15) 对应：
        # 式(7) 的 Δ 是"把 B 这路知识按最终 λ 融进来"造成的得分变化
        b_sparse = torch.cat([build_proto(model, uf, d_mask),
                              build_proto(model, uf, b_mask)], 0)
        _, out_b = predict_agg(model, uf, [sparse_s, b_sparse], args.lam, dev)
        delta = out_b[:, 1] - out0[:, 1]

        dpos = delta[d_mask].clamp(min=0)
        eta = torch.log(dpos.exp().mean() + 1e-8) if dpos.numel() else \
            torch.zeros((), device=dev)

        # 距离约束：C 必须贴近 B 核心
        near = torch.zeros_like(c_mask)
        if b_mask.any() and c_mask.any():
            cd = uf['cd']
            b_xyz = cd[b_mask].cpu().numpy()
            c_idx = torch.nonzero(c_mask).squeeze(1)
            dist, _ = cKDTree(b_xyz).query(cd[c_idx].cpu().numpy(), k=1)
            near[c_idx[torch.from_numpy(dist < args.near_r).to(dev)]] = True

        v = delta - eta
        keep3 = c_mask & near & (delta > 0)
        keep4 = c_mask & near & (v > 0)
        prob_b = Fs.softmax(out_b, dim=1)[:, 1]
        g = v.clamp(min=0)
        w_c = torch.where(keep4, prob_b * g / (1.0 + g), torch.zeros_like(g))

        # 式(13)：B 的权重由它救活的 C 决定，而不是它自己的置信度。
        # 做真正的 B–C 邻接：B 周围没有可靠 C 时 w_b → 0，直接被剔除。
        w_b = torch.zeros_like(p0)
        b_idx = torch.nonzero(b_mask).squeeze(1)
        k_idx = torch.nonzero(keep4).squeeze(1)
        if b_idx.numel() and k_idx.numel():
            cd = uf['cd']
            wc_np = w_c[k_idx].detach().cpu().numpy()
            tot = float(wc_np.sum()) + 1e-8
            nb = cKDTree(cd[k_idx].cpu().numpy()).query_ball_point(
                cd[b_idx].cpu().numpy(), r=args.near_r)
            share = np.array([wc_np[list(ix)].sum() if len(ix) else 0.0
                              for ix in nb]) / tot
            w_b[b_idx] = p0[b_idx] * torch.from_numpy(share).to(dev).float()

        # 诊断：C 里有多少真的是前景、B 的精度、C 能覆盖多少真前景
        lab = uf['yl'].bool()
        n_lab = max(int(lab.sum()), 1)
        infos.append(dict(
            uf=uf, p0=p0, b_mask=b_mask, d_mask=d_mask,
            delta=delta, eta=float(eta), keep3=keep3, keep4=keep4,
            w_c=w_c, w_b=w_b, near=float(near.float().mean()),
            b_prec=float((lab & b_mask).sum()) / max(int(b_mask.sum()), 1),
            c_prec=float((lab & keep4).sum()) / max(int(keep4.sum()), 1),
            c_rec=float((lab & keep4).sum()) / n_lab,
            c3_prec=float((lab & keep3).sum()) / max(int(keep3.sum()), 1),
        ))
    return infos


def assemble(model, infos, args, mode):
    """按口径拼出 (bg K + fg K) 的 sparse。

    mode='naive' : 前景 = prob>fg_hi；背景 = prob<bg_lo（现有基线方法）
    mode='oasb'  : 前景 = B 槽 75% + C 槽 25%，C 由 Δ>0 选
    mode='csr'   : 前景 = B 槽 75% + C 槽 25%，C 由 V>0 选，B 按 w_b 加权
    """
    n_c = int(round(K * args.c_ratio))
    n_b = K - n_c
    bgs, fgs, stats = [], [], dict(n_b=0, n_c=0, eta=0.0, dmean=0.0, wsum=0.0)

    for inf in infos:
        uf = inf['uf']
        if mode == 'naive':
            bgs.append(build_proto(model, uf, inf['d_mask']))
            fgs.append(build_proto(model, uf, inf['p0'] > args.fg_hi))
            stats['n_b'] += int((inf['p0'] > args.fg_hi).sum())
            continue

        bgs.append(build_proto(model, uf, inf['d_mask']))
        keep = inf['keep3'] if mode == 'oasb' else inf['keep4']

        # B 槽位：CSR 时按 w_b 加权抽，OASB 时直接按 p0 抽
        b_scores = inf['w_b'] if mode == 'csr' else inf['p0']
        b_pick = weighted_pick(b_scores, inf['b_mask'], n_b)
        b_sel = torch.zeros_like(inf['b_mask'])
        b_sel[b_pick] = True

        # C 槽位：按 w_c 加权抽；C 不足时用 B 的点补齐，保持原型数恒为 K
        c_scores = inf['w_c'] if mode == 'csr' else inf['delta'].clamp(min=0)
        c_pick = weighted_pick(c_scores, keep, n_c)
        if c_pick.numel() < n_c:
            pad = weighted_pick(b_scores, inf['b_mask'] & ~b_sel, n_c - c_pick.numel())
            c_pick = torch.cat([c_pick, pad])
        c_sel = torch.zeros_like(inf['b_mask'])
        c_sel[c_pick] = True

        fgs.append(torch.cat([build_proto(model, uf, b_sel, n_b),
                              build_proto(model, uf, c_sel, n_c)], 0))
        stats['n_b'] += int(b_sel.sum())
        stats['n_c'] += int(keep.sum())
        stats['eta'] += inf['eta']
        stats['dmean'] += float(inf['delta'][keep].mean()) if keep.any() else 0.0
        stats['wsum'] += float(inf['w_c'].sum())

    n = max(len(infos), 1)
    for k in ('eta', 'dmean', 'wsum'):
        stats[k] /= n
    return torch.cat([torch.stack(bgs).mean(0), torch.stack(fgs).mean(0)], 0), stats


def lambda_adaptive(args, conf_s, g_u):
    """自适应融合权重。

    两个因子相乘：
      1 - conf_s   S 越自信越不需要 U（Q5 的负增益靠这里压掉）
      g_u/(κ+g_u)  U 里可靠桥梁越多给得越多；没有桥梁时退化为 Baseline
    lam_max 封顶，避免单边放大。
    """
    return args.lam_max * max(0.0, 1.0 - conf_s) * (g_u / (args.kappa + g_u))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=10)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(CACHE, 'ocsb_advanced.json'))
    ap.add_argument('--fg-hi', type=float, default=0.7)
    ap.add_argument('--bg-lo', type=float, default=0.2)
    ap.add_argument('--tau-b', type=float, default=0.5, help='C 候选的概率上限')
    ap.add_argument('--c-lo', type=float, default=0.2, help='C 候选的概率下限（排除背景）')
    ap.add_argument('--near-r', type=float, default=0.3, help='C/B 邻接的距离上限(m)')
    ap.add_argument('--c-ratio', type=float, default=0.25, help='前景槽位给 C 的比例')
    ap.add_argument('--lam', type=float, default=0.7, help='Step 2/3/4 的固定 λ')
    ap.add_argument('--lam-max', type=float, default=0.7, help='λ_adap 上限')
    ap.add_argument('--kappa', type=float, default=0.1)
    args = ap.parse_args()

    names = load_class_names()
    eps = build_episodes(args.per_class, args.shard, args.shards)
    model, cfg = load_model()
    dev = torch.device('cuda:0')

    rows = []
    ious = {s: [] for s in STEPS}
    cm = {s: {c: [0, 0, 0] for c in NEW_IDS} for s in STEPS}
    lam_used = {s: [] for s in STEPS}
    diag_rows = []

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

        u_feats = []
        for u in us:
            u_x, u_bin = episode_tensors(u, c)
            uf = feats_one(model, u_x, u_bin, dev)
            uf['seg_d'] = seg_downsample(uf['xyz'], u['seg'].to(dev), uf['cd'])
            u_feats.append(uf)

        infos = analyze_blocks(model, u_feats, sparse_s, args, dev)
        sp_naive, st_n = assemble(model, infos, args, 'naive')
        sp_oasb, st_o = assemble(model, infos, args, 'oasb')
        sp_csr, st_c = assemble(model, infos, args, 'csr')

        # conf_s：三个候选，先都记下来看谁能区分难易任务
        pred0, _ = predict_agg(model, qf, [sparse_s], 0.0, dev)
        pq = Fs.softmax(pred0, dim=0)[1]      # pred0 是 [2, N]，类别维是 dim=0
        frac_q = float((pq > 0.5).float().mean())
        frac_s = float((s_bin > 0).float().mean())
        conf_a = float(torch.maximum(pq, 1 - pq).mean())      # 全点裕度（实测恒 1）
        sel = pq[pq > 0.5]
        conf_b = float(sel.mean()) if sel.numel() else 0.0     # 预测前景点的均值
        conf_c = float(np.exp(-3.0 * abs(frac_q - frac_s)))    # 前景占比一致性
        lam_a = lambda_adaptive(args, conf_c, st_c['wsum'])

        def run(step, sparse_list, lam):
            pred, _ = predict_agg(model, qf, sparse_list, lam, dev)
            tp, fp, fn, iou = iou_counts(pred, q_map)
            ious[step].append(iou)
            cm[step][c][0] += tp
            cm[step][c][1] += fp
            cm[step][c][2] += fn
            lam_used[step].append(lam)
            rows.append(dict(ep=ei, step=step, cls=c, iou=iou,
                             lam=lam, tp=tp, fp=fp, fn=fn))

        run('Baseline', [sparse_s], 0.0)
        run('Naive-B', [sparse_s, sp_naive], args.lam)
        run('OASB', [sparse_s, sp_oasb], args.lam)
        run('CSR', [sparse_s, sp_csr], args.lam)
        run('OASB+CSR+adap', [sparse_s, sp_csr], lam_a)

        diag_rows.append(dict(
            ep=ei, cls=c, lam_adap=lam_a,
            conf_a=conf_a, conf_b=conf_b, conf_c=conf_c,
            frac_q=frac_q, frac_s=frac_s,
            **{'n_b': st_c['n_b'], 'n_c': st_c['n_c'],
               'eta': st_c['eta'], 'dmean': st_c['dmean'],
               'wsum': st_c['wsum'],
               'b_prec': float(np.mean([i['b_prec'] for i in infos])),
               'c_prec': float(np.mean([i['c_prec'] for i in infos])),
               'c3_prec': float(np.mean([i['c3_prec'] for i in infos])),
               'c_rec': float(np.mean([i['c_rec'] for i in infos]))}))
        if (ei + 1) % 10 == 0:
            print('  %d / %d' % (ei + 1, len(eps)), flush=True)
            json.dump(dict(rows=rows, done=ei + 1, steps=STEPS),
                      open(args.out + '.partial', 'w'))

    out = {}
    print('\n%-18s %9s %9s %9s %9s' % ('step', 'Arith', 'Cumul', 'dCumul', 'mean λ'))
    for st in STEPS:
        arith = float(np.mean(ious[st]))
        per = [cm[st][c][0] / max(sum(cm[st][c]), 1) for c in NEW_IDS]
        cumul = float(np.mean(per))
        out[st] = dict(arith=arith, cumul=cumul, per_class=per, n_ep=len(ious[st]))
        print('%-18s %9.4f %9.4f %+9.4f %9.3f' % (
            st, arith, cumul, cumul - out['Baseline']['cumul'],
            float(np.mean(lam_used[st]))))

    print('\n逐类 Cumul')
    print('%-14s' % 'class' + ''.join('%18s' % s for s in STEPS))
    for i, cc in enumerate(NEW_IDS):
        print('%-14s' % names[cc] + ''.join(
            '%18.4f' % out[s]['per_class'][i] for s in STEPS))

    print('\n诊断（均值）：')
    print('  n_b=%.1f  n_c=%.1f  η=%.3f  Δ=%.3f  Σw_c=%.2f' % (
        np.mean([d['n_b'] for d in diag_rows]),
        np.mean([d['n_c'] for d in diag_rows]),
        np.mean([d['eta'] for d in diag_rows]),
        np.mean([d['dmean'] for d in diag_rows]),
        np.mean([d['wsum'] for d in diag_rows])))
    print('  精度：B=%.3f  C(Δ>0)=%.3f  C(V>0)=%.3f  C 覆盖真前景=%.3f' % (
        np.mean([d['b_prec'] for d in diag_rows]),
        np.mean([d['c3_prec'] for d in diag_rows]),
        np.mean([d['c_prec'] for d in diag_rows]),
        np.mean([d['c_rec'] for d in diag_rows])))
    print('  conf 候选：a=%.3f  b=%.3f  c=%.3f   λ_adap(用 c)=%.3f' % (
        np.mean([d['conf_a'] for d in diag_rows]),
        np.mean([d['conf_b'] for d in diag_rows]),
        np.mean([d['conf_c'] for d in diag_rows]),
        np.mean([d['lam_adap'] for d in diag_rows])))

    out['rows'] = rows
    out['diags'] = diag_rows
    json.dump(out, open(args.out, 'w'), indent=1)
    print('\n->', args.out)


if __name__ == '__main__':
    main()
