"""生成干净的 pred_S / pred_U 缓存。

修复三处算法级问题 + 一处泄漏：

A. 不再把两个 U 块的 [2K, C] 原型矩阵按行平均。
   两个房间的 FPS 原型在物理部位和语义通道上没有对应关系，按行平均会
   糊掉特征、缩短 L2 模长、并让坏块污染好块。改为按块级可靠度择优。

B. B_clean 只在 p0 > fg_hi 的高置信池里按 scorer 取 Top keep_ratio；
   C_rec 单独从 0.3 < p0 <= 0.7 且贴近 B_clean 的点里取，最多占 c_ratio 槽位。
   不再把 p0 > 0.2 的一切混在一起按比例取。

C. 同时缓存两种 pred_U：
     solo  = predict_agg(qf, [sparse_u], lam=1.0)          无 S 锚定
     fused = predict_agg(qf, [sparse_s, sparse_u], lam=0.3/0.5/0.7)
   后者在相关性层融合，避免 HCA/KPConv 在没有 S 锚定时放大误差。

泄漏 2：每个 episode 用「没见过它」的那一折训练的 scorer 打分
   （g_ep 落在 A 折 → 用 B 折训练的 scorer；反之亦然）。
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as Fs
from scipy.spatial import cKDTree

from ocsb_core import (CACHE, K, build_episodes, build_proto, episode_tensors,
                       feats_one, load_model, predict_agg, prepare_block,
                       seed_everything, seg_downsample)
from v3_build_cache import max_cos, seg_stats
from train_bridge_scorer import BridgeScorer


def load_scorer(path, dev):
    s = BridgeScorer().to(dev)
    s.load_state_dict(torch.load(path, map_location=dev)['state'])
    s.eval()
    return s


def raw_feats(model, uf, p0, sel, s_fg, s_bg, rev_fg, dev):
    """对 sel 内的点算 10 维原始特征（未标准化）。"""
    f = uf['fd'][sel]
    sim_fg = max_cos(f, s_fg)
    sim_bg = max_cos(f, s_bg)
    sim_base = max_cos(f, model.base_prototypes)
    sp_mean, sp_std = seg_stats(uf['seg_d'], p0)
    sp_mean, sp_std = sp_mean[sel], sp_std[sel]

    core = p0 > 0.7
    if int(core.sum()) >= 3:
        sim_core = max_cos(f, build_proto(model, uf, core))
        d, _ = cKDTree(uf['cd'][core].cpu().numpy()).query(
            uf['cd'][sel].cpu().numpy(), k=1)
        dist_core = torch.from_numpy(d).to(dev).float()
    else:
        sim_core = torch.zeros(f.shape[0], device=dev)
        dist_core = torch.full((f.shape[0],), 10.0, device=dev)

    return torch.stack([p0[sel], sp_mean, sp_std, sim_fg, sim_bg, sim_base,
                        sim_fg - sim_bg, torch.full_like(sim_fg, rev_fg),
                        dist_core, sim_core], dim=1)


def pick_protos(model, uf, p0, s_fg, s_bg, rev_fg, scorer, args, dev):
    """按 B/C 两档构造前景掩码，返回 (fg_mask, stats)。

    B_clean 只在 p0 > fg_hi 里取；C_rec 只在模糊带且贴近 B_clean 里取。
    """
    n_c = int(round(K * args.c_ratio))
    sel = p0 > args.c_lo
    if int(sel.sum()) < 10:
        return p0 > args.fg_hi, torch.zeros(p0.shape, dtype=torch.bool,
                                            device=p0.device),             dict(b_prec=0.0, c_prec=0.0, n_b=0, n_c=0)

    cols = raw_feats(model, uf, p0, sel, s_fg, s_bg, rev_fg, dev)
    mu = cols.mean(0, keepdim=True)
    sd = cols.std(0, keepdim=True).clamp(min=1e-6)
    cols = (cols - mu) / sd
    with torch.no_grad():
        score = scorer(cols)                      # 对应 sel 内每个点

    b_in = (p0[sel] > args.fg_hi)
    c_in = (p0[sel] > args.c_lo) & (p0[sel] <= args.tau_b)

    # --- B_clean：高置信池内 Top keep_ratio ---
    b_sel = torch.zeros(p0.shape, dtype=torch.bool, device=p0.device)
    b_idx = torch.nonzero(sel).squeeze(1)[b_in]
    if b_idx.numel() >= 5:
        sc_b = score[b_in]
        n_keep = max(int(round(b_idx.numel() * args.keep_ratio)), 10)
        n_keep = min(n_keep, b_idx.numel())
        top = torch.topk(sc_b, n_keep).indices
        b_sel[b_idx[top]] = True
    else:
        b_sel[b_idx] = True

    # --- C_rec：模糊带里贴着 B_clean 的 ---
    c_sel = torch.zeros(p0.shape, dtype=torch.bool, device=p0.device)
    c_idx = torch.nonzero(sel).squeeze(1)[c_in]
    if c_idx.numel() >= 3 and b_sel.any():
        cd = uf['cd']
        dist, _ = cKDTree(cd[b_sel].cpu().numpy()).query(
            cd[c_idx].cpu().numpy(), k=1)
        near = torch.from_numpy(dist < args.near_r).to(dev)
        cand = c_idx[near]
        if cand.numel():
            sc_c = score[torch.isin(torch.nonzero(sel).squeeze(1), cand)]
            take = min(n_c, cand.numel())
            pick = torch.topk(sc_c, take).indices
            c_sel[cand[pick]] = True

    lab = uf['yl']
    stats = dict(
        b_prec=float(lab[b_sel].float().mean()) if b_sel.any() else 0.0,
        c_prec=float(lab[c_sel].float().mean()) if c_sel.any() else 0.0,
        n_b=int(b_sel.sum()), n_c=int(c_sel.sum()))
    return b_sel, c_sel, stats


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
    ap.add_argument('--fuse-lams', default='0.3,0.5,0.7')
    ap.add_argument('--out', default=os.path.join(CACHE, 'v3g'))
    args = ap.parse_args()

    eps = build_episodes(args.per_class, args.shard, args.shards)
    n_all = len(build_episodes(args.per_class))
    half = n_all // 2
    model, cfg = load_model()
    dev = torch.device('cuda:0')
    # 折外 scorer：g_ep < half 属 A 折 → 用 B 训练的
    sco_a = load_scorer(os.path.join(CACHE, 'scorer_trainedB.pt'), dev)
    sco_b = load_scorer(os.path.join(CACHE, 'scorer_trainedA.pt'), dev)
    fuse_lams = [float(x) for x in args.fuse_lams.split(',')]

    recs, stats_all = [], []
    for ei, m in enumerate(eps):
        g_ep = args.shard + args.shards * ei
        seed_everything(10000 + g_ep)
        scorer = sco_a if g_ep < half else sco_b
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

        blocks = []
        for u in us:
            u_x, u_bin = episode_tensors(u, c)
            uf = feats_one(model, u_x, u_bin, dev)
            uf['seg_d'] = seg_downsample(uf['xyz'], u['seg'].to(dev), uf['cd'])
            _, out0 = predict_agg(model, uf, [sparse_s], 0.0, dev)
            p0 = Fs.softmax(out0, dim=1)[:, 1]
            b_sp = torch.cat([build_proto(model, uf, p0 < args.bg_lo),
                              build_proto(model, uf, p0 > args.fg_hi)], 0)
            ps, _ = predict_agg(model, sf, [b_sp], 1.0, dev)
            ps = Fs.softmax(ps, dim=0)[1]
            rev = float(ps[s_bin > 0].mean()) if (s_bin > 0).any() else 0.0
            b_sel, c_sel, st = pick_protos(model, uf, p0, s_fg, s_bg, rev,
                                           scorer, args, dev)
            # 块级可靠度：B 池内 R_s 均值 + rev_fg
            sel = p0 > args.fg_hi
            if int(sel.sum()) >= 5:
                rs = max_cos(uf['fd'][sel], s_fg) - max_cos(uf['fd'][sel], s_bg)
                rel = float(rs.mean()) + rev
            else:
                rel = rev
            blocks.append(dict(uf=uf, p0=p0, b=b_sel, c=c_sel, rev=rev,
                               rel=rel, st=st))
            stats_all.append(st)

        # Bug A：按可靠度择优，而不是跨块按行平均
        best = max(range(len(blocks)), key=lambda i: blocks[i]['rel'])
        bst = blocks[best]
        n_c = int(round(K * args.c_ratio))
        n_b = K - n_c
        sparse_u = torch.cat([
            build_proto(model, bst['uf'], bst['p0'] < args.bg_lo),
            build_proto(model, bst['uf'], bst['b'], n_b),
            build_proto(model, bst['uf'], bst['c'], n_c)], 0)

        pred_s, out_s = predict_agg(model, qf, [sparse_s], 0.0, dev)
        pred_u_solo, _ = predict_agg(model, qf, [sparse_u], 1.0, dev)
        fused = {}
        for lam in fuse_lams:
            fused[lam], _ = predict_agg(model, qf, [sparse_s, sparse_u], lam, dev)

        p0s = Fs.softmax(pred_s, dim=0)[1]
        p0s_d = Fs.softmax(out_s, dim=1)[:, 1]
        p0u = Fs.softmax(pred_u_solo, dim=0)[1]
        fg_sel = p0s_d > 0.5
        sim_fg_q = (float(max_cos(qf['fd'][fg_sel], s_fg).mean())
                    if fg_sel.any() else 0.0)
        frac_q = float((p0s_d > args.fg_hi).float().mean())
        frac_s = float((s_bin > 0).float().mean())
        p0s_np = p0s.cpu().numpy()
        ent = float(-(p0s_np * np.log(np.clip(p0s_np, 1e-8, None))
                      + (1 - p0s_np) * np.log(np.clip(1 - p0s_np, 1e-8, None))).mean())
        x_task = np.array([sim_fg_q, frac_q, frac_s, frac_q / (frac_s + 1e-4),
                           ent, float(np.quantile(np.abs(p0s_np - 0.5), 0.9)),
                           float(np.mean([b['rev'] for b in blocks])),
                           float(np.mean([float((b['p0'] > args.fg_hi).float().mean())
                                          for b in blocks]))], dtype=np.float32)

        rec = dict(g_ep=g_ep, cls=c,
                   pred_s=pred_s.cpu().numpy().astype(np.float32),
                   pred_u_solo=pred_u_solo.cpu().numpy().astype(np.float32),
                   p0s=p0s.cpu().numpy().astype(np.float32),
                   p0u=p0u.cpu().numpy().astype(np.float32),
                   x_task=x_task,
                   gt=q_map.cpu().numpy().astype(np.int64),
                   best_block=best)
        for lam in fuse_lams:
            rec['pred_u_l%.1f' % lam] = fused[lam].cpu().numpy().astype(np.float32)
        recs.append(rec)

        if (ei + 1) % 20 == 0:
            print('  %d / %d' % (ei + 1, len(eps)), flush=True)

    torch.save(recs, args.out + '_s%d.pt' % args.shard)
    b = np.mean([s['b_prec'] for s in stats_all])
    c_ = np.mean([s['c_prec'] for s in stats_all])
    print('B_clean 精度 %.4f   C_rec 精度 %.4f   (%d eps)' % (b, c_, len(recs)))
    print('-> %s_s%d.pt' % (args.out, args.shard))


if __name__ == '__main__':
    main()
