"""生成 pred_S / pred_U 缓存（含连续权重与单向保护）。

相比 v3g_build_cache 的增量：
1. 保留 BridgeScorer 的连续权重：每个原型槽位的权重取簇内平均 w_u，
   而不是把选中的点一律当 1.0。权重通过 `proto_weights` 乘在相关性上。
2. 同时生成单向保护版本（mode='one_sided'），对应 P2。
3. 门控（P3）留给下游：这里只产出四路预测，不决定开关。

四路预测
--------
pred_s          仅 S 原型
pred_u_l0.3     U 原型按 λ=0.3 双边融合（不加权，对照）
pred_u_w0.3     加权版双边融合
pred_u_w0.3_os  加权版 + 单向保 TP
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
from lib.pointops2.functions import pointops


def load_scorer(path, dev):
    s = BridgeScorer().to(dev)
    s.load_state_dict(torch.load(path, map_location=dev)['state'])
    s.eval()
    return s


def raw_feats(model, uf, p0, sel, s_fg, s_bg, rev_fg, dev):
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


def build_proto_w(model, uf, mask, weight, k=K):
    """FPS + 聚类，额外返回每个簇的平均权重。"""
    if int(mask.sum()) == 0:
        return (torch.zeros(k, model.feat_dim, device=weight.device),
                torch.zeros(k, device=weight.device))
    coord = uf['cd'][mask].contiguous()
    feat = uf['fd'][mask]
    w = weight[mask]
    n = feat.shape[0]
    if n <= k:
        pad = torch.zeros(k - n, model.feat_dim, device=feat.device)
        cw = torch.zeros(k, device=feat.device)
        cw[:n] = w
        return torch.cat([feat, pad], 0), cw
    off = torch.tensor([n], dtype=torch.int32, device=coord.device)
    fps = pointops.furthestsampling(
        coord, off, torch.cuda.IntTensor([k])).long()
    seeds = feat[fps]
    dist = torch.linalg.norm(feat[:, None, :] - seeds[None, :, :], dim=2)
    assign = torch.argmin(dist, dim=1)
    protos = torch.zeros(k, model.feat_dim, device=feat.device)
    cw = torch.zeros(k, device=feat.device)
    for i in range(k):
        sel = torch.nonzero(assign == i).squeeze(1)
        if sel.numel() == 0:
            protos[i] = feat[fps[i]]
            cw[i] = w[fps[i]]
        else:
            protos[i] = feat[sel].mean(0)
            cw[i] = w[sel].mean()
    return protos, cw


def pick_protos(model, uf, p0, s_fg, s_bg, rev_fg, scorer, args, dev):
    """B 从 p0>fg_hi 取 Top keep_ratio；C 从模糊带邻接取。返回 (b,c,w_full,stats)。"""
    n_c = int(round(K * args.c_ratio))
    sel = p0 > args.c_lo
    w_full = torch.zeros_like(p0)
    if int(sel.sum()) < 10:
        return p0 > args.fg_hi, torch.zeros(p0.shape, dtype=torch.bool, device=dev), \
            w_full, dict(b_prec=0.0, c_prec=0.0, n_b=0, n_c=0)

    cols = raw_feats(model, uf, p0, sel, s_fg, s_bg, rev_fg, dev)
    mu = cols.mean(0, keepdim=True)
    sd = cols.std(0, keepdim=True).clamp(min=1e-6)
    with torch.no_grad():
        score = scorer((cols - mu) / sd)
    w_full[sel] = score

    b_in = p0[sel] > args.fg_hi
    c_in = (p0[sel] > args.c_lo) & (p0[sel] <= args.tau_b)
    b_sel = torch.zeros(p0.shape, dtype=torch.bool, device=dev)
    b_idx = torch.nonzero(sel).squeeze(1)[b_in]
    if b_idx.numel() >= 5:
        sc_b = score[b_in]
        n_keep = min(max(int(round(b_idx.numel() * args.keep_ratio)), 10),
                     b_idx.numel())
        b_sel[b_idx[torch.topk(sc_b, n_keep).indices]] = True
    else:
        b_sel[b_idx] = True

    c_sel = torch.zeros(p0.shape, dtype=torch.bool, device=dev)
    c_idx = torch.nonzero(sel).squeeze(1)[c_in]
    if c_idx.numel() >= 3 and b_sel.any():
        cd = uf['cd']
        dist, _ = cKDTree(cd[b_sel].cpu().numpy()).query(
            cd[c_idx].cpu().numpy(), k=1)
        cand = c_idx[torch.from_numpy(dist < args.near_r).to(dev)]
        if cand.numel():
            sc_c = w_full[cand]
            take = min(n_c, cand.numel())
            c_sel[cand[torch.topk(sc_c, take).indices]] = True

    lab = uf['yl']
    stats = dict(
        b_prec=float(lab[b_sel].float().mean()) if b_sel.any() else 0.0,
        c_prec=float(lab[c_sel].float().mean()) if c_sel.any() else 0.0,
        n_b=int(b_sel.sum()), n_c=int(c_sel.sum()))
    return b_sel, c_sel, w_full, stats


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
    ap.add_argument('--lam', type=float, default=0.3)
    ap.add_argument('--out', default=os.path.join(CACHE, 'v3h'))
    args = ap.parse_args()

    eps = build_episodes(args.per_class, args.shard, args.shards)
    half = len(build_episodes(args.per_class)) // 2
    model, cfg = load_model()
    dev = torch.device('cuda:0')
    sco_a = load_scorer(os.path.join(CACHE, 'scorer_trainedB.pt'), dev)
    sco_b = load_scorer(os.path.join(CACHE, 'scorer_trainedA.pt'), dev)

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
            b_sel, c_sel, w_full, st = pick_protos(
                model, uf, p0, s_fg, s_bg, rev, scorer, args, dev)
            sel_hi = p0 > args.fg_hi
            rel = (float((max_cos(uf['fd'][sel_hi], s_fg)
                          - max_cos(uf['fd'][sel_hi], s_bg)).mean()) + rev
                   if int(sel_hi.sum()) >= 5 else rev)
            blocks.append(dict(uf=uf, p0=p0, b=b_sel, c=c_sel, w=w_full,
                               rev=rev, rel=rel, st=st))
            stats_all.append(st)

        best = max(range(len(blocks)), key=lambda i: blocks[i]['rel'])
        bt = blocks[best]
        n_c = int(round(K * args.c_ratio))
        n_b = K - n_c
        bg_p = build_proto(model, bt['uf'], bt['p0'] < args.bg_lo)
        b_p, b_w = build_proto_w(model, bt['uf'], bt['b'], bt['w'], n_b)
        c_p, c_w = build_proto_w(model, bt['uf'], bt['c'], bt['w'], n_c)
        sparse_u = torch.cat([bg_p, b_p, c_p], 0)
        proto_w = torch.cat([torch.ones(K, device=dev), b_w, c_w], 0)

        pred_s, out_s = predict_agg(model, qf, [sparse_s], 0.0, dev)
        pred_u_l03, _ = predict_agg(model, qf, [sparse_s, sparse_u], args.lam, dev)
        pred_u_w, _ = predict_agg(model, qf, [sparse_s, sparse_u], args.lam, dev,
                                  proto_weights=proto_w)
        pred_u_w_os, _ = predict_agg(model, qf, [sparse_s, sparse_u], args.lam, dev,
                                    mode='one_sided', proto_weights=proto_w)

        p0s = Fs.softmax(pred_s, dim=0)[1]          # 原始分辨率
        p0s_d = Fs.softmax(out_s, dim=1)[:, 1]      # 降采样，与 qf['fd'] 对齐
        assert p0s_d.shape[0] == qf['fd'].shape[0], '分辨率不匹配'
        p0s_np = p0s.cpu().numpy()
        fg_sel = p0s_d > 0.5
        rec = dict(
            g_ep=g_ep, cls=c,
            pred_s=pred_s.cpu().numpy().astype(np.float32),
            pred_u_l03=pred_u_l03.cpu().numpy().astype(np.float32),
            pred_u_w03=pred_u_w.cpu().numpy().astype(np.float32),
            pred_u_w03_os=pred_u_w_os.cpu().numpy().astype(np.float32),
            x_task=np.array([
                float(max_cos(qf['fd'][fg_sel], s_fg).mean()) if fg_sel.any() else 0.0,
                float((p0s > args.fg_hi).float().mean()),
                float((s_bin > 0).float().mean()),
                float((p0s > args.fg_hi).float().mean())
                / float((s_bin > 0).float().mean() + 1e-4),
                float(-(p0s_np * np.log(np.clip(p0s_np, 1e-8, None))
                        + (1 - p0s_np) * np.log(np.clip(1 - p0s_np, 1e-8, None))).mean()),
                float(np.quantile(np.abs(p0s_np - 0.5), 0.9)),
                float(np.mean([b['rev'] for b in blocks])),
                float(np.mean([float((b['p0'] > args.fg_hi).float().mean())
                               for b in blocks]))], dtype=np.float32),
            gt=q_map.cpu().numpy().astype(np.int64),
            w_mean=float(proto_w[K:].mean()))
        recs.append(rec)
        if (ei + 1) % 20 == 0:
            print('  %d / %d' % (ei + 1, len(eps)), flush=True)

    torch.save(recs, args.out + '_s%d.pt' % args.shard)
    print('B_clean %.4f  C_rec %.4f  mean_w %.4f  (%d eps)' % (
        np.mean([s['b_prec'] for s in stats_all]),
        np.mean([s['c_prec'] for s in stats_all]),
        np.mean([r['w_mean'] for r in recs]), len(recs)))
    print('-> %s_s%d.pt' % (args.out, args.shard))


if __name__ == '__main__':
    main()
