"""缓存 OmissionGate 的训练样本：pred_S / pred_U / 任务级与点级特征。

预测器设计
----------
`pred_U` 用训练好的 BridgeScorer 重新构造 U 原型集：
  1. 取与缓存时一致的 10 维特征（任务内 z-score）
  2. scorer 打分，取 Top-K 作为提纯后的前景原型
  3. 背景仍取 `p0 < bg_lo`
  4. 与 `sparse_s` 分别跑 `predict_agg`，得到两条独立预测

缓存内容（按 episode）
- pred_S, pred_U : [2, N_raw] 原始分辨率 logits
- p0s, p0u       : 两条预测的前景概率（点级特征）
- x_task         : 8 维任务级特征
- gt             : 查询集二值标签
- cls, g_ep
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


def score_and_pick(model, uf, p0, s_fg, s_bg, rev_fg, scorer, args, dev):
    """用 BridgeScorer 给候选点打分，返回提纯后的前景掩码。"""
    sel = p0 > args.c_lo
    if int(sel.sum()) < 20:
        return p0 > args.fg_hi
    f = uf['fd'][sel]
    sim_fg = max_cos(f, s_fg)
    sim_bg = max_cos(f, s_bg)
    sim_base = max_cos(f, model.base_prototypes)
    sp_mean, sp_std = seg_stats(uf['seg_d'], p0)
    sp_mean, sp_std = sp_mean[sel], sp_std[sel]

    core = p0 > args.fg_hi
    if int(core.sum()) >= 3:
        core_proto = build_proto(model, uf, core)
        sim_core = max_cos(f, core_proto)
        d, _ = cKDTree(uf['cd'][core].cpu().numpy()).query(
            uf['cd'][sel].cpu().numpy(), k=1)
        dist_core = torch.from_numpy(d).to(dev).float()
    else:
        sim_core = torch.zeros(f.shape[0], device=dev)
        dist_core = torch.full((f.shape[0],), 10.0, device=dev)

    cols = torch.stack([p0[sel], sp_mean, sp_std, sim_fg, sim_bg, sim_base,
                        sim_fg - sim_bg, torch.full_like(sim_fg, rev_fg),
                        dist_core, sim_core], dim=1)
    mu = cols.mean(0, keepdim=True)
    sd = cols.std(0, keepdim=True).clamp(min=1e-6)
    cols = (cols - mu) / sd

    with torch.no_grad():
        w = scorer(cols)
    k = min(args.topk, int(sel.sum()))
    top = torch.topk(w, k).indices
    mask = torch.zeros_like(p0)
    mask[torch.nonzero(sel).squeeze(1)[top]] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=56)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--fg-hi', type=float, default=0.7)
    ap.add_argument('--bg-lo', type=float, default=0.2)
    ap.add_argument('--c-lo', type=float, default=0.2)
    ap.add_argument('--topk', type=int, default=60, help='提纯后保留多少个原型点')
    ap.add_argument('--scorer', default=os.path.join(CACHE, 'bridge_scorer.pt'))
    ap.add_argument('--out', default=os.path.join(CACHE, 'v3_gate'))
    args = ap.parse_args()

    eps = build_episodes(args.per_class, args.shard, args.shards)
    model, cfg = load_model()
    dev = torch.device('cuda:0')
    scorer = BridgeScorer().to(dev)
    scorer.load_state_dict(torch.load(args.scorer, map_location=dev)['state'])
    scorer.eval()

    recs = []
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

        uf_list, p0_list, rev_list = [], [], []
        for u in us:
            u_x, u_bin = episode_tensors(u, c)
            uf = feats_one(model, u_x, u_bin, dev)
            uf['seg_d'] = seg_downsample(uf['xyz'], u['seg'].to(dev), uf['cd'])
            _, out0 = predict_agg(model, uf, [sparse_s], 0.0, dev)
            p0 = Fs.softmax(out0, dim=1)[:, 1]
            # 块级 rev_fg：与训练缓存时的定义完全一致
            b_sp = torch.cat([build_proto(model, uf, p0 < args.bg_lo),
                              build_proto(model, uf, p0 > args.fg_hi)], 0)
            ps, _ = predict_agg(model, sf, [b_sp], 1.0, dev)
            ps = Fs.softmax(ps, dim=0)[1]
            rev = float(ps[s_bin > 0].mean()) if (s_bin > 0).any() else 0.0
            uf_list.append(uf)
            p0_list.append(p0)
            rev_list.append(rev)

        # 提纯后的 U 原型集
        bg_stack, fg_stack = [], []
        for uf, p0, rev in zip(uf_list, p0_list, rev_list):
            keep = score_and_pick(model, uf, p0, s_fg, s_bg, rev, scorer, args, dev)
            bg_stack.append(build_proto(model, uf, p0 < args.bg_lo))
            fg_stack.append(build_proto(model, uf, keep))
        sparse_u = torch.cat([torch.stack(bg_stack).mean(0),
                              torch.stack(fg_stack).mean(0)], 0)

        pred_s, out_s = predict_agg(model, qf, [sparse_s], 0.0, dev)
        pred_u, out_u = predict_agg(model, qf, [sparse_u], 1.0, dev)
        # 点级特征与 GT 在原始分辨率上；sim_fg_q 要和降采样特征对齐
        p0s = Fs.softmax(pred_s, dim=0)[1]
        p0u = Fs.softmax(pred_u, dim=0)[1]
        p0s_d = Fs.softmax(out_s, dim=1)[:, 1]

        # 任务级 8 维
        fg_sel = p0s_d > 0.5
        sim_fg_q = (float(max_cos(qf['fd'][fg_sel], s_fg).mean())
                    if fg_sel.any() else 0.0)
        frac_q = float((p0s_d > args.fg_hi).float().mean())
        frac_s = float((s_bin > 0).float().mean())
        p0s_np = p0s.cpu().numpy()
        ent = float(-(p0s_np * np.log(np.clip(p0s_np, 1e-8, None))
                      + (1 - p0s_np) * np.log(np.clip(1 - p0s_np, 1e-8, None))).mean())
        margin_p90 = float(np.quantile(np.abs(p0s_np - 0.5), 0.9))
        valid_ratio = float(np.mean([float((p > args.fg_hi).float().mean())
                                     for p in p0_list]))
        x_task = np.array([sim_fg_q, frac_q, frac_s, frac_q / (frac_s + 1e-4),
                           ent, margin_p90, float(np.mean(rev_list)),
                           valid_ratio], dtype=np.float32)

        recs.append(dict(g_ep=g_ep, cls=c,
                         pred_s=pred_s.cpu().numpy().astype(np.float32),
                         pred_u=pred_u.cpu().numpy().astype(np.float32),
                         p0s=p0s.cpu().numpy().astype(np.float32),
                         p0u=p0u.cpu().numpy().astype(np.float32),
                         x_task=x_task,
                         gt=q_map.cpu().numpy().astype(np.int64)))

        if (ei + 1) % 20 == 0:
            print('  %d / %d' % (ei + 1, len(eps)), flush=True)

    torch.save(recs, args.out + '_s%d.pt' % args.shard)
    print('-> %s_s%d.pt  (%d eps)' % (args.out, args.shard, len(recs)))


if __name__ == '__main__':
    main()
