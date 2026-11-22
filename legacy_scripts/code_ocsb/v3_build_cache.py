"""缓存 BridgeScorer 的训练特征（类别无关的相关性空间统计量）。

对每个 U 块的候选点（B 与邻接模糊带），提取 10 维特征：

  1 p0            初始前景概率
  2 sp_mean_p0    所在超点的 p0 均值
  3 sp_std_p0     所在超点的 p0 标准差
  4 sim_fg        与 S 前景原型的最大余弦
  5 sim_bg        与 S 背景原型的最大余弦
  6 sim_base      与 10 个基类原型的最大余弦
  7 R_s           sim_fg − sim_bg
  8 rev_fg        块级：该块的 B 原型在 S 真前景上的平均激活
  9 dist_to_core  到 B 核心（p0>fg_hi）的最近三维距离
 10 sim_to_core   与 B 核心原型的最大余弦

标签 `y_u` = 该降采样点是否真前景（`yl`）。
每个 episode 内对特征做 z-score（类别无关标准化），使训练与类别无关。

输出 npz 供 `train_bridge_scorer.py` 秒级加载。
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as Fs
from scipy.spatial import cKDTree

from ocsb_core import (CACHE, build_episodes, build_proto, episode_tensors,
                       feats_one, load_model, predict_agg, prepare_block,
                       seed_everything, seg_downsample)


def max_cos(f, protos):
    if protos.shape[0] == 0:
        return torch.zeros(f.shape[0], device=f.device)
    return Fs.cosine_similarity(f[:, None, :], protos[None, :, :], dim=2).max(1)[0]


def seg_stats(seg_d, p0):
    sg = seg_d.detach().cpu().numpy()
    uniq, inv = np.unique(sg, return_inverse=True)
    p = p0.detach().cpu().numpy()
    cnt = np.bincount(inv).astype(np.float64)
    s1 = np.bincount(inv, weights=p)
    s2 = np.bincount(inv, weights=p ** 2)
    mean = s1 / np.maximum(cnt, 1)
    var = np.maximum(s2 / np.maximum(cnt, 1) - mean ** 2, 0.0)
    return (torch.from_numpy(mean[inv]).to(p0.device).float(),
            torch.from_numpy(np.sqrt(var)[inv]).to(p0.device).float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=56)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--fg-hi', type=float, default=0.7)
    ap.add_argument('--bg-lo', type=float, default=0.2)
    ap.add_argument('--c-lo', type=float, default=0.2, help='候选下界（含 B 与模糊带）')
    ap.add_argument('--out', default=os.path.join(CACHE, 'v3_feat_u'))
    args = ap.parse_args()

    eps = build_episodes(args.per_class, args.shard, args.shards)
    model, cfg = load_model()
    dev = torch.device('cuda:0')

    X, Y, EP, CLS, BLK = [], [], [], [], []

    for ei, m in enumerate(eps):
        g_ep = args.shard + args.shards * ei
        seed_everything(10000 + g_ep)
        c = m['class']
        s = prepare_block(m['support'], cfg)
        us = [prepare_block(nm, cfg) for nm in m['u']]
        s_x, s_bin = episode_tensors(s, c)
        sf = feats_one(model, s_x, s_bin, dev)
        fg_s = sf['yl'].bool()
        s_fg = build_proto(model, sf, fg_s)
        s_bg = build_proto(model, sf, ~fg_s)
        sparse_s = torch.cat([s_bg, s_fg], 0)

        ep_rows, ep_lab = [], []
        for bi, u in enumerate(us):
            u_x, u_bin = episode_tensors(u, c)
            uf = feats_one(model, u_x, u_bin, dev)
            seg_d = seg_downsample(uf['xyz'], u['seg'].to(dev), uf['cd'])
            _, out0 = predict_agg(model, uf, [sparse_s], 0.0, dev)
            p0 = Fs.softmax(out0, dim=1)[:, 1]

            # 块级 rev_fg：B 原型在 S 真前景上的平均激活
            b_sp = torch.cat([build_proto(model, uf, p0 < args.bg_lo),
                              build_proto(model, uf, p0 > args.fg_hi)], 0)
            pred_s, _ = predict_agg(model, sf, [b_sp], 1.0, dev)
            pS = Fs.softmax(pred_s, dim=0)[1]
            rev_fg = float(pS[s_bin > 0].mean()) if (s_bin > 0).any() else 0.0

            sel = p0 > args.c_lo
            if int(sel.sum()) < 20:
                continue
            f = uf['fd'][sel]
            sim_fg = max_cos(f, s_fg)
            sim_bg = max_cos(f, s_bg)
            sim_base = max_cos(f, model.base_prototypes)

            sp_mean, sp_std = seg_stats(seg_d, p0)
            sp_mean, sp_std = sp_mean[sel], sp_std[sel]

            # 到 B 核心的距离与相似度
            core = p0 > args.fg_hi
            if int(core.sum()) >= 3:
                cd = uf['cd']
                core_proto = build_proto(model, uf, core)
                sim_core = max_cos(f, core_proto)
                d, _ = cKDTree(cd[core].cpu().numpy()).query(
                    uf['cd'][sel].cpu().numpy(), k=1)
                dist_core = torch.from_numpy(d).to(dev).float()
            else:
                sim_core = torch.zeros(f.shape[0], device=dev)
                dist_core = torch.full((f.shape[0],), 10.0, device=dev)

            cols = torch.stack([
                p0[sel], sp_mean, sp_std, sim_fg, sim_bg, sim_base,
                sim_fg - sim_bg,
                torch.full_like(sim_fg, rev_fg),
                dist_core, sim_core,
            ], dim=1)                                   # [N, 10]

            ep_rows.append(cols.detach().cpu())
            ep_lab.append(uf['yl'][sel].detach().cpu())

        if not ep_rows:
            continue
        rows = torch.cat(ep_rows)
        lab = torch.cat(ep_lab)
        # 任务内 z-score（类别无关标准化），只用非常数维度
        mu = rows.mean(0, keepdim=True)
        sd = rows.std(0, keepdim=True).clamp(min=1e-6)
        rows = (rows - mu) / sd
        X.append(rows)
        Y.append(lab)
        EP.append(torch.full((rows.shape[0],), g_ep, dtype=torch.long))
        CLS.append(torch.full((rows.shape[0],), c, dtype=torch.long))

        if (ei + 1) % 20 == 0:
            print('  %d / %d' % (ei + 1, len(eps)), flush=True)

    np.savez_compressed(
        args.out,
        X=torch.cat(X).numpy().astype(np.float32),
        Y=torch.cat(Y).numpy().astype(np.int64),
        EP=torch.cat(EP).numpy(), CLS=torch.cat(CLS).numpy())
    n = sum(len(a) for a in X)
    y = torch.cat(Y)
    print('\n%d 个点  %d 个 episode  shard %d/%d' % (
        n, len(eps), args.shard, args.shards))
    print('正样本比例 %.4f' % float(y.float().mean()))
    print('->', args.out + '.npz')


if __name__ == '__main__':
    main()
