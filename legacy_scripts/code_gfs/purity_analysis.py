"""离线提纯信号分析：哪些信号能把 B 的精度从 0.475 拉上去。

只读已有缓存，不训练、不改主管道。

分析对象
--------
B = U 块中初始 p0 > fg_hi 的点（基准精度约 0.475），标签取降采样真值 `yl`。
对每个 B 点计算一组候选信号，统计每个信号对 B_pos / B_neg 的 AUC，
以及取 Top-50% / 30% / 10% 后的实际精度。

信号设计
--------
特征级（逐点可算，不额外前向）：
  p0        初始前景概率
  sim_fg    与支持集前景原型的最大余弦
  sim_bg    与支持集背景原型的最大余弦
  R_s       sim_fg - sim_bg，即"反向支持验证"（真桥梁该更像 S_fg）
  sim_base  与 10 个基类原型的最大余弦（假桥梁多为 cabinet/wall/floor）
  R_base    sim_fg - sim_base
  sim_du    与 U 内高置信背景原型的最大余弦
  R_du      sim_fg - sim_du
前向级：
  delta     融合 B 这一路参考集后该点的前景 logit 变化（式 7）
超点级：
  sp_mean / sp_std / sp_min  超点内 p0 的均值/标准差/最小值

关于 eta：
  η_c(B) 按式(8) 是"每个 B 一个标量"，对逐点排序等价于常数偏移，
  AUC 恒为 0.5，所以它不能作为逐点提纯信号，只能做整体门控。
  这里用 sim_du / sim_base 作为它的逐点替代。
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as Fs
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

from ocsb_core import (CACHE, NEW_IDS, build_episodes, build_proto,
                       episode_tensors, feats_one, load_model, predict_agg,
                       prepare_block, seed_everything)

SIGNALS = ['p0', 'delta', 'sim_fg', 'sim_bg', 'R_s',
           'sim_base', 'R_base', 'sim_du', 'R_du',
           'sp_mean', 'sp_std', 'sp_min']


def max_cos(f, protos):
    """f: (N, C)  protos: (K, C) -> (N,) 每个点对原型集的最大余弦相似度。"""
    if protos.shape[0] == 0:
        return torch.zeros(f.shape[0], device=f.device)
    return Fs.cosine_similarity(f[:, None, :], protos[None, :, :], dim=2).max(1)[0]


def seg_downsample(xyz, seg, cd):
    if xyz.shape[0] == cd.shape[0]:
        return seg
    tree = cKDTree(xyz.detach().cpu().numpy())
    _, idx = tree.query(cd.detach().cpu().numpy())
    return torch.from_numpy(np.asarray(seg.cpu().numpy())[idx]).to(cd.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=10)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--fg-hi', type=float, default=0.7)
    ap.add_argument('--bg-lo', type=float, default=0.2)
    ap.add_argument('--lam', type=float, default=0.7, help='算 Δ 时的融合强度')
    ap.add_argument('--out', default=os.path.join(CACHE, 'purity_analysis.json'))
    args = ap.parse_args()

    eps = build_episodes(args.per_class, args.shard, args.shards)
    model, cfg = load_model()
    dev = torch.device('cuda:0')

    # 每个信号收集 (值, 标签)
    buf = {s: [] for s in SIGNALS}
    labels = []
    n_ep = 0

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
        s_bg, s_fg = sparse_s[:100], sparse_s[100:]

        for u in us:
            u_x, u_bin = episode_tensors(u, c)
            uf = feats_one(model, u_x, u_bin, dev)
            seg_d = seg_downsample(uf['xyz'], u['seg'].to(dev), uf['cd'])

            _, out0 = predict_agg(model, uf, [sparse_s], 0.0, dev)
            p0 = Fs.softmax(out0, dim=1)[:, 1]
            b_mask = p0 > args.fg_hi
            if int(b_mask.sum()) < 5:
                continue

            d_mask = p0 < args.bg_lo
            b_sparse = torch.cat([build_proto(model, uf, d_mask),
                                  build_proto(model, uf, b_mask)], 0)
            _, out_b = predict_agg(model, uf, [sparse_s, b_sparse], args.lam, dev)
            delta = out_b[:, 1] - out0[:, 1]

            f = uf['fd'][b_mask]
            lab = uf['yl'][b_mask].float()

            sim_fg = max_cos(f, s_fg)
            sim_bg = max_cos(f, s_bg)
            sim_base = max_cos(f, model.base_prototypes)
            du_proto = build_proto(model, uf, d_mask)
            sim_du = max_cos(f, du_proto)

            # 超点级聚合：用 bincount 一次性算组内均值/标准差/最小值
            sg_all = seg_d.detach().cpu().numpy()
            uniq, inv = np.unique(sg_all, return_inverse=True)
            p_np = p0.detach().cpu().numpy()
            cnt = np.bincount(inv).astype(np.float64)
            s1 = np.bincount(inv, weights=p_np)
            s2 = np.bincount(inv, weights=p_np ** 2)
            mean = s1 / np.maximum(cnt, 1)
            var = np.maximum(s2 / np.maximum(cnt, 1) - mean ** 2, 0.0)
            std = np.sqrt(var)
            gmin = np.full(len(uniq), np.inf)
            np.minimum.at(gmin, inv, p_np)
            sp_mean = torch.from_numpy(mean[inv]).to(dev).float()
            sp_std = torch.from_numpy(std[inv]).to(dev).float()
            sp_min = torch.from_numpy(gmin[inv]).to(dev).float()

            vals = dict(
                p0=p0[b_mask], delta=delta[b_mask],
                sim_fg=sim_fg, sim_bg=sim_bg, **{'R_s': sim_fg - sim_bg},
                sim_base=sim_base, **{'R_base': sim_fg - sim_base},
                sim_du=sim_du, **{'R_du': sim_fg - sim_du},
                sp_mean=sp_mean[b_mask], sp_std=sp_std[b_mask],
                sp_min=sp_min[b_mask])
            for k in SIGNALS:
                buf[k].append(vals[k].detach().cpu())
            labels.append(lab.detach().cpu())
        n_ep += 1
        if (ei + 1) % 10 == 0:
            print('  %d / %d' % (ei + 1, len(eps)), flush=True)

    cat = {k: torch.cat(v).numpy() for k, v in buf.items()}
    y = torch.cat(labels).numpy()
    n = len(y)
    base_prec = float(y.mean())
    print('\nepisodes %d  B 点总数 %d  B 基准精度 %.4f' % (n_ep, n, base_prec))

    print('\n=== B 提纯信号（AUC 越高越能把 B_pos 排到前面） ===')
    print('%-9s %8s %11s %11s %11s' % (
        'signal', 'AUC', 'Top50%', 'Top30%', 'Top10%'))
    rows = {}
    for k in SIGNALS:
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
        rows[k] = dict(auc=auc, top50=line[0], top30=line[1], top10=line[2])
        print('%-9s %8.4f %11.4f %11.4f %11.4f' % (
            k, auc, line[0], line[1], line[2]))

    # 反向信号（越低越像前景）也试一下
    print('\n（反向：取负号后的 Top-k）')
    for k in ['sim_bg', 'sim_base', 'sim_du']:
        v = -cat[k]
        line = []
        for frac in (0.5, 0.3, 0.1):
            kk = max(int(len(v) * frac), 1)
            idx = np.argsort(-v)[:kk]
            line.append(float(y[idx].mean()))
        print('%-9s %8.4f %11.4f %11.4f %11.4f' % (
            '-' + k, float(roc_auc_score(y, v)), line[0], line[1], line[2]))

    json.dump(dict(n=n, base_prec=base_prec, rows=rows,
                   n_ep=n_ep), open(args.out, 'w'), indent=1)
    print('\n->', args.out)


if __name__ == '__main__':
    main()
