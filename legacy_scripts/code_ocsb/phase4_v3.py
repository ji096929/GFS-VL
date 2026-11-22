"""Phase 4 v3：统一口径的 2 折 CV 盲测消融表。

与 v2 的区别：第四行不再读 Phase 2 的固定 baseline，而是
  1) 在训练折选出 Phase 1 的最优 (agg, alpha, theta)
  2) 在该配置的 base 上，再在训练折选出 Phase 2 的最优桥接配置
  3) 把"base + 桥接"一起应用到盲测折

这样第三、四行是同一套超参选择流程，第四行相对第三行的差值就是桥接本身的贡献。
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase1_smooth import load_all, sp_stats  # noqa: E402
from phase2_v2 import adjacent, load_eps, score_adj  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
CKPT = '/data01/jh/weights/sc0_1w1s.pth'
AGGS = ('mean', 'median', 'p75', 'trim')
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
THETAS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6)
TAUS = (0.5, 1.0)
KINDS = ('cos', 'vq', 'rgb', 'rgbn')
TOPS = (0.1, 0.2, 0.3)
DMAX = 0.10


def iou_of(pred, y):
    tp = int((pred & (y > 0)).sum())
    un = int((pred | (y > 0)).sum())
    return tp / max(un, 1)


def cum_of(preds, data):
    tp_c, un_c = {}, {}
    for pred, d in zip(preds, data):
        c = d['cls']
        tp_c[c] = tp_c.get(c, 0) + int((pred & (d['y'] > 0)).sum())
        un_c[c] = un_c.get(c, 0) + int((pred | (d['y'] > 0)).sum())
    return float(np.mean([tp_c[c] / max(un_c[c], 1) for c in tp_c])), tp_c, un_c


def main():
    torch.manual_seed(0)
    # ---- Phase 1 数据 ----
    d1 = load_all(560)
    N = len(d1)
    print('phase1 episodes:', N, flush=True)
    ms = {agg: [sp_stats(d['margin'], d['seg'], d['n_reg'], agg) for d in d1]
          for agg in AGGS}
    base_ious = np.array([iou_of(d['margin'] > 0, d['y']) for d in d1])
    base_cum = cum_of([d['margin'] > 0 for d in d1], d1)[0]
    print('baseline arith %.4f cum %.4f' % (base_ious.mean(), base_cum),
          flush=True)

    # ---- Phase 2 数据 ----
    table = torch.load(CKPT, map_location='cpu')['state_dict'][
        'module.base_prototypes'].float().cuda()
    d2 = load_eps(560, table)
    print('phase2 episodes:', len(d2), flush=True)

    half = N // 2
    folds = [list(range(half)), list(range(half, N))]

    # 预计算每个 episode 的 B/C/adj（按 tau 分组）
    adj_cache = {}
    for tau in TAUS:
        for ei, e in enumerate(d2):
            B = np.where(e['m'] >= tau)[0]
            C = np.where((e['m'] < 0.5) & (e['m'] > -3.0))[0]
            adj_cache[(tau, ei)] = (B, adjacent(e, B, C, DMAX))

    def base_ell(cfg, i):
        agg, alpha, theta = cfg
        return ((1 - alpha) * d1[i]['margin'] + alpha * ms[agg][i][d1[i]['seg']]
                - theta)

    def bridge_sel(i, tau, kind, top):
        e = d2[i]
        B, adj = adj_cache[(tau, i)]
        if len(adj) == 0 or len(B) == 0:
            return np.zeros(0, dtype=np.int64)
        s = score_adj(e, adj, B, kind)
        if kind == 'vq':
            keep = s > 0
            adj2 = adj[keep.cpu().numpy()]
            s2 = s[keep]
        else:
            adj2, s2 = adj, s
        if len(adj2) == 0:
            return np.zeros(0, dtype=np.int64)
        kk = max(int(len(adj2) * top), 1)
        return adj2[torch.topk(s2, kk).indices.cpu().numpy()]

    # 预计算所有桥接选择
    sel_cache = {}
    for tau in TAUS:
        for kind in KINDS:
            for top in TOPS:
                sel_cache[(tau, kind, top)] = [
                    bridge_sel(i, tau, kind, top) for i in range(N)]

    # ---- 2 折 CV ----
    rows = {}
    blind_base = np.zeros(N)
    blind_full = np.zeros(N)
    preds_base = [None] * N
    preds_full = [None] * N
    picks = []

    for f in (0, 1):
        tr, te = folds[f], folds[1 - f]
        # 1) 训练折选 Phase 1 配置
        best_base, best_score = None, -1e9
        for agg in AGGS:
            for alpha in ALPHAS:
                for theta in THETAS:
                    cfg = (agg, alpha, theta)
                    sc = np.mean([iou_of(base_ell(cfg, i) > 0, d1[i]['y'])
                                  for i in tr])
                    if sc > best_score:
                        best_score, best_base = sc, cfg
        # 2) 训练折选桥接配置（固定 best_base）
        best_bridge, best_bs = None, best_score
        for tau in TAUS:
            for kind in KINDS:
                for top in TOPS:
                    sc = 0.0
                    for i in tr:
                        ell = base_ell(best_base, i).copy()
                        sel = sel_cache[(tau, kind, top)][i]
                        if len(sel):
                            ell[np.isin(d1[i]['seg'], sel)] = 1.0
                        sc += iou_of(ell > 0, d1[i]['y'])
                    sc /= len(tr)
                    if sc > best_bs:
                        best_bs, best_bridge = sc, (tau, kind, top)
        # 3) 盲测折
        for i in te:
            preds_base[i] = base_ell(best_base, i) > 0
            ell = base_ell(best_base, i).copy()
            if best_bridge is not None:
                sel = sel_cache[best_bridge][i]
                if len(sel):
                    ell[np.isin(d1[i]['seg'], sel)] = 1.0
            preds_full[i] = ell > 0
            blind_base[i] = iou_of(preds_base[i], d1[i]['y'])
            blind_full[i] = iou_of(preds_full[i], d1[i]['y'])
        picks.append({'fold': f, 'base': best_base, 'bridge': best_bridge,
                      'train_base': float(best_score),
                      'train_full': float(best_bs)})
        print('fold %d: base %s  bridge %s  train %.4f -> %.4f'
              % (f, best_base, best_bridge, best_score, best_bs), flush=True)

    def stat(preds, blind, tag):
        d = blind - base_ious
        pse = d.std(ddof=1) / np.sqrt(N)
        cum, tp_c, un_c = cum_of(preds, d1)
        per = {int(c): float(tp_c[c] / max(un_c[c], 1)) for c in sorted(tp_c)}
        print('%-28s arith %.4f (d %+.4f) cum %.4f (d %+.4f) pSE %.4f t %+.2f'
              % (tag, blind.mean(), d.mean(), cum, cum - base_cum, pse,
                 d.mean() / pse if pse else 0), flush=True)
        return {'name': tag, 'arith': float(blind.mean()), 'cum': cum,
                'delta': float(d.mean()), 'cum_delta': float(cum - base_cum),
                'pse': float(pse),
                't': float(d.mean() / pse) if pse else 0.0,
                'per_class': per}

    print('\n=== blind 560 (unified protocol) ===', flush=True)
    _, btp, bun = cum_of([d['margin'] > 0 for d in d1], d1)
    rows['baseline'] = {'name': 'Baseline', 'arith': float(base_ious.mean()),
                        'cum': base_cum, 'delta': 0.0, 'cum_delta': 0.0,
                        'pse': 0.0, 't': 0.0,
                        'per_class': {int(c): float(btp[c] / max(bun[c], 1))
                                      for c in sorted(btp)}}
    rows['sp'] = stat(preds_base, blind_base, '+ query SP consistency (CV)')
    rows['bridge'] = stat(preds_full, blind_full,
                          '+ spatial-adjacent bridge (CV)')

    print('\n=== per-class (cum IoU) ===')
    cls_ids = sorted(rows['baseline']['per_class'])
    print('%-32s %s' % ('version', ' '.join('%6d' % c for c in cls_ids)))
    for k in ('baseline', 'sp', 'bridge'):
        r = rows[k]
        print('%-32s %s' % (r['name'], ' '.join(
            '%6.3f' % r['per_class'][c] for c in cls_ids)))

    json.dump({'rows': [rows['baseline'], rows['sp'], rows['bridge']],
               'picks': picks},
              open(os.path.join(CACHE, 'phase4_v3.json'), 'w'), indent=1)
    print('->', os.path.join(CACHE, 'phase4_v3.json'))


if __name__ == '__main__':
    main()
