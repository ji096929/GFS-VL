"""Phase 2：查询集内部三维邻接约束下的自桥接 A -> B_Q -> C_Q。

B_Q   查询集内 m(R) >= tau_seed 的高置信种子
C_Q   初始背景 (m(R) < tau_seed 且 > -3.0) 且与某个 B 的最小点距 < d_max
D_Q   查询集内 m(R) < -2.0 的纯背景（负对照，用于 CSR）

先跑生死检验（C_Q 内 Top 10/20/30% 的 Precision vs I0），再跑端到端。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

CACHE = '/data01/jh/data/ocsb_cache'
INTRA = os.path.join(CACHE, 'intra')
CKPT = '/data01/jh/weights/sc0_1w1s.pth'

TAU_SEEDS = (0.5, 1.0)
D_MAX = (0.05, 0.10, 0.15, 0.25)
TOPS = (0.1, 0.2, 0.3)


def to_t(a):
    if a.dtype == np.float16:
        return torch.from_numpy(a.astype(np.float32)).cuda()
    if a.dtype.kind in 'iu':
        return torch.from_numpy(a.astype(np.int64)).cuda()
    return torch.from_numpy(a.astype(np.float32)).cuda()


def iou_of(pred, y):
    tp = int((pred & (y > 0)).sum())
    un = int((pred | (y > 0)).sum())
    return tp / max(un, 1)


def load_eps(n):
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))[:n]
    eps = []
    for i, m in enumerate(metas):
        v = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
        a = np.load(os.path.join(INTRA, 'intra_%03d.npz' % i))
        z = F.normalize(to_t(v['q_h']), dim=-1)
        margin = to_t(v['q_margin']).cpu().numpy()
        seg = a['q_seg'].astype(np.int64)
        coord = v['q_coord'].astype(np.float64)
        y = v['q_y'].astype(np.int64)
        uniq, inv = np.unique(seg, return_inverse=True)
        n_reg = len(uniq)
        cnt = np.bincount(inv, minlength=n_reg).astype(np.float64)
        m_sp = np.bincount(inv, weights=margin, minlength=n_reg) / np.maximum(cnt, 1)
        fg_frac = np.bincount(inv, weights=y.astype(np.float64),
                              minlength=n_reg) / np.maximum(cnt, 1)
        z_sp = torch.zeros(n_reg, z.shape[1], device=z.device)
        z_sp.index_add_(0, to_t(inv), z)
        z_sp = F.normalize(z_sp / torch.clamp(
            to_t(cnt).unsqueeze(1), min=1), dim=-1)
        scene = float((v['q_coord_full'].max(0) - v['q_coord_full'].min(0)).max())
        eps.append({'m': m_sp, 'inv': inv, 'n_reg': n_reg, 'y': y,
                    'coord': coord, 'seg': seg, 'z_sp': z_sp,
                    'fg_frac': fg_frac, 'cls': int(m['class']),
                    'scene': scene})
        if (i + 1) % 100 == 0:
            print('  loaded %d / %d' % (i + 1, len(metas)), flush=True)
    return eps


def adjacent(e, B, C, dmax_phys):
    """返回 C 中与 B 最小点距 < dmax_phys(米) 的超点 id（q_coord 为米制）。"""
    if len(B) == 0 or len(C) == 0:
        return np.zeros(0, dtype=np.int64)
    in_b = np.isin(e['inv'], B)
    if not in_b.any():
        return np.zeros(0, dtype=np.int64)
    tree_b = cKDTree(e['coord'][in_b])
    keep = []
    for c in C:
        pts = e['coord'][e['inv'] == c]
        if len(pts) == 0:
            continue
        if float(tree_b.query(pts)[0].min()) < dmax_phys:
            keep.append(c)
    return np.array(keep, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=560)
    ap.add_argument('--out', default=os.path.join(CACHE, 'phase2.json'))
    args = ap.parse_args()

    torch.manual_seed(0)
    eps = load_eps(args.n)
    print('episodes:', len(eps), flush=True)

    base_ious = [iou_of(e['m'][e['inv']] > 0.5, e['y']) for e in eps]
    I0 = float(np.mean(base_ious))
    print('baseline (Phase1 alpha=1 theta=0.5) arith %.4f' % I0, flush=True)

    out = {'I0': I0, 'death': [], 'e2e': []}

    # ---------------- 生死检验 ----------------
    print('\n=== death test on C_Q_adj (Precision vs I0=%.4f) ===' % I0,
          flush=True)
    for tau in TAU_SEEDS:
        # 预计算每个 episode 的 adj
        cache_adj = {}
        for ei, e in enumerate(eps):
            B = np.where(e['m'] >= tau)[0]
            C = np.where((e['m'] < tau) & (e['m'] > -3.0))[0]
            for dmax in D_MAX:
                cache_adj[(ei, dmax)] = adjacent(e, B, C, dmax)
        for dmax in D_MAX:
            acc = {t: [0, 0] for t in TOPS}
            n_cand = 0
            for ei, e in enumerate(eps):
                adj = cache_adj[(ei, dmax)]
                if len(adj) < 3:
                    continue
                n_cand += len(adj)
                B = np.where(e['m'] >= tau)[0]
                if len(B) == 0:
                    continue
                sim = (e['z_sp'][adj] @ e['z_sp'][B].t()).max(1).values
                order = torch.argsort(-sim).cpu().numpy()
                for t in TOPS:
                    kk = max(int(len(adj) * t), 1)
                    sel = adj[order[:kk]]
                    yy = e['fg_frac'][sel] > 0.5
                    acc[t][0] += int(yy.sum())
                    acc[t][1] += int((~yy).sum())
            if n_cand == 0:
                continue
            row = {'tau': tau, 'dmax': dmax, 'n_cand': n_cand}
            line = 'tau %.1f dmax %.2f  n=%6d  ' % (tau, dmax, n_cand)
            for t in TOPS:
                tp, fp = acc[t]
                p = tp / max(tp + fp, 1)
                row['p_top%d' % int(100 * t)] = p
                line += 'T%d%%:%.3f%s ' % (100 * t, p, '*' if p > I0 else ' ')
            print(line, flush=True)
            out['death'].append(row)

    # ---------------- 端到端 ----------------
    print('\n=== end-to-end (Phase1 + adjacent recovery) ===', flush=True)
    for tau in TAU_SEEDS:
        for dmax in D_MAX:
            for top in TOPS:
                ious = []
                for ei, e in enumerate(eps):
                    ell = e['m'][e['inv']].copy()
                    B = np.where(e['m'] >= tau)[0]
                    C = np.where((e['m'] < tau) & (e['m'] > -3.0))[0]
                    adj = adjacent(e, B, C, dmax)
                    if len(adj) and len(B):
                        sim = (e['z_sp'][adj] @ e['z_sp'][B].t()).max(1).values
                        kk = max(int(len(adj) * top), 1)
                        sel = adj[torch.topk(sim, kk).indices.cpu().numpy()]
                        ell[np.isin(e['inv'], sel)] = 1.0
                    ious.append(iou_of(ell > 0.5, e['y']))
                d = np.array(ious) - np.array(base_ious)
                pse = d.std(ddof=1) / np.sqrt(len(d))
                out['e2e'].append({'tau': tau, 'dmax': dmax, 'top': top,
                                   'arith': float(np.mean(ious)),
                                   'delta': float(d.mean()),
                                   'pse': float(pse),
                                   't': float(d.mean() / pse) if pse else 0})
                print('tau %.1f dmax %.2f top %.0f%%  arith %.4f  d %+.4f  '
                      'pSE %.4f  t %+.2f'
                      % (tau, dmax, 100 * top, np.mean(ious), d.mean(), pse,
                         d.mean() / pse if pse else 0), flush=True)

    json.dump(out, open(args.out, 'w'), indent=1)
    print('->', args.out)


if __name__ == '__main__':
    main()
