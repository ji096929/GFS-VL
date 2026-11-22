"""Phase 2 完整版 v2：四种匹配度 + CSR 验证 + 2 折 CV + Cumul。

打分：
  S_cos   max_B z_B·z_C
  S_vq    max_B [ z_B·z_C - max_{D in N} z_B·z_D ]   （CSR 净贡献）
  S_rgb   max_B [(cos_rgb + exp(-|dz|/sigma)) / 2]
  S_rgbn  max_B [(cos_rgb + exp(-|dz|/sigma) + |n_B·n_C|) / 3]  （+法向）

负对照 N = D_Q ∪ D_S ∪ P_base
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
SCORES = ('cos', 'vq', 'rgb', 'rgbn')
Z_SIGMA = 0.10


def to_t(a):
    if a.dtype == np.float16:
        return torch.from_numpy(a.astype(np.float32)).cuda()
    if a.dtype.kind in 'iu':
        return torch.from_numpy(a.astype(np.int64)).cuda()
    return torch.from_numpy(a.astype(np.float32)).cuda()


def sp_mean_norm(z, seg, n_reg):
    out = torch.zeros(n_reg, z.shape[1], dtype=z.dtype, device=z.device)
    out.index_add_(0, seg, z)
    cnt = torch.bincount(seg, minlength=n_reg).clamp_min(1).to(z.dtype)
    return F.normalize(out / cnt[:, None], dim=-1)


def sp_normals(coord, inv, n_reg):
    """超点级法向：超点内点的 PCA 最小特征向量。"""
    n = np.zeros((n_reg, 3))
    for r in range(n_reg):
        pts = coord[inv == r]
        if len(pts) < 3:
            n[r] = [0.0, 0.0, 1.0]
            continue
        p = pts - pts.mean(0)
        w, v = np.linalg.eigh(p.T @ p)
        n[r] = v[:, 0]
    return n


def iou_of(pred, y):
    tp = int((pred & (y > 0)).sum())
    un = int((pred | (y > 0)).sum())
    return tp / max(un, 1)


def cum_of(preds, ys, cls_list):
    tp_c, un_c = {}, {}
    for pred, y, c in zip(preds, ys, cls_list):
        tp_c[c] = tp_c.get(c, 0) + int((pred & (y > 0)).sum())
        un_c[c] = un_c.get(c, 0) + int((pred | (y > 0)).sum())
    return float(np.mean([tp_c[c] / max(un_c[c], 1) for c in tp_c]))


def load_eps(n, table):
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))[:n]
    eps = []
    for i, m in enumerate(metas):
        v = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
        a = np.load(os.path.join(INTRA, 'intra_%03d.npz' % i))
        z = F.normalize(to_t(v['q_h']), dim=-1)
        margin = to_t(v['q_margin']).cpu().numpy()
        seg = a['q_seg'].astype(np.int64)
        coord = v['q_coord'].astype(np.float64)
        rgb = a['q_rgb'].astype(np.float64) / 255.0
        y = v['q_y'].astype(np.int64)
        uniq, inv = np.unique(seg, return_inverse=True)
        n_reg = len(uniq)
        cnt = np.bincount(inv, minlength=n_reg).astype(np.float64)
        m_sp = np.bincount(inv, weights=margin, minlength=n_reg) / np.maximum(cnt, 1)
        fg_frac = np.bincount(inv, weights=y.astype(np.float64),
                              minlength=n_reg) / np.maximum(cnt, 1)
        z_sp = sp_mean_norm(z, to_t(inv), n_reg)
        rgb_sp = np.zeros((n_reg, 3))
        coord_sp = np.zeros((n_reg, 3))
        for r in range(n_reg):
            sel = inv == r
            rgb_sp[r] = rgb[sel].mean(0)
            coord_sp[r] = coord[sel].mean(0)
        nrm_sp = sp_normals(coord, inv, n_reg)
        # 支持集背景 + P_base
        z_s = F.normalize(to_t(v['s_h']), dim=-1)
        s_seg = to_t(v['s_seg'].astype(np.int64))
        n_s = int(s_seg.max().item()) + 1
        z_sR = sp_mean_norm(z_s, s_seg, n_s)
        s_y = to_t(v['s_y'].astype(np.float32))
        fg_s = torch.zeros(n_s, device=z_s.device)
        fg_s.index_add_(0, s_seg, (s_y > 0.5).float())
        cnt_s = torch.bincount(s_seg, minlength=n_s).clamp_min(1).float()
        fg_s = fg_s / cnt_s
        D_S = z_sR[fg_s <= 0.0]
        P_base = F.normalize(table.to(z_s.dtype), dim=-1)
        # D_Q：查询集内 m(R) < -2.0 的纯背景超点
        D_Q = z_sp[torch.from_numpy(m_sp < -2.0).to(z_sp.device)]
        N_neg = torch.cat([x for x in (D_Q, D_S, P_base)
                           if x.shape[0] > 0], 0)
        eps.append({'m': m_sp, 'inv': inv, 'n_reg': n_reg, 'y': y,
                    'coord': coord, 'seg': seg, 'z_sp': z_sp,
                    'rgb_sp': rgb_sp, 'coord_sp': coord_sp,
                    'nrm_sp': nrm_sp, 'fg_frac': fg_frac,
                    'cls': int(m['class']), 'N_neg': N_neg,
                    'u_y': v['u_y'].astype(np.int64),
                    'u_seg': v['u_seg'].astype(np.int64),
                    'u_h': v['u_h'].astype(np.float16)})
        if (i + 1) % 100 == 0:
            print('  loaded %d / %d' % (i + 1, len(metas)), flush=True)
    return eps


def adjacent(e, B, C, dmax):
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
        if float(tree_b.query(pts)[0].min()) < dmax:
            keep.append(c)
    return np.array(keep, dtype=np.int64)


def score_adj(e, adj, B, kind):
    zc = e['z_sp'][adj]
    zb = e['z_sp'][B]
    sim = zc @ zb.t()
    if kind == 'cos':
        return sim.max(1).values
    if kind == 'vq':
        N = e['N_neg']
        if N.shape[0] == 0:
            return sim.max(1).values
        eta = (zb @ N.t()).max(1).values
        return (sim - eta.unsqueeze(0)).max(1).values
    rgc = e['rgb_sp'][adj]
    rgb = e['rgb_sp'][B]
    rgc_n = rgc / (np.linalg.norm(rgc, axis=1, keepdims=True) + 1e-6)
    rgb_n = rgb / (np.linalg.norm(rgb, axis=1, keepdims=True) + 1e-6)
    cos_rgb = torch.from_numpy(rgc_n @ rgb_n.T).to(zc.dtype).cuda()
    dz = np.abs(e['coord_sp'][adj][:, None, 2] - e['coord_sp'][B][None, :, 2])
    w_z = torch.from_numpy(np.exp(-dz / Z_SIGMA)).to(zc.dtype).cuda()
    if kind == 'rgb':
        return (0.5 * (cos_rgb + w_z)).max(1).values
    cos_n = torch.from_numpy(
        np.abs(e['nrm_sp'][adj] @ e['nrm_sp'][B].T)).to(zc.dtype).cuda()
    return ((cos_rgb + w_z + cos_n) / 3.0).max(1).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=560)
    ap.add_argument('--out', default=os.path.join(CACHE, 'phase2_v2.json'))
    args = ap.parse_args()

    torch.manual_seed(0)
    table = torch.load(CKPT, map_location='cpu')['state_dict'][
        'module.base_prototypes'].float().cuda()
    eps = load_eps(args.n, table)
    N = len(eps)
    print('episodes:', N, flush=True)

    base_ious = [iou_of(e['m'][e['inv']] > 0.5, e['y']) for e in eps]
    base_cum = cum_of([e['m'][e['inv']] > 0.5 for e in eps],
                      [e['y'] for e in eps], [e['cls'] for e in eps])
    I0 = float(np.mean(base_ious))
    print('baseline (a=1,theta=0.5) arith %.4f cum %.4f' % (I0, base_cum),
          flush=True)

    out = {'I0': I0, 'base_cum': base_cum, 'death': [], 'e2e': []}

    # ---------------- 生死检验 ----------------
    print('\n=== death test (Precision vs I0/(1+I0)=%.4f) ==='
          % (I0 / (1.0 + I0)), flush=True)
    THR = I0 / (1.0 + I0)
    for tau in TAU_SEEDS:
        cache_adj = {}
        for ei, e in enumerate(eps):
            B = np.where(e['m'] >= tau)[0]
            C = np.where((e['m'] < 0.5) & (e['m'] > -3.0))[0]
            for dmax in D_MAX:
                cache_adj[(ei, dmax)] = (B, adjacent(e, B, C, dmax))
        for kind in SCORES:
            for dmax in D_MAX:
                acc = {t: [0, 0] for t in TOPS}
                n_cand = 0
                for ei, e in enumerate(eps):
                    B, adj = cache_adj[(ei, dmax)]
                    if len(adj) < 3 or len(B) == 0:
                        continue
                    n_cand += len(adj)
                    s = score_adj(e, adj, B, kind).cpu().numpy()
                    order = np.argsort(-s)
                    for t in TOPS:
                        kk = max(int(len(adj) * t), 1)
                        sel = adj[order[:kk]]
                        yy = e['fg_frac'][sel] > 0.5
                        acc[t][0] += int(yy.sum())
                        acc[t][1] += int((~yy).sum())
                if n_cand == 0:
                    continue
                row = {'tau': tau, 'dmax': dmax, 'score': kind,
                       'n_cand': n_cand}
                line = '%s tau %.1f dmax %.2f n=%6d  ' % (kind, tau, dmax,
                                                           n_cand)
                for t in TOPS:
                    tp, fp = acc[t]
                    p = tp / max(tp + fp, 1)
                    row['p_top%d' % int(100 * t)] = p
                    line += 'T%d%%:%.3f%s ' % (100 * t, p,
                                               '*' if p > THR else ' ')
                print(line, flush=True)
                out['death'].append(row)

    # ---------------- 端到端（CSR 门控 V_Q>0） ----------------
    print('\n=== end-to-end with CSR gate ===', flush=True)
    configs = []
    for tau in TAU_SEEDS:
        B_all, adj_all = {}, {}
        for ei, e in enumerate(eps):
            B = np.where(e['m'] >= tau)[0]
            C = np.where((e['m'] < 0.5) & (e['m'] > -3.0))[0]
            B_all[ei] = B
            adj_all[ei] = adjacent(e, B, C, 0.10)   # 固定 dmax 以省时
        for kind in SCORES:
            for top in TOPS:
                ious, preds = [], []
                n_rec = 0
                for ei, e in enumerate(eps):
                    ell = e['m'][e['inv']].copy()
                    B, adj = B_all[ei], adj_all[ei]
                    if len(adj) and len(B):
                        s = score_adj(e, adj, B, kind)
                        if kind == 'vq':
                            keep = s > 0
                            adj2 = adj[keep.cpu().numpy()]
                            s2 = s[keep]
                        else:
                            adj2, s2 = adj, s
                        if len(adj2):
                            kk = max(int(len(adj2) * top), 1)
                            sel = adj2[torch.topk(s2, kk).indices
                                       .cpu().numpy()]
                            ell[np.isin(e['inv'], sel)] = 1.0
                            n_rec += len(sel)
                    pred = ell > 0.5
                    ious.append(iou_of(pred, e['y']))
                    preds.append(pred)
                cum = cum_of(preds, [e['y'] for e in eps],
                             [e['cls'] for e in eps])
                configs.append({'tau': tau, 'dmax': 0.10, 'score': kind,
                                'top': top, 'ious': np.array(ious),
                                'cum': cum, 'preds': preds})
                d = np.array(ious) - np.array(base_ious)
                pse = d.std(ddof=1) / np.sqrt(N)
                print('%-4s tau %.1f top %.0f%%  arith %.4f (d %+.4f t %+.2f)'
                      '  cum %.4f  rec=%d'
                      % (kind, tau, 100 * top, np.mean(ious), d.mean(),
                         d.mean() / pse if pse else 0, cum, n_rec),
                      flush=True)

    # ---------------- 2 折 CV ----------------
    half = N // 2
    folds = [list(range(half)), list(range(half, N))]
    blind = np.zeros(N)
    preds_blind = [None] * N
    picks = []
    for f in (0, 1):
        tr, te = folds[f], folds[1 - f]
        best = max(configs, key=lambda r: r['ious'][tr].mean())
        blind[te] = best['ious'][te]
        for i in te:
            preds_blind[i] = best['preds'][i]
        picks.append({'fold': f, 'tau': best['tau'], 'score': best['score'],
                      'top': best['top'],
                      'train': float(best['ious'][tr].mean()),
                      'cum_train': float(best['cum'])})
        print('fold %d: %s tau %.1f top %.0f%%  train %.4f  blind %.4f'
              % (f, best['score'], best['tau'], 100 * best['top'],
                 best['ious'][tr].mean(), best['ious'][te].mean()),
              flush=True)
    d = blind - np.array(base_ious)
    pse = d.std(ddof=1) / np.sqrt(N)
    blind_cum = cum_of(preds_blind, [e['y'] for e in eps],
                       [e['cls'] for e in eps])
    print('\n=== blind 560 ===')
    print('arith %.4f (d %+.4f)  pSE %.4f  t %+.2f  cum %.4f (d %+.4f)'
          % (blind.mean(), d.mean(), pse, d.mean() / pse if pse else 0,
             blind_cum, blind_cum - base_cum), flush=True)
    out['blind'] = {'arith': float(blind.mean()), 'delta': float(d.mean()),
                    'pse': float(pse),
                    't': float(d.mean() / pse) if pse else 0.0,
                    'cum': float(blind_cum),
                    'cum_delta': float(blind_cum - base_cum)}
    out['picks'] = picks
    out['configs'] = [{'tau': r['tau'], 'score': r['score'],
                       'top': r['top'], 'arith': float(r['ious'].mean()),
                       'cum': r['cum']} for r in configs]
    json.dump(out, open(args.out, 'w'), indent=1)
    print('->', args.out)


if __name__ == '__main__':
    main()
