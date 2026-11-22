"""Phase 3 v2：β_orth / 边界腐蚀 / 自适应阈值，全部带 Cumul 与 2 折 CV。"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

CACHE = '/data01/jh/data/ocsb_cache'
INTRA = os.path.join(CACHE, 'intra')

BETAS = (0.0, 0.2, 0.5, 0.8, 1.0)
ERODE = (0.0, 0.2, 0.3)
LAMS = (0.0, 0.5, 1.0, 2.0)
THETA0 = (0.4, 0.5, 0.6)
TAU = 0.1


DEV = torch.device('cuda:0')


def to_t(a):
    if a.dtype == np.float16:
        return torch.from_numpy(a.astype(np.float32)).to(DEV)
    if a.dtype.kind in 'iu':
        return torch.from_numpy(a.astype(np.int64)).to(DEV)
    return torch.from_numpy(a.astype(np.float32)).to(DEV)


def iou_of(pred, y):
    tp = int((pred & (y > 0)).sum())
    un = int((pred | (y > 0)).sum())
    return tp / max(un, 1)


def eval_preds(preds, data):
    ious, tp_c, un_c = [], {}, {}
    for pred, d in zip(preds, data):
        ious.append(iou_of(pred, d['y']))
        c = d['cls']
        tp_c[c] = tp_c.get(c, 0) + int((pred & (d['y'] > 0)).sum())
        un_c[c] = un_c.get(c, 0) + int((pred | (d['y'] > 0)).sum())
    cum = float(np.mean([tp_c[c] / max(un_c[c], 1) for c in tp_c]))
    return np.array(ious), cum


def cv_pick(configs, base, N, data=None):
    half = N // 2
    folds = [list(range(half)), list(range(half, N))]
    blind = np.zeros(N)
    preds_blind = [None] * N
    picks = []
    for f in (0, 1):
        tr, te = folds[f], folds[1 - f]
        best = max(configs, key=lambda r: r['ious'][tr].mean())
        blind[te] = best['ious'][te]
        if data is not None and 'preds' in best:
            for i in te:
                preds_blind[i] = best['preds'][i]
        picks.append({'fold': f, 'cfg': best['cfg'],
                      'train': float(best['ious'][tr].mean()),
                      'test': float(best['ious'][te].mean())})
    d = blind - base
    pse = d.std(ddof=1) / np.sqrt(N)
    stat = {'arith': float(blind.mean()), 'delta': float(d.mean()),
            'pse': float(pse),
            't': float(d.mean() / pse) if pse else 0.0}
    if data is not None and all(p is not None for p in preds_blind):
        _, cum = eval_preds(preds_blind, data)
        stat['cum'] = cum
    return blind, stat, picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=560)
    ap.add_argument('--out', default=os.path.join(CACHE, 'phase3_v2.json'))
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    if args.cpu:
        global DEV
        DEV = torch.device('cpu')

    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))[:args.n]
    data = []
    for i, m in enumerate(metas):
        v = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
        a = np.load(os.path.join(INTRA, 'intra_%03d.npz' % i))
        s_h = to_t(v['s_h'])
        q_h = to_t(v['q_h'])
        s_y = to_t(v['s_y'].astype(np.float32))
        seg = a['q_seg'].astype(np.int64)
        margin = to_t(v['q_margin']).cpu().numpy()
        y = v['q_y'].astype(np.int64)
        uniq, inv = np.unique(seg, return_inverse=True)
        n_reg = len(uniq)
        cnt = np.bincount(inv, minlength=n_reg).astype(np.float64)
        m_sp = np.bincount(inv, weights=margin, minlength=n_reg) / np.maximum(cnt, 1)
        data.append({'s_h': s_h, 'q_h': q_h, 's_y': s_y, 'seg': inv,
                     'n_reg': n_reg, 'y': y, 'cls': int(m['class']),
                     'margin': margin, 'm_sp': m_sp})
        if (i + 1) % 100 == 0:
            print('  loaded %d / %d' % (i + 1, len(metas)), flush=True)
    N = len(data)
    print('episodes:', N, flush=True)

    base_preds = [d['margin'] > 0 for d in data]
    base_ious, base_cum = eval_preds(base_preds, data)
    print('baseline (margin>0) arith %.4f cum %.4f'
          % (base_ious.mean(), base_cum), flush=True)

    out = {'baseline': {'arith': float(base_ious.mean()), 'cum': base_cum},
           'beta': [], 'erode': [], 'adapt': []}

    # ---------- 1) beta_orth ----------
    print('\n=== beta_orth ===', flush=True)
    cfgs = []
    for beta in BETAS:
        preds = []
        for d in data:
            z_s = F.normalize(d['s_h'], dim=-1)
            z_q = F.normalize(d['q_h'], dim=-1)
            fg = d['s_y'] > 0.5
            if fg.sum() == 0 or (~fg).sum() == 0:
                preds.append(d['margin'] > 0)
                continue
            p_bg = F.normalize(z_s[~fg].mean(0, keepdim=True), dim=-1)
            if beta > 0:
                z_s = F.normalize(z_s - beta * (z_s @ p_bg.t()) * p_bg, dim=-1)
                z_q = F.normalize(z_q - beta * (z_q @ p_bg.t()) * p_bg, dim=-1)
            p_fg = F.normalize(z_s[fg].mean(0, keepdim=True), dim=-1)
            s = (z_q @ p_fg.t()).squeeze(1) / TAU
            m_sp = np.bincount(d['seg'], weights=s.cpu().numpy().astype(np.float64),
                               minlength=d['n_reg']) / np.maximum(
                np.bincount(d['seg'], minlength=d['n_reg']), 1)
            preds.append(m_sp[d['seg']] > 0.5)
        ious, cum = eval_preds(preds, data)
        cfgs.append({'cfg': beta, 'ious': ious, 'cum': cum,
                     'preds': preds})
        print(' beta %.1f  arith %.4f (d %+.4f)  cum %.4f'
              % (beta, ious.mean(), ious.mean() - base_ious.mean(), cum),
              flush=True)
    blind, stat, picks = cv_pick(cfgs, base_ious, N, data)
    out['beta'] = [{'beta': c['cfg'], 'arith': float(c['ious'].mean()),
                    'cum': c['cum']} for c in cfgs]
    out['beta_cv'] = dict(stat, picks=picks)

    # ---------- 2) 边界腐蚀 ----------
    print('\n=== boundary erosion ===', flush=True)
    cfgs = []
    for frac in ERODE:
        preds = []
        for d in data:
            z_s = F.normalize(d['s_h'], dim=-1)
            z_q = F.normalize(d['q_h'], dim=-1)
            fg = d['s_y'] > 0.5
            if fg.sum() < 5 or (~fg).sum() == 0:
                preds.append(d['margin'] > 0)
                continue
            p_bg = F.normalize(z_s[~fg].mean(0, keepdim=True), dim=-1)
            sim_bg = (z_s[fg] @ p_bg.t()).squeeze(1)
            if frac > 0:
                th = float(np.quantile(sim_bg.cpu().numpy(), 1.0 - frac))
                keep = sim_bg < th
            else:
                keep = torch.ones_like(sim_bg, dtype=torch.bool)
            if int(keep.sum()) < 3:
                preds.append(d['margin'] > 0)
                continue
            p_fg = F.normalize(z_s[fg][keep].mean(0, keepdim=True), dim=-1)
            s = (z_q @ p_fg.t()).squeeze(1) / TAU
            m_sp = np.bincount(d['seg'], weights=s.cpu().numpy().astype(np.float64),
                               minlength=d['n_reg']) / np.maximum(
                np.bincount(d['seg'], minlength=d['n_reg']), 1)
            preds.append(m_sp[d['seg']] > 0.5)
        ious, cum = eval_preds(preds, data)
        cfgs.append({'cfg': frac, 'ious': ious, 'cum': cum,
                     'preds': preds})
        print(' erode %.1f  arith %.4f (d %+.4f)  cum %.4f'
              % (frac, ious.mean(), ious.mean() - base_ious.mean(), cum),
              flush=True)
    out['erode'] = [{'frac': c['cfg'], 'arith': float(c['ious'].mean()),
                     'cum': c['cum']} for c in cfgs]

    # ---------- 3) 自适应阈值 ----------
    print('\n=== adaptive theta ===', flush=True)
    cfgs = []
    for th0 in THETA0:
        for lam in LAMS:
            preds = []
            for d in data:
                fg0 = float((d['m_sp'] > th0).mean())
                th = th0 + lam * max(fg0 - 0.15, 0.0)
                preds.append(d['m_sp'][d['seg']] > th)
            ious, cum = eval_preds(preds, data)
            cfgs.append({'cfg': (th0, lam), 'ious': ious, 'cum': cum,
                         'preds': preds})
            print(' th0 %.1f lam %.1f  arith %.4f (d %+.4f)  cum %.4f'
                  % (th0, lam, ious.mean(), ious.mean() - base_ious.mean(),
                     cum), flush=True)
    blind, stat, picks = cv_pick(cfgs, base_ious, N, data)
    out['adapt'] = [{'theta0': c['cfg'][0], 'lam': c['cfg'][1],
                     'arith': float(c['ious'].mean()), 'cum': c['cum']}
                    for c in cfgs]
    out['adapt_cv'] = dict(stat, picks=picks)

    json.dump(out, open(args.out, 'w'), indent=1)
    print('->', args.out)


if __name__ == '__main__':
    main()
