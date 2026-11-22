"""Phase 2: does the CLIP referee break the 0.610 AUC ceiling?

Two variants are recorded:
  clip_*_r -- raw CLIP features
  clip_*_c -- WARM-style episode centering (subtract pool mean, renormalise)
plus q_c as the F0-only baseline. Dual-expert arbitration keeps a bridge only
if V_orth > 0 (one-vote veto).
"""
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

FILES = {'cross': '/data01/jh/data/cx2/records.jsonl',
         'same': '/data01/jh/data/cs2/records.jsonl'}


def auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    ok = ~np.isnan(s)
    if len(np.unique(y[ok])) < 2:
        return np.nan
    return roc_auc_score(y[ok], s[ok])


def topk(recs, key, k, mask=None):
    out = []
    for r in recs:
        y = r.get('B_gt')
        s = r.get(key)
        if not y or not s or len(y) != len(s):
            continue
        s = np.asarray(s, float)
        if mask is not None:
            keep = mask(y, s, r)
            if not keep.any():
                continue
            y = [y[i] for i in range(len(y)) if keep[i]]
            s = s[keep]
            if not y:
                continue
        order = np.argsort(s)[::-1][:k]
        out.append(np.mean([y[i] for i in order]))
    return np.mean(out) if out else np.nan


for scen, path in FILES.items():
    try:
        recs = [json.loads(l) for l in open(path)]
    except Exception:
        print('%s: not ready' % scen)
        continue
    recs = [r for r in recs if r.get('B_gt') and r.get('clip_V_c')]
    if len(recs) < 10:
        print('%s: only %d usable' % (scen, len(recs)))
        continue

    pos = sum(sum(r['B_gt']) for r in recs)
    neg = sum(len(r['B_gt']) - sum(r['B_gt']) for r in recs)
    print('\n' + '=' * 76)
    print('%s  eps=%d  B_pos %d  B_neg %d  base precision %.3f'
          % (scen, len(recs), pos, neg, pos / max(pos + neg, 1)))

    print('\n  %-18s %10s %12s %7s %7s %7s %7s'
          % ('score', 'pooled', 'per-ep', 'T1', 'T2', 'T3', 'T5'))
    for key in ('clip_qc', 'clip_s_pos_r', 'clip_eta_r', 'clip_V_r',
                'clip_s_pos_c', 'clip_eta_c', 'clip_V_c'):
        py, ps, per = [], [], []
        for r in recs:
            y = [int(v) for v in r['B_gt']]
            s = r.get(key)
            if not s or len(y) != len(s):
                continue
            py += y
            ps += list(s)
            a = auc(y, s)
            if not np.isnan(a):
                per.append(a)
        if not ps:
            continue
        row = '  %-18s %10.3f %12.3f' % (key, auc(py, ps), np.mean(per))
        for k in (1, 2, 3, 5):
            row += ' %7.3f' % topk(recs, key, k)
        print(row)

    # dual-expert: keep only bridges where V_c > 0, then rank by q_c
    print('\n  dual-expert arbitration (V_c > 0 one-vote veto, then rank by q_c):')
    py, ps, per, kept, tot = [], [], [], 0, 0
    for r in recs:
        y = [int(v) for v in r['B_gt']]
        v = np.asarray(r['clip_V_c'], float)
        q = np.asarray(r['clip_qc'], float)
        keep = v > 0
        tot += len(y)
        kept += int(keep.sum())
        if keep.sum() == 0:
            continue
        yy = [y[i] for i in range(len(y)) if keep[i]]
        qq = q[keep]
        py += yy
        ps += list(qq)
        a = auc(yy, qq)
        if not np.isnan(a):
            per.append(a)
    print('    kept %.1f%% of candidates; pooled AUC %.3f; per-ep %.3f'
          % (100 * kept / max(tot, 1), auc(py, ps) if ps else np.nan,
             np.mean(per) if per else np.nan))
    # precision of the vetoed set
    prec_kept = []
    for r in recs:
        y = np.asarray([int(v) for v in r['B_gt']])
        v = np.asarray(r['clip_V_c'], float)
        if (v > 0).any():
            prec_kept.append(y[v > 0].mean())
    if prec_kept:
        print('    mean precision after veto %.3f (baseline %.3f)'
              % (np.mean(prec_kept), pos / max(pos + neg, 1)))
