"""Phase 2: does the physical-appearance referee separate B_pos from B_neg
better than F0-derived signals (whose ceiling was 0.610)?

Feedback is per candidate bridge:
  phys_s_pos -- cosine similarity to the support foreground prototype in
                (RGB, relative-height) space
  phys_eta   -- negative-control risk against high-confidence background in U
  phys_V     -- s_pos - eta  (net orthogonal contribution)
  phys_qc    -- the original q_c, for reference
"""
import glob
import json
import sys

import numpy as np

FILES = {'cross': '/data01/jh/data/orth_x/records.jsonl',
         'same': '/data01/jh/data/orth_s/records.jsonl'}


def auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return np.nan
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


def topk_prec(recs, score_key, k):
    out = []
    for r in recs:
        y = r.get('B_gt')
        s = r.get(score_key)
        if not y or not s or len(y) != len(s) or len(y) == 0:
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
    recs = [r for r in recs if r.get('phys_V') and r.get('B_gt')]
    if len(recs) < 10:
        print('%s: only %d usable' % (scen, len(recs)))
        continue

    print('\n' + '=' * 70)
    print('%s  episodes=%d' % (scen, len(recs)))
    pos = sum(sum(r['B_gt']) for r in recs)
    neg = sum(len(r['B_gt']) - sum(r['B_gt']) for r in recs)
    print('candidates: B_pos %d  B_neg %d  (precision %.3f)'
          % (pos, neg, pos / max(pos + neg, 1)))

    print('\n  score            pooled AUC   per-ep AUC   Top1   Top2   Top3   Top5')
    for name in ('phys_qc', 'phys_s_pos', 'phys_eta', 'phys_V'):
        py, ps, per = [], [], []
        for r in recs:
            y = [int(v) for v in r['B_gt']]
            s = r.get(name)
            if not s or len(y) != len(s):
                continue
            py += y
            ps += list(s)
            a = auc(y, s)
            if not np.isnan(a):
                per.append(a)
        if not ps:
            continue
        pooled = auc(py, ps)
        row = '  %-16s %10.3f %12.3f' % (name, pooled, np.mean(per))
        for k in (1, 2, 3, 5):
            row += ' %6.3f' % topk_prec(recs, name, k)
        print(row)

    # joint score with q_c
    print('\n  joint = z(q_c) + gamma * z(V_orth):')
    print('  %-7s %10s %12s %8s %8s' % ('gamma', 'pooled', 'per-ep', 'T2', 'T3'))
    for gamma in (0.0, 0.25, 0.5, 1.0, 2.0):
        py, ps, per = [], [], []
        for r in recs:
            y = [int(v) for v in r['B_gt']]
            qc = r.get('phys_qc')
            v = r.get('phys_V')
            if not qc or not v or len(y) != len(qc):
                continue
            qc = np.array(qc, float)
            v = np.array(v, float)
            zq = (qc - qc.mean()) / (qc.std() + 1e-6)
            zv = (v - v.mean()) / (v.std() + 1e-6)
            sj = zq + gamma * zv
            py += y
            ps += list(sj)
            a = auc(y, sj)
            if not np.isnan(a):
                per.append(a)
        if ps:
            print('  %-7.2f %10.3f %12.3f %8.3f %8.3f'
                  % (gamma, auc(py, ps), np.mean(per),
                     np.nan, np.nan))
