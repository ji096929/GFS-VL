"""Task 1.5 completion: real query-correlation gate for the bridge injection.

For each episode, if the strongest candidate B is only weakly correlated with
any query point, set lambda=0 (fall back to baseline). Compare thresholds.
"""
import glob
import json
import sys

import numpy as np

pattern = sys.argv[1] if len(sys.argv) > 1 else '/data01/jh/data/diag_gate_s*/records.jsonl'
recs = []
for f in sorted(glob.glob(pattern)):
    try:
        for line in open(f):
            recs.append(json.loads(line))
    except Exception:
        pass
print('episodes:', len(recs))

rel_all = []
for r in recs:
    rel = r.get('B_query_rel')
    if rel:
        rel_all += rel
if rel_all:
    a = np.array(rel_all)
    print('B_query_rel: mean %.3f  median %.3f  p10 %.3f  p90 %.3f  max %.3f'
          % (a.mean(), np.median(a), np.percentile(a, 10),
             np.percentile(a, 90), a.max()))

base = [r['oracle_base_iou'] for r in recs if r.get('oracle_base_iou') is not None]
base_m = float(np.mean(base)) if base else float('nan')
print('\nbaseline: %.4f (n=%d)' % (base_m, len(base)))

print('\n## Query-correlation gate (top3 lam=0.3)')
print('thr    kept  IoU     delta')
for thr in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
    vals, kept = [], 0
    for r in recs:
        b = r.get('oracle_base_iou')
        e = r.get('e2e_top3_lam0.3')
        rel = r.get('B_query_rel')
        if b is None or e is None or not rel:
            continue
        if max(rel) >= thr:
            vals.append(e)
            kept += 1
        else:
            vals.append(b)
    if vals:
        print('%.1f  %4d  %.4f  %+.4f'
              % (thr, kept, np.mean(vals), np.mean(vals) - base_m))

print('\n## Query-correlation as a B selection score (Top-K)')
for K in (1, 2, 3):
    precs = []
    for r in recs:
        rel = r.get('B_query_rel')
        gts = r.get('B_gt')
        if not rel or not gts or len(rel) != len(gts):
            continue
        order = np.argsort(rel)[::-1][:K]
        precs.append(np.mean([gts[i] for i in order]))
    if precs:
        print('top%d %.3f (n=%d)' % (K, np.mean(precs), len(precs)))
