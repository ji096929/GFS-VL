"""Top-K bridge truncation + Oracle injection analysis."""
import glob
import json
import sys

import numpy as np

pattern = sys.argv[1] if len(sys.argv) > 1 else '/data01/jh/data/diag_or_s*/records.jsonl'
recs = []
for f in sorted(glob.glob(pattern)):
    try:
        for line in open(f):
            recs.append(json.loads(line))
    except Exception as e:
        print('skip', f, e)

print('episodes:', len(recs))

# ---- Top-K truncation: precision of top-K B by q_c(R) ----
print('\n## Top-K truncation (B sorted by q_c(R))')
print('K   precision   avg_n_selected')
for K in (1, 3, 5, 10, 20):
    precs = []
    for r in recs:
        q = r.get('B_q')
        gt = r.get('B_gt')
        if not q or not gt:
            continue
        order = np.argsort(q)[::-1][:K]
        precs.append(np.mean([gt[i] for i in order]))
    if precs:
        print('%-3d %.3f' % (K, np.mean(precs)))

# ---- tau=0.6 + Top-K combined ----
print('\n## tau=0.6 + Top-K')
for K in (3, 5, 10):
    precs = []
    for r in recs:
        q = r.get('B_q')
        gt = r.get('B_gt')
        if not q or not gt:
            continue
        idx = [i for i, v in enumerate(q) if v >= 0.6]
        if not idx:
            continue
        idx = sorted(idx, key=lambda i: -q[i])[:K]
        precs.append(np.mean([gt[i] for i in idx]))
    if precs:
        print('K=%-2d precision=%.3f (n=%d eps with B)' % (K, np.mean(precs), len(precs)))

# ---- Oracle injection into query ----
print('\n## Oracle bridge injection (query IoU)')
base = [r['oracle_base_iou'] for r in recs if r.get('oracle_base_iou') is not None]
print('base (no bridge): %.4f (n=%d)' % (np.mean(base), len(base)))
for lam in ('0.5', '1.0', '2.0'):
    key = 'oracle_iou_lam' + lam
    v = [r[key] for r in recs if r.get(key) is not None]
    if v:
        print('logit-lam=%-4s %.4f (n=%d)  delta=%+.4f'
              % (lam, np.mean(v), len(v), np.mean(v) - np.mean(base)))

for key, label in (('oracle_iou_2a', '2A point-pool'),
                   ('oracle_iou_2c_0.3', '2C fuse 0.3'),
                   ('oracle_iou_2c_0.5', '2C fuse 0.5')):
    v = [r[key] for r in recs if r.get(key) is not None]
    if v:
        print('%-16s %.4f (n=%d)  delta=%+.4f'
              % (label, np.mean(v), len(v), np.mean(v) - np.mean(base)))

# error count
errs = [r for r in recs if 'oracle_err' in r]
print('\noracle errors:', len(errs), (errs[0]['oracle_err'] if errs else ''))
