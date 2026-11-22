"""Per-class breakdown of the overnight experiments (offline)."""
import glob
import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

pattern = sys.argv[1] if len(sys.argv) > 1 else '/data01/jh/data/diag_on_s*/records.jsonl'
recs = []
for f in sorted(glob.glob(pattern)):
    try:
        for line in open(f):
            recs.append(json.loads(line))
    except Exception:
        pass

NAMES = {2: 'floor', 3: 'chair', 5: 'desk', 6: 'bed', 7: 'bookshelf',
         10: 'bathtub', 12: 'curtain', 13: 'counter', 14: 'door', 19: 'cabinet'}
by = defaultdict(list)
for r in recs:
    by[r['class']].append(r)
print('episodes:', len(recs), 'classes:', len(by))

print('\n## Task 1 per-class Oracle IoU')
print('class      name        n   base    B_pos0.5  d(B_pos)  B+C0.3   d(B+C)')
for c in sorted(by):
    rs = by[c]
    base = [r['oracle_base_iou'] for r in rs
            if r.get('oracle_base_iou') is not None]
    b5 = [r['oracle_iou_2c_0.5'] for r in rs
          if r.get('oracle_iou_2c_0.5') is not None]
    bc = [r['oracle_iou_2c_bc_0.3'] for r in rs
          if r.get('oracle_iou_2c_bc_0.3') is not None]
    if not base:
        continue
    mb = np.mean(base)
    print('%-10s %-10s %3d  %.4f  %.4f  %+.4f  %.4f  %+.4f'
          % (c, NAMES.get(c, '?'), len(rs), mb,
             np.mean(b5) if b5 else float('nan'),
             (np.mean(b5) - mb) if b5 else float('nan'),
             np.mean(bc) if bc else float('nan'),
             (np.mean(bc) - mb) if bc else float('nan')))

print('\n## Task 2 per-class signal AUC (B_pos vs B_neg)')
print('class      name        n_ep  multi_max  q_p90   eta_intra  spread')
for c in sorted(by):
    rs = by[c]
    row = []
    for sig in ('multi_max', 'q_p90', 'eta_intra', 'spread'):
        pos, neg = [], []
        for r in rs:
            p = r.get('B_pos_' + sig)
            n = r.get('B_neg_' + sig)
            if not p or not n:
                continue
            p = [x for x in p if x is not None and not np.isnan(x)]
            n = [x for x in n if x is not None and not np.isnan(x)]
            pos += p
            neg += n
        if pos and neg:
            auc = roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg)
            row.append(max(auc, 1 - auc))
        else:
            row.append(float('nan'))
    print('%-10s %-10s %3d  %.3f      %.3f   %.3f      %.3f'
          % (c, NAMES.get(c, '?'), len(rs), row[0], row[1], row[2], row[3]))

print('\n## Task 3 per-class CSR (pool_bg)')
print('class      name        n_ep  recall  false   precision')
for c in sorted(by):
    rs = by[c]
    vals = []
    for metric in ('recall', 'false', 'precision'):
        v = [r['csr_pool_bg_' + metric] for r in rs
             if r.get('csr_pool_bg_' + metric) is not None]
        vals.append(np.mean(v) if v else float('nan'))
    print('%-10s %-10s %3d  %.3f   %.3f   %.3f'
          % (c, NAMES.get(c, '?'), len(rs), vals[0], vals[1], vals[2]))

print('\n## Task 4 per-class end-to-end')
print('class      name        n_ep  base    e2e_top3_0.2  d      e2e_bc_0.2  d')
for c in sorted(by):
    rs = by[c]
    base = [r['oracle_base_iou'] for r in rs
            if r.get('oracle_base_iou') is not None]
    t3 = [r['e2e_top3_lam0.2'] for r in rs
          if r.get('e2e_top3_lam0.2') is not None]
    bc = [r['e2e_bc_lam0.2'] for r in rs
          if r.get('e2e_bc_lam0.2') is not None]
    if not base:
        continue
    mb = np.mean(base)
    print('%-10s %-10s %3d  %.4f  %.4f        %+.4f  %.4f      %+.4f'
          % (c, NAMES.get(c, '?'), len(rs), mb,
             np.mean(t3) if t3 else float('nan'),
             (np.mean(t3) - mb) if t3 else float('nan'),
             np.mean(bc) if bc else float('nan'),
             (np.mean(bc) - mb) if bc else float('nan')))
