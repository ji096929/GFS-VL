"""Compare cross-scene vs same-scene U without relying on episode pairing.

Pairing turned out to be impossible: np.random.choice(population, size,
replace=False) consumes a number of draws that depends on population size, so
the two runs diverge right after the first U sampling. Instead we compare the
two pools as wholes, and stratify by class because class difficulty dominates
everything else here.
"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def load(pattern):
    recs = []
    for f in sorted(glob.glob(pattern)):
        try:
            for line in open(f):
                recs.append(json.loads(line))
        except Exception:
            pass
    return recs


cross = load(sys.argv[1] if len(sys.argv) > 1
             else '/data01/jh/data/diag_gate_s12*/records.jsonl')
same = load(sys.argv[2] if len(sys.argv) > 2
            else '/data01/jh/data/diag_same_s12*/records.jsonl')
print('cross-scene: %d episodes' % len(cross))
print('same-scene : %d episodes' % len(same))
if not same:
    raise SystemExit('no same-scene records yet')

CLASS_NAMES = {2: 'floor', 3: 'chair', 4: 'table', 5: 'desk', 6: 'bed',
               7: 'bookshelf', 8: 'sofa', 9: 'sink', 10: 'bathtub',
               11: 'toilet', 12: 'curtain', 13: 'counter', 14: 'door',
               15: 'wall', 16: 'window', 17: 'books', 18: 'picture',
               19: 'cabinet', 20: 'otherfurniture'}


def summary(recs, label):
    base = np.array([r.get('oracle_base_iou') for r in recs], dtype=float)
    nB = np.array([r.get('n_B', 0) for r in recs], dtype=float)
    npos = np.array([r.get('n_B_pos', 0) for r in recs], dtype=float)
    print('\n## %s' % label)
    print('episodes %d   baseline IoU %.4f +/- %.3f'
          % (len(recs), np.nanmean(base), np.nanstd(base)))
    print('n_B %.1f   B_pos %.1f   precision %.3f'
          % (nB.mean(), npos.mean(), (npos / np.maximum(nB, 1)).mean()))
    print('%-22s %+8s' % ('oracle config', 'delta'))
    for k in ('oracle_iou_2c_0.1', 'oracle_iou_2c_0.2', 'oracle_iou_2c_0.3',
              'oracle_iou_2c_0.4', 'oracle_iou_2c_0.5', 'oracle_iou_2c_0.7',
              'oracle_iou_2a', 'oracle_iou_2c_centroid_0.3'):
        v = np.array([r.get(k, np.nan) for r in recs], dtype=float)
        ok = ~np.isnan(v) & ~np.isnan(base)
        if ok.sum() < 5:
            continue
        print('%-22s %+8.4f'
              % (k.replace('oracle_iou_', ''), (v[ok] - base[ok]).mean()))
    print('%-22s %+8.4f' % ('e2e qc_top3_lam0.3',
                            np.nanmean([r.get('e2e_qc_top3_lam0.3', np.nan)
                                        - r.get('oracle_base_iou', np.nan)
                                        for r in recs])))
    return base


summary(cross, 'cross-scene')
summary(same, 'same-scene')

print('\n## Per-class baseline and B precision (classes with >=8 eps in both)')
print('%-12s %5s %5s | %8s %8s | %8s %8s | %8s %8s'
      % ('class', 'nC', 'nS', 'baseC', 'baseS', 'precC', 'precS',
         'gainC', 'gainS'))
for c in sorted(CLASS_NAMES):
    rc = [r for r in cross if r['class'] == c]
    rs = [r for r in same if r['class'] == c]
    if len(rc) < 8 or len(rs) < 8:
        continue
    bc = np.array([r['oracle_base_iou'] for r in rc])
    bs = np.array([r['oracle_base_iou'] for r in rs])
    pc = np.mean([r['n_B_pos'] / max(r['n_B'], 1) for r in rc])
    ps = np.mean([r['n_B_pos'] / max(r['n_B'], 1) for r in rs])
    gc = np.nanmean([r.get('oracle_iou_2c_0.5', np.nan) - r['oracle_base_iou']
                     for r in rc])
    gs = np.nanmean([r.get('oracle_iou_2c_0.5', np.nan) - r['oracle_base_iou']
                     for r in rs])
    print('%-12s %5d %5d | %8.4f %8.4f | %8.3f %8.3f | %+8.4f %+8.4f'
          % (CLASS_NAMES[c], len(rc), len(rs), bc.mean(), bs.mean(),
             pc, ps, gc, gs))

print('\n## Signal AUC')
print('%-12s %8s %8s %8s %8s' % ('signal', 'AUCc', 'AUCs', 'dC', 'dS'))


def cohens_d(pos, neg):
    p, n = np.array(pos), np.array(neg)
    sp = np.sqrt(((len(p) - 1) * p.std(ddof=1) ** 2
                  + (len(n) - 1) * n.std(ddof=1) ** 2)
                 / max(len(p) + len(n) - 2, 1))
    return (p.mean() - n.mean()) / max(sp, 1e-8)


for name in ('q_p90', 'margin', 'multi_max', 'base_abs', 'local_rel',
             'eta_intra', 'q_std', 'spread'):
    row = []
    for recs in (cross, same):
        pos, neg = [], []
        for r in recs:
            pos += [v for v in (r.get('B_pos_' + name) or [])
                    if v is not None and not np.isnan(v)]
            neg += [v for v in (r.get('B_neg_' + name) or [])
                    if v is not None and not np.isnan(v)]
        if len(pos) < 50 or len(neg) < 50:
            row.append((np.nan, np.nan))
            continue
        a = roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg)
        row.append((max(a, 1 - a), cohens_d(pos, neg)))
    if np.isnan(row[0][0]) or np.isnan(row[1][0]):
        continue
    print('%-12s %8.3f %8.3f %+8.3f %+8.3f'
          % (name, row[0][0], row[1][0], row[0][1], row[1][1]))

print('\n## Correlation between B confidence and query accuracy')
for label, recs in (('cross', cross), ('same', same)):
    qm, iou = [], []
    for r in recs:
        qs = (r.get('B_pos_q_p90') or []) + (r.get('B_neg_q_p90') or [])
        if not qs:
            continue
        qm.append(np.mean(qs))
        iou.append(r.get('oracle_base_iou', np.nan))
    qm, iou = np.array(qm), np.array(iou)
    ok = ~np.isnan(iou)
    print('%-6s corr(mean B q_c, baseline IoU) = %+.3f  (n=%d)'
          % (label, np.corrcoef(qm[ok], iou[ok])[0, 1], ok.sum()))
