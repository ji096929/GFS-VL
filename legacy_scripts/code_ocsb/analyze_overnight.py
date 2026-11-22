"""Overnight report analysis: Task 1/2/3/4 tables."""
import glob
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

pattern = sys.argv[1] if len(sys.argv) > 1 else '/data01/jh/data/diag_on_s*/records.jsonl'
recs = []
for f in sorted(glob.glob(pattern)):
    try:
        for line in open(f):
            recs.append(json.loads(line))
    except Exception as e:
        print('skip', f, e)
print('episodes:', len(recs))

base = [r['oracle_base_iou'] for r in recs if r.get('oracle_base_iou') is not None]
base_m = float(np.mean(base)) if base else float('nan')
print('\nbaseline query IoU: %.4f (n=%d)' % (base_m, len(base)))

print('\n## Task 1: Oracle ceilings')
print('config            IoU     delta    n')
configs = [
    ('B_pos lam0.1', 'oracle_iou_2c_0.1'),
    ('B_pos lam0.2', 'oracle_iou_2c_0.2'),
    ('B_pos lam0.3', 'oracle_iou_2c_0.3'),
    ('B_pos lam0.4', 'oracle_iou_2c_0.4'),
    ('B_pos lam0.5', 'oracle_iou_2c_0.5'),
    ('B_pos lam0.7', 'oracle_iou_2c_0.7'),
    ('B+C lam0.2', 'oracle_iou_2c_bc_0.2'),
    ('B+C lam0.3', 'oracle_iou_2c_bc_0.3'),
    ('AllFG lam0.2', 'oracle_iou_2c_allfg_0.2'),
    ('AllFG lam0.3', 'oracle_iou_2c_allfg_0.3'),
    ('centroid lam0.2', 'oracle_iou_2c_centroid_0.2'),
    ('centroid lam0.3', 'oracle_iou_2c_centroid_0.3'),
    ('centroid lam0.5', 'oracle_iou_2c_centroid_0.5'),
]
for name, key in configs:
    v = [r[key] for r in recs if r.get(key) is not None]
    if v:
        print('%-16s %.4f  %+.4f  %d' % (name, np.mean(v), np.mean(v) - base_m, len(v)))

d = [r['oracle_iou_2c_0.3'] - r['oracle_base_iou'] for r in recs
     if r.get('oracle_iou_2c_0.3') is not None and r.get('oracle_base_iou') is not None]
if d:
    print('B_pos lam0.3 per-episode: up %d, flat %d, down %d'
          % (sum(x > 1e-6 for x in d), sum(abs(x) <= 1e-6 for x in d),
             sum(x < -1e-6 for x in d)))

best = []
for r in recs:
    vals = [r.get('oracle_iou_2c_%.1f' % l) for l in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7)]
    vals = [v for v in vals if v is not None]
    if vals:
        best.append(max(vals))
if best:
    print('adaptive lambda (oracle best): %.4f delta=%+.4f'
          % (np.mean(best), np.mean(best) - base_m))

# episode-level gating: fall back to baseline when the bridge score is low
print('\nEpisode-level gating (unsupervised score threshold):')
for thresh in (0.4, 0.5, 0.6, 0.7):
    for key, tag in (('e2e_top3_lam0.3', 'top3-lam0.3'),
                     ('e2e_top2_lam0.3', 'top2-lam0.3')):
        vals = []
        for r in recs:
            b = r.get('oracle_base_iou')
            e = r.get(key)
            sc = r.get('_e2e_score')
            if b is None or e is None or not sc:
                continue
            vals.append(e if max(sc) >= thresh else b)
        if vals:
            print('thr=%.1f %-12s %.4f delta=%+.4f (n=%d)'
                  % (thresh, tag, np.mean(vals), np.mean(vals) - base_m,
                     len(vals)))

print('\n## Task 2: orthogonal signal AUC (B_pos vs B_neg)')
print('signal          pooled  per-ep  n_pos  n_neg')
signals = ['delta_base', 'base_abs', 'multi_max', 'margin', 'eta_intra',
           'spread', 'base_prob', 'size', 'q_std', 'q_p90',
           'sim_pool_bg', 'rel_pool_bg', 'delta_bg_risk',
           'spread_t005', 'spread_t02', 'spread_multi',
           'promiscuity', 'delta_mean', 'feat_var', 'local_rel']
for sig in signals:
    pos_all, neg_all, per = [], [], []
    for r in recs:
        p = r.get('B_pos_' + sig)
        n = r.get('B_neg_' + sig)
        if not p or not n:
            continue
        p = [x for x in p if x is not None and not np.isnan(x)]
        n = [x for x in n if x is not None and not np.isnan(x)]
        if not p or not n:
            continue
        pos_all += p
        neg_all += n
        try:
            per.append(roc_auc_score([1] * len(p) + [0] * len(n), p + n))
        except Exception:
            pass
    if not pos_all:
        print('%-15s n/a' % sig)
        continue
    y = [1] * len(pos_all) + [0] * len(neg_all)
    auc = roc_auc_score(y, pos_all + neg_all)
    auc_best = max(auc, 1 - auc)
    direction = 'keep' if auc >= 0.5 else 'flip'
    print('%-15s %.3f   %.3f   %s   %d   %d'
          % (sig, auc_best, np.mean(per) if per else float('nan'),
             direction, len(pos_all), len(neg_all)))

print('\n## Task 2b: Top-K precision by unsupervised score (multi_max)')
for K in (1, 2, 3):
    precs = []
    for r in recs:
        sc = r.get('_e2e_score')
        gt = r.get('B_gt')
        if not sc or not gt:
            continue
        order = np.argsort(sc)[::-1][:K]
        precs.append(np.mean([gt[i] for i in order]))
    if precs:
        print('Top-%d precision %.3f (n=%d)' % (K, np.mean(precs), len(precs)))

print('\n## Task 3: CSR recovery (C_pos recall / false recovery / precision)')
for eta in ('none', 'support_D', 'base_prob', 'pool_bg', 'combined'):
    row = []
    for metric in ('recall', 'false', 'precision'):
        key = 'csr_%s_%s' % (eta, metric)
        v = [r[key] for r in recs if r.get(key) is not None]
        row.append(np.mean(v) if v else float('nan'))
    print('%-10s recall=%.3f false=%.3f precision=%.3f' % (eta, row[0], row[1], row[2]))

print('\n## Task 4: end-to-end (no GT)')
for key in ('e2e_top2_lam0.2', 'e2e_top2_lam0.3',
            'e2e_top3_lam0.2', 'e2e_top3_lam0.3',
            'e2e_bc_lam0.2', 'e2e_bc_lam0.3',
            'e2e_qc_top2_lam0.2', 'e2e_qc_top2_lam0.3',
            'e2e_qc_top3_lam0.2', 'e2e_qc_top3_lam0.3'):
    v = [r[key] for r in recs if r.get(key) is not None]
    if v:
        print('%-18s %.4f  %+.4f (n=%d)'
              % (key, np.mean(v), np.mean(v) - base_m, len(v)))
