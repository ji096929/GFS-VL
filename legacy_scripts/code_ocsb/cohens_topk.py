"""Task 2 completion: Cohen's d + per-signal Top-K B precision."""
import glob
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

pattern = sys.argv[1] if len(sys.argv) > 1 else '/data01/jh/data/diag_502_s*/records.jsonl'
recs = []
for f in sorted(glob.glob(pattern)):
    try:
        for line in open(f):
            recs.append(json.loads(line))
    except Exception:
        pass
print('episodes:', len(recs))

signals = ['delta_base', 'base_abs', 'multi_max', 'margin', 'eta_intra',
           'spread', 'base_prob', 'size', 'q_std', 'q_p90',
           'sim_pool_bg', 'rel_pool_bg', 'delta_bg_risk',
           'spread_t005', 'spread_t02', 'spread_multi',
           'promiscuity', 'delta_mean', 'feat_var', 'local_rel']

print('\nsignal          pooled  per-ep  cohens_d  top1   top2   top3   dir')
print('-' * 78)
for sig in signals:
    pos_all, neg_all, per = [], [], []
    precs = {1: [], 2: [], 3: []}
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
        except ValueError:
            pass
        vals = np.array(p + n)
        gts = np.array([1] * len(p) + [0] * len(n))
        order = np.argsort(vals)[::-1]
        for K in (1, 2, 3):
            if len(order) >= K:
                precs[K].append(gts[order[:K]].mean())
    if not pos_all or not neg_all:
        print('%-15s n/a' % sig)
        continue
    pv = np.array(pos_all)
    nv = np.array(neg_all)
    sp = np.sqrt(
        ((len(pv) - 1) * pv.std(ddof=1) ** 2
         + (len(nv) - 1) * nv.std(ddof=1) ** 2) / (len(pv) + len(nv) - 2)
    )
    d = (pv.mean() - nv.mean()) / max(sp, 1e-8)
    y = [1] * len(pv) + [0] * len(nv)
    auc = roc_auc_score(y, list(pv) + list(nv))
    direction = 'keep' if auc >= 0.5 else 'flip'
    print('%-15s %.3f   %.3f   %+.3f     %.3f  %.3f  %.3f  %s'
          % (sig, max(auc, 1 - auc),
             np.mean(per) if per else float('nan'), d,
             np.mean(precs[1]) if precs[1] else float('nan'),
             np.mean(precs[2]) if precs[2] else float('nan'),
             np.mean(precs[3]) if precs[3] else float('nan'),
             direction))
