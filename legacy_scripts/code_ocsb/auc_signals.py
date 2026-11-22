"""AUC of candidate negative-control signals: B_pos vs B_neg per episode."""
import glob
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

pattern = sys.argv[1] if len(sys.argv) > 1 else '/data01/jh/data/diag_up_s*/records.jsonl'
recs = []
for f in sorted(glob.glob(pattern)):
    try:
        for line in open(f):
            recs.append(json.loads(line))
    except Exception as e:
        print('skip', f, e)

print('episodes:', len(recs))
signals = ['delta_base', 'base_abs', 'multi_max', 'margin', 'eta_intra', 'spread']
for sig in signals:
    pos_all, neg_all, per_ep = [], [], []
    for r in recs:
        p = r.get('B_pos_' + sig)
        n = r.get('B_neg_' + sig)
        if not p or not n:
            continue
        pos_all += p
        neg_all += n
        y = [1] * len(p) + [0] * len(n)
        try:
            per_ep.append(roc_auc_score(y, p + n))
        except ValueError:
            pass
    if not pos_all or not neg_all:
        print('%-12s n/a' % sig)
        continue
    y = [1] * len(pos_all) + [0] * len(neg_all)
    pooled = roc_auc_score(y, pos_all + neg_all)
    print('%-12s pooled=%.3f  per-ep=%.3f  n_ep=%d  n_pos=%d  n_neg=%d'
          % (sig, pooled, np.mean(per_ep) if per_ep else float('nan'),
             len(per_ep), len(pos_all), len(neg_all)))

# also report the old support-negative signal for comparison
bp, bn = [], []
for r in recs:
    a = r.get('mean_sim_Bpos_D')
    b = r.get('mean_sim_Bneg_D')
    if a is not None and b is not None:
        bp.append(a)
        bn.append(b)
if bp:
    y = [1] * len(bp) + [0] * len(bn)
    print('%-12s pooled=%.3f  (episode-level means, old signal)'
          % ('support_D', roc_auc_score(y, bp + bn)))
