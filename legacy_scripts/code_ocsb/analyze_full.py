"""Task 2/3 on the full pool: old 502 (v4/v5) + new full-field run."""
import glob
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


def load(patterns):
    recs = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            try:
                for line in open(f):
                    recs.append(json.loads(line))
            except Exception:
                pass
    return recs


old = load(['/data01/jh/data/diag_v4_s*/records.jsonl',
            '/data01/jh/data/diag_v5_s*/records.jsonl'])
new = load(sys.argv[1:] or ['/data01/jh/data/diag_full_s*/records.jsonl'])
print('old:', len(old), ' new:', len(new), ' total:', len(old) + len(new))

print('\n## Task 2: signal AUC on the full pool (intersection signals)')
print('pool      signal        pooled  best    per-ep  n_pos   n_neg')
for pool, tag in ((old, 'old-502'), (old + new, 'all')):
    for sig in ('multi_max', 'delta_base', 'eta_intra', 'spread'):
        pos, neg, per = [], [], []
        for r in pool:
            p = r.get('B_pos_' + sig)
            n = r.get('B_neg_' + sig)
            if not p or not n:
                continue
            p = [x for x in p if x is not None and not np.isnan(x)]
            n = [x for x in n if x is not None and not np.isnan(x)]
            if not p or not n:
                continue
            pos += p
            neg += n
            try:
                per.append(roc_auc_score([1] * len(p) + [0] * len(n), p + n))
            except ValueError:
                pass
        if pos and neg:
            y = [1] * len(pos) + [0] * len(neg)
            auc = roc_auc_score(y, pos + neg)
            print('%-9s %-13s %.3f   %.3f   %.3f   %d   %d'
                  % (tag, sig, auc, max(auc, 1 - auc),
                     np.mean(per) if per else float('nan'), len(pos), len(neg)))

print('\n## Task 2b: Top-K precision by q_c vs multi_max (full pool)')
for score_key, tag in (('B_q_placeholder', 'q_c(missing)'),
                       ('B_pos_multi_max', 'multi_max(only pos)')):
    pass
# q_c per B is only stored in the new run as B_q
precs = {}
for K in (1, 2, 3):
    qc, mm = [], []
    for r in new:
        q = r.get('B_q')
        gt = r.get('B_gt')
        if not q or not gt:
            continue
        order = np.argsort(q)[::-1][:K]
        qc.append(np.mean([gt[i] for i in order]))
        sc = r.get('_e2e_score_combo') or r.get('_e2e_score')
        if sc:
            order2 = np.argsort(sc)[::-1][:K]
            mm.append(np.mean([gt[i] for i in order2]))
    print('K=%d  q_c %.3f (n=%d)   combo %.3f (n=%d)'
          % (K, np.mean(qc) if qc else float('nan'), len(qc),
             np.mean(mm) if mm else float('nan'), len(mm)))

print('\n## Task 3: CSR on the new pool (all eta variants)')
for eta in ('none', 'support_D', 'base_prob', 'pool_bg', 'combined'):
    row = []
    for metric in ('recall', 'false', 'precision'):
        v = [r['csr_%s_%s' % (eta, metric)] for r in new
             if r.get('csr_%s_%s' % (eta, metric)) is not None]
        row.append(np.mean(v) if v else float('nan'))
    print('%-10s recall=%.3f false=%.3f precision=%.3f'
          % (eta, row[0], row[1], row[2]))
