"""How does B precision decay as K grows, for the scores we could use?

The end-to-end runs only evaluated K in {2,3} while the oracle used every B_pos
(~33 per episode on the same-scene runs). If precision holds up to much larger
K, the low realisation rate is a K problem, not a selection problem.
"""
import glob
import json
import sys

import numpy as np

pattern = sys.argv[1] if len(sys.argv) > 1 else '/data01/jh/data/diag_same_s12[13]/records.jsonl'
recs = []
for f in sorted(glob.glob(pattern)):
    try:
        for line in open(f):
            recs.append(json.loads(line))
    except Exception:
        pass
print('episodes:', len(recs))

SCORES = ('B_q', 'B_query_rel', '_e2e_score', '_e2e_score_combo')


def topk_prec(rec, score_name, K):
    gts = rec.get('B_gt')
    sc = rec.get(score_name)
    if not gts or sc is None or len(sc) != len(gts):
        return None
    order = np.argsort(sc)[::-1][:K]
    return float(np.mean([gts[i] for i in order]))


print('\n## Top-K precision by score (same-scene)')
print('%-14s %6s %6s %6s %6s %6s %6s %6s %6s'
      % ('score', 'K=1', '2', '3', '5', '10', '20', '40', 'all'))
for s in SCORES:
    row = []
    for K in (1, 2, 3, 5, 10, 20, 40, 10 ** 9):
        v = [topk_prec(r, s, K) for r in recs]
        v = [x for x in v if x is not None]
        row.append(np.mean(v) if v else np.nan)
    print('%-14s ' % s + ' '.join('%6.3f' % x for x in row))

print('\n## How many B are there (so "all" is comparable)')
nB = np.array([r.get('n_B', 0) for r in recs], dtype=float)
npos = np.array([r.get('n_B_pos', 0) for r in recs], dtype=float)
print('n_B mean %.1f median %.1f   n_B_pos mean %.1f' % (nB.mean(), np.median(nB), npos.mean()))

print('\n## Fraction of total B_pos captured at each K')
for K in (3, 5, 10, 20, 40):
    if K == 10 ** 9:
        continue
    frac = [min(K, r.get('n_B', 0)) / max(r.get('n_B_pos', 1), 1) for r in recs]
    print('K=%-3d captures %.1f%% of available B_pos' % (K, 100 * np.mean(frac)))

print('\n## Expected precision at matched recall')
# oracle uses all B_pos, so compare precision at the K that recovers the same
# number of positives on average
for s in ("B_q", "B_query_rel", "_e2e_score_combo"):
    best = None
    for K in (5, 10, 15, 20, 30, 40, 60):
        v = [topk_prec(r, s, K) for r in recs]
        v = [x for x in v if x is not None]
        cov = np.mean([min(K, r.get('n_B', 0)) / max(r.get('n_B_pos', 1), 1) for r in recs])
        if cov >= 0.95:
            best = (K, np.mean(v), cov)
            break
    if best:
        print('%-14s K=%d precision %.3f (captures %.0f%% of B_pos)'
              % (s, best[0], best[1], 100 * best[2]))
