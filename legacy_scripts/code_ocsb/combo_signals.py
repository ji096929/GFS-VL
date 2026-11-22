"""Task 2c: linear combination search over orthogonal signals.

Per-bridge signals are combined after per-episode z-scoring, then evaluated by
per-episode AUC and Top-K precision.
"""
import glob
import json
import itertools
import sys

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
print('episodes:', len(recs))

# direction that makes each signal point toward "true bridge"
SIGNS = {
    'multi_max': 1.0,
    'q_p90': 1.0,
    'delta_base': -1.0,
    'eta_intra': -1.0,
    'spread': -1.0,
    'q_std': -1.0,
    'base_abs': 1.0,
    'margin': 1.0,
    'size': -1.0,
    'q_c': 1.0,
}

# build per-episode arrays: keys present for both B_pos and B_neg
episodes = []
for r in recs:
    avail = {}
    for sig, sgn in SIGNS.items():
        p = r.get('B_pos_' + sig)
        n = r.get('B_neg_' + sig)
        if not p or not n:
            continue
        p = np.array(p, dtype=float)
        n = np.array(n, dtype=float)
        # fill NaN (e.g. eta_intra has no same-block bg) with the median so
        # every signal keeps the same length per episode
        both = np.concatenate([p, n])
        med = float(np.nanmedian(both)) if not np.all(np.isnan(both)) else 0.0
        p = np.where(np.isnan(p), med, p)
        n = np.where(np.isnan(n), med, n)
        avail[sig] = (sgn * p, sgn * n)
    # q_c(B) is stored as B_q with B_gt, split into pos/neg
    if 'B_q' in r and 'B_gt' in r:
        qv = np.array(r['B_q'], dtype=float)
        gv = np.array(r['B_gt'], dtype=float)
        if len(qv) == len(gv) and len(qv) > 0:
            p = qv[gv == 1]
            n = qv[gv == 0]
            if len(p) > 0 and len(n) > 0:
                avail['q_c'] = (1.0 * p, 1.0 * n)
    if avail:
        episodes.append(avail)

print('episodes with signals:', len(episodes))
sigs = sorted(SIGNS)


def evaluate(weights):
    """weights: dict sig -> w. Returns (per-ep AUC, top1/2/3 precision)."""
    aucs, p1, p2, p3 = [], [], [], []
    for ep in episodes:
        use = [s for s in weights if s in ep]
        if not use:
            continue
        pos = None
        neg = None
        for s in use:
            p, n = ep[s]
            pz = (p - np.mean(np.concatenate([p, n]))) / (np.std(np.concatenate([p, n])) + 1e-8)
            nz = (n - np.mean(np.concatenate([p, n]))) / (np.std(np.concatenate([p, n])) + 1e-8)
            pos = weights[s] * pz if pos is None else pos + weights[s] * pz
            neg = weights[s] * nz if neg is None else neg + weights[s] * nz
        y = [1] * len(pos) + [0] * len(neg)
        try:
            aucs.append(roc_auc_score(y, list(pos) + list(neg)))
        except ValueError:
            pass
        allsc = np.concatenate([pos, neg])
        allgt = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        order = np.argsort(allsc)[::-1]
        for K, acc in ((1, p1), (2, p2), (3, p3)):
            if len(order) >= K:
                acc.append(np.mean(allgt[order[:K]]))
    return (np.mean(aucs) if aucs else float('nan'),
            np.mean(p1) if p1 else float('nan'),
            np.mean(p2) if p2 else float('nan'),
            np.mean(p3) if p3 else float('nan'))


print('\n## single signals')
rows = []
for s in sigs:
    a, t1, t2, t3 = evaluate({s: 1.0})
    rows.append((a, s, t1, t2, t3))
    print('%-12s per-ep AUC %.3f  top1 %.3f top2 %.3f top3 %.3f'
          % (s, a, t1, t2, t3))

print('\n## top single signals sorted by AUC')
for a, s, t1, t2, t3 in sorted(rows, reverse=True)[:5]:
    print('%-12s AUC %.3f  top2 %.3f top3 %.3f' % (s, a, t2, t3))

print('\n## pairwise combinations (weights 0.5/0.5 and 0.7/0.3)')
best = []
for s1, s2 in itertools.combinations(sigs, 2):
    for w in ((0.5, 0.5), (0.7, 0.3), (0.3, 0.7)):
        a, t1, t2, t3 = evaluate({s1: w[0], s2: w[1]})
        if not np.isnan(a):
            best.append((a, t2, t3, '%s*%.1f + %s*%.1f' % (s1, w[0], s2, w[1])))
for a, t2, t3, name in sorted(best, reverse=True)[:10]:
    print('AUC %.3f  top2 %.3f top3 %.3f  %s' % (a, t2, t3, name))

print('\n## triple combinations (equal weights, top-6 signals)')
top6 = [s for _, s, _, _, _ in sorted(rows, reverse=True)[:6]]
best3 = []
for combo in itertools.combinations(top6, 3):
    a, t1, t2, t3 = evaluate({s: 1.0 for s in combo})
    if not np.isnan(a):
        best3.append((a, t2, t3, ' + '.join(combo)))
for a, t2, t3, name in sorted(best3, reverse=True)[:8]:
    print('AUC %.3f  top2 %.3f top3 %.3f  %s' % (a, t2, t3, name))
