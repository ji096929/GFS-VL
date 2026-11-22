"""Task 2f: cross-episode B stability as an orthogonal signal.

A superpoint is identified by (block name, raw seg id). A true bridge should be
selected as B repeatedly across episodes that use the same U block; a false
bridge (boundary fragment) should be selected only sporadically.
"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

pattern = sys.argv[1] if len(sys.argv) > 1 else '/data01/jh/data/diag_stab_s*/records.jsonl'
recs = []
for f in sorted(glob.glob(pattern)):
    try:
        for line in open(f):
            recs.append(json.loads(line))
    except Exception:
        pass
print('episodes:', len(recs))

MOD = 10 ** 7
appear = defaultdict(int)
ep_seen = defaultdict(set)
for ep, r in enumerate(recs):
    for blk, s in zip(r.get('B_block', []), r.get('B_seg', [])):
        key = (blk, int(s) % MOD)
        appear[key] += 1
        ep_seen[key].add(ep)

counts = list(appear.values())
print('unique (block,seg):', len(appear))
print('appear count: mean %.2f  median %.0f  max %d'
      % (np.mean(counts), np.median(counts), max(counts)))
print('superpoints appearing >= 3 episodes: %d (%.1f%%)'
      % (sum(c >= 3 for c in counts),
         100.0 * np.mean([c >= 3 for c in counts])))


def score_of(r):
    blks = r.get('B_block', [])
    segs = r.get('B_seg', [])
    return [appear[(b, int(s) % MOD)] for b, s in zip(blks, segs)]


pos_all, neg_all, per = [], [], []
for r in recs:
    scores = score_of(r)
    gts = r.get('B_gt', [])
    if not scores or not gts or len(scores) != len(gts):
        continue
    try:
        per.append(roc_auc_score(gts, scores))
    except ValueError:
        pass
    for sc, gt in zip(scores, gts):
        (pos_all if gt == 1 else neg_all).append(sc)

pv = np.array(pos_all, dtype=float)
nv = np.array(neg_all, dtype=float)
sp = np.sqrt(
    ((len(pv) - 1) * pv.std(ddof=1) ** 2
     + (len(nv) - 1) * nv.std(ddof=1) ** 2) / (len(pv) + len(nv) - 2)
)
d = (pv.mean() - nv.mean()) / max(sp, 1e-8)
auc = roc_auc_score([1] * len(pv) + [0] * len(nv), list(pv) + list(nv))
print('\nstability signal:')
print('  pooled AUC %.3f  per-ep AUC %.3f  cohens d %+.3f'
      % (max(auc, 1 - auc), np.mean(per), d))
print('  mean pos %.2f  mean neg %.2f' % (pv.mean(), nv.mean()))

print('\nTop-K precision by stability:')
for K in (1, 2, 3):
    precs = []
    for r in recs:
        scores = score_of(r)
        gts = r.get('B_gt', [])
        if not scores or not gts or len(scores) != len(gts):
            continue
        order = np.argsort(scores)[::-1][:K]
        precs.append(np.mean([gts[i] for i in order]))
    print('  top%d %.3f (n=%d)' % (K, np.mean(precs), len(precs)))

print('\nCombined with q_p90 / multi_max (z-score, 50:50):')
for base_sig in ('q_p90', 'multi_max', 'margin'):
    pos_all2, neg_all2, per2 = [], [], []
    for r in recs:
        scores = score_of(r)
        p = r.get('B_pos_' + base_sig)
        n = r.get('B_neg_' + base_sig)
        gts = r.get('B_gt', [])
        if (not scores or not p or not n or not gts
                or len(scores) != len(gts)):
            continue
        base = np.array(p + n, dtype=float)
        stab = np.array(scores, dtype=float)
        if len(base) != len(stab):
            continue
        allv = np.concatenate([base, stab])
        mu, sd = allv.mean(), allv.std() + 1e-8
        combo = 0.5 * (base - mu) / sd + 0.5 * (stab - mu) / sd
        g = np.array(gts)
        try:
            per2.append(roc_auc_score(g, combo))
        except ValueError:
            pass
        for sc, gt in zip(combo, g):
            (pos_all2 if gt == 1 else neg_all2).append(sc)
    if pos_all2 and neg_all2:
        auc2 = roc_auc_score(
            [1] * len(pos_all2) + [0] * len(neg_all2), pos_all2 + neg_all2
        )
        print('  %s + stability: pooled %.3f  per-ep %.3f'
              % (base_sig, max(auc2, 1 - auc2), np.mean(per2)))
