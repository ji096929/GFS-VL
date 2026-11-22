"""Block/scene-level diagnostics for the bridge set B.

The 21 superpoint-level signals all cap at AUC 0.615. This script tests the one
granularity that was never tried: aggregate evidence at the U-block level, where
per-block sample counts are large enough for stable statistics.

Questions:
  1. Are U blocks ever the same scene as the query block? (leakage check)
  2. How concentrated is B across the |U| sampled blocks?
  3. Can block-level aggregates predict per-block B precision?
  4. Does restricting to high-purity blocks lift the oracle gain?
  5. Does cross-block agreement (same class seen in several U blocks) select
     better bridges than any single-superpoint score?
"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

pattern = sys.argv[1] if len(sys.argv) > 1 else '/data01/jh/data/diag_gate_s*/records.jsonl'
recs = []
for f in sorted(glob.glob(pattern)):
    try:
        for line in open(f):
            recs.append(json.loads(line))
    except Exception:
        pass
print('episodes:', len(recs))


def scene_of(block):
    return block.rsplit('_block_', 1)[0]


# ---------- 1. leakage check ----------
same_scene = n_u_same = total_u = 0
for r in recs:
    q = scene_of(r['query'])
    us = r.get('u') or []
    total_u += len(us)
    k = sum(scene_of(b) == q for b in us)
    n_u_same += k
    if k:
        same_scene += 1
print('\n## 1. U / query scene overlap')
print('episodes with >=1 U block in the query scene: %d / %d'
      % (same_scene, len(recs)))
print('U blocks sharing the query scene: %d / %d' % (n_u_same, total_u))

# ---------- 2. B concentration over U blocks ----------
nblocks, frac_top1 = [], []
for r in recs:
    bb = r.get('B_block') or []
    if not bb:
        continue
    cnt = defaultdict(int)
    for b in bb:
        cnt[b] += 1
    nblocks.append(len(cnt))
    frac_top1.append(max(cnt.values()) / len(bb))
print('\n## 2. B spread over sampled U blocks')
print('distinct blocks contributing B: mean %.2f  median %.1f  max %d'
      % (np.mean(nblocks), np.median(nblocks), max(nblocks)))
print('share of B from the single most frequent block: mean %.3f  median %.3f'
      % (np.mean(frac_top1), np.median(frac_top1)))

# ---------- 3. block-level aggregates vs per-block purity ----------
# Collect per (episode, block) records.
rows = []  # dict per (ep, block)
for ei, r in enumerate(recs):
    bb = r.get('B_block') or []
    gts = r.get('B_gt') or []
    qs = r.get('B_q') or []
    rel = r.get('B_query_rel') or []
    if not bb or len(bb) != len(gts):
        continue
    cls = r['class']
    nU = len(r.get('u') or []) or 1
    idx = defaultdict(list)
    for i, b in enumerate(bb):
        idx[b].append(i)
    for b, ii in idx.items():
        g = np.array([gts[i] for i in ii])
        q = np.array([qs[i] for i in ii]) if len(qs) == len(gts) else None
        rl = np.array([rel[i] for i in ii]) if len(rel) == len(gts) else None
        rows.append(dict(
            ep=ei, cls=cls, block=b, n=len(ii), npos=int(g.sum()),
            purity=float(g.mean()),
            q_mean=float(q.mean()) if q is not None else np.nan,
            q_std=float(q.std()) if q is not None else np.nan,
            rel_mean=float(rl.mean()) if rl is not None else np.nan,
            block_share=len(ii) / len(bb),
            nU=nU,
        ))
print('\n## 3. per-(episode, block) rows: %d' % len(rows))

# AUC: can a block-level aggregate predict whether the whole block is pure?
# Binary label = block fully positive (purity == 1) vs not.
y = np.array([1 if r['purity'] == 1.0 else 0 for r in rows])
print('blocks fully positive: %d / %d (%.3f)' % (y.sum(), len(y), y.mean()))
for name in ('n', 'q_mean', 'q_std', 'rel_mean', 'block_share'):
    x = np.array([r[name] for r in rows])
    ok = ~np.isnan(x)
    if ok.sum() < 10 or y[ok].sum() in (0, ok.sum()):
        print('%-12s n/a' % name)
        continue
    from sklearn.metrics import roc_auc_score
    a = roc_auc_score(y[ok], x[ok])
    print('%-12s AUC %.3f' % (name, max(a, 1 - a)))

# ---------- 4. oracle gain when restricted to pure blocks ----------
# Approximate: oracle_iou_2c_* was computed on ALL B_pos bridges. We cannot
# recompute the model forward here, so instead measure how much of B_pos lives
# in pure blocks vs mixed blocks, and how the per-episode gain correlates with
# the purity of the blocks it used.
base = np.array([r.get('oracle_base_iou') for r in recs], dtype=float)
gain05 = np.array([r.get('oracle_iou_2c_0.5') for r in recs], dtype=float)
blk_purity = defaultdict(list)
for r in rows:
    blk_purity[r['ep']].append(r['purity'])
ep_purity = np.array([np.mean(blk_purity[i]) if blk_purity[i] else np.nan
                      for i in range(len(recs))])
ok = ~np.isnan(base) & ~np.isnan(gain05) & ~np.isnan(ep_purity)
print('\n## 4. episode gain vs mean block purity')
print('n=%d  corr(purity, gain) = %.3f'
      % (ok.sum(), np.corrcoef(ep_purity[ok], (gain05 - base)[ok])[0, 1]))
for lo, hi in ((0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01)):
    m = ok & (ep_purity >= lo) & (ep_purity < hi)
    if m.sum():
        print('purity [%.1f,%.1f)  n=%3d  base %.4f  gain %+.4f'
              % (lo, hi, m.sum(), base[m].mean(),
                 (gain05[m] - base[m]).mean()))

# ---------- 5. cross-block agreement ----------
# For each B superpoint, count how many distinct U blocks in the same episode
# also contain a top-scoring superpoint of the same class. Rationale: a class
# that shows up consistently across independent scenes is a safer bridge.
agree_pos, agree_neg = [], []
for r in recs:
    bb = r.get('B_block') or []
    gts = r.get('B_gt') or []
    qs = r.get('B_q') or []
    if not bb or len(bb) != len(gts) or len(qs) != len(gts):
        continue
    cnt = defaultdict(int)
    for b in bb:
        cnt[b] += 1
    for b, g in zip(bb, gts):
        (agree_pos if g == 1 else agree_neg).append(cnt[b])
if agree_pos and agree_neg:
    p, n = np.array(agree_pos), np.array(agree_neg)
    print('\n## 5. same-block B count as a signal')
    print('B_pos: mean %.2f  B_neg: mean %.2f' % (p.mean(), n.mean()))
    from sklearn.metrics import roc_auc_score
    a = roc_auc_score([1] * len(p) + [0] * len(n), list(p) + list(n))
    print('pooled AUC %.3f  (flip: %.3f)' % (a, 1 - a))

# ---------- 6. repeat scenes across episodes ----------
scene_use = defaultdict(int)
for r in recs:
    for b in (r.get('u') or []):
        scene_use[scene_of(b)] += 1
vals = np.array(list(scene_use.values()))
print('\n## 6. U scene reuse')
print('distinct U scenes: %d  uses/scene: median %.1f  p90 %.1f  max %d'
      % (len(vals), np.median(vals), np.percentile(vals, 90), vals.max()))
