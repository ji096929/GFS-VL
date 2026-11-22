"""Control for the difficulty confound in the cross/same-scene comparison.

same_scene happens to have an easier episode mix: baseline IoU 0.4238 vs
0.3618, and more floor/bed/cabinet. Higher baselines may mechanically allow
larger bridge gains. Two controls:
  1. reweight same_scene to the cross_scene class distribution
  2. stratify by baseline IoU bucket and compare within bucket
Also checks whether gain scales with baseline at all.
"""
import glob
import json
import sys

import numpy as np


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
print('cross %d   same %d' % (len(cross), len(same)))
if not same:
    raise SystemExit('no same-scene records')

LAM = 'oracle_iou_2c_0.5'


def gain(r, key=LAM):
    a, b = r.get(key), r.get('oracle_base_iou')
    if a is None or b is None:
        return np.nan
    return a - b


def gstat(recs, key=LAM):
    v = np.array([gain(r, key) for r in recs], dtype=float)
    v = v[~np.isnan(v)]
    return (v.mean() if len(v) else np.nan), len(v)


# ---------- 1. class reweighting ----------
print('\n## 1. class-reweighted gain (%s)' % LAM)
ccounts = {}
for r in cross:
    ccounts[r['class']] = ccounts.get(r['class'], 0) + 1
tot = sum(ccounts.values())
w = {c: n / tot for c, n in ccounts.items()}
num, den, missing = 0.0, 0.0, []
for c, wc in sorted(w.items()):
    rs = [r for r in same if r['class'] == c]
    rc = [r for r in cross if r['class'] == c]
    if len(rs) < 5:
        missing.append(c)
        continue
    gs, _ = gstat(rs)
    gc, _ = gstat(rc)
    num += wc * gs
    den += wc
    print('class %2d  w=%.3f  cross %+.4f (%3d)  same %+.4f (%3d)'
          % (c, wc, gc, len(rc), gs, len(rs)))
print('reweighted same-scene gain: %+.4f  (covered weight %.2f, missing %s)'
      % (num / max(den, 1e-9), den, missing))
print('cross-scene overall gain: %+.4f' % gstat(cross)[0])
print('same-scene naive gain    : %+.4f' % gstat(same)[0])

# ---------- 2. baseline stratification ----------
print('\n## 2. gain vs baseline IoU bucket')
edges = [0.0, 0.1, 0.3, 0.5, 0.7, 1.01]
print('%-14s | %-18s | %-18s' % ('baseline', 'cross', 'same'))
print('%-14s | %6s %5s %6s | %6s %5s %6s'
      % ('bucket', 'n', 'base', 'gain', 'n', 'base', 'gain'))
for lo, hi in zip(edges[:-1], edges[1:]):
    rc = [r for r in cross if lo <= r['oracle_base_iou'] < hi]
    rs = [r for r in same if lo <= r['oracle_base_iou'] < hi]
    gc, nc = gstat(rc)
    gs, ns = gstat(rs)
    bc = np.mean([r['oracle_base_iou'] for r in rc]) if rc else np.nan
    bs = np.mean([r['oracle_base_iou'] for r in rs]) if rs else np.nan
    print('[%.1f,%.1f)     | %6d %5.2f %+6.3f | %6d %5.2f %+6.3f'
          % (lo, hi, nc, bc, gc, ns, bs, gs))

print('\n## 3. does gain scale with baseline?')
for label, recs in (('cross', cross), ('same', same)):
    b = np.array([r['oracle_base_iou'] for r in recs], dtype=float)
    g = np.array([gain(r) for r in recs], dtype=float)
    ok = ~np.isnan(g)
    print('%-6s corr(baseline, gain) = %+.3f   mean gain %+.4f'
          % (label, np.corrcoef(b[ok], g[ok])[0, 1], g[ok].mean()))

# ---------- 4. matched subsample ----------
# Keep only baseline in [0.15, 0.75] and classes present in both, then compare.
common = sorted({r['class'] for r in cross} & {r['class'] for r in same})
print('\n## 4. restricted to classes present in both, baseline in [0.15,0.75]')
for label, recs in (('cross', cross), ('same', same)):
    sel = [r for r in recs
           if r['class'] in common and 0.15 <= r['oracle_base_iou'] <= 0.75]
    g, n = gstat(sel)
    b = np.mean([r['oracle_base_iou'] for r in sel])
    print('%-6s n=%d  base %.4f  gain %+.4f' % (label, n, b, g))

# ---------- 5. per-100-episode stability ----------
print('\n## 5. gain by shard (is the same-scene effect uniform?)')
for label, recs in (('cross', cross), ('same', same)):
    groups = {}
    for r in recs:
        s = (r.get('support') or '').split('_block_')[0]
        groups.setdefault(s[-2:], []).append(r)
    line = ' '.join('%s:%+.3f(%d)' % (k, gstat(v)[0], len(v))
                    for k, v in sorted(groups.items()))
    print('%-6s %s' % (label, line))
