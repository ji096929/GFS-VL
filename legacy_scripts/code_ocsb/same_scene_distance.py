"""Does the same-scene gain decay with distance between query and U blocks?

Only meaningful within a scene (blocks in different scenes live in unrelated
coordinate frames). If the gain comes from spatial proximity, episodes whose U
blocks sit far from the query should gain less.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

pattern = sys.argv[1] if len(sys.argv) > 1 else '/data01/jh/data/diag_same_s12*/records.jsonl'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'

recs = []
for f in sorted(glob.glob(pattern)):
    try:
        for line in open(f):
            recs.append(json.loads(line))
    except Exception:
        pass
print('episodes:', len(recs))
if not recs:
    raise SystemExit('no records')

names = set()
for r in recs:
    names.add(r['query'])
    names.update(r.get('u') or [])
print('unique blocks:', len(names))

cent = {}
for n in names:
    p = os.path.join(DATA, n + '.npy')
    if not os.path.exists(p):
        continue
    d = np.load(p, mmap_mode='r')
    xyz = np.asarray(d[:, :3]).astype(np.float64)
    cent[n] = xyz.mean(0)

print('centroids:', len(cent), 'missing:', len(names) - len(cent))


def scene_of(n):
    return n.split('_block_')[0]


rows = []
for r in recs:
    q = r['query']
    us = r.get('u') or []
    if q not in cent or not us:
        continue
    same = [u for u in us if scene_of(u) == scene_of(q) and u in cent]
    dist = [float(np.linalg.norm(cent[q] - cent[u])) for u in same]
    b = r.get('oracle_base_iou')
    a = r.get('oracle_iou_2c_0.5')
    if b is None or a is None or not dist:
        continue
    rows.append(dict(dist=float(np.mean(dist)), dmin=float(np.min(dist)),
                     gain=a - b, base=b, n=len(dist),
                     nB=r.get('n_B', 0), npos=r.get('n_B_pos', 0),
                     prec=r.get('n_B_pos', 0) / max(r.get('n_B', 1), 1)))
print('episodes with same-scene U distance: %d' % len(rows))
if len(rows) < 20:
    raise SystemExit('too few')

D = np.array([x['dist'] for x in rows])
G = np.array([x['gain'] for x in rows])
B = np.array([x['base'] for x in rows])
P = np.array([x['prec'] for x in rows])
print('\ndistance query->U (m): mean %.2f  median %.2f  p10 %.2f  p90 %.2f  max %.2f'
      % (D.mean(), np.median(D), np.percentile(D, 10),
         np.percentile(D, 90), D.max()))


def corr(x, y):
    return np.corrcoef(x, y)[0, 1]


print('corr(dist, gain)      = %+.3f' % corr(D, G))
print('corr(dist, n_B)       = %+.3f' % corr(D, np.array([x['nB'] for x in rows])))
print('corr(dist, B precision)=%+.3f' % corr(D, P))
print('corr(dist, baseline)  = %+.3f' % corr(D, B))

print('\n## gain by distance quartile')
q = np.quantile(D, [0, .25, .5, .75, 1.0])
print('%-16s %5s %6s %8s' % ('range (m)', 'n', 'base', 'gain'))
for i in range(4):
    m = (D >= q[i]) & (D <= q[i + 1] if i == 3 else D < q[i + 1])
    if m.sum() < 3:
        continue
    print('[%.2f,%.2f]%s %5d %6.3f %+8.4f'
          % (q[i], q[i + 1], ' ' * max(0, 6 - len('%.2f' % q[i + 1])),
             m.sum(), B[m].mean(), G[m].mean()))

# partial out the baseline effect, since baseline correlates with gain
if len(rows) > 10:
    coef = np.polyfit(B, G, 1)
    resid = G - np.polyval(coef, B)
    print('\npartial corr(dist, gain | baseline) = %+.3f' % corr(D, resid))

print('\n## how many same-scene blocks per episode')
ns = np.array([x['n'] for x in rows])
print('mean %.2f  counts: %s'
      % (ns.mean(), {int(k): int(v) for k, v in
                     zip(*np.unique(ns, return_counts=True))}))
