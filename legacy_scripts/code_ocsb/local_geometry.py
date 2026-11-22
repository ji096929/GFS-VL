"""Task 2g: superpoint local geometry (curvature / planarity) as a signal.

Reads the block npy files directly (col 7 = official seg id), computes PCA
eigenvalues per superpoint, and evaluates whether curvature / planarity
separate B_pos from B_neg. No model forward pass needed.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

pattern = sys.argv[1] if len(sys.argv) > 1 else '/data01/jh/data/diag_stab_s*/records.jsonl'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
MOD = 10 ** 7

recs = []
for f in sorted(glob.glob(pattern)):
    try:
        for line in open(f):
            recs.append(json.loads(line))
    except Exception:
        pass
print('episodes:', len(recs))

# unique (block, seg)
keys = set()
for r in recs:
    for b, s in zip(r.get('B_block', []), r.get('B_seg', [])):
        keys.add((b, int(s) % MOD))
print('unique superpoints:', len(keys))

by_block = defaultdict(list)
for b, seg in keys:
    by_block[b].append(seg)

geo = {}
missing = 0
for b, segs in by_block.items():
    path = os.path.join(DATA, b + '.npy')
    if not os.path.exists(path):
        missing += len(segs)
        continue
    d = np.load(path, mmap_mode='r')
    if d.shape[1] < 8:
        missing += len(segs)
        continue
    segcol = np.asarray(d[:, 7]).astype(np.int64)
    xyz_all = np.asarray(d[:, :3]).astype(np.float64)
    for seg in segs:
        mask = segcol == seg
        n = int(mask.sum())
        if n < 3:
            continue
        xyz = xyz_all[mask]
        xyz = xyz - xyz.mean(0)
        cov = xyz.T @ xyz / max(n - 1, 1)
        ev = np.linalg.eigvalsh(cov)
        ev = np.clip(ev, 0.0, None)
        l3, l2, l1 = ev[0], ev[1], ev[2]
        curv = l3 / max(l1 + l2 + l3, 1e-12)
        planar = (l2 - l3) / max(l1, 1e-12)
        geo[(b, seg)] = (curv, planar, np.log(float(n)))

print('computed:', len(geo), 'missing:', missing)

FIELDS = [('curvature', 0), ('planarity', 1), ('log_size', 2)]
for name, idx in FIELDS:
    pos, neg, per = [], [], []
    for r in recs:
        gts = r.get('B_gt', [])
        blks = r.get('B_block', [])
        segs = r.get('B_seg', [])
        if not gts or not blks or len(gts) != len(blks):
            continue
        scores, valid = [], []
        for i, (b, s) in enumerate(zip(blks, segs)):
            k = (b, int(s) % MOD)
            if k in geo:
                scores.append(geo[k][idx])
                valid.append(i)
        if len(scores) < 3:
            continue
        y = [gts[i] for i in valid]
        if len(set(y)) < 2:
            continue
        try:
            per.append(roc_auc_score(y, scores))
        except ValueError:
            pass
        for sc, gt in zip(scores, y):
            (pos if gt == 1 else neg).append(sc)
    if not pos or not neg:
        print('%-12s n/a' % name)
        continue
    pv = np.array(pos)
    nv = np.array(neg)
    sp = np.sqrt(
        ((len(pv) - 1) * pv.std(ddof=1) ** 2
         + (len(nv) - 1) * nv.std(ddof=1) ** 2) / (len(pv) + len(nv) - 2)
    )
    d = (pv.mean() - nv.mean()) / max(sp, 1e-8)
    auc = roc_auc_score([1] * len(pv) + [0] * len(nv), list(pv) + list(nv))
    print('%-12s pooled %.3f  per-ep %.3f  d %+.3f  n_pos %d  n_neg %d'
          % (name, max(auc, 1 - auc), np.mean(per), d, len(pv), len(nv)))

# top-K by curvature (and by -planarity)
for name, idx, sign in (('curvature', 0, 1), ('planarity', 1, -1)):
    for K in (1, 2, 3):
        precs = []
        for r in recs:
            gts = r.get('B_gt', [])
            blks = r.get('B_block', [])
            segs = r.get('B_seg', [])
            if not gts or not blks or len(gts) != len(blks):
                continue
            sc = []
            for b, s in zip(blks, segs):
                k = (b, int(s) % MOD)
                sc.append(sign * geo[k][idx] if k in geo else -1e9)
            order = np.argsort(sc)[::-1][:K]
            precs.append(np.mean([gts[i] for i in order]))
        if precs:
            print('%-10s top%d %.3f (n=%d)' % (name, K, np.mean(precs), len(precs)))
