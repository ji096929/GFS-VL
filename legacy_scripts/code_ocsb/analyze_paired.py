"""Paired analysis over the fixed manifest.

All runs consume the same manifest prefix, so episode i is the same (S, Q) in
every run. That lets us compute per-episode differences and a paired SE, which
is far tighter than the 0.6-0.7pp unpaired SE of the free-running shards.

Usage:
  python analyze_paired.py <run_dir_1> <run_dir_2> ...
Each dir must contain records.jsonl produced with --manifest.
"""
import json
import os
import sys

import numpy as np


def load(d):
    p = os.path.join(d, 'records.jsonl')
    out = {}
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        out[r['manifest_id']] = r
    return out


runs = {}
for d in sys.argv[1:]:
    tag = os.path.basename(d.rstrip('/'))
    runs[tag] = load(d)
    print('%-10s %d episodes' % (tag, len(runs[tag])))

common = set.intersection(*[set(v) for v in runs.values()])
print('common episodes: %d\n' % len(common))
if not common:
    raise SystemExit('no overlap; did all runs use the same manifest?')

ids = sorted(common)


def series(tag, key):
    return np.array([runs[tag][i].get(key, np.nan) for i in ids], dtype=float)


def paired_se(x, y):
    """SE of mean(x - y) over paired episodes."""
    d = x - y
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return np.nan, np.nan, 0
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d)


print('## baseline sanity (must be identical across runs)')
for tag in runs:
    b = series(tag, 'oracle_base_iou')
    print('%-10s mean %.4f  n %d' % (tag, np.nanmean(b), np.sum(~np.isnan(b))))
base_tags = list(runs)
ref = series(base_tags[0], 'oracle_base_iou')
for tag in base_tags[1:]:
    cur = series(tag, 'oracle_base_iou')
    d = np.abs(cur - ref)
    print('  %s vs %s: max |diff| %.2e' % (tag, base_tags[0], np.nanmax(d)))

print('\n## Oracle 2C (paired delta)')
print('%-10s %8s %9s %+9s %8s %8s' % ('run', 'cfg', 'IoU', 'delta', 'pSE', 'n'))
for tag in runs:
    b = series(tag, 'oracle_base_iou')
    for key in ('oracle_iou_2c_0.3', 'oracle_iou_2c_0.5', 'oracle_iou_2c_0.7'):
        v = series(tag, key)
        m, se, n = paired_se(v, b)
        print('%-10s %8s %9.4f %+9.4f %8.4f %8d'
              % (tag, key[-3:], np.nanmean(v), m, se, n))

print('\n## End-to-end (paired delta)')
for tag in runs:
    b = series(tag, 'oracle_base_iou')
    rows = []
    for K in (3, 5, 10, 20):
        v = series(tag, 'e2e_qc_top%d_lam0.3' % K)
        m, se, n = paired_se(v, b)
        rows.append((K, m, se))
    print('%-10s ' % tag + '  '.join('K%d %+.4f±%.4f' % r for r in rows))

print('\n## fps vs balanced (paired, same setting)')
pairs = [('x_fps', 'x_bal'), ('s_fps', 's_bal')]
for a, btag in pairs:
    if a not in runs or btag not in runs:
        continue
    print('-- %s vs %s' % (btag, a))
    for key in ('oracle_iou_2c_0.5', 'e2e_qc_top5_lam0.3',
                'e2e_qc_top20_lam0.3'):
        have_a = sum(1 for i in ids if key in runs[a][i])
        have_b = sum(1 for i in ids if key in runs[btag][i])
        if have_a < len(ids) // 2 or have_b < len(ids) // 2:
            continue
        x = series(btag, key)
        y = series(a, key)
        m, se, n = paired_se(x, y)
        print('   %-24s %+.4f ± %.4f (paired, n=%d)' % (key, m, se, n))

print('\n## point counts (balanced should equalise them)')
for tag in runs:
    for K in (5, 20):
        v = series(tag, 'e2e_top%d_npts' % K)
        if np.isnan(v).all():
            continue
        print('%-10s K=%d npts mean %.0f median %.0f p90 %.0f'
              % (tag, K, np.nanmean(v), np.nanmedian(v), np.nanpercentile(v, 90)))
