"""Summarize diag_bc.py records: overall + per-class tables."""
import json
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else '/data01/jh/data/diag_100'
recs = [json.loads(l) for l in open(path + '/records.jsonl')]
print('episodes:', len(recs))

taus = ['0.5', '0.6', '0.7', '0.8']
print('\n## Overall')
print('| tau | B_nonempty | B_precision | omission |')
print('|---|---:|---:|---:|')
for t in taus:
    vals = [r['taus'][t] for r in recs if t in r['taus']]
    if not vals:
        continue
    ne = np.mean([v['n_B'] > 0 for v in vals])
    pr = np.mean([v['precision'] for v in vals])
    om = np.mean([v['omission'] for v in vals])
    print('| %s | %.3f | %.3f | %.3f |' % (t, ne, pr, om))

adv = [r['adv_single'] for r in recs if r.get('adv_single') is not None]
print('\nadvantage_single: %s (n=%d)' % (
    ('%.3f' % np.mean(adv)) if adv else 'n/a', len(adv)))
adv_m = [r['adv_multi'] for r in recs if r.get('adv_multi') is not None]
print('advantage_multi: %s (n=%d)' % (
    ('%.3f' % np.mean(adv_m)) if adv_m else 'n/a', len(adv_m)))

print('\n## Per class (tau=0.6)')
print('| class | n_ep | B_nonempty | B_precision | omission | advantage |')
print('|---|---:|---:|---:|---:|---:|')
by = defaultdict(list)
for r in recs:
    by[r['class']].append(r)
for c in sorted(by):
    rs = by[c]
    vals = [r['taus']['0.6'] for r in rs]
    advs = [r['adv_single'] for r in rs if r.get('adv_single') is not None]
    print('| %d | %d | %.2f | %.2f | %.2f | %s |' % (
        c, len(rs),
        np.mean([v['n_B'] > 0 for v in vals]),
        np.mean([v['precision'] for v in vals]),
        np.mean([v['omission'] for v in vals]),
        ('%.2f' % np.mean(advs)) if advs else 'n/a',
    ))
