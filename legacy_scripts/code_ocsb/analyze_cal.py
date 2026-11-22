"""Task 10: does margin-space calibration / a threshold-free soft gate beat the
raw end-to-end fusion?

All comparisons are within one run (same process, same episodes), so the paired
difference is the only number that matters.
"""
import json
import sys

import numpy as np

FILES = {'cross': '/data01/jh/data/cal_x/records.jsonl',
         'same': '/data01/jh/data/cal_s/records.jsonl'}

CONFIGS = [
    ('qc_top5_lam0.2', 'e2e_qc_top5_lam0.2', 'raw'),
    ('qc_top5_lam0.3', 'e2e_qc_top5_lam0.3', 'raw'),
    ('calA_lam0.2', 'calA_top5_lam0.2', 'demean'),
    ('calA_lam0.3', 'calA_top5_lam0.3', 'demean'),
    ('calB_lam0.2', 'calB_top5_lam0.2', 'zscore'),
    ('calB_lam0.3', 'calB_top5_lam0.3', 'zscore'),
    ('soft3', 'soft3_top5', 'soft'),
    ('soft5', 'soft5_top5', 'soft'),
    ('guard_lam0.2', 'guard_top5_lam0.2', 'guard'),
    ('guard_lam0.3', 'guard_top5_lam0.3', 'guard'),
    ('oracle_2c_0.5', 'oracle_iou_2c_0.5', 'oracle'),
]


def load(p):
    return [json.loads(l) for l in open(p)]


for scen, path in FILES.items():
    try:
        recs = load(path)
    except Exception:
        print('%s: not ready' % scen)
        continue
    recs = [r for r in recs if r.get('oracle_base_iou') is not None]
    if len(recs) < 10:
        print('%s: only %d records' % (scen, len(recs)))
        continue
    base = np.array([r['oracle_base_iou'] for r in recs])
    print('\n' + '=' * 72)
    print('%s  n=%d  baseline %.4f' % (scen, len(recs), base.mean()))
    print('%-18s %-8s %9s %+9s %8s %6s'
          % ('config', 'kind', 'IoU', 'delta', 'pSE', 'n'))
    for name, key, kind in CONFIGS:
        v = [(r[key], r['oracle_base_iou']) for r in recs
             if r.get(key) is not None]
        if len(v) < 10:
            continue
        a = np.array([x[0] for x in v])
        b = np.array([x[1] for x in v])
        d = a - b
        print('%-18s %-8s %9.4f %+9.4f %8.4f %6d'
              % (name, kind, a.mean(), d.mean(),
                 d.std(ddof=1) / np.sqrt(len(d)), len(d)))

    # how often does calibration change the outcome sign?
    k_raw, k_a, k_b = ('e2e_qc_top5_lam0.3', 'calA_top5_lam0.3',
                       'calB_top5_lam0.3')
    pairs = [(r[k_raw], r[k_a], r[k_b], r['oracle_base_iou']) for r in recs
             if r.get(k_raw) is not None and r.get(k_a) is not None]
    if pairs:
        raw = np.array([p[0] - p[3] for p in pairs])
        ca = np.array([p[1] - p[3] for p in pairs])
        cb = np.array([p[2] - p[3] for p in pairs])
        print('\n  raw gains %5.1f%% of episodes; calA %5.1f%%; calB %5.1f%%'
              % (100 * (raw > 0).mean(), 100 * (ca > 0).mean(),
                 100 * (cb > 0).mean()))
        print('  calA - raw: %+.4f   calB - raw: %+.4f'
              % ((ca - raw).mean(), (cb - raw).mean()))

    # the shift statistic itself
    sh = np.array([r.get('cal_shift', np.nan) for r in recs], float)
    ms = np.array([r.get('cal_meanS', np.nan) for r in recs], float)
    mb = np.array([r.get('cal_meanB', np.nan) for r in recs], float)
    ok = ~np.isnan(sh)
    if ok.sum() > 10:
        print('\n  margin mean: S %.3f  B %.3f  shift applied %.3f (nonzero %d%%)'
              % (np.nanmean(ms), np.nanmean(mb), np.nanmean(sh),
                 100 * (sh[ok] > 0).mean()))
        print('  corr(shift, e2e delta) = %+.3f'
              % np.corrcoef(sh[ok], (np.array(
                  [r.get(k_raw, np.nan) for r in recs])[ok] - base[ok]))[0, 1])
