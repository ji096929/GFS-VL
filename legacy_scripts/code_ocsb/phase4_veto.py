"""Task 14: orthogonal veto as the episode gate, after fixing the eta scale.

eta was computed as logmeanexp(relu(...)/tau) without multiplying back by tau,
which inflated it 5-10x and made V_orth > 0 reject 95% of bridges. With the
fix, V is on the same scale as the cosine similarity and the veto is meaningful.
"""
import json
import sys

import numpy as np

FILES = {'cross': '/data01/jh/data/vx/records.jsonl',
         'same': '/data01/jh/data/vs/records.jsonl'}


def load(p):
    return [json.loads(l) for l in open(p)]


for scen, path in FILES.items():
    try:
        recs = load(path)
    except Exception:
        print('%s: not ready' % scen)
        continue
    recs = [r for r in recs if r.get('veto_top1_lam0.3') is not None]
    if len(recs) < 10:
        print('%s: only %d usable' % (scen, len(recs)))
        continue

    base = np.array([r['oracle_base_iou'] for r in recs])
    print('\n' + '=' * 78)
    print('%s  n=%d  baseline %.4f' % (scen, len(recs), base.mean()))

    # distribution of V after the tau fix
    V = np.concatenate([np.asarray(r['clip_V_c'], float) for r in recs
                        if r.get('clip_V_c')])
    print('  V_orth (tau-fixed): mean %.3f  median %.3f  frac>0 %.1f%%'
          % (V.mean(), np.median(V), 100 * (V > 0).mean()))
    kept = [r.get('veto_pos_nB', 0) / max(len(r.get('clip_V_c', [])), 1)
            for r in recs if r.get('clip_V_c')]
    print('  strict veto keeps %.1f%% of candidates on average'
          % (100 * np.mean(kept)))
    opened = sum(1 for r in recs
                 if r.get('veto_pos_nB', 0) > 0) / len(recs)
    print('  episodes that keep at least one bridge: %.0f%%'
          % (100 * opened))

    print('\n  %-22s %9s %+9s %8s %6s'
          % ('config', 'IoU', 'delta', 'pSE', 'n'))
    for k in (1, 2, 3, 5):
        for lam in ('0.2', '0.3'):
            key = 'veto_top%d_lam%s' % (k, lam)
            v = [(r[key], r['oracle_base_iou']) for r in recs
                 if r.get(key) is not None]
            if len(v) < 10:
                continue
            a = np.array([x[0] for x in v])
            b = np.array([x[1] for x in v])
            d = a - b
            print('  %-22s %9.4f %+9.4f %8.4f %6d'
                  % (key, a.mean(), d.mean(),
                     d.std(ddof=1) / np.sqrt(len(d)), len(d)))
    for lam in ('0.2', '0.3'):
        key = 'veto_pos_lam%s' % lam
        v = [(r[key], r['oracle_base_iou']) for r in recs
             if r.get(key) is not None]
        if len(v) < 10:
            continue
        a = np.array([x[0] for x in v])
        b = np.array([x[1] for x in v])
        d = a - b
        print('  %-22s %9.4f %+9.4f %8.4f %6d'
              % (key, a.mean(), d.mean(),
                 d.std(ddof=1) / np.sqrt(len(d)), len(d)))

    # reference: raw top-K by q_c, and oracle
    print('\n  reference:')
    for k in (1, 2, 3, 5):
        for lam in ('0.3',):
            key = 'e2e_qc_top%d_lam%s' % (k, lam)
            v = [(r[key], r['oracle_base_iou']) for r in recs
                 if r.get(key) is not None]
            if len(v) < 10:
                continue
            a = np.array([x[0] for x in v])
            b = np.array([x[1] for x in v])
            d = a - b
            print('  %-22s %9.4f %+9.4f %8.4f %6d'
                  % (key, a.mean(), d.mean(),
                     d.std(ddof=1) / np.sqrt(len(d)), len(d)))
    v = [(r['oracle_iou_2c_0.5'], r['oracle_base_iou']) for r in recs
         if r.get('oracle_iou_2c_0.5') is not None]
    if v:
        a = np.array([x[0] for x in v])
        b = np.array([x[1] for x in v])
        d = a - b
        print('  %-22s %9.4f %+9.4f %8.4f %6d'
              % ('oracle_2c_0.5', a.mean(), d.mean(),
                 d.std(ddof=1) / np.sqrt(len(d)), len(d)))
