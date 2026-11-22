"""Step 1: add the end-to-end column to the baseline-stratified table, plus
the two idealised gating ceilings.

The oracle column told us where bridges *could* help. This script answers the
life-or-death question: inside the low-baseline buckets, does the *unsupervised*
end-to-end path also gain? If B stays clean when the support set is weak, then
the global -1.58pp is purely the high-baseline episodes dragging it down and a
gate can fix it. If B collapses there too, the idea is dead.
"""
import json
import sys

import numpy as np

FILES = {'cross': '/data01/jh/data/base_x/records.jsonl',
         'same': '/data01/jh/data/base_s/records.jsonl'}
LAMS = ('0.2', '0.3')
KS = (5, 20)
BUCKETS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]


def load(p):
    return [json.loads(l) for l in open(p)]


def key(k, lam):
    # the qc path was never given a proto suffix (only the combo path was)
    return 'e2e_qc_top%d_lam%s' % (k, lam)


for scen, path in FILES.items():
    recs = load(path)
    base = np.array([r['oracle_base_iou'] for r in recs])
    print('\n' + '=' * 78)
    print('%s  n=%d  mean %.4f  median %.4f'
          % (scen, len(recs), base.mean(), np.median(base)))

    # ---------- bucket table ----------
    hdr = '%-14s %4s %7s %8s' % ('baseline', 'n', 'B_prec', 'Oracle')
    for k in KS:
        for lam in LAMS:
            hdr += ' %5s' % ('K%d.%s' % (k, lam))
    print(hdr)
    for lo, hi in BUCKETS:
        idx = [i for i, r in enumerate(recs) if lo <= r['oracle_base_iou'] < hi]
        if not idx:
            continue
        n = len(idx)
        bp = np.mean([recs[i]['n_B_pos'] / max(recs[i]['n_B'], 1) for i in idx])
        orc = np.mean([recs[i]['oracle_iou_2c_0.5'] - recs[i]['oracle_base_iou']
                       for i in idx if recs[i].get('oracle_iou_2c_0.5')
                       is not None])
        line = '%-14s %4d %7.3f %+8.4f' % ('[%.1f,%.1f)' % (lo, min(hi, 1.0)),
                                          n, bp, orc)
        for k in KS:
            for lam in LAMS:
                v = [recs[i][key(k, lam)] - recs[i]['oracle_base_iou']
                     for i in idx if recs[i].get(key(k, lam)) is not None]
                line += ' %+5.2f' % (100 * np.mean(v)) if v else '   -  '
        print(line)

    # ---------- ceiling 1: GT baseline gate ----------
    print('\n  GT-baseline gate (open bridge only when baseline < tau):')
    print('  %-6s %5s %9s %9s %9s' % ('tau', 'keep', 'K5.3', 'K20.3', 'oracle'))
    for tau in (0.2, 0.3, 0.4, 0.5, 0.6):
        ys, y20, yo = [], [], []
        for r in recs:
            b = r['oracle_base_iou']
            gate = b < tau
            o = r.get('oracle_iou_2c_0.5')
            for arr, k, lam in ((ys, 5, '0.3'), (y20, 20, '0.3')):
                e = r.get(key(k, lam))
                if e is not None:
                    arr.append(e if gate else b)
            if o is not None:
                yo.append(o if gate else b)
        if ys:
            print('  %-6.1f %5d %+9.4f %+9.4f %+9.4f'
                  % (tau, sum(1 for r in recs if r['oracle_base_iou'] < tau),
                     np.mean(ys) - base.mean(),
                     np.mean(y20) - base.mean(),
                     np.mean(yo) - base.mean()))

    # ---------- ceiling 2: per-episode optimal gate on e2e ----------
    print('\n  Per-episode optimal gate on e2e (open iff that episode gains):')
    for k in KS:
        for lam in LAMS:
            d = []
            pos = 0
            tot = 0
            for r in recs:
                e = r.get(key(k, lam))
                if e is None:
                    continue
                delta = e - r['oracle_base_iou']
                d.append(max(delta, 0.0))
                pos += delta > 0
                tot += 1
            if tot:
                print('  K=%-2d lam%s: %5.2f%% episodes gain, ceiling %+.4fpp'
                      % (k, lam, 100 * pos / tot, np.mean(d)))

    # ---------- how much of the loss comes from the high bucket ----------
    hi = [i for i, v in enumerate(base) if v >= 0.6]
    lo = [i for i, v in enumerate(base) if v < 0.4]
    if hi and lo:
        for k in (5,):
            for lam in ('0.3',):
                dl = np.mean([recs[i][key(k, lam)] - recs[i]['oracle_base_iou']
                              for i in lo if recs[i].get(key(k, lam)) is not None])
                dh = np.mean([recs[i][key(k, lam)] - recs[i]['oracle_base_iou']
                              for i in hi if recs[i].get(key(k, lam)) is not None])
                print('\n  K=%d lam%s: low-baseline(<0.4) n=%d %+.4f | '
                      'high-baseline(>=0.6) n=%d %+.4f'
                      % (k, lam, len(lo), dl, len(hi), dh))
