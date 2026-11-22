"""Phase 3: does the CLIP consensus gate predict when bridging helps?

agree_orth -- CLIP-space agreement between the direct path's Q_hat_S and the
              support foreground prototype (minus the best background match)
gain_orth  -- how much more CLIP-consistent Q_hat_B is than Q_hat_S
"""
import json
import sys

import numpy as np

FILES = {'cross': '/data01/jh/data/vx/records.jsonl',
         'same': '/data01/jh/data/vs/records.jsonl'}
E2E = 'e2e_qc_top5_lam0.3'


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 10:
        return np.nan
    return np.corrcoef(x[ok], y[ok])[0, 1]


for scen, path in FILES.items():
    try:
        recs = [json.loads(l) for l in open(path)]
    except Exception:
        print('%s: not ready' % scen)
        continue
    recs = [r for r in recs if r.get('agree_orth') is not None]
    if len(recs) < 10:
        print('%s: only %d usable' % (scen, len(recs)))
        continue

    base = np.array([r['oracle_base_iou'] for r in recs])
    delta = np.array([(r.get(E2E, np.nan) - r['oracle_base_iou'])
                      for r in recs])
    agree = np.array([r['agree_orth'] for r in recs])
    gain = np.array([r.get('gain_orth', np.nan) for r in recs])

    print('\n' + '=' * 70)
    print('%s  n=%d  baseline %.4f  e2e delta %+.4f'
          % (scen, len(recs), base.mean(), np.nanmean(delta)))

    print('\n  corr(agree_orth, baseline IoU) = %+.3f' % corr(agree, base))
    print('  corr(agree_orth, e2e delta)    = %+.3f' % corr(agree, delta))
    print('  corr(gain_orth,  e2e delta)    = %+.3f' % corr(gain, delta))
    print('  corr(gain_orth,  baseline)     = %+.3f' % corr(gain, base))

    # gating rule from the objective: close when Agree_orth is high; open only
    # when it is low AND a reliable bridge exists (V>0 handled in Phase 2)
    print('\n  gate: open iff agree < tau  (LOO-free, just the curve)')
    print('  %-6s %6s %10s %10s' % ('tau', 'open%', 'mean dIoU', 'vs none'))
    none = np.nanmean(delta)
    for tau in (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
        use = agree < tau
        v = np.where(use, np.nan_to_num(delta), 0.0)
        print('  %-6.3f %5.0f%% %+10.4f %+10.4f'
              % (tau, 100 * use.mean(), v.mean(), v.mean() - none))

    print('\n  gate with gain guard: open iff agree < tau AND gain > 0')
    print('  %-6s %6s %10s' % ('tau', 'open%', 'mean dIoU'))
    for tau in (0.05, 0.1, 0.2, 0.3):
        use = (agree < tau) & (np.nan_to_num(gain) > 0)
        v = np.where(use, np.nan_to_num(delta), 0.0)
        print('  %-6.3f %5.0f%% %+10.4f' % (tau, 100 * use.mean(), v.mean()))
