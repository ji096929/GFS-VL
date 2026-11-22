"""Step 2/3: can an unsupervised episode-level signal predict (a) the baseline
IoU and (b) whether bridging will help?

Everything here uses only quantities available at test time: the direct-path
logits on Q, the bridge-path logits on Q, and the bridge confidences in U.
No GT of U or Q is touched.

Gating is evaluated with leave-one-out so the threshold is never fitted on the
episode it is scored on.
"""
import json
import sys

import numpy as np

FILES = {'cross': '/data01/jh/data/gate_x/records.jsonl',
         'same': '/data01/jh/data/gate_s/records.jsonl'}
E2E = 'e2e_qc_top5_lam0.3'
FEATS = ['S_fg_ratio', 'S_p95', 'S_p90', 'S_conf_fg', 'S_mean',
         'B_fg_ratio', 'B_p95', 'B_conf_fg', 'agree_SB', 'B_fg_gain']


def load(p):
    return [json.loads(l) for l in open(p)]


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 10:
        return np.nan
    return np.corrcoef(x[ok], y[ok])[0, 1]


for scen, path in FILES.items():
    recs = load(path)
    keep = [r for r in recs if r.get(E2E) is not None
            and r.get('S_fg_ratio') is not None]
    print('\n' + '=' * 74)
    print('%s  n=%d' % (scen, len(keep)))
    if not keep:
        continue
    base = np.array([r['oracle_base_iou'] for r in keep])
    delta = np.array([r[E2E] - r['oracle_base_iou'] for r in keep])
    print('baseline %.4f  e2e delta %+.4f  gain%% %.1f'
          % (base.mean(), delta.mean(), 100 * (delta > 0).mean()))

    print('\n  feature            corr(base)  corr(delta)   mean     sd')
    for f in FEATS:
        v = np.array([r.get(f, np.nan) for r in keep], float)
        print('  %-18s %+10.3f  %+11.3f  %8.3f %6.3f'
              % (f, corr(v, base), corr(v, delta),
                 np.nanmean(v), np.nanstd(v)))

    # ---------- single-feature gates, LOO ----------
    print('\n  single-feature gate (open bridge iff feat < tau), LOO:')
    print('  %-18s %8s %10s %10s %10s'
          % ('feature', 'tau*', 'LOO open%', 'LOO dIoU', 'in-sample'))
    for f in FEATS:
        v = np.array([r.get(f, np.nan) for r in keep], float)
        ok = ~np.isnan(v)
        if ok.sum() < 30:
            continue
        cand = np.quantile(v[ok], np.linspace(0.1, 0.9, 33))
        # in-sample best
        best_t, best_g = None, -9
        for t in cand:
            g = np.where(v < t, delta, 0.0)
            m = g.mean()
            if m > best_g:
                best_g, best_t = m, t
        # LOO
        loo = []
        opened = []
        for i in range(len(keep)):
            tr = np.ones(len(keep), bool)
            tr[i] = False
            btr, dtr, vtr = base[tr], delta[tr], v[tr]
            bt, bg = None, -9
            for t in cand:
                g = np.where(vtr < t, dtr, 0.0).mean()
                if g > bg:
                    bg, bt = g, t
            use = v[i] < bt
            loo.append(delta[i] if use else 0.0)
            opened.append(use)
        print('  %-18s %8.3f %9.0f%% %+10.4f %+10.4f'
              % (f, best_t, 100 * np.mean(opened), np.mean(loo), best_g))

    # ---------- two-feature rule: weak direct edge AND strong recovery -----
    print('\n  two-feature rule (S_fg_ratio < a AND B_fg_gain > b), LOO:')
    a_grid = np.quantile(np.array([r['S_fg_ratio'] for r in keep]),
                         np.linspace(0.2, 0.9, 15))
    b_grid = np.quantile(np.array([r['B_fg_gain'] for r in keep]),
                         np.linspace(0.1, 0.9, 15))
    S = np.array([r['S_fg_ratio'] for r in keep], float)
    G = np.array([r['B_fg_gain'] for r in keep], float)
    loo = []
    opened = 0
    for i in range(len(keep)):
        tr = np.ones(len(keep), bool)
        tr[i] = False
        best, bg = (None, None), -9
        for a in a_grid:
            for b in b_grid:
                pred = (S[tr] < a) & (G[tr] > b)
                g = np.where(pred, delta[tr], 0.0).mean()
                if g > bg:
                    bg, best = g, (a, b)
        a, b = best
        use = (S[i] < a) and (G[i] > b)
        loo.append(delta[i] if use else 0.0)
        opened += use
    print('  best-in-sample %+.4f   LOO %+.4f   open %.0f%%'
          % (bg, np.mean(loo), 100 * opened / len(keep)))

    # reference: the GT gate we already know works
    for tau in (0.4, 0.5):
        g = np.where(base < tau, delta, 0.0)
        print('  [reference] GT baseline<%.1f gate: %+.4f (open %d%%)'
              % (tau, g.mean(), 100 * (base < tau).mean()))
