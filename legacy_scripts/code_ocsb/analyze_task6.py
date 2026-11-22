"""Task 6 analysis: drop-off anatomy, asymmetric fusion, RSV gating, K sweep.

Usage:
  python analyze_task6.py <cross_glob> <same_glob>
"""
import glob
import json
import sys

import numpy as np


def load(p):
    r = []
    for f in sorted(glob.glob(p)):
        try:
            for line in open(f):
                try:
                    r.append(json.loads(line))
                except Exception:
                    pass
        except Exception:
            pass
    return r


CROSS = load(sys.argv[1] if len(sys.argv) > 1
             else '/data01/jh/data/diag_t6x_s12*/records.jsonl')
SAME = load(sys.argv[2] if len(sys.argv) > 2
            else '/data01/jh/data/diag_t6s_s13*/records.jsonl')
print('cross %d   same %d' % (len(CROSS), len(SAME)))


def gain(r, key, base='oracle_base_iou'):
    a, b = r.get(key), r.get(base)
    if a is None or b is None:
        return np.nan
    return a - b


def col(recs, key):
    return np.array([r.get(key, np.nan) for r in recs], dtype=float)


for tag, recs in (('CROSS', CROSS), ('SAME', SAME)):
    if not recs:
        continue
    base = col(recs, 'oracle_base_iou')
    print('\n' + '=' * 66)
    print('%s  n=%d  baseline %.4f' % (tag, len(recs), np.nanmean(base)))

    # ---------- 1. drop-off anatomy on oracle 2c 0.5 ----------
    g05 = col(recs, 'oracle_iou_2c_0.5')
    tp5 = col(recs, 'oracle_2c05_tp')
    fp5 = col(recs, 'oracle_2c05_fp')
    fn5 = col(recs, 'oracle_2c05_fn')
    rec5 = col(recs, 'oracle_2c05_rec')
    pre5 = col(recs, 'oracle_2c05_prec')
    b_rec = col(recs, 'oracle_base_rec')
    b_pre = col(recs, 'oracle_base_prec')
    ok = (~np.isnan(g05) & ~np.isnan(rec5) & ~np.isnan(b_rec))
    d = g05 - base
    down = ok & (d < 0)
    up = ok & (d > 0)
    print('\n## 1. oracle 2c lam0.5 drop-off anatomy')
    print('episodes: up %d  flat %d  down %d (%.1f%%)'
          % (up.sum(), (ok & (np.abs(d) < 1e-9)).sum(), down.sum(),
             100 * down.sum() / max(ok.sum(), 1)))
    if down.sum():
        drec = (rec5 - b_rec)[down]
        dpre = (pre5 - b_pre)[down]
        print('down episodes: dRecall %+.4f (median %+.4f)  dPrecision %+.4f (median %+.4f)'
              % (drec.mean(), np.median(drec), dpre.mean(), np.median(dpre)))
        only_rec = (drec < 0) & (dpre >= 0)
        only_pre = (dpre < 0) & (drec >= 0)
        both = (drec < 0) & (dpre < 0)
        print('   recall-only loss %d (%.0f%%)  precision-only %d (%.0f%%)  both %d (%.0f%%)'
              % (only_rec.sum(), 100 * only_rec.sum() / down.sum(),
                 only_pre.sum(), 100 * only_pre.sum() / down.sum(),
                 both.sum(), 100 * both.sum() / down.sum()))
        dtp = (tp5 - col(recs, 'oracle_base_tp'))[down]
        dfp = (fp5 - col(recs, 'oracle_base_fp'))[down]
        dfn = (fn5 - col(recs, 'oracle_base_fn'))[down]
        btp = col(recs, 'oracle_base_tp')[down]
        # fraction of already-correct foreground that the bridge destroys
        lost = -dtp / np.maximum(btp, 1)
        print('   dTP median %+.0f (mean %+.0f)  dFP median %+.0f (mean %+.0f)'
              % (np.median(dtp), dtp.mean(), np.median(dfp), dfp.mean()))
        print('   fraction of base TP destroyed: median %.3f  mean %.3f'
              % (np.median(lost), lost.mean()))
        print('   dFN median %+.0f  -> net recall change driven by TP loss'
              % np.median(dfn))
    if up.sum():
        print('up episodes:   dRecall %+.4f  dPrecision %+.4f'
              % ((rec5 - b_rec)[up].mean(), (pre5 - b_pre)[up].mean()))

    # ---------- 2. asymmetric fusion vs symmetric ----------
    print('\n## 2. fusion operators (oracle, delta vs baseline)')
    print('%-24s %9s %9s' % ('config', 'delta', 'n'))
    cfgs = ['oracle_iou_2c_0.3', 'oracle_iou_2c_0.5', 'oracle_iou_2c_0.7']
    for t in (0.0, 0.5, 1.0):
        cfgs += ['oracle_iou_omit%g_lam0.3' % t, 'oracle_iou_omit%g_lam0.5' % t]
    for t in (0.0, 0.5, 1.0, 1.5):
        cfgs += ['oracle_iou_boost%g_lam0.3' % t, 'oracle_iou_boost%g_lam0.5' % t]
    for k in cfgs:
        v = col(recs, k)
        m = ~np.isnan(v) & ~np.isnan(base)
        if m.sum() < 5:
            continue
        print('%-24s %+9.4f %9d' % (k.replace('oracle_iou_', ''),
                                    (v[m] - base[m]).mean(), m.sum()))

    # ---------- 3. RSV correlation and gating ----------
    print('\n## 3. reverse-support validation (RSV)')
    for n in ('bpos', 'ball'):
        ri = col(recs, 'rsv_iou_' + n)
        rf = col(recs, 'rsv_fpr_' + n)
        m = ~np.isnan(ri) & ~np.isnan(g05) & ~np.isnan(base)
        if m.sum() < 10:
            continue
        g = (g05 - base)[m]
        print('%-6s n=%d  rsv_iou mean %.3f  rsv_fpr mean %.3f'
              % (n, m.sum(), ri[m].mean(), rf[m].mean()))
        print('   corr(rsv_iou, gain) %+.3f   corr(rsv_fpr, gain) %+.3f'
              % (np.corrcoef(ri[m], g)[0, 1],
                 np.corrcoef(rf[m], g)[0, 1]))

    # gate: only inject when the bridge can reproduce the support block
    for n in ('ball', 'bpos'):
        ri = col(recs, 'rsv_iou_' + n)
        if np.isnan(ri).all():
            continue
        print('   gate by rsv_iou_%s:' % n)
        print('     %-8s %6s %9s %+9s' % ('tau', 'kept', 'IoU', 'delta'))
        for tau in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            vals = []
            for r in recs:
                b = r.get('oracle_base_iou')
                a = r.get('oracle_iou_2c_0.5')
                q = r.get('rsv_iou_' + n)
                if b is None or a is None or q is None:
                    continue
                vals.append(a if q >= tau else b)
            if not vals:
                continue
            v = np.array(vals)
            print('     %-8.1f %6d %9.4f %+9.4f'
                  % (tau, len(v), v.mean(), v.mean() - np.nanmean(base)))

    # ---------- 4. K sweep ----------
    print('\n## 4. end-to-end K sweep (delta vs baseline)')
    print('%-6s %10s %10s %10s %10s' % ('K', 'combo.2', 'combo.3',
                                        'qc.2', 'qc.3'))
    for K in (3, 5, 10, 15, 20):
        row = []
        for pre in ('e2e_top%d', 'e2e_qc_top%d'):
            for lam in (0.2, 0.3):
                v = col(recs, (pre % K) + '_lam%.1f' % lam)
                m = ~np.isnan(v) & ~np.isnan(base)
                row.append((v[m] - base[m]).mean() if m.sum() > 5 else np.nan)
        print('%-6d %+10.4f %+10.4f %+10.4f %+10.4f' % (K, *row))
    for lam in (0.1, 0.2, 0.3):
        v = col(recs, 'e2e_all_lam%.1f' % lam)
        m = ~np.isnan(v) & ~np.isnan(base)
        if m.sum() > 5:
            print('All_B lam%.1f  %+.4f (n=%d)'
                  % (lam, (v[m] - base[m]).mean(), m.sum()))
    print('\n   asymmetric at K=15/20:')
    for K in (15, 20):
        for pre in ('e2e_top%d', 'e2e_qc_top%d'):
            for suf in ('omit_lam0.3', 'boost_lam0.3'):
                v = col(recs, pre % K + '_' + suf)
                m = ~np.isnan(v) & ~np.isnan(base)
                if m.sum() > 5:
                    print('   %-24s %+.4f' % (pre % K + '_' + suf,
                                              (v[m] - base[m]).mean()))
    print()
