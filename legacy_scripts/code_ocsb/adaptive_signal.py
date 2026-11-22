"""Task 2e: per-class adaptive signal selection.

For each novel class, pick the signal with the highest per-episode AUC (and
optionally with a train/holdout split to avoid selection bias), then report
the class-level AUC and Top-K precision under that signal.
"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

pattern = sys.argv[1] if len(sys.argv) > 1 else '/data01/jh/data/diag_502_s*/records.jsonl'
recs = []
for f in sorted(glob.glob(pattern)):
    try:
        for line in open(f):
            recs.append(json.loads(line))
    except Exception:
        pass
print('episodes:', len(recs))

SIGNALS = ['delta_base', 'base_abs', 'multi_max', 'margin', 'eta_intra',
           'spread', 'base_prob', 'size', 'q_std', 'q_p90',
           'sim_pool_bg', 'rel_pool_bg', 'delta_bg_risk',
           'spread_t005', 'spread_t02', 'spread_multi',
           'promiscuity', 'delta_mean', 'feat_var', 'local_rel', 'q_c']


def collect(rs, sig):
    """Return (pos, neg, per_ep_aucs, topk_precisions)."""
    pos, neg, per = [], [], []
    topk = {1: [], 2: [], 3: []}
    for r in rs:
        if sig == 'q_c':
            qv = r.get('B_q')
            gv = r.get('B_gt')
            if not qv or not gv:
                continue
            p = [qv[i] for i in range(len(qv)) if gv[i] == 1]
            n = [qv[i] for i in range(len(qv)) if gv[i] == 0]
        else:
            p = r.get('B_pos_' + sig)
            n = r.get('B_neg_' + sig)
            if not p or not n:
                continue
            p = [x for x in p if x is not None and not np.isnan(x)]
            n = [x for x in n if x is not None and not np.isnan(x)]
        if not p or not n:
            continue
        pos += p
        neg += n
        try:
            per.append(roc_auc_score([1] * len(p) + [0] * len(n), p + n))
        except ValueError:
            pass
        vals = np.array(p + n)
        gts = np.array([1] * len(p) + [0] * len(n))
        order = np.argsort(vals)[::-1]
        for K in (1, 2, 3):
            if len(order) >= K:
                topk[K].append(gts[order[:K]].mean())
    if not pos or not neg:
        return None
    auc = roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg)
    return {
        'auc': max(auc, 1 - auc),
        'per': np.mean(per) if per else float('nan'),
        'top': {K: (np.mean(v) if v else float('nan'))
                for K, v in topk.items()},
        'n_pos': len(pos), 'n_neg': len(neg),
    }


by_class = defaultdict(list)
for r in recs:
    by_class[r['class']].append(r)

NAMES = {2: 'floor', 3: 'chair', 5: 'desk', 6: 'bed', 7: 'bookshelf',
         10: 'bathtub', 12: 'curtain', 13: 'counter', 14: 'door',
         19: 'cabinet'}

print('\n## Per-class best signal (full data; selection bias noted)')
print('class  name        n   best_sig       AUC    top1   top2   top3')
rows = []
for c in sorted(by_class):
    rs = by_class[c]
    best = None
    for sig in SIGNALS:
        res = collect(rs, sig)
        if res and (best is None or res['auc'] > best[1]['auc']):
            best = (sig, res)
    if best:
        sig, res = best
        print('%-5d %-10s %3d  %-13s %.3f  %.3f  %.3f  %.3f'
              % (c, NAMES.get(c, '?'), len(rs), sig, res['auc'],
                 res['top'][1], res['top'][2], res['top'][3]))
        rows.append((c, len(rs), sig, res))

# weighted average over classes
n = sum(r[1] for r in rows)
w_auc = sum(r[1] * r[3]['auc'] for r in rows) / max(n, 1)
w_t1 = sum(r[1] * r[3]['top'][1] for r in rows) / max(n, 1)
w_t2 = sum(r[1] * r[3]['top'][2] for r in rows) / max(n, 1)
w_t3 = sum(r[1] * r[3]['top'][3] for r in rows) / max(n, 1)
print('\nweighted adaptive: AUC %.3f  top1 %.3f  top2 %.3f  top3 %.3f'
      % (w_auc, w_t1, w_t2, w_t3))

# holdout check: pick on the first half of episodes per class, evaluate on second
print('\n## Holdout check (select on first half, evaluate on second)')
print('class  name        n_tr  n_te  best_sig       test_AUC  test_top3')
test_aucs, test_t3 = [], []
for c in sorted(by_class):
    rs = by_class[c]
    half = len(rs) // 2
    if half < 2 or len(rs) - half < 2:
        continue
    train, test = rs[:half], rs[half:]
    best = None
    for sig in SIGNALS:
        res = collect(train, sig)
        if res and (best is None or res['auc'] > best[1]['auc']):
            best = (sig, res)
    if best is None:
        continue
    sig = best[0]
    te = collect(test, sig)
    if te is None:
        continue
    print('%-5d %-10s %4d  %4d  %-13s %.4f    %.3f'
          % (c, NAMES.get(c, '?'), half, len(rs) - half, sig, te['auc'],
             te['top'][3]))
    test_aucs += [te['auc']] * (len(rs) - half)
    test_t3 += [te['top'][3]] * (len(rs) - half)
if test_aucs:
    print('\nholdout adaptive: AUC %.3f  top3 %.3f (n=%d)'
          % (np.mean(test_aucs), np.mean(test_t3), len(test_aucs)))
