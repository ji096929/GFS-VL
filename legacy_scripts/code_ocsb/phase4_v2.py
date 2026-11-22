"""Phase 4 v2：按 2 折 CV 盲测口径生成消融表。

每个组件在训练折上选超参，在盲测折上评估，两折拼接成 560 的盲测结果。
  Baseline           无超参
  + 阈值校准          θ ∈ {0.3,0.4,0.5,0.6}
  + 查询集超点一致性  agg×α×θ（144 配置）
  + 空间邻接自桥接    读 phase2_v2.json 的盲测 per-episode IoU
  + 支持集去污染      读 phase3_v2.json（对当前 pipeline 无接口，等同上一行）
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase1_smooth import load_all, sp_stats  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
AGGS = ('mean', 'median', 'p75', 'trim')
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
THETAS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6)
CAL_THETAS = (0.2, 0.3, 0.4, 0.5, 0.6)


def iou_of(pred, y):
    tp = int((pred & (y > 0)).sum())
    un = int((pred | (y > 0)).sum())
    return tp / max(un, 1)


def cum_of(preds, data):
    tp_c, un_c = {}, {}
    for pred, d in zip(preds, data):
        c = d['cls']
        tp_c[c] = tp_c.get(c, 0) + int((pred & (d['y'] > 0)).sum())
        un_c[c] = un_c.get(c, 0) + int((pred | (d['y'] > 0)).sum())
    return float(np.mean([tp_c[c] / max(un_c[c], 1) for c in tp_c])), tp_c, un_c


def cv_blind(configs, data, base_ious):
    N = len(data)
    half = N // 2
    folds = [list(range(half)), list(range(half, N))]
    blind = np.zeros(N)
    preds_blind = [None] * N
    picks = []
    for f in (0, 1):
        tr, te = folds[f], folds[1 - f]
        best = max(configs, key=lambda r: r['ious'][tr].mean())
        blind[te] = best['ious'][te]
        for i in te:
            preds_blind[i] = best['preds'][i]
        picks.append({'fold': f, 'cfg': best['cfg'],
                      'train': float(best['ious'][tr].mean()),
                      'test': float(best['ious'][te].mean())})
    d = blind - base_ious
    pse = d.std(ddof=1) / np.sqrt(N)
    cum, tp_c, un_c = cum_of(preds_blind, data)
    base_cum = cum_of([d0['margin'] > 0 for d0 in data], data)[0]
    per_class = {int(c): float(tp_c[c] / max(un_c[c], 1)) for c in sorted(tp_c)}
    return {'arith': float(blind.mean()), 'cum': cum,
            'delta': float(d.mean()), 'cum_delta': float(cum - base_cum),
            'pse': float(pse),
            't': float(d.mean() / pse) if pse else 0.0,
            'picks': picks, 'per_class': per_class}


def main():
    data = load_all(560)
    N = len(data)
    print('episodes:', N)

    base_ious = np.array([iou_of(d['margin'] > 0, d['y']) for d in data])
    base_preds = [d['margin'] > 0 for d in data]
    base_cum, base_tp, base_un = cum_of(base_preds, data)
    print('baseline arith %.4f cum %.4f' % (base_ious.mean(), base_cum))

    rows = [{'name': 'Baseline (COSeg)', 'arith': float(base_ious.mean()),
             'cum': base_cum, 'delta': 0.0, 'cum_delta': 0.0,
             'pse': 0.0, 't': 0.0,
             'per_class': {int(c): float(base_tp[c] / max(base_un[c], 1))
                           for c in sorted(base_tp)}}]

    # ---- 2) 阈值校准 ----
    cfgs = []
    for th in CAL_THETAS:
        preds = [d['margin'] > th for d in data]
        ious = np.array([iou_of(p, d['y']) for p, d in zip(preds, data)])
        cfgs.append({'cfg': th, 'ious': ious, 'preds': preds})
    r = cv_blind(cfgs, data, base_ious)
    r['name'] = '+ threshold calib (CV theta)'
    rows.append(r)
    print('%-34s arith %.4f (d %+.4f) cum %.4f (d %+.4f) pSE %.4f t %+.2f'
          % (r['name'], r['arith'], r['delta'], r['cum'], r['cum_delta'],
             r['pse'], r['t']))

    # ---- 3) 查询集超点一致性 ----
    ms = {agg: [sp_stats(d['margin'], d['seg'], d['n_reg'], agg) for d in data]
          for agg in AGGS}
    cfgs = []
    for agg in AGGS:
        for alpha in ALPHAS:
            for theta in THETAS:
                preds = []
                for i, d in enumerate(data):
                    ell = ((1 - alpha) * d['margin']
                           + alpha * ms[agg][i][d['seg']])
                    preds.append(ell > theta)
                ious = np.array([iou_of(p, d['y'])
                                 for p, d in zip(preds, data)])
                cfgs.append({'cfg': (agg, alpha, theta), 'ious': ious,
                             'preds': preds})
    r = cv_blind(cfgs, data, base_ious)
    r['name'] = '+ query SP consistency (CV)'
    rows.append(r)
    print('%-34s arith %.4f (d %+.4f) cum %.4f (d %+.4f) pSE %.4f t %+.2f'
          % (r['name'], r['arith'], r['delta'], r['cum'], r['cum_delta'],
             r['pse'], r['t']))

    # ---- 4) 邻接桥接（读 phase2_v2 盲测） ----
    p2 = None
    f2 = os.path.join(CACHE, 'phase2_v2.json')
    if os.path.exists(f2):
        p2 = json.load(open(f2))
    if p2 and p2.get('blind'):
        r4 = {'name': '+ spatial-adjacent bridge (CV)',
              'arith': p2['blind']['arith'], 'cum': p2['blind'].get('cum', 0),
              'delta': p2['blind']['arith'] - float(base_ious.mean()),
              'cum_delta': p2['blind'].get('cum', 0) - base_cum,
              'pse': p2['blind']['pse'], 't': p2['blind']['t'],
              'picks': p2.get('picks'), 'per_class': None,
              'bridge_only': {'arith': p2['blind']['delta'],
                              'cum': p2['blind'].get('cum_delta', 0)}}
    else:
        r4 = {'name': '+ spatial-adjacent bridge (CV)', 'arith': rows[2]['arith'],
              'cum': rows[2]['cum'], 'delta': rows[2]['delta'],
              'cum_delta': rows[2]['cum_delta'], 'pse': rows[2]['pse'],
              't': rows[2]['t'], 'per_class': None,
              'note': 'phase2_v2.json missing'}
    rows.append(r4)
    print('%-34s arith %.4f (d %+.4f) cum %.4f (d %+.4f) pSE %.4f t %+.2f'
          % (r4['name'], r4['arith'], r4['delta'], r4['cum'],
             r4['cum_delta'], r4['pse'], r4['t']))

    # ---- 5) 支持集去污染（无接口，等同上一行） ----
    r5 = dict(r4)
    r5['name'] = '+ support decontamination (CV)'
    r5['note'] = ('phase3_v2: beta/erode only affect support-prototype '
                  'matching, no interface to this pipeline')
    f3 = os.path.join(CACHE, 'phase3_v2.json')
    if os.path.exists(f3):
        p3 = json.load(open(f3))
        r5['phase3'] = {'beta': p3.get('beta'), 'erode': p3.get('erode'),
                        'adapt_cv': p3.get('adapt_cv')}
    rows.append(r5)
    print('%-34s arith %.4f (d %+.4f) cum %.4f (d %+.4f) pSE %.4f t %+.2f'
          % (r5['name'], r5['arith'], r5['delta'], r5['cum'],
             r5['cum_delta'], r5['pse'], r5['t']))

    print('\n=== per-class (cum IoU) ===')
    cls_ids = sorted(rows[0]['per_class'])
    print('%-34s %s' % ('version', ' '.join('%6d' % c for c in cls_ids)))
    for r in rows:
        if r.get('per_class') is None:
            print('%-34s %s' % (r['name'], '(blinded, per-class in source json)'))
            continue
        print('%-34s %s' % (r['name'],
                            ' '.join('%6.3f' % r['per_class'][c]
                                     for c in cls_ids)))

    json.dump(rows, open(os.path.join(CACHE, 'phase4_v2.json'), 'w'), indent=1)
    print('\n->', os.path.join(CACHE, 'phase4_v2.json'))


if __name__ == '__main__':
    main()
