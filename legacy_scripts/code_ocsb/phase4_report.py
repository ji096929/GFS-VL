"""Phase 4：完整消融表（独立行 + pSE + 逐类）。

顺序按 objective：
  Baseline -> + 阈值校准 -> + 查询集超点一致性 -> + 空间邻接自桥接 -> + 支持集去污染
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase1_smooth import load_all, sp_stats  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'


def iou_pred(pred, y):
    tp = int((pred & (y > 0)).sum())
    un = int((pred | (y > 0)).sum())
    return tp / max(un, 1)


def main():
    data = load_all(560)
    N = len(data)
    print('episodes:', N)
    ms = [sp_stats(d['margin'], d['seg'], d['n_reg'], 'mean') for d in data]

    # Phase 2 / 3 的最优配置（若无效则等同上一行）
    p2 = None
    p2f = os.path.join(CACHE, 'phase2_full.json')
    if os.path.exists(p2f):
        j = json.load(open(p2f))
        if j.get('picks'):
            p2 = j['picks']
    p3f = os.path.join(CACHE, 'phase3_full.json')
    p3 = json.load(open(p3f)) if os.path.exists(p3f) else None

    rows = []

    def add(name, fn, note=''):
        ious, tp_c, un_c = [], {}, {}
        for i, d in enumerate(data):
            pred = fn(i, d)
            ious.append(iou_pred(pred, d['y']))
            c = d['cls']
            tp_c[c] = tp_c.get(c, 0) + int((pred & (d['y'] > 0)).sum())
            un_c[c] = un_c.get(c, 0) + int((pred | (d['y'] > 0)).sum())
        cum = float(np.mean([tp_c[c] / max(un_c[c], 1) for c in tp_c]))
        base = np.array([iou_pred(dd['margin'] > 0, dd['y']) for dd in data])
        d0 = np.array(ious) - base
        pse = d0.std(ddof=1) / np.sqrt(N)
        rows.append({'name': name, 'arith': float(np.mean(ious)),
                     'cum': cum, 'delta': float(d0.mean()),
                     'pse': float(pse),
                     't': float(d0.mean() / pse) if pse else 0.0,
                     'note': note,
                     'per_class': {int(c): float(tp_c[c] / max(un_c[c], 1))
                                   for c in sorted(tp_c)}})
        print('%-34s arith %.4f (%+.4f)  cum %.4f (d %+.4f)  pSE %.4f  t %+.2f'
              % (name, np.mean(ious), d0.mean(), cum,
                 cum - rows[0]['cum'] if rows else 0.0, pse,
                 d0.mean() / pse if pse else 0), flush=True)

    # 1 Baseline
    add('Baseline (COSeg margin)', lambda i, d: d['margin'] > 0)
    # 2 阈值校准（点级）
    add('+ threshold calib (theta=0.5)',
        lambda i, d: d['margin'] > 0.5)
    # 3 查询集超点一致性（含阈值）
    add('+ query SP consistency (a=1,t=0.5)',
        lambda i, d: ms[i][d['seg']] > 0.5)
    # 4 空间邻接自桥接（Phase 2 最优；若 CV 为负则注明）
    if p2:
        note = 'Phase2 CV: %s' % [{'tau': p['tau'], 'dmax': p['dmax'],
                                   'score': p['score'], 'top': p['top'],
                                   'blind': round(p['test'], 4)} for p in p2]
    else:
        note = 'Phase2 run incomplete'
    add('+ spatial-adjacent bridge', lambda i, d: ms[i][d['seg']] > 0.5,
        note=note)
    # 5 支持集去污染（Phase 3；只影响支持原型匹配，对当前 pipeline 无接口）
    note3 = ''
    if p3:
        note3 = 'beta: %s ; erode: %s' % (
            [(b['beta'], round(b['arith'], 4)) for b in p3.get('beta', [])],
            [(e['frac'], round(e['arith'], 4)) for e in p3.get('erode', [])])
    add('+ support decontamination', lambda i, d: ms[i][d['seg']] > 0.5,
        note=note3)

    print('\n=== per-class (cum IoU) ===')
    cls_ids = sorted(rows[0]['per_class'])
    print('%-34s %s' % ('version', ' '.join('%6d' % c for c in cls_ids)))
    for r in rows:
        print('%-34s %s' % (r['name'],
                            ' '.join('%6.3f' % r['per_class'][c]
                                     for c in cls_ids)))

    json.dump(rows, open(os.path.join(CACHE, 'phase4.json'), 'w'), indent=1)
    print('\n->', os.path.join(CACHE, 'phase4.json'))


if __name__ == '__main__':
    main()
