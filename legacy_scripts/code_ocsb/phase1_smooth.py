"""Phase 1：查询集超点内部一致性平滑 + 2 折交叉验证。

ell(i) = (1-alpha) * q_margin(i) + alpha * m(R_i) - theta
m(R) 支持 mean / median / p75 / trimmed_mean。

交叉验证：前 280 选 (agg, alpha, theta)，后 280 盲测；反向再来一次，
拼成 560 集的盲测结果。
"""
import argparse
import json
import os
import sys

import numpy as np

CACHE = '/data01/jh/data/ocsb_cache'
INTRA = os.path.join(CACHE, 'intra')
AGGS = ('mean', 'median', 'p75', 'trim')
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
THETAS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6)


def sp_stats(margin, seg, n_reg, agg):
    """按超点聚合。返回 [n_reg]。"""
    if n_reg == 0:
        return np.zeros(0)
    if agg == 'mean':
        cnt = np.bincount(seg, minlength=n_reg).astype(np.float64)
        s = np.bincount(seg, weights=margin, minlength=n_reg)
        return s / np.maximum(cnt, 1)
    order = np.argsort(seg, kind='stable')
    ss = seg[order]
    mm = margin[order]
    bounds = np.searchsorted(ss, np.arange(n_reg + 1))
    out = np.zeros(n_reg)
    for r in range(n_reg):
        v = mm[bounds[r]:bounds[r + 1]]
        if len(v) == 0:
            continue
        if agg == 'median':
            out[r] = np.median(v)
        elif agg == 'p75':
            out[r] = np.percentile(v, 75)
        else:  # trim
            if len(v) < 5:
                out[r] = v.mean()
            else:
                k = max(int(len(v) * 0.1), 1)
                out[r] = v.mean() if k * 2 >= len(v) else v[k:-k].mean()
    return out


def load_all(n):
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))[:n]
    data = []
    for i, m in enumerate(metas):
        v = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
        p = os.path.join(INTRA, 'intra_%03d.npz' % i)
        if not os.path.exists(p):
            continue
        a = np.load(p)
        margin = v['q_margin_full'].astype(np.float64)
        seg = a['q_seg_full'].astype(np.int64)
        if len(seg) != len(margin):
            continue
        uniq, inv = np.unique(seg, return_inverse=True)
        data.append({'margin': margin, 'y': v['q_y_full'].astype(np.int64),
                     'seg': inv, 'n_reg': len(uniq), 'cls': m['class'],
                     'ep': i})
    return data


def iou_of(pred, y):
    tp = int((pred & (y > 0)).sum())
    un = int((pred | (y > 0)).sum())
    return tp / max(un, 1)


def stat_block(data, idx, ious, base):
    d = np.array(ious) - np.array(base)
    pse = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan')
    return {'arith': float(np.mean(ious)), 'delta': float(d.mean()),
            'pse': float(pse),
            't': float(d.mean() / pse) if pse else 0.0}


def cum_miou(data, idx, preds):
    tp_c, un_c = {}, {}
    for i, pred in zip(idx, preds):
        c = data[i]['cls']
        y = data[i]['y']
        tp_c[c] = tp_c.get(c, 0) + int((pred & (y > 0)).sum())
        un_c[c] = un_c.get(c, 0) + int((pred | (y > 0)).sum())
    return float(np.mean([tp_c[c] / max(un_c[c], 1) for c in tp_c]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=560)
    ap.add_argument('--out', default=os.path.join(CACHE, 'phase1.json'))
    args = ap.parse_args()

    data = load_all(args.n)
    print('episodes with intra cache:', len(data), flush=True)
    N = len(data)

    # 预计算每种聚合的超点统计
    m_stats = {}
    for agg in AGGS:
        m_stats[agg] = [sp_stats(d['margin'], d['seg'], d['n_reg'], agg)
                        for d in data]
        print('  agg %s done' % agg, flush=True)

    # 基线
    base_ious = [iou_of(d['margin'] > 0, d['y']) for d in data]
    print('baseline arith %.4f' % np.mean(base_ious), flush=True)

    # 超点级理论上限
    up1, up2 = [], []
    I0 = float(np.mean(base_ious))
    for d in data:
        y = d['y'] > 0
        seg = d['seg']
        # (1) 每个超点赋真实主类
        n_reg = d['n_reg']
        fg_cnt = np.bincount(seg, weights=y.astype(np.float64), minlength=n_reg)
        cnt = np.bincount(seg, minlength=n_reg).astype(np.float64)
        frac = fg_cnt / np.maximum(cnt, 1)
        pred1 = (frac[seg] > 0.5)
        up1.append(iou_of(pred1, d['y']))
        # (2) 超点内最优正负号翻转（真前景占比 > I0 才翻正）
        pred2 = (frac[seg] > I0)
        up2.append(iou_of(pred2, d['y']))
    print('upper bound (majority label) arith %.4f' % np.mean(up1))
    print('upper bound (per-sp flip)    arith %.4f' % np.mean(up2))

    # 2 折交叉验证
    half = N // 2
    folds = [list(range(half)), list(range(half, N))]

    results = []
    for agg in AGGS:
        ms = m_stats[agg]
        for alpha in ALPHAS:
            # 预计算 ell_base（不含 theta）
            ells = []
            for i, d in enumerate(data):
                m = ms[i][d['seg']]
                ells.append((1.0 - alpha) * d['margin'] + alpha * m)
            for theta in THETAS:
                ious = [iou_of(ells[i] > theta, data[i]['y'])
                        for i in range(N)]
                results.append({'agg': agg, 'alpha': alpha, 'theta': theta,
                                'ious': np.array(ious)})
    print('configs:', len(results), flush=True)

    blind_ious = np.zeros(N)
    picks = []
    for f in (0, 1):
        tr, te = folds[f], folds[1 - f]
        best = max(results, key=lambda r: r['ious'][tr].mean())
        picks.append({'fold': f, 'agg': best['agg'], 'alpha': best['alpha'],
                      'theta': best['theta'],
                      'train': float(best['ious'][tr].mean()),
                      'test': float(best['ious'][te].mean())})
        blind_ious[te] = best['ious'][te]
        print('fold %d: %s alpha %.1f theta %.1f  train %.4f  blind %.4f'
              % (f, best['agg'], best['alpha'], best['theta'],
                 best['ious'][tr].mean(), best['ious'][te].mean()),
              flush=True)

    res = {
        'baseline': {'arith': float(np.mean(base_ious)),
                     'cum': cum_miou(data, range(N),
                                     [data[i]['margin'] > 0 for i in range(N)])},
        'upper_majority': float(np.mean(up1)),
        'upper_flip': float(np.mean(up2)),
        'blind': stat_block(data, range(N), blind_ious, base_ious),
        'blind_cum': None,
        'picks': picks,
    }
    # 盲测的累计 mIoU：用两次 fold 的最优配置分别算
    cum_parts = []
    for f in (0, 1):
        tr, te = folds[f], folds[1 - f]
        p = [x for x in picks if x['fold'] == f][0]
        r = next(r for r in results
                 if r['agg'] == p['agg'] and r['alpha'] == p['alpha']
                 and r['theta'] == p['theta'])
        cum_parts.append((te, r['ious']))
    # 直接重算：对每个 fold 的测试集用对应配置
    preds = {}
    for f in (0, 1):
        te = folds[1 - f]
        p = picks[f]
        agg, alpha, theta = p['agg'], p['alpha'], p['theta']
        ms = m_stats[agg]
        for i in te:
            d = data[i]
            m = ms[i][d['seg']]
            ell = (1 - alpha) * d['margin'] + alpha * m
            preds[i] = ell > theta
    res['blind_cum'] = cum_miou(data, range(N), [preds[i] for i in range(N)])
    res['all_configs'] = [{'agg': r['agg'], 'alpha': r['alpha'],
                           'theta': r['theta'],
                           'arith': float(r['ious'].mean())}
                          for r in results]

    print('\n=== blind 560 ===')
    print('arith %.4f  (d %+.4f)  pSE %.4f  t %.2f'
          % (res['blind']['arith'], res['blind']['delta'],
             res['blind']['pse'], res['blind']['t']))
    print('cum %.4f  (baseline cum %.4f)'
          % (res['blind_cum'], res['baseline']['cum']))
    json.dump(res, open(args.out, 'w'), indent=1)
    print('->', args.out)


if __name__ == '__main__':
    main()
