"""Phase 3：自适应防爆阈值 + 支持集去污染诊断。

主实验：任务级自适应阈值
    theta_ep = theta_0 + lam * relu(fg_ratio - 0.15)
其中 fg_ratio 是在 theta_0 下的预测前景占比。2 折交叉验证。

诊断：背景正交投影与边界腐蚀只改变"支持原型匹配"这一类打分，而
Phase 1/2 都不依赖支持原型，所以它们对当前 pipeline 没有直接作用。
这里只测它们对支持集前景/背景可分性的影响，供后续参考。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

CACHE = '/data01/jh/data/ocsb_cache'
INTRA = os.path.join(CACHE, 'intra')

THETA0 = (0.4, 0.5, 0.6)
LAMS = (0.0, 0.5, 1.0, 2.0, 4.0)
FG_THR = 0.15


def iou_of(pred, y):
    tp = int((pred & (y > 0)).sum())
    un = int((pred | (y > 0)).sum())
    return tp / max(un, 1)


def load(n):
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))[:n]
    eps = []
    for i, m in enumerate(metas):
        v = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
        a = np.load(os.path.join(INTRA, 'intra_%03d.npz' % i))
        margin = v['q_margin'].astype(np.float32)
        seg = a['q_seg'].astype(np.int64)
        uniq, inv = np.unique(seg, return_inverse=True)
        n_reg = len(uniq)
        cnt = np.bincount(inv, minlength=n_reg).astype(np.float64)
        m_sp = np.bincount(inv, weights=margin, minlength=n_reg) / np.maximum(cnt, 1)
        eps.append({'m': m_sp, 'inv': inv, 'y': v['q_y'].astype(np.int64),
                    'cls': int(m['class']), 'cn': cnt})
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=560)
    ap.add_argument('--out', default=os.path.join(CACHE, 'phase3.json'))
    args = ap.parse_args()

    eps = load(args.n)
    N = len(eps)
    print('episodes:', N, flush=True)

    # 配置：(theta0, lam)
    configs = []
    for th0 in THETA0:
        for lam in LAMS:
            ious = []
            fr = []
            for e in eps:
                m = e['m']
                fg0 = float((m > th0).mean())
                th = th0 + lam * max(fg0 - FG_THR, 0.0)
                ious.append(iou_of(m[e['inv']] > th, e['y']))
                fr.append(fg0)
            configs.append({'theta0': th0, 'lam': lam,
                            'ious': np.array(ious),
                            'fg_mean': float(np.mean(fr))})
            print('theta0 %.1f lam %.1f  arith %.4f  (fg_ratio %.3f)'
                  % (th0, lam, np.mean(ious), np.mean(fr)), flush=True)

    # 2 折 CV
    half = N // 2
    folds = [list(range(half)), list(range(half, N))]
    # baseline 用 theta0=0.5, lam=0
    base = next(c for c in configs if c['theta0'] == 0.5 and c['lam'] == 0.0)
    base_ious = base['ious']
    blind = np.zeros(N)
    picks = []
    for f in (0, 1):
        tr, te = folds[f], folds[1 - f]
        best = max(configs, key=lambda r: r['ious'][tr].mean())
        blind[te] = best['ious'][te]
        picks.append({'fold': f, 'theta0': best['theta0'], 'lam': best['lam'],
                      'train': float(best['ious'][tr].mean()),
                      'test': float(best['ious'][te].mean())})
        print('fold %d: theta0 %.1f lam %.1f  train %.4f blind %.4f'
              % (f, best['theta0'], best['lam'],
                 best['ious'][tr].mean(), best['ious'][te].mean()), flush=True)

    d = blind - base_ious
    pse = d.std(ddof=1) / np.sqrt(N)
    print('\n=== blind 560 (vs theta0=0.5 lam=0) ===')
    print('blind arith %.4f  baseline %.4f  d %+.4f  pSE %.4f  t %+.2f'
          % (blind.mean(), base_ious.mean(), d.mean(), pse,
             d.mean() / pse if pse else 0), flush=True)

    out = {'configs': [{'theta0': c['theta0'], 'lam': c['lam'],
                        'arith': float(c['ious'].mean()),
                        'fg_mean': c['fg_mean']} for c in configs],
           'blind': {'arith': float(blind.mean()),
                     'delta': float(d.mean()), 'pse': float(pse),
                     't': float(d.mean() / pse) if pse else 0.0},
           'base_arith': float(base_ious.mean()), 'picks': picks}
    json.dump(out, open(args.out, 'w'), indent=1)
    print('->', args.out)


if __name__ == '__main__':
    main()
