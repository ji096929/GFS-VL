"""核查 +2.13pp 的两个乐观来源。

1. 插值控制：baseline 用原始点级 margin，Step4 用 downsampled margin 插值。
   如果 interp(margin_down) 本身的 IoU 就不同于 baseline，那提升里混入了插值效应。
2. 50/50 交叉验证：前 50 个 episode 扫 gamma，后 50 个报告，避免超参泄漏。
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ocsb_faithful import OCSB  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'


def to_t(a):
    if a.dtype == np.float16:
        return torch.from_numpy(a.astype(np.float32)).cuda()
    if a.dtype.kind in 'iu':
        return torch.from_numpy(a.astype(np.int64)).cuda()
    return torch.from_numpy(a.astype(np.float32)).cuda()


def interp(cf, cd, val, k=3):
    d, idx = cKDTree(cd).query(cf, k=k)
    w = 1.0 / (d + 1e-8)
    w /= w.sum(1, keepdims=True)
    return (val[idx] * w).sum(1)


def iou_from_pred(pred, y):
    tp = int((pred & (y > 0)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y > 0)).sum())
    return tp / max(tp + fp + fn, 1)


def iou_direct(ell_full, ep):
    """直接用原始点级得分，不插值。"""
    return iou_from_pred(ell_full.cpu().numpy() > 0,
                         ep['q_y_full'].cpu().numpy())


def iou_down(ell_down, ep):
    """downsampled 得分插值回原始点。"""
    e = interp(ep['q_coord_full'].cpu().numpy().astype(np.float64),
               ep['q_coord'].cpu().numpy().astype(np.float64),
               ell_down.cpu().numpy().astype(np.float64))
    return iou_from_pred(e > 0, ep['q_y_full'].cpu().numpy())


def main():
    files = sorted(glob.glob(os.path.join(CACHE, 'val_*.npz')))
    eps = []
    for f in files:
        d = np.load(f)
        eps.append({k: to_t(d[k]) for k in d.files})
    dim = int(eps[0]['s_h'].shape[1])
    print('episodes:', len(eps))

    # ---- 1. 插值控制 ----
    full_ious, down_ious = [], []
    for ep in eps:
        full_ious.append(iou_direct(ep['q_margin_full'], ep))
        down_ious.append(iou_down(ep['q_margin'], ep))
    full = float(np.mean(full_ious))
    down = float(np.mean(down_ious))
    print('\n[1] 插值控制')
    print('    baseline: 原始点级 margin   IoU %.4f' % full)
    print('    control : downsampled margin 插值回原始点  IoU %.4f  (%+.4f)'
          % (down, down - full))
    print('    -> 后续 Step4 的增益必须减去这个零点偏移')

    # ---- 2. 前 50 扫 gamma，后 50 验证 ----
    m = OCSB(dim, tau=0.1, tau_b=0.6, fusion_mode='A', gamma_fuse=1.0).cuda()
    deltas = []
    with torch.no_grad():
        for ep in eps:
            out = m(ep, enable_csr=True)
            deltas.append(out['delta_mem'].detach())

    gammas = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

    def split_eval(idx_a, idx_b, tag):
        best, rows = None, []
        for g in gammas:
            ia = [iou_down(ep['q_margin'] + g * deltas[i], ep)
                  for i, ep in enumerate(eps) if i in idx_a]
            va = float(np.mean(ia)) - float(np.mean(
                [down_ious[i] for i in idx_a]))
            rows.append((g, va))
            if best is None or va > best[1]:
                best = (g, va)
        g_star = best[0]
        ib = [iou_down(ep['q_margin'] + g_star * deltas[i], ep)
              for i, ep in enumerate(eps) if i in idx_b]
        base_b = float(np.mean([down_ious[i] for i in idx_b]))
        vb = float(np.mean(ib)) - base_b
        # 配对 SE
        diffs = np.array(ib) - np.array([down_ious[i] for i in idx_b])
        pse = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
        print('\n[2] %s' % tag)
        print('    A 段扫出 gamma* = %.2f (A 段 d %+.4f)' % (g_star, best[1]))
        print('    B 段验证: IoU %.4f vs control %.4f  d %+.4f  pSE %.4f  t %.2f'
              % (float(np.mean(ib)), base_b, vb, pse, vb / pse if pse else 0))
        return {'gamma': g_star, 'delta_b': vb, 'pse': pse}

    n = len(eps)
    half = n // 2
    r1 = split_eval(set(range(half)), set(range(half, n)), '前50扫 -> 后50测')
    r2 = split_eval(set(range(half, n)), set(range(half)), '后50扫 -> 前50测')

    print('\n[3] 汇总')
    print('    同集扫(之前报的) d %+.4f' % (float(np.mean(
        [iou_down(ep['q_margin'] + 2.0 * deltas[i], ep)
         for i, ep in enumerate(eps)])) - down))
    print('    交叉验证据此: %+.4f / %+.4f' % (r1['delta_b'], r2['delta_b']))
    json.dump({'interp_control': down - full, 'split1': r1, 'split2': r2},
              open(os.path.join(CACHE, 'verify_gain.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
