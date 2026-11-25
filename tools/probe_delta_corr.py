#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XL-2 诊断 2A-1：ScanNet200 5-shot 的两组增量是否救同一批类。

Δ_sym    = iou(sym_CE)      - iou(base_CE)
Δ_lovasz = iou(base_Lovasz) - iou(base_CE)

输出：
  * Pearson / Spearman 相关系数（45 novel 类）
  * Top-10 重叠类（两边都排前 10）
  * 优势缩水 Top-10：按 (sym_ce - base_ce) - (sym_lovasz - base_lovasz) 降序
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/GFS-VL')
from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import CLASS_LABELS_BASE_NOVEL

R = '/data01/jh/code/GFS-VL/exp/scannet200/'
NB = 12
NAMES = list(CLASS_LABELS_BASE_NOVEL)
RUNS = [
    ('base_ce',     R + 'base_k5_102400_s42/result/val.pth'),
    ('sym_ce',      R + 'sym_k5_102400_s42/result/val.pth'),
    ('base_lovasz', R + 'base_lovasz_k5_102400_s42/result/val.pth'),
    ('sym_lovasz',  R + 'sym_lovasz_k5_102400_s42/result/val.pth'),
]

def iou_of(p):
    d = torch.load(p, map_location='cpu', weights_only=False)
    it = d['intersection'].astype(np.float64); un = d['union'].astype(np.float64)
    return np.where(un > 0, it / np.maximum(un, 1), np.nan)

def rank(x):
    return np.argsort(np.argsort(x)).astype(np.float64)

def main():
    iou = {}
    for tag, p in RUNS:
        try:
            iou[tag] = iou_of(p)
        except Exception as e:
            print('load fail %s: %s' % (tag, e)); return
    d_sym = iou['sym_ce'][NB:] - iou['base_ce'][NB:]
    d_lov = iou['base_lovasz'][NB:] - iou['base_ce'][NB:]
    ok = ~np.isnan(d_sym) & ~np.isnan(d_lov)
    a, b = d_sym[ok], d_lov[ok]
    print('=== 两组增量的相关（%d 个类）===' % ok.sum())
    print('  Pearson  r = %+.4f' % np.corrcoef(a, b)[0, 1])
    print('  Spearman r = %+.4f' % np.corrcoef(rank(a), rank(b))[0, 1])
    print('  Δ_sym    均值 %+.4f  中位 %+.4f  >0 类数 %d' % (a.mean(), np.median(a), int((a > 0).sum())))
    print('  Δ_lovasz 均值 %+.4f  中位 %+.4f  >0 类数 %d' % (b.mean(), np.median(b), int((b > 0).sum())))

    idx = np.arange(len(NAMES))[NB:][ok]
    top_sym = set(idx[np.argsort(-a)[:10]].tolist())
    top_lov = set(idx[np.argsort(-b)[:10]].tolist())
    inter = sorted(top_sym & top_lov)
    print('\n=== Top-10 重叠 ===')
    print('  Δ_sym Top-10:    ' + ', '.join(NAMES[i] for i in sorted(top_sym, key=lambda x: -d_sym[x-NB])))
    print('  Δ_lovasz Top-10: ' + ', '.join(NAMES[i] for i in sorted(top_lov, key=lambda x: -d_lov[x-NB])))
    print('  两者都进 Top-10: %d 个 -> %s' % (len(inter), ', '.join(NAMES[i] for i in inter)))

    shrink = {i: (iou['sym_ce'][i] - iou['base_ce'][i]) - (iou['sym_lovasz'][i] - iou['base_lovasz'][i])
              for i in idx}
    order = sorted(shrink.items(), key=lambda kv: -kv[1])
    print('\n=== 优势缩水 Top-10（sym 相对 base 的增量，CE 减 Lovász）===')
    print('%-22s %9s %9s %9s %9s %9s' % ('class', 'base_ce', 'sym_ce', 'base_lov', 'sym_lov', '缩水'))
    for i, s in order[:10]:
        print('%-22s %9.4f %9.4f %9.4f %9.4f %+9.4f' % (
            NAMES[i], iou['base_ce'][i], iou['sym_ce'][i],
            iou['base_lovasz'][i], iou['sym_lovasz'][i], s))

    print('\n=== 四模型整体 ===')
    for tag in ('base_ce','sym_ce','base_lovasz','sym_lovasz'):
        v = iou[tag][NB:]; v = v[~np.isnan(v)]
        print('  %-12s novel mIoU %.4f' % (tag, v.mean()))

if __name__ == '__main__':
    main()
