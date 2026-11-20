# -*- coding: utf-8 -*-
"""从 val.pth 的逐场景 inter/union/target 计算每类 Precision / Recall / IoU。

定义（与 intersection_and_union 一致）：
  I = TP, target = TP + FN, union = TP + FP + FN
  P = I / (I + FP) = I / (union - target + I)
  R = I / (TP + FN) = I / target
"""
import sys, torch, numpy as np

RUNS = {
    'base_k5': '/data01/jh/code/GFS-VL/exp/scannet200/base_k5_s42/result/val.pth',
    'sym_k5':  '/data01/jh/code/GFS-VL/exp/scannet200/sym_xl_k5_s42/result/val.pth',
}
NAMES = {
    3: 'bookshelf', 16: 'shelf', 28: 'book', 7: 'cabinet', 11: 'wall',
    9: 'chair', 44: 'stool', 34: 'office chair', 49: 'armchair',
    5: 'table', 0: 'refrigerator', 1: 'desk', 4: 'bed', 8: 'door',
}

data = {}
for tag, path in RUNS.items():
    try:
        d = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print('[skip] %s: %s' % (tag, e))
        continue
    I = d['intersection'].astype(np.float64)
    U = d['union'].astype(np.float64)
    T = d['target'].astype(np.float64)
    data[tag] = {
        'P': I / (U - T + I + 1e-10),
        'R': I / (T + 1e-10),
        'IoU': I / (U + 1e-10),
        'TP': I, 'FP': U - T, 'FN': T - I,
    }

if len(data) < 2:
    print('need both runs'); sys.exit(1)

print('%-15s %18s %18s %18s' % ('class', 'base_k5 P / R', 'sym_k5 P / R', 'dP / dR'))
print('-' * 74)
for idx, name in sorted(NAMES.items()):
    row = []
    for tag in ('base_k5', 'sym_k5'):
        row.append('%.3f / %.3f' % (data[tag]['P'][idx], data[tag]['R'][idx]))
    dP = data['sym_k5']['P'][idx] - data['base_k5']['P'][idx]
    dR = data['sym_k5']['R'][idx] - data['base_k5']['R'][idx]
    print('%-15s %18s %18s %+8.3f / %+8.3f' % (name, row[0], row[1], dP, dR))

print()
print('%-15s %14s %14s' % ('class', 'base FP', 'sym FP'))
print('-' * 46)
for idx, name in sorted(NAMES.items()):
    print('%-15s %14.0f %14.0f' % (
        name, data['base_k5']['FP'][idx], data['sym_k5']['FP'][idx]))
