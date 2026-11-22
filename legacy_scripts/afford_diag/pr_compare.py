# -*- coding: utf-8 -*-
import torch, numpy as np

def load(p):
    d = torch.load(p, map_location='cpu', weights_only=False)
    I = d['intersection'].astype(np.float64)
    U = d['union'].astype(np.float64)
    T = d['target'].astype(np.float64)
    return dict(I=I, FP=U - T, FN=T - I,
                P=I / (U - T + I + 1e-10),
                R=I / (T + 1e-10),
                IoU=I / (U + 1e-10))

b = load('/data01/jh/afford_diag/pr_base_k5.pth')
s = load('/data01/jh/afford_diag/pr_sym_k5.pth')

names = {3: 'bookshelf', 16: 'shelf', 28: 'book', 7: 'cabinet',
         11: 'wall', 9: 'chair', 44: 'stool', 34: 'office chair',
         49: 'armchair', 5: 'table', 0: 'refrigerator'}

print('%-14s %13s %13s %13s %13s' % ('class', 'base P/R', 'sym P/R', 'dP', 'dR'))
print('-' * 72)
for i, n in sorted(names.items()):
    print('%-14s %6.3f/%6.3f %6.3f/%6.3f %+7.3f %+7.3f   IoU %.3f->%.3f'
          % (n, b['P'][i], b['R'][i], s['P'][i], s['R'][i],
             s['P'][i] - b['P'][i], s['R'][i] - b['R'][i],
             b['IoU'][i], s['IoU'][i]))

print()
print('%-14s %11s %11s %11s' % ('class', 'base FP', 'sym FP', 'dFP'))
print('-' * 50)
for i, n in sorted(names.items()):
    print('%-14s %11.0f %11.0f %+11.0f'
          % (n, b['FP'][i], s['FP'][i], s['FP'][i] - b['FP'][i]))

print()
print('%-14s %11s %11s %11s' % ('class', 'base FN', 'sym FN', 'dFN'))
print('-' * 50)
for i, n in sorted(names.items()):
    print('%-14s %11.0f %11.0f %+11.0f'
          % (n, b['FN'][i], s['FN'][i], s['FN'][i] - b['FN'][i]))
