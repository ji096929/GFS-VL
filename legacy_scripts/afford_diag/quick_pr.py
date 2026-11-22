# -*- coding: utf-8 -*-
import sys, torch, numpy as np
p = sys.argv[1]
d = torch.load(p, map_location='cpu', weights_only=False)
I = d['intersection'].astype(np.float64)
U = d['union'].astype(np.float64)
T = d['target'].astype(np.float64)
P = I / (U - T + I + 1e-10)
R = I / (T + 1e-10)
IoU = I / (U + 1e-10)
names = {3: 'bookshelf', 16: 'shelf', 28: 'book', 7: 'cabinet',
         11: 'wall', 9: 'chair', 44: 'stool', 34: 'office chair',
         49: 'armchair', 5: 'table'}
print('%-14s %9s %9s %9s %7s %7s %7s' % ('class', 'TP', 'FP', 'FN', 'P', 'R', 'IoU'))
print('-' * 70)
for i, n in sorted(names.items()):
    print('%-14s %9.0f %9.0f %9.0f %7.3f %7.3f %7.3f'
          % (n, I[i], U[i] - T[i], T[i] - I[i], P[i], R[i], IoU[i]))
