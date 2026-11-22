'''第 3 批 CVD7 的选择：sym_xl 是否优于 xl。'''
import os
import re
import sys

import numpy as np

D = '/data01/jh/afford_diag/xl_runs'
NB, NC = 12, 57


def parse(fn):
    p = os.path.join(D, fn + '.log')
    if not os.path.exists(p):
        return None
    d = {}
    for line in open(p, errors='ignore'):
        m = re.search(r'Class_(\d+) - (.+?) Final Result: iou ([\d.]+)', line)
        if m:
            d[int(m.group(1))] = (m.group(2), float(m.group(3)))
    if len(d) != NC:
        return None
    v = np.array([d[i][1] for i in range(NC)])
    names = [d[i][0] for i in range(NC)]
    b, n = v[:NB].mean(), v[NB:].mean()
    return dict(v=v, names=names, HM=2 * b * n / (b + n))


s = [parse('sym_xl_s%d' % x) for x in (42, 7)]
o = [parse('sc200_k1_XLu_s42_resume'), parse('xl_s7_eval')]
if any(x is None for x in s) or any(x is None for x in o):
    print('xl')
    sys.exit(0)
shm = float(np.mean([x['HM'] for x in s]))
sb = float(np.mean([x['v'][x['names'].index('bookshelf')] for x in s]))
ohm = float(np.mean([x['HM'] for x in o]))
ob = float(np.mean([x['v'][x['names'].index('bookshelf')] for x in o]))
print('sym HM=%.4f bookshelf=%.4f | xl HM=%.4f bookshelf=%.4f'
      % (shm, sb, ohm, ob), file=sys.stderr)
print('sym' if (shm > ohm and sb > ob) else 'xl')
