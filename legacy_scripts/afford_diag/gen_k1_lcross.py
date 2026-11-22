# -*- coding: utf-8 -*-
d = '/data01/jh/code/GFS-VL/configs/scannet200/'
s = open(d + 'semseg-pt-v3m1-0-gfsregistrain_k1_sym.py', encoding='utf-8').read()

anchor = '    base_cross=True,'
assert s.count(anchor) == 1
s = s.replace(anchor, '    base_cross=True,\n    lam_cross=0.3,')

anchor2 = 'param_dicts = [dict(keyword=\"backbone\", lr=0.0001)]'
assert s.count(anchor2) == 1
s = s.replace(anchor2, anchor2 + '\nlam_cross = 0.3')

open(d + 'semseg-pt-v3m1-0-gfsregistrain_k1_lcross.py', 'w', encoding='utf-8').write(s)
print('written k1_lcross.py (L_cross + 原版 Adam)')
