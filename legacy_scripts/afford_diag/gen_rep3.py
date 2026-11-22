# -*- coding: utf-8 -*-
d = '/data01/jh/code/GFS-VL/configs/scannet200/'
s = open(d + 'semseg-pt-v3m1-0-gfsregistrain_k1_sym.py', encoding='utf-8').read()
a = '    base_cross=True,'
assert s.count(a) == 1
s = s.replace(a, '    base_cross=True,\n    rep_dec0=True,')
open(d + 'semseg-pt-v3m1-0-gfsregistrain_k1_rep3.py', 'w', encoding='utf-8').write(s)
print('written k1_rep3.py')
