# -*- coding: utf-8 -*-
d = '/data01/jh/code/GFS-VL/configs/scannet200/'
src = open(d + 'semseg-pt-v3m1-0-gfsregistrain_k5_sym.py', encoding='utf-8').read()

# 模型里启用辅助损失
anchor = '    base_cross=True,'
assert src.count(anchor) == 1
s = src.replace(anchor, '    base_cross=True,\n    lam_cross=0.3,')

# 顶层也放一份，给 GFS_VL_Trainer 读
anchor2 = 'param_dicts = [dict(keyword=\"backbone\", lr=0.0001)]'
assert s.count(anchor2) == 1
s = s.replace(anchor2, anchor2 + '\nlam_cross = 0.3')

open(d + 'semseg-pt-v3m1-0-gfsregistrain_k5_lcross.py', 'w', encoding='utf-8').write(s)
print('written k5_lcross.py')
