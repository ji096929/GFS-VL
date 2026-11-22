# -*- coding: utf-8 -*-
d = '/data01/jh/code/GFS-VL/configs/scannet200/'
s = open(d + 'semseg-pt-v3m1-0-gfsregistrain_k5_lcross.py', encoding='utf-8').read()

o1 = 'optimizer = dict(type="Adam", lr=0.001)'
assert s.count(o1) == 1
s = s.replace(o1, 'optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.0001)')

o2 = '''scheduler = dict(
    type="MultiStepLR",
    milestones=[50, 100],
    gamma=0.5,
)'''
assert s.count(o2) == 1
s = s.replace(o2, '''scheduler = dict(
    type="CosineAnnealingLR",
    eta_min=1e-5,
)''')

o3 = 'dict(type="CheckpointSaver_MultiRegis", save_freq=None)'
assert s.count(o3) == 1
s = s.replace(o3, 'dict(type="CheckpointSaver_MultiRegis", save_freq=4)')

open(d + 'semseg-pt-v3m1-0-gfsregistrain_k5_lcross_cos.py', 'w', encoding='utf-8').write(s)
print('written k5_lcross_cos.py')
