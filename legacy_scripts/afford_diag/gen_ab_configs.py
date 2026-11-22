# -*- coding: utf-8 -*-
d = '/data01/jh/code/GFS-VL/configs/scannet200/'
sym = open(d + 'semseg-pt-v3m1-0-gfsregistrain_k5_sym.py', encoding='utf-8').read()
bas = open(d + 'semseg-pt-v3m1-0-gfsregistrain_k5.py', encoding='utf-8').read()

# ---------- 实验 A: detach + 旧头降 LR ----------
a = sym
anchor = '    base_cross=True,'
assert a.count(anchor) == 1, 'A anchor1=%d' % a.count(anchor)
a = a.replace(anchor, '    base_cross=True,\n    detach_cross=True,')

old_pd = 'param_dicts = [dict(keyword="backbone", lr=0.0001)]'
new_pd = ('param_dicts = [\n'
          '    dict(keyword="novel_seg_head", lr=0.001),\n'
          '    dict(keyword="backbone", lr=0.0001),\n'
          '    dict(keyword="seg_head", lr=0.0001),\n'
          ']')
assert a.count(old_pd) == 1, 'A anchor2=%d' % a.count(old_pd)
a = a.replace(old_pd, new_pd)
open(d + 'semseg-pt-v3m1-0-gfsregistrain_k5_detach.py', 'w', encoding='utf-8').write(a)
print('written k5_detach.py')

# ---------- 实验 B: cosine + AdamW + save_freq ----------
def cosify(s, tag):
    o1 = 'optimizer = dict(type="Adam", lr=0.001)'
    assert s.count(o1) == 1, tag + ' opt=%d' % s.count(o1)
    s = s.replace(o1, 'optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.0001)')

    o2 = ('scheduler = dict(\n'
          '    type="MultiStepLR",\n'
          '    milestones=[50, 100],\n'
          '    gamma=0.5,\n'
          ')')
    assert s.count(o2) == 1, tag + ' sched=%d' % s.count(o2)
    s = s.replace(o2, 'scheduler = dict(\n    type="CosineAnnealingLR",\n    eta_min=1e-5,\n)')

    o3 = 'dict(type="CheckpointSaver_MultiRegis", save_freq=None)'
    assert s.count(o3) == 1, tag + ' savefreq=%d' % s.count(o3)
    s = s.replace(o3, 'dict(type="CheckpointSaver_MultiRegis", save_freq=4)')
    return s

open(d + 'semseg-pt-v3m1-0-gfsregistrain_k5_cos_sym.py', 'w', encoding='utf-8').write(cosify(sym, 'sym'))
open(d + 'semseg-pt-v3m1-0-gfsregistrain_k5_cos_base.py', 'w', encoding='utf-8').write(cosify(bas, 'base'))
print('written k5_cos_sym.py / k5_cos_base.py')
