d = '/data01/jh/code/GFS-VL/configs/scannet200/'
for src, dst in [('semseg-pt-v3m1-0-gfsregistrain_k5.py',
                  'semseg-pt-v3m1-0-gfsregistrain_k5_102400.py'),
                 ('semseg-pt-v3m1-0-gfsregistrain_k5_sym.py',
                  'semseg-pt-v3m1-0-gfsregistrain_k5_sym_102400.py')]:
    s = open(d + src, encoding='utf-8').read()
    assert s.count('point_max=65536') == 1, src
    s = s.replace('point_max=65536', 'point_max=102400')
    open(d + dst, 'w', encoding='utf-8').write(s)
    print('written', dst)
