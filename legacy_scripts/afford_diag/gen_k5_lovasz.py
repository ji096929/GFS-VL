d = '/data01/jh/code/GFS-VL/configs/scannet200/'
key = 'param_dicts = [dict(keyword=\"backbone\", lr=0.0001)]'

for src, dst in [('semseg-pt-v3m1-0-gfsregistrain_k5_102400.py',
                  'semseg-pt-v3m1-0-gfsregistrain_k5_lovasz_102400.py'),
                 ('semseg-pt-v3m1-0-gfsregistrain_k5_sym_102400.py',
                  'semseg-pt-v3m1-0-gfsregistrain_k5_sym_lovasz_102400.py')]:
    s = open(d + src, encoding='utf-8').read()
    assert s.count(key) == 1, src
    s = s.replace(key, key + '\nuse_lovasz = True')
    open(d + dst, 'w', encoding='utf-8').write(s)
    print('written', dst)
