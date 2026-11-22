d = '/data01/jh/code/GFS-VL/configs/scannet200/'
key = '    base_cross=True,'
pd = 'param_dicts = [dict(keyword=\"backbone\", lr=0.0001)]'

for src, dst in [('semseg-pt-v3m1-0-gfsregistrain_k1_sym.py',
                  'semseg-pt-v3m1-0-gfsregistrain_k1_lcross01.py'),
                 ('semseg-pt-v3m1-0-gfsregistrain_k5_sym_102400.py',
                  'semseg-pt-v3m1-0-gfsregistrain_k5_sym_lcross01_102400.py')]:
    s = open(d + src, encoding='utf-8').read()
    assert s.count(key) == 1, src
    s = s.replace(key, key + '\n    lam_cross=0.1,')
    assert s.count(pd) == 1, src
    s = s.replace(pd, pd + '\nlam_cross = 0.1')
    open(d + dst, 'w', encoding='utf-8').write(s)
    print('written', dst)
