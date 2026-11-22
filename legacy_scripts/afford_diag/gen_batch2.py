d = '/data01/jh/code/GFS-VL/configs/scannet200/'
base = open(d + 'semseg-pt-v3m1-0-gfsregistrain_k1.py', encoding='utf-8').read()
sym = open(d + 'semseg-pt-v3m1-0-gfsregistrain_k1_sym.py', encoding='utf-8').read()

def add(src, line):
    key = 'param_dicts = [dict(keyword=\"backbone\", lr=0.0001)]'
    assert src.count(key) == 1
    return src.replace(key, key + '\n' + line)

open(d + 'semseg-pt-v3m1-0-gfsregistrain_k1_lovasz.py', 'w').write(
    add(base, 'use_lovasz = True'))
open(d + 'semseg-pt-v3m1-0-gfsregistrain_k1_sym_lovasz.py', 'w').write(
    add(sym, 'use_lovasz = True'))
open(d + 'semseg-pt-v3m1-0-gfsregistrain_k1_la.py', 'w').write(
    add(base, 'use_la = True\nla_tau = 0.5'))
open(d + 'semseg-pt-v3m1-0-gfsregistrain_k1_sym_la.py', 'w').write(
    add(sym, 'use_la = True\nla_tau = 0.5'))
print('4 configs written')
