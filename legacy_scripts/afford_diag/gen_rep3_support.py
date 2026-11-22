d = '/data01/jh/code/GFS-VL/configs/scannet200/'
s = open(d + 'semseg-pt-v3m1-0-gfsregistrain_k1_rep3.py', encoding='utf-8').read()
open(d + 'semseg-pt-v3m1-0-gfsregistrain_k1_rep3_support.py', 'w', encoding='utf-8').write(s)
print('written')
