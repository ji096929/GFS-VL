p = '/data01/jh/afford_diag/xl_erank.py'
s = open(p, encoding='utf-8').read()
o1 = 'CFG_XL = ROOT + "/configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_xl.py"'
n1 = o1 + '\nCFG_REP3 = ROOT + "/configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_rep3.py"'
assert s.count(o1) == 1
s = s.replace(o1, n1)
o2 = '    "test_online": (CFG_BASE, ROOT + "/exp/scannet200/sc200_k1_ONLINE_s42/model/regis1_model_last.pth"),'
n2 = o2 + '\n    "rep3_s42": (CFG_REP3, ROOT + "/exp/scannet200/rep3_s42/model/regis1_model_last.pth"),'
assert s.count(o2) == 1
s = s.replace(o2, n2)
open(p, 'w', encoding='utf-8').write(s)
print('cfg added')
