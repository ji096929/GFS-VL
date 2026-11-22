d = '/data01/jh/code/GFS-VL/configs/scannet200/'
s = open(d + 'semseg-pt-v3m1-0-gfsregistrain_k1.py', encoding='utf-8').read()
a = '    num_novel_classes=len(CLASS_LABELS_BASE_NOVEL) - len(CLASS_LABELS_BASE),'
assert s.count(a) == 1
s = s.replace(a, a + '\n    ln_only=True,')
open(d + 'semseg-pt-v3m1-0-gfsregistrain_k1_lnonly.py', 'w', encoding='utf-8').write(s)
print('written k1_lnonly.py')
