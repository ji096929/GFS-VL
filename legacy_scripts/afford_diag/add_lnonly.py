p = '/data01/jh/code/GFS-VL/pointcept/models/default.py'
s = open(p, encoding='utf-8').read()

o1 = '        rep_dec0=False,'
n1 = '        rep_dec0=False,\n        ln_only=False,'
assert s.count(o1) == 1
s = s.replace(o1, n1)

o2 = '        self.rep_dec0 = bool(rep_dec0)'
n2 = '        self.rep_dec0 = bool(rep_dec0)\n        self.ln_only = bool(ln_only)'
assert s.count(o2) == 1
s = s.replace(o2, n2)

o3 = '''        else:
            self.novel_seg_head = nn.Linear(
                backbone_out_channels, num_novel_classes, bias=False
            )'''
n3 = '''        elif self.ln_only:
            self.novel_seg_head = nn.Sequential(
                nn.LayerNorm(backbone_out_channels),
                nn.Linear(backbone_out_channels, num_novel_classes, bias=False),
            )
        else:
            self.novel_seg_head = nn.Linear(
                backbone_out_channels, num_novel_classes, bias=False
            )'''
assert s.count(o3) == 1
s = s.replace(o3, n3)

open(p, 'w', encoding='utf-8').write(s)
print('ln_only added')
