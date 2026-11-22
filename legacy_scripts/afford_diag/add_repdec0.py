# -*- coding: utf-8 -*-
p = '/data01/jh/code/GFS-VL/pointcept/models/default.py'
s = open(p, encoding='utf-8').read()

# 1) 构造参数
o1 = '''        lam_cross=0.0,'''
n1 = '''        lam_cross=0.0,
        rep_dec0=False,'''
assert s.count(o1) == 1
s = s.replace(o1, n1)

# 2) 属性 + 模块
o2 = '''        self.lam_cross = float(lam_cross)'''
n2 = '''        self.lam_cross = float(lam_cross)
        self.rep_dec0 = bool(rep_dec0)'''
assert s.count(o2) == 1
s = s.replace(o2, n2)

# 3) 在 cross_layers 分支里，如果 rep_dec0 则用 dec0 复制两份做投影
o3 = '''            self.dec0_norm = nn.LayerNorm(backbone_out_channels)'''
n3 = '''            self.dec0_norm = nn.LayerNorm(backbone_out_channels)
            if self.rep_dec0:
                # dec0_rep3 对照：dec0 复制两份走独立投影，
                # 结构与 sym_xl 完全相同，只是特征源只有 dec0
                self.rep_norms = nn.ModuleList(
                    [nn.LayerNorm(backbone_out_channels)
                     for _ in self.cross_layers])
                self.rep_projs = nn.ModuleList(
                    [nn.Linear(backbone_out_channels, cross_dim)
                     for _ in self.cross_layers])
                self.rep_outs = nn.ModuleList(
                    [nn.LayerNorm(cross_dim) for _ in self.cross_layers])'''
assert s.count(o3) == 1
s = s.replace(o3, n3)

# 4) _cross_projs 里分支
o4 = '''    def _cross_projs(self, feat):
        ln0 = self.dec0_norm(feat.float())
        projs = []
        for s in self.cross_layers:'''
n4 = '''    def _cross_projs(self, feat):
        ln0 = self.dec0_norm(feat.float())
        projs = []
        if self.rep_dec0:
            for i in range(len(self.cross_layers)):
                _h = feat.float()
                if self.detach_cross:
                    _h = _h.detach()
                h = self.rep_norms[i](_h)
                h = torch.nn.functional.gelu(self.rep_projs[i](h))
                projs.append(self.rep_outs[i](h))
            return ln0, projs
        for s in self.cross_layers:'''
assert s.count(o4) == 1
s = s.replace(o4, n4)

open(p, 'w', encoding='utf-8').write(s)
print('rep_dec0 added')
