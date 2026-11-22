# -*- coding: utf-8 -*-
p = '/data01/jh/code/GFS-VL/pointcept/models/default.py'
s = open(p, encoding='utf-8').read()

# 1) 构造参数
old1 = '''        depth_gate=False,'''
new1 = '''        depth_gate=False,
        detach_cross=False,'''
assert s.count(old1) == 1, 'anchor1=%d' % s.count(old1)
s = s.replace(old1, new1)

# 2) 属性
old2 = '''        self.depth_gate = bool(depth_gate)'''
new2 = '''        self.depth_gate = bool(depth_gate)
        self.detach_cross = bool(detach_cross)'''
assert s.count(old2) == 1, 'anchor2=%d' % s.count(old2)
s = s.replace(old2, new2)

# 3) _cross_projs 里按开关切断 dec1/dec2 的反传
old3 = '''            h = self.cross_norms[str(s)](self._to_full(s).float())'''
new3 = '''            _h = self._to_full(s).float()
            if self.detach_cross:
                _h = _h.detach()
            h = self.cross_norms[str(s)](_h)'''
assert s.count(old3) == 1, 'anchor3=%d' % s.count(old3)
s = s.replace(old3, new3)

open(p, 'w', encoding='utf-8').write(s)
print('detach_cross added')
