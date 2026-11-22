# -*- coding: utf-8 -*-
p = '/data01/jh/code/GFS-VL/pointcept/models/default.py'
s = open(p, encoding='utf-8').read()

# 1) 构造参数
old1 = '''        detach_cross=False,'''
new1 = '''        detach_cross=False,
        lam_cross=0.0,'''
assert s.count(old1) == 1, 'a1=%d' % s.count(old1)
s = s.replace(old1, new1)

# 2) 属性
old2 = '''        self.detach_cross = bool(detach_cross)'''
new2 = '''        self.detach_cross = bool(detach_cross)
        self.lam_cross = float(lam_cross)'''
assert s.count(old2) == 1, 'a2=%d' % s.count(old2)
s = s.replace(old2, new2)

# 3) base_cross 分支里产出 cross_logits（仅训练期）
old3 = '''                if self.base_cross:
                    base_seg_logits = base_seg_logits + self.base_cross_head(
                        torch.cat(projs, dim=1).to(feat.dtype)
                    )'''
new3 = '''                if self.base_cross:
                    _cf = torch.cat(projs, dim=1).to(feat.dtype)
                    _bc = self.base_cross_head(_cf)
                    base_seg_logits = base_seg_logits + _bc
                    # L_cross: 在纯跨层子空间做辅助监督，消除 base_cross_head
                    # 被 seg_head(dec0) 遮住梯度的问题（推理时不用）。
                    if self.training and self.lam_cross > 0:
                        _cn = torch.nn.functional.linear(
                            _cf, self.novel_seg_head.weight[:, 64:]
                        )
                        cross_logits = torch.cat([_bc, _cn], dim=1)'''
assert s.count(old3) == 1, 'a3=%d' % s.count(old3)
s = s.replace(old3, new3)

# 4) 初始化 cross_logits 并在训练分支返回
old4 = '''            base_seg_logits = self.seg_head(feat)
            decorr_loss = None'''
new4 = '''            base_seg_logits = self.seg_head(feat)
            decorr_loss = None
            cross_logits = None'''
assert s.count(old4) == 1, 'a4=%d' % s.count(old4)
s = s.replace(old4, new4)

old5 = '''            out = dict(seg_logits=seg_logits)
            if decorr_loss is not None:
                out["decorr_loss"] = decorr_loss
            return out'''
new5 = '''            out = dict(seg_logits=seg_logits)
            if decorr_loss is not None:
                out["decorr_loss"] = decorr_loss
            if cross_logits is not None:
                out["cross_logits"] = cross_logits
            return out'''
assert s.count(old5) == 1, 'a5=%d' % s.count(old5)
s = s.replace(old5, new5)

open(p, 'w', encoding='utf-8').write(s)
print('default.py patched')

# ---------- train.py ----------
p2 = '/data01/jh/code/GFS-VL/pointcept/engines/train.py'
t = open(p2, encoding='utf-8').read()

old6 = '''            decorr_loss = output_dict.pop("decorr_loss", None)'''
new6 = '''            decorr_loss = output_dict.pop("decorr_loss", None)
            cross_logits = output_dict.pop("cross_logits", None)'''
assert t.count(old6) == 1, 'a6=%d' % t.count(old6)
t = t.replace(old6, new6)

old7 = '''            if decorr_loss is not None:
                loss = loss + decorr_loss'''
new7 = '''            if decorr_loss is not None:
                loss = loss + decorr_loss
            if cross_logits is not None:
                _lam = float(getattr(self.cfg, "lam_cross", 0.3))
                loss = loss + _lam * criteria(cross_logits, new_target)'''
assert t.count(old7) == 1, 'a7=%d' % t.count(old7)
t = t.replace(old7, new7)

open(p2, 'w', encoding='utf-8').write(t)
print('train.py patched')
