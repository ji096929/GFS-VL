# -*- coding: utf-8 -*-
p = '/data01/jh/code/GFS-VL/pointcept/engines/train.py'
s = open(p, encoding='utf-8').read()

old = '''            similarities = torch.cosine_similarity(
                feat_neg_one[:, None, :],
                updated_prototypes[None, :, :],
                dim=-1,
            )
            max_similarity, max_class_idx = similarities.max(dim=1)'''

new = '''            # 分块计算：大场景（原始点数）整场景都是 -1，M 极大，
            # (M,45,512) 广播会撑爆显存。分块不改变结果。
            _chunk = 16384
            _ms, _mc = [], []
            for _s in range(0, feat_neg_one.shape[0], _chunk):
                _f = feat_neg_one[_s:_s + _chunk]
                _sim = torch.cosine_similarity(
                    _f[:, None, :],
                    updated_prototypes[None, :, :],
                    dim=-1,
                )
                _a, _b = _sim.max(dim=1)
                _ms.append(_a)
                _mc.append(_b)
            if _ms:
                max_similarity = torch.cat(_ms)
                max_class_idx = torch.cat(_mc)
            else:
                max_similarity = torch.empty(0, device=feat_neg_one.device)
                max_class_idx = torch.empty(
                    0, dtype=torch.long, device=feat_neg_one.device)'''

assert s.count(old) == 1, 'anchor=%d' % s.count(old)
s = s.replace(old, new)
open(p, 'w', encoding='utf-8').write(s)
print('chunked cosine_similarity')
