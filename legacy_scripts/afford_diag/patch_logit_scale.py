# -*- coding: utf-8 -*-
p = '/data01/jh/code/GFS-VL/pointcept/engines/test.py'
s = open(p, encoding='utf-8').read()

old = '''                    pred_part = self.model(input_dict, **kwargs)[
                        \"seg_logits\"
                    ]  # (n, k)
                    pred_part = F.softmax(pred_part, -1)'''

new = '''                    pred_part = self.model(input_dict, **kwargs)[
                        \"seg_logits\"
                    ]  # (n, k)
                    # 推理期 logit 校准：可对 novel 段整体缩放，
                    # 用于验证 Base/Novel 分支的 logit 尺度是否失衡。
                    _ls = float(getattr(self.cfg, \"novel_logit_scale\", 1.0))
                    if _ls != 1.0:
                        _nb = int(self.cfg.data.num_bases)
                        pred_part = pred_part.clone()
                        pred_part[:, _nb:] = pred_part[:, _nb:] * _ls
                    pred_part = F.softmax(pred_part, -1)'''

assert s.count(old) == 1, 'anchor=%d' % s.count(old)
s = s.replace(old, new)
open(p, 'w', encoding='utf-8').write(s)
print('logit scale patched')
