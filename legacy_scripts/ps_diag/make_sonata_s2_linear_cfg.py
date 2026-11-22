# -*- coding: utf-8 -*-
"""从实测跑出 0.4122 的 base_lovasz_s42 存档配置派生 sonata_small 版。

改动（与 Stage-1a 严格对齐）：
  1. in_channels 6 -> 9
  2. backbone 加 embedding_type='linear'、norm_all_ln=True
  3. feat_keys ('color','normal') -> ('coord','color','normal')  共 8 处
  4. model 加 freeze_enc（A=True 冻结 / B=False 全解冻）

保留：param_dicts=[keyword='backbone', lr=0.0001]、use_lovasz=True、
      RegisTrainSegmentor 裸线性头（无 cross_layers / base_cross）。
"""
import io

SRC = "/data01/jh/code/GFS-VL/exp/scannet200/base_lovasz_s42/config.py"
BASE = "/data01/jh/code/GFS-VL/configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_base_lovasz_sonata_small"

src = io.open(SRC, encoding="utf-8").read()

# 1-2: backbone 参数（注意 config dump 里是 8 空格缩进）
old = "        type='PT-v3m1',\n        in_channels=6,\n"
new = ("        type='PT-v3m1',\n"
       "        in_channels=9,\n"
       "        embedding_type='linear',\n"
       "        norm_all_ln=True,\n")
assert src.count(old) == 1, "backbone 锚点 %d" % src.count(old)
src = src.replace(old, new)

# 3: feat_keys
n = src.count("feat_keys=('color', 'normal')")
assert n == 8, "feat_keys 出现 %d 次，预期 8" % n
src = src.replace("feat_keys=('color', 'normal')",
                  "feat_keys=('coord', 'color', 'normal')")

# 4: freeze_enc
anchor = "    type='RegisTrainSegmentor',\n"
assert src.count(anchor) == 1, "model 锚点 %d" % src.count(anchor)
a = src.replace(anchor, anchor + "    freeze_enc=True,  # Stage-2A：encoder 冻结\n")
b = src.replace(anchor, anchor + "    freeze_enc=False,  # Stage-2B：encoder 全解冻\n")

io.open(BASE + "_a.py", "w", encoding="utf-8").write(a)
io.open(BASE + "_b.py", "w", encoding="utf-8").write(b)
print("已写:\n  %s_a.py\n  %s_b.py" % (BASE, BASE))
