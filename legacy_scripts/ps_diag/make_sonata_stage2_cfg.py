# -*- coding: utf-8 -*-
"""派生 Stage-2 的 sonata_small 配置（A: enc 冻结 / B: enc 解冻）。

相对 semseg-pt-v3m1-0-gfsregistrain_k1_sym_lovasz.py：
  1. in_channels 6 -> 9
  2. + embedding_type="linear"、norm_all_ln=True
  3. feat_keys ("color","normal") -> ("coord","color","normal")  6 处
  4. A 版加 freeze_enc=True；B 版保持 False（全解冻，block 由 param_dicts 走 0.0006）
"""
import io

SRC = ("/data01/jh/code/GFS-VL/configs/scannet200/"
       "semseg-pt-v3m1-0-gfsregistrain_k1_sym_lovasz.py")
BASE = SRC[:-3]

src = io.open(SRC, encoding="utf-8").read()

old = "        type=\"PT-v3m1\",\n        in_channels=6,\n"
new = ("        type=\"PT-v3m1\",\n"
       "        in_channels=9,\n"
       "        embedding_type=\"linear\",\n"
       "        norm_all_ln=True,\n")
assert src.count(old) == 1, "backbone 锚点 %d" % src.count(old)
src = src.replace(old, new)

n = src.count("feat_keys=(\"color\", \"normal\"),")
assert n >= 6, "feat_keys 出现 %d 次，预期 >=6" % n
src = src.replace("feat_keys=(\"color\", \"normal\"),",
                  "feat_keys=(\"coord\", \"color\", \"normal\"),")

# A: enc 冻结
old_a = "    cross_dims=(64, 128),\n"
assert src.count(old_a) == 1
a = src.replace(old_a, "    cross_dims=(64, 128),\n    freeze_enc=True,  # Stage-2A：encoder 冻结\n")
io.open(BASE + "_sonata_small_a.py", "w", encoding="utf-8").write(a)

# B: enc 解冻
b = src.replace(old_a, "    cross_dims=(64, 128),\n    freeze_enc=False,  # Stage-2B：encoder 全解冻\n")
io.open(BASE + "_sonata_small_b.py", "w", encoding="utf-8").write(b)

print("已写:\n  %s_sonata_small_a.py\n  %s_sonata_small_b.py" % (BASE, BASE))
