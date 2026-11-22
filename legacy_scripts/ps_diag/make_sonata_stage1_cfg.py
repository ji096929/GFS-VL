# -*- coding: utf-8 -*-
"""从 scratch 版 Stage-1 config 派生 sonata_small 版。

改动（6 处）：
  1. in_channels 6 -> 9
  2. 加 embedding_type="linear"（对齐 sonata 的 Linear(9->32)）
  3. 加 norm_all_ln=True（sonata 全 LayerNorm，state_dict 无 running_mean）
  4. 加 freeze_enc=True（Stage-1a：冻结预训练 encoder，只训 decoder + Base 头）
  5. feat_keys ("color","normal") -> ("coord","color","normal") 共 3 处
  6. epoch 800 -> 40；num_worker 24 -> 6
"""
import io

SRC = "/data01/jh/code/GFS-VL/configs/scannet200/semseg-pt-v3m1-0-gfspretrain.py"
DST = "/data01/jh/code/GFS-VL/configs/scannet200/semseg-pt-v3m1-0-gfspretrain_sonata_small.py"

src = io.open(SRC, encoding="utf-8").read()

# 1-3: backbone 参数
old = "        type=\"PT-v3m1\",\n        in_channels=6,\n"
new = ("        type=\"PT-v3m1\",\n"
       "        in_channels=9,\n"
       "        embedding_type=\"linear\",\n"
       "        norm_all_ln=True,\n")
assert src.count(old) == 1, "backbone 锚点 %d" % src.count(old)
src = src.replace(old, new)

# 4: freeze_enc 属于 DefaultSegmentorV2，不是 backbone
old = "    backbone_out_channels=64,\n    backbone=dict("
new = "    backbone_out_channels=64,\n    freeze_enc=True,  # Stage-1a：冻结 sonata_small 预训练 encoder\n    backbone=dict("
assert src.count(old) == 1, "freeze_enc 锚点 %d" % src.count(old)
src = src.replace(old, new)

# 5: feat_keys 三处
n = src.count("feat_keys=(\"color\", \"normal\"),")
assert n == 3, "feat_keys 出现 %d 次，预期 3" % n
src = src.replace("feat_keys=(\"color\", \"normal\"),",
                  "feat_keys=(\"coord\", \"color\", \"normal\"),")

# 6: epoch / num_worker
assert src.count("epoch = 800") == 1
src = src.replace("epoch = 800", "epoch = 40")
assert src.count("num_worker = 24") == 1
src = src.replace("num_worker = 24", "num_worker = 6  # 历史教训：>15 会 timeout")

# 训练增广里的 RandomScale/RandomFlip 对 9 维 feat 无影响（feat 由 Collect 拼）
io.open(DST, "w", encoding="utf-8").write(src)
print("已写 %s" % DST)
