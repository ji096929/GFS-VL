# -*- coding: utf-8 -*-
"""让 shuffle_orders 只在训练时生效，eval 完全确定。

训练：保持随机（数据增强不变，与历史 bit-for-bit 一致）
评估：序列化顺序固定 → 同一 checkpoint 两次评估必须 0 误差
"""
import io
import shutil

P = "/data01/jh/code/GFS-VL/pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py"
BAK = P + ".bak_deteval"

src = io.open(P, encoding="utf-8").read()
if "shuffle_orders and self.training" in src:
    print("已修，跳过")
    raise SystemExit(0)

# 1) SerializedPooling 里的额外 randperm
old1 = "        if self.shuffle_orders:\n            perm = torch.randperm(code.shape[0])\n"
new1 = ("        # eval 时关闭随机序列化，否则同一权重两次评估会有 0.x 的抖动\n"
        "        if self.shuffle_orders and self.training:\n"
        "            perm = torch.randperm(code.shape[0])\n")
assert src.count(old1) == 1, "pooling 锚点 %d" % src.count(old1)
src = src.replace(old1, new1)

# 2) 顶层 serialization 的 shuffle
old2 = "            order=self.order, shuffle_orders=self.shuffle_orders\n"
new2 = ("            order=self.order,\n"
        "            shuffle_orders=self.shuffle_orders and self.training,\n")
assert src.count(old2) == 1, "serialization 锚点 %d" % src.count(old2)
src = src.replace(old2, new2)

shutil.copy2(P, BAK)
io.open(P, "w", encoding="utf-8").write(src)
print("已修 shuffle_orders（备份 %s）" % BAK)
