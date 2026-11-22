# -*- coding: utf-8 -*-
"""把 sonata_small.pth 转成 GFS-VL 的 checkpoint 格式。

sonata 的 key:  embedding.* / enc.*
我们 DefaultSegmentorV2 期望: backbone.embedding.* / backbone.enc.* / seg_head.* / dec.*

missing 只剩 backbone.dec.* 与 seg_head.*，由 Stage-1 从头训。
"""
import torch

SRC = "/data01/jh/code/sonata/ckpt/sonata_small.pth"
DST = "/data01/jh/weights/sonata_small_ptv3.pth"

d = torch.load(SRC, map_location="cpu", weights_only=False)
sd = d["state_dict"]
new = {}
for k, v in sd.items():
    if k.startswith(("embedding.", "enc.")):
        new["backbone." + k] = v
    else:
        print("跳过非骨干键:", k)
d["state_dict"] = new
d["epoch"] = -1
d["iter"] = -1
torch.save(d, DST)
n = sum(v.numel() for v in new.values())
print("写出 %s  键 %d  params %.2fM" % (DST, len(new), n / 1e6))
