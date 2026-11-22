# -*- coding: utf-8 -*-
"""为 ScanNet++ 派生 sym_lovasz（跨层残差双头）配置。

从 k1/k5 的 base_lovasz 版派生，只加 4 个 RegisTrainSegmentor 参数：
  base_cross=True, cross_layers=(1,2), cross_dims=(64,128), cross_dim=64
dec_channels=(64,64,128,256) -> dec1=64, dec2=128，与 cross_dims 对应。
"""
import io

BASE = "/data01/jh/code/GFS-VL/configs/scannetpp/semseg-pt-v3m1-0-gfsregistrain"

for tag in ("k1", "k5"):
    src_path = "%s_%s.py" % (BASE, tag)
    dst_path = "%s_%s_sym.py" % (BASE, tag)
    src = io.open(src_path, encoding="utf-8").read()

    anchor = '    type="RegisTrainSegmentor",\n'
    assert src.count(anchor) == 1, "%s model 锚点 %d" % (tag, src.count(anchor))
    add = (
        anchor
        + "    base_cross=True,\n"
        + "    cross_layers=(1, 2),\n"
        + "    cross_dims=(64, 128),\n"
        + "    cross_dim=64,\n"
    )
    src = src.replace(anchor, add)
    io.open(dst_path, "w", encoding="utf-8").write(src)
    print("已写 %s" % dst_path)
