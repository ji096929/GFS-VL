# -*- coding: utf-8 -*-
"""把 freeze_enc 补到 DefaultSegmentorV2（Stage-1 用的是这个类）。"""
import io
import shutil

P = "/data01/jh/code/GFS-VL/pointcept/models/default.py"
BAK = P + ".bak_dsv2enc"

src = io.open(P, encoding="utf-8").read()
if "class DefaultSegmentorV2" not in src:
    raise SystemExit("找不到 DefaultSegmentorV2")

old = (
    "class DefaultSegmentorV2(nn.Module):\n"
    "    def __init__(\n"
    "        self,\n"
    "        num_classes,\n"
    "        backbone_out_channels,\n"
    "        backbone=None,\n"
    "        criteria=None,\n"
    "    ):\n"
    "        super().__init__()\n"
    "        self.seg_head = (\n"
    "            nn.Linear(backbone_out_channels, num_classes)\n"
    "            if num_classes > 0\n"
    "            else nn.Identity()\n"
    "        )\n"
    "        self.backbone = build_model(backbone)\n"
    "        self.criteria = build_criteria(criteria)\n"
)
new = (
    "class DefaultSegmentorV2(nn.Module):\n"
    "    def __init__(\n"
    "        self,\n"
    "        num_classes,\n"
    "        backbone_out_channels,\n"
    "        backbone=None,\n"
    "        criteria=None,\n"
    "        freeze_enc=False,\n"
    "    ):\n"
    "        super().__init__()\n"
    "        self.seg_head = (\n"
    "            nn.Linear(backbone_out_channels, num_classes)\n"
    "            if num_classes > 0\n"
    "            else nn.Identity()\n"
    "        )\n"
    "        self.backbone = build_model(backbone)\n"
    "        self.criteria = build_criteria(criteria)\n"
    "\n"
    "        # freeze_enc：只冻结 encoder（embedding + enc），decoder 与读出头照常训练。\n"
    "        # 用于 sonata_small 接入的 Stage-1a。\n"
    "        self.freeze_enc = bool(freeze_enc)\n"
    "        if self.freeze_enc:\n"
    "            _frozen, _train = 0, 0\n"
    "            for _name, _p in self.backbone.named_parameters():\n"
    "                if _name.startswith((\"embedding.\", \"enc.\")):\n"
    "                    _p.requires_grad = False\n"
    "                    _frozen += _p.numel()\n"
    "                else:\n"
    "                    _train += _p.numel()\n"
    "            print(\"freeze_enc: 冻结 %.2fM，可训练 %.2fM\"\n"
    "                  % (_frozen / 1e6, _train / 1e6))\n"
)
assert src.count(old) == 1, "DefaultSegmentorV2 锚点 %d" % src.count(old)
src = src.replace(old, new)

if not __import__("os").path.exists(BAK):
    shutil.copy2(P, BAK)
io.open(P, "w", encoding="utf-8").write(src)
print("已给 DefaultSegmentorV2 加 freeze_enc（备份 %s）" % BAK)
