# -*- coding: utf-8 -*-
"""给 DefaultSegmentorV2 加 freeze_enc：只冻结 encoder，decoder 与读出头照常训练。

Stage-1a 用：sonata_small 的预训练 encoder 冻结，只训新初始化的 decoder + Base 头。
"""
import io
import os
import shutil

P = "/data01/jh/code/GFS-VL/pointcept/models/default.py"
BAK = P + ".bak_freezeenc"

src = io.open(P, encoding="utf-8").read()
if "freeze_enc" in src:
    print("已存在 freeze_enc，跳过")
    raise SystemExit(0)

old_sig = "        freeze_backbone=False,\n    ):"
new_sig = "        freeze_backbone=False,\n        freeze_enc=False,\n    ):"
assert src.count(old_sig) == 1, "构造参数锚点不唯一: %d" % src.count(old_sig)
src = src.replace(old_sig, new_sig)

old_body = (
    "        self.freeze_backbone = bool(freeze_backbone)\n"
    "        if self.freeze_backbone:\n"
    "            for _p in self.backbone.parameters():\n"
    "                _p.requires_grad = False\n"
)
assert src.count(old_body) == 1, "冻结逻辑锚点不唯一: %d" % src.count(old_body)
new_body = old_body + (
    "\n"
    "        # freeze_enc：只冻结 encoder（embedding + enc），decoder 与读出头照常训练。\n"
    "        # 用于 sonata_small 接入的 Stage-1a（冻结预训练编码器，只训新 decoder）。\n"
    "        self.freeze_enc = bool(freeze_enc)\n"
    "        if self.freeze_enc and not self.freeze_backbone:\n"
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
src = src.replace(old_body, new_body)

if not os.path.exists(BAK):
    shutil.copy2(P, BAK)
io.open(P, "w", encoding="utf-8").write(src)
print("已改 %s（备份 %s）" % (P, BAK))
