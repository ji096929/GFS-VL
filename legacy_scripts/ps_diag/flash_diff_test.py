# -*- coding: utf-8 -*-
"""对比 enable_flash=True/False 的数值差异（同权重、同数据、同 patch 配置）。

关键：PTv3 在 enable_flash=False 时会把 patch_size 动态设为
      min(batch 内最小点数, patch_size_max)，而 True 时固定 patch_size_max。
      如果 batch 内所有样本点数都 >= patch_size_max，两者 patch_size 相同，
      数值应只差浮点累积顺序（1e-3 级）；否则是结构性差异。
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/data01/jh/code/GFS-VL")

from pointcept.datasets import build_dataset, collate_fn  # noqa: E402
from pointcept.models import build_model  # noqa: E402
from pointcept.utils.config import Config  # noqa: E402

CFG = "/data01/jh/code/GFS-VL/configs/scannet200/semseg-pt-v3m1-0-gfspretrain.py"
CKPT = "/data01/jh/code/GFS-VL/exp/scannet200/ptv3_bk/model/model_best.pth"


def main():
    torch.manual_seed(0)
    cfg = Config.fromfile(CFG)

    ds = build_dataset(cfg.data.val)
    d = ds[0]
    inp = collate_fn([d])
    for k in inp:
        if isinstance(inp[k], torch.Tensor):
            inp[k] = inp[k].cuda()
    n = int(inp["offset"][-1])
    print("场景点数 %d  feat %s" % (n, tuple(inp["feat"].shape)))

    outs = {}
    ps = {}
    for flash in (False, True):
        m = build_model(dict(cfg.model.backbone, enable_flash=flash)).cuda().eval()
        sd = torch.load(CKPT, map_location="cpu")["state_dict"]
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        bk = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        miss, unexp = m.load_state_dict(bk, strict=False)
        print("flash=%-5s 加载: missing %d unexpected %d" % (flash, len(miss), len(unexp)))

        # 记录实际 patch_size
        def hook(mod, args):
            ps[flash] = mod.patch_size
        m.enc.enc4.block0.attn.register_forward_pre_hook(hook)

        with torch.no_grad():
            out = m(inp)
        outs[flash] = out.feat.detach().float().clone()
        del m
        torch.cuda.empty_cache()

    a, b = outs[False], outs[True]
    print()
    print("patch_size:  False -> %s   True -> %s"
          % (ps.get(False), ps.get(True)))
    print("输出形状:    False %s   True %s" % (tuple(a.shape), tuple(b.shape)))
    if a.shape == b.shape:
        d_ = (a - b).abs()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=1).mean()
        print("max_err %.3e   mean_err %.3e   cosine %.8f"
              % (d_.max().item(), d_.mean().item(), cos.item()))
    else:
        print("形状不同，无法逐位比对")


if __name__ == "__main__":
    main()
