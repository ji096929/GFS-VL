# -*- coding: utf-8 -*-
"""排查：eval 模式下我们的模型两次前向是否完全一致（drop_path / shuffle_orders / traceable）。"""
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/data01/jh/code/sonata")
sys.path.insert(0, "/data01/jh/code/GFS-VL")
import sonata  # noqa: E402
from pointcept.models import build_model  # noqa: E402

CKPT = "/data01/jh/code/sonata/ckpt/sonata_small.pth"
SCENE = "/data01/jh/data/scannet200_gfs/train/scene0000_00"

CFG = dict(
    type="PT-v3m1", in_channels=9, embedding_type="linear", norm_all_ln=True,
    order=["z", "z-trans", "hilbert", "hilbert-trans"], stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2), enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32), enc_patch_size=(1024,) * 5,
    dec_depths=(2, 2, 2, 2), dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16), dec_patch_size=(1024,) * 4,
    mlp_ratio=4, qkv_bias=True, drop_path=0.3, shuffle_orders=True,
    pre_norm=True, enable_rpe=False, enable_flash=False,
    pdnorm_bn=False, pdnorm_ln=False,
)


def resolve(mod, path):
    cur = mod
    for p in path.split("."):
        cur = getattr(cur, p)
    return cur


def run(model, pt, tag):
    out = {}

    def mk(name):
        def hook(m, i, o):
            out[name] = o.feat.detach().float().clone()
        return hook

    hs = []
    for t in ("enc.enc0", "enc.enc4"):
        h = resolve(model, t).register_forward_hook(mk(t))
        hs.append(h)
    with torch.no_grad():
        model(copy.deepcopy(pt))
    for h in hs:
        h.remove()
    return out


def main():
    torch.manual_seed(0)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    om = build_model(CFG).cuda().eval()
    om.load_state_dict(sd, strict=False)

    coord = np.load(os.path.join(SCENE, "coord.npy")).astype(np.float32)
    color = np.load(os.path.join(SCENE, "color.npy")).astype(np.float32)
    normal = np.load(os.path.join(SCENE, "normal.npy")).astype(np.float32)
    pt = sonata.transform.default()(dict(coord=coord, color=color, normal=normal))
    for k in list(pt.keys()):
        if isinstance(pt[k], torch.Tensor):
            pt[k] = pt[k].cuda()

    print("shuffle_orders attr:", getattr(om, "shuffle_orders", "n/a"))
    for t in ("enc.enc0", "enc.enc4"):
        m = resolve(om, t)
        print("  %-12s block0.drop_path p=%s" % (t, getattr(getattr(m, "block0", m), "drop_path", "n/a")))

    a = run(om, pt, "run1")
    b = run(om, pt, "run2")
    print()
    print("=== 同一模型两次前向（eval） ===")
    for k in a:
        d = (a[k] - b[k]).abs()
        print("  %-10s max %.3e  mean %.3e" % (k, d.max().item(), d.mean().item()))


if __name__ == "__main__":
    main()
