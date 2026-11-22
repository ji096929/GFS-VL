# -*- coding: utf-8 -*-
"""验证 hypothesis：enc1+ 的误差只是体素行排列不同（Z-order vs 字典序），
用 pooling_inverse 展开回原始点顺序后两边应该逐位一致。
"""
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
    mlp_ratio=4, qkv_bias=True, drop_path=0.3, shuffle_orders=False,
    pre_norm=True, enable_rpe=False, enable_flash=False,
    pdnorm_bn=False, pdnorm_ln=False,
)

TARGETS = ["enc.enc1", "enc.enc2", "enc.enc3", "enc.enc4"]


def resolve(mod, path):
    cur = mod
    for p in path.split("."):
        cur = getattr(cur, p)
    return cur


def main():
    torch.manual_seed(0)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    sm = sonata.load(CKPT, custom_config=dict(
        enc_patch_size=[1024] * 5, enable_flash=False,
        shuffle_orders=False)).cuda().eval()
    om = build_model(CFG).cuda().eval()
    om.load_state_dict(sd, strict=False)

    coord = np.load(os.path.join(SCENE, "coord.npy")).astype(np.float32)
    color = np.load(os.path.join(SCENE, "color.npy")).astype(np.float32)
    normal = np.load(os.path.join(SCENE, "normal.npy")).astype(np.float32)
    pt = sonata.transform.default()(dict(coord=coord, color=color, normal=normal))
    for k in list(pt.keys()):
        if isinstance(pt[k], torch.Tensor):
            pt[k] = pt[k].cuda()
    n0 = len(pt["coord"])

    grab = {"s": {}, "o": {}}

    def mk(store, name):
        def hook(mod, inp, out):
            store[name] = (out.feat.detach().float().clone(),
                           out.pooling_inverse.detach().clone()
                           if "pooling_inverse" in out.keys() else None)
        return hook

    for t in TARGETS:
        resolve(sm, t).register_forward_hook(mk(grab["s"], t))
        resolve(om, t).register_forward_hook(mk(grab["o"], t))

    with torch.no_grad():
        sm(copy.deepcopy(pt))
        om(copy.deepcopy(pt))

    print("%-12s %-18s %-18s %12s %12s"
          % ("layer", "voxels(s)", "voxels(o)", "raw_max", "expanded_max"))
    for t in TARGETS:
        fs, is_ = grab["s"][t]
        fo, io = grab["o"][t]
        raw = (fs - fo).abs().max().item() if fs.shape == fo.shape else float("nan")
        if is_ is None or io is None:
            print("%-12s  no pooling_inverse" % t)
            continue
        # 展开回输入这一层的父点数
        es = fs[is_]
        eo = fo[io]
        if es.shape != eo.shape:
            print("%-12s %-18s %-18s %12.3e  SHAPE %s vs %s"
                  % (t, tuple(fs.shape), tuple(fo.shape), raw,
                     tuple(es.shape), tuple(eo.shape)))
            continue
        d = (es - eo).abs()
        print("%-12s %-18s %-18s %12.3e %12.3e"
              % (t, tuple(fs.shape), tuple(fo.shape), raw, d.max().item()))


if __name__ == "__main__":
    main()
