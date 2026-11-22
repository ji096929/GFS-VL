# -*- coding: utf-8 -*-
"""Sonata-Small -> PTv3 权重注入 + 1 场景特征一致性验收。

gate: 同一份 point dict 喂两个模型，enc4 输出 max_err < 1e-4 才允许开训。

为什么要这个 gate：
  sonata_small 的 state_dict 里 running_mean 键数量为 0（全套 LayerNorm），
  而我们原版 PTv3 的 embedding.stem.norm 和 enc.enc1..4.down.norm.0 是 BatchNorm1d。
  strict=False 加载不会报任何 warning（LN 与 BN 的 weight/bias 形状完全相同），
  但前向会用 running_mean=0/var=1 归一化，预训练特征直接失效。
  所以必须显式比对前向输出。
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
    mlp_ratio=4, qkv_bias=True, drop_path=0.3, shuffle_orders=True,
    pre_norm=True, enable_rpe=False, enable_flash=False,
    pdnorm_bn=False, pdnorm_ln=False,
)


def main():
    torch.manual_seed(0)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]

    sm = sonata.load(CKPT, custom_config=dict(
        enc_patch_size=[1024] * 5, enable_flash=False)).cuda().eval()
    om = build_model(CFG).cuda().eval()

    miss, unexp = om.load_state_dict(sd, strict=False)
    print("missing  %3d: %s" % (len(miss), list(miss)[:6]))
    print("unexpected %3d: %s" % (len(unexp), list(unexp)[:6]))
    bad_miss = [k for k in miss if not k.startswith("dec.")]
    if unexp:
        raise SystemExit("FAIL: 有意外键 %s" % list(unexp)[:5])
    if bad_miss:
        raise SystemExit("FAIL: encoder/embedding 缺键 %s" % bad_miss[:5])
    print("权重映射 OK：encoder 全命中，missing 全部是 dec.*（从头训）")

    coord = np.load(os.path.join(SCENE, "coord.npy")).astype(np.float32)
    color = np.load(os.path.join(SCENE, "color.npy")).astype(np.float32)
    normal = np.load(os.path.join(SCENE, "normal.npy")).astype(np.float32)

    pt = sonata.transform.default()(dict(coord=coord, color=color, normal=normal))
    for k in list(pt.keys()):
        if isinstance(pt[k], torch.Tensor):
            pt[k] = pt[k].cuda()
    print("输入 %d 点  feat %s  grid_coord %s"
          % (len(pt["coord"]), tuple(pt["feat"].shape),
             tuple(pt["grid_coord"].shape)))

    feats = {}

    def mk(name):
        def hook(mod, inp, out):
            feats[name] = out.feat.detach().float().clone()
        return hook

    sm.enc.enc4.register_forward_hook(mk("s"))
    om.enc.enc4.register_forward_hook(mk("o"))

    with torch.no_grad():
        sm(copy.deepcopy(pt))
        om(copy.deepcopy(pt))

    a, b = feats["s"], feats["o"]
    print("sonata enc4 %s   ours enc4 %s" % (tuple(a.shape), tuple(b.shape)))
    if a.shape != b.shape:
        raise SystemExit("FAIL: 形状不一致")
    d = (a - b).abs()
    mx, mean = d.max().item(), d.mean().item()
    print("max_err %.3e   mean_err %.3e" % (mx, mean))
    print("rms sonata %.4f   ours %.4f" % (a.pow(2).mean().sqrt().item(),
                                          b.pow(2).mean().sqrt().item()))
    print("GATE", "PASS" if mx < 1e-4 else "FAIL")


if __name__ == "__main__":
    main()
