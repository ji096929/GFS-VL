# -*- coding: utf-8 -*-
"""格 1 探针：Sonata（PTv3 架构，纯 encoder）冻结特征 + 线性头。

Sonata 的接口（已核实）：
  - model: sonata.model.PointTransformerV3, enc_mode=True（无 decoder）
  - 输入: dict(coord, grid_coord, feat, offset)，feat = coord+color+normal = 9 通道
  - 输出: Point，point.feat = 512 维 @ 1/16 分辨率
  - 需用每层 GridPooling 的 pooling_inverse 逐层反池化回全分辨率

用法:
  python sonata_probe.py --dry 2          # 干跑 2 个场景，只验形状
  python sonata_probe.py --extract        # 全量提取特征（train+val）
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/data01/jh/code/sonata")
sys.path.insert(0, "/data01/jh/code/GFS-VL")
sys.path.insert(0, "/data01/jh/code/GFS-VL/pointcept/models/PLA")

CKPT = "/data01/jh/code/sonata/ckpt/sonata.pth"
DATA = "/data01/jh/data/scannet200_gfs"
OUT = "/data01/jh/data/sonata_feat"   # 1201 场景 × 95 MB ≈ 114 GB，超出 /dev/shm 容量


def load_sonata():
    from sonata.model import PointTransformerV3

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    cfg["enable_flash"] = False   # 环境无 flash_attn，与 Stage-2 一致
    print("cfg:", {k: cfg[k] for k in ("in_channels", "enc_channels", "enc_mode")}, flush=True)
    model = PointTransformerV3(**cfg)
    miss, unexp = model.load_state_dict(ck["state_dict"], strict=False)
    print("load missing=%d unexpected=%d" % (len(miss), len(unexp)), flush=True)
    model.eval().cuda()
    return model


def attach_inverse_hooks(model):
    """每层 GridPooling 存 pooling_inverse，用于逐层反池化。"""
    invs = []

    def make_hook():
        def h(module, inputs, output):
            invs.append(output.pooling_inverse)

        return h

    n = 0
    for i in range(len(model.enc)):
        layer = model.enc[i]
        if "down" in layer._modules:
            layer._modules["down"].register_forward_hook(make_hook())
            n += 1
    print("hook 挂在 %d 个 GridPooling 上" % n, flush=True)
    return invs


def build_tf():
    """用 Sonata 自带的 transform，feat_keys 含 coord（9 通道）。"""
    from sonata.transform import Compose

    return Compose([
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train",
             return_grid_coord=True),
        dict(type="CenterShift", apply_z=False),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "segment"),
             feat_keys=("coord", "color", "normal")),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", type=int, default=0, help="干跑 N 个场景验形状")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--split", default="train")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()

    model = load_sonata()
    invs = attach_inverse_hooks(model)
    tf = build_tf()

    scenes = sorted(os.listdir(os.path.join(DATA, args.split)))
    # 按分片切片（此前漏了这一步，导致三片重复跑同一批场景）
    scenes = scenes[args.shard::args.shards]
    n_scene = args.dry if args.dry > 0 else len(scenes)
    print("%s 场景 %d / 总 %d" % (args.split, n_scene, len(scenes)), flush=True)

    t0 = time.time()
    for i in range(n_scene):
        sc = scenes[i]
        # 提前跳过已完成场景，避免重复加载
        if args.extract:
            _out = os.path.join(OUT, "%s_%s.npz" % (args.split, sc))
            if os.path.exists(_out):
                continue
        d0 = os.path.join(DATA, args.split, sc)
        raw = {
            "coord": np.load(os.path.join(d0, "coord.npy")).astype(np.float32),
            "color": np.load(os.path.join(d0, "color.npy")),
            "normal": np.load(os.path.join(d0, "normal.npy")),
            "segment": np.load(os.path.join(d0, "segment200.npy")).reshape(-1).astype(np.int64),
            "name": sc,
        }
        n_orig = len(raw["coord"])
        invs.clear()
        item = tf(raw)
        inp = {}
        for k, v in item.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if isinstance(v, torch.Tensor):
                # Sonata 的 Point 用 offset 分 batch，所有字段保持 (N, D) / (N,)
                inp[k] = v.cuda()
        with torch.no_grad():
            point = model(inp)
        feat = point.feat.float()
        print("  %s: 原始 %d 点 -> voxel %d, feat %s, 收 %d 层 inverse"
              % (sc, n_orig, feat.shape[0], tuple(feat.shape), len(invs)), flush=True)
        # 逐层反池化
        f = feat
        for inv in reversed(invs):
            f = f[inv]
        print("     反池化后: %s (原始 %d)" % (tuple(f.shape), n_orig), flush=True)
        if args.dry:
            continue
        # 全量提取时按场景存盘
        if args.extract:
            os.makedirs(OUT, exist_ok=True)
            out_path = os.path.join(OUT, "%s_%s.npz" % (args.split, sc))
            if os.path.exists(out_path):
                continue          # 已有则跳过，便于断点续跑
            np.savez(
                out_path,
                feat=f.cpu().numpy().astype(np.float16),
                gt=np.asarray(item["segment"]).astype(np.int16),
            )
        if (i + 1) % 50 == 0:
            print("  %d/%d  %.0fs" % (i + 1, n_scene, time.time() - t0), flush=True)

    print("done %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
