# -*- coding: utf-8 -*-
"""纯 3D 1-shot 机理诊断：验证死结 1（1232 维无约束线性拟合过拟合）。

对照 sonata_1shot.py 的 0.1594（40 轮 AdamW + wcls 加权 + 无约束线性头）：
  本脚本不改数据、不用 VLM、不换骨干，只换分类器：
    Base 12 类 : 线性头（CrossEntropy，**去掉 wcls 加权**，不让新类撑爆权重）
    Novel 45 类: **L2 归一化余弦原型**（1-shot support 的类均值）+ 温度 tau
  两部分分数拼在一起 argmax。

用法:
  python sonata_proto.py --tau 20
  python sonata_proto.py --tau 20 --limit 10 --val-limit 3   # 干跑
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/data01/jh/code/sonata")
sys.path.insert(0, "/data01/jh/code/GFS-VL")
sys.path.insert(0, "/data01/jh/code/GFS-VL/pointcept/models/PLA")
sys.path.insert(0, "/data01/jh/code/ps_diag")

import sonata  # noqa: E402
import vlm_common as V  # noqa: E402
from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
    CLASS_LABELS_200)  # noqa: E402

CKPT = "/data01/jh/code/sonata/ckpt/sonata.pth"
DATA_SHM = "/dev/shm/sc200_train"
DATA_HDD = "/data01/jh/data/scannet200_gfs"
REGIS = "/data01/jh/data/scannet200_gfs/sc200_regis_1_10.txt"
NO, NC = 12, 57


def build_model():
    cfg = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    m = sonata.load(CKPT, custom_config=cfg).cuda().eval()
    return m


def feat_gpu(model, coord, color, normal):
    pt = dict(coord=np.asarray(coord, dtype=np.float32),
              color=np.asarray(color, dtype=np.float32),
              normal=np.asarray(normal, dtype=np.float32))
    pt = sonata.transform.default()(pt)
    for k in list(pt.keys()):
        if isinstance(pt[k], torch.Tensor):
            pt[k] = pt[k].cuda(non_blocking=True)
    base = model(pt)
    chain, cur = [], base
    while "pooling_parent" in cur.keys():
        chain.append((cur["pooling_parent"], cur["pooling_inverse"]))
        cur = cur["pooling_parent"]
    top = cur
    inv = top["inverse"]
    f = base.feat
    for par, iv in chain:
        f = torch.cat([par.feat, f[iv]], dim=-1)
    return f[inv]


def scene_dir(split, sc):
    p = os.path.join(DATA_SHM, split, sc)
    return p if os.path.isdir(p) else os.path.join(DATA_HDD, split, sc)


def load_raw(d):
    return (np.load(os.path.join(d, "coord.npy")).astype(np.float32),
            np.load(os.path.join(d, "color.npy")).astype(np.float32),
            np.load(os.path.join(d, "normal.npy")).astype(np.float32),
            np.load(os.path.join(d, "segment200.npy")).reshape(-1))


def cid_to_gfs(cid):
    return V.CLASS_NAMES.index(CLASS_LABELS_200[cid])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scene", type=int, default=50)
    ap.add_argument("--tau", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda"
    model = build_model()
    rng = np.random.RandomState(0)

    # ---------- Base：1201 场景的 12 类 ----------
    root = os.path.join(DATA_SHM, "train")
    if not os.path.isdir(root):
        root = os.path.join(DATA_HDD, "train")
    scenes = sorted(os.listdir(root))
    if args.limit > 0:
        scenes = scenes[:args.limit]
    print("Base 场景 %d" % len(scenes), flush=True)

    bfeats, bgts = [], []
    t0 = time.time()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i, sc in enumerate(scenes):
            coord, color, normal, seg200 = load_raw(scene_dir("train", sc))
            gfs = V.convert_to_gfs(seg200)
            f = feat_gpu(model, coord, color, normal)
            idx = []
            for c in range(NO):
                w = np.where(gfs == c)[0]
                if len(w) == 0:
                    continue
                if len(w) > args.per_scene:
                    w = rng.choice(w, args.per_scene, replace=False)
                idx.append(w)
            if not idx:
                continue
            idx = np.concatenate(idx)
            sel = torch.from_numpy(idx).to(f.device)
            bfeats.append(f[sel].float().cpu())
            bgts.append(gfs[idx])
            if (i + 1) % 200 == 0:
                print("  base %d/%d  %.0fs" % (i + 1, len(scenes), time.time() - t0),
                      flush=True)

    # ---------- Novel：45 support 场景 -> 类原型 ----------
    protos = np.zeros((45, 1232), dtype=np.float32)
    have = np.zeros(45, dtype=bool)
    for line in open(REGIS):
        cid, rel = line.strip().split("\t")
        gfs_c = cid_to_gfs(int(cid))
        coord, color, normal, seg200 = load_raw(os.path.join(DATA_HDD, rel))
        gfs = V.convert_to_gfs(seg200)
        w = np.where(gfs == gfs_c)[0]
        if len(w) == 0:
            continue
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            f = feat_gpu(model, coord, color, normal)
        sel = torch.from_numpy(w).to(f.device)
        v = f[sel].float().mean(0).cpu().numpy()
        protos[gfs_c - NO] = v
        have[gfs_c - NO] = True
    print("Novel 原型覆盖 %d/45" % have.sum(), flush=True)

    # ---------- Base 线性头（无 wcls 加权） ----------
    Xb = torch.cat(bfeats, 0)
    yb = torch.tensor(np.concatenate(bgts, 0), dtype=torch.long, device=dev)
    Xb = Xb.to(dev)
    mu = Xb.mean(0, keepdim=True)
    sd = Xb.std(0, keepdim=True) + 1e-6
    W = torch.zeros(1232, NO, requires_grad=True, device=dev)
    b = torch.zeros(NO, requires_grad=True, device=dev)
    opt = torch.optim.AdamW([W, b], lr=1e-3, weight_decay=5e-4)
    lossf = torch.nn.CrossEntropyLoss()          # ← 无类别加权
    n = len(yb)
    for ep in range(40):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, 8192):
            idx = perm[s:s + 8192]
            opt.zero_grad()
            loss = lossf((Xb[idx] - mu) / sd @ W + b, yb[idx])
            loss.backward()
            opt.step()
    print("Base 线性头训练完成", flush=True)

    P = torch.tensor(protos, device=dev)
    P = F.normalize(P, dim=1)

    # 用 Base 训练集校准两个分支的尺度（避免手调 tau 压垮 base）
    with torch.no_grad():
        bl_tr = (Xb - mu) / sd @ W + b
        cs_tr = F.normalize(Xb, dim=1) @ P.T
        scale = float(bl_tr.std() / (cs_tr.std() + 1e-6))
    print("尺度校准: base_std=%.3f cos_std=%.3f -> scale=%.3f"
          % (float(bl_tr.std()), float(cs_tr.std()), scale), flush=True)

    # ---------- 评估 ----------
    root = os.path.join(DATA_SHM, "val")
    if not os.path.isdir(root):
        root = os.path.join(DATA_HDD, "val")
    scenes = sorted(os.listdir(root))
    if args.val_limit > 0:
        scenes = scenes[:args.val_limit]
    inter = np.zeros(NC)
    union = np.zeros(NC)
    print("评估 %d 个 val 场景（tau=%.1f）" % (len(scenes), args.tau), flush=True)
    t0 = time.time()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i, sc in enumerate(scenes):
            coord, color, normal, seg200 = load_raw(scene_dir("val", sc))
            gfs = V.convert_to_gfs(seg200)
            f = feat_gpu(model, coord, color, normal).float()
            base_logits = (f - mu) / sd @ W + b
            cos = F.normalize(f, dim=1) @ P.T * scale
            all_scores = torch.cat([base_logits, cos], dim=1)
            p = all_scores.argmax(1).cpu().numpy()
            valid = gfs >= 0
            p, g = p[valid], gfs[valid]
            for c in range(NC):
                tp = ((p == c) & (g == c)).sum()
                fp = ((p == c) & (g != c)).sum()
                fn = ((p != c) & (g == c)).sum()
                inter[c] += tp
                union[c] += tp + fp + fn
            if (i + 1) % 100 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(scenes), time.time() - t0), flush=True)

    iou = inter / np.maximum(union, 1)
    base = iou[:NO].mean()
    novel = iou[NO:].mean()
    hm = 2 * base * novel / max(base + novel, 1e-9)
    print()
    print("=== Sonata 1232d · 纯 3D 1-shot · 余弦原型（tau=%.1f）===" % args.tau)
    print("Base  (12 类) = %.4f" % base)
    print("Novel (45 类) = %.4f" % novel)
    print("HM            = %.4f" % hm)
    print("mIoU          = %.4f" % iou.mean())


if __name__ == "__main__":
    main()
