# -*- coding: utf-8 -*-
"""格 1：Sonata 1232 维多尺度特征的 57 类线性探针（正确版）。

修复的 4 个 bug：
  1. 采样：原来按场景顺序遍历，第一个场景就把常见类的 CAP 吃满
     → 改为「每场景每类最多 PER_SCENE 点」，跨 1201 场景均匀覆盖
  2. 标签：原来直接用 segment200.npy（-1..172 原始 200 类）
     → 用 vlm_common.convert_to_gfs 映射到 57 类（0..11 base, 12..56 novel, -1 ignore）
  3. 特征：原来只取 enc4 单层 512 维（1/16 分辨率）
     → 复用 build_sonata_region.feat_sonata()，沿 pooling_parent 链拼 5 层 = 1232 维全分辨率
  4. 存储：原来落盘 114 GB npz
     → 完全在线，训练特征留在内存（每场景每类 50 点，约 6 GB）

用法:
  python sonata_lp2.py --per-scene 50            # 训练+评估全流程
  python sonata_lp2.py --per-scene 50 --limit 5  # 干跑 5 个场景
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
sys.path.insert(0, "/data01/jh/code/ps_diag")

import vlm_common as V  # noqa: E402
from build_sonata_region import feat_sonata  # noqa: E402

CKPT = "/data01/jh/code/sonata/ckpt/sonata.pth"
DATA = "/data01/jh/data/scannet200_gfs"
NO, NC = 12, 57
EPOCH = 40
LR = 1e-3
WD = 5e-4


def build_model():
    import sonata

    cfg = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    model = sonata.load(CKPT, custom_config=cfg).cuda().eval()
    n = sum(p.numel() for p in model.parameters())
    print("[sonata] %.2fM params" % (n / 1e6), flush=True)
    return model


def collect(model, split, per_scene, limit=0, rng=None):
    """逐场景前向 -> 1232 维全分辨率特征 -> 每类最多 per_scene 点。"""
    scenes = sorted(os.listdir(os.path.join(DATA, split)))
    if limit > 0:
        scenes = scenes[:limit]
    if rng is None:
        rng = np.random.RandomState(0)

    feats, gts = [], []
    t0 = time.time()
    for i, sc in enumerate(scenes):
        d0 = os.path.join(DATA, split, sc)
        coord = np.load(os.path.join(d0, "coord.npy")).astype(np.float32)
        color = np.load(os.path.join(d0, "color.npy")).astype(np.float32)
        normal = np.load(os.path.join(d0, "normal.npy")).astype(np.float32)
        seg200 = np.load(os.path.join(d0, "segment200.npy")).reshape(-1)
        gfs = V.convert_to_gfs(seg200)              # 0..11 base, 12..56 novel, -1 ignore

        f = feat_sonata(model, coord, color, normal)  # (N, 1232) 原始点数
        assert f.shape[0] == len(gfs), "特征与标签点数不一致"

        idx = []
        for c in range(NC):
            w = np.where(gfs == c)[0]
            if len(w) == 0:
                continue
            if len(w) > per_scene:
                w = rng.choice(w, per_scene, replace=False)
            idx.append(w)
        if not idx:
            continue
        idx = np.concatenate(idx)
        feats.append(f[idx])
        gts.append(gfs[idx])
        if (i + 1) % 100 == 0:
            print("  %d/%d  %.0fs  累计 %d 点" % (i + 1, len(scenes), time.time() - t0,
                                                sum(len(g) for g in gts)), flush=True)

    X = np.concatenate(feats, 0).astype(np.float16)
    y = np.concatenate(gts, 0).astype(np.int64)
    print("  -> %s: %d 点, %d 维, %.0fs" % (split, len(y), X.shape[1], time.time() - t0),
          flush=True)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scene", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-per-scene", type=int, default=200)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda"
    model = build_model()

    Xtr, ytr = collect(model, "train", args.per_scene, args.limit)
    # 验证集不落盘、不全量缓存：边前向边评估
    mu = Xtr.astype(np.float32).mean(0, keepdims=True)
    sd = Xtr.astype(np.float32).std(0, keepdims=True) + 1e-6
    Xtr = ((Xtr.astype(np.float32) - mu) / sd).astype(np.float32)

    Xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(ytr, device=dev)
    D = Xtr.shape[1]
    W = torch.zeros(D, NC, requires_grad=True, device=dev)
    b = torch.zeros(NC, requires_grad=True, device=dev)
    opt = torch.optim.AdamW([W, b], lr=LR, weight_decay=WD)
    lossf = torch.nn.CrossEntropyLoss()

    # 类别不均衡：用频次的倒数开方做 loss 权重
    freq = np.bincount(ytr, minlength=NC).astype(np.float64)
    wcls = np.where(freq > 0, 1.0 / np.sqrt(np.maximum(freq, 1)), 0.0)
    wcls = wcls / wcls.sum() * NC
    lossf = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(wcls, dtype=torch.float32, device=dev))

    n = len(yt)
    bs = 8192
    for ep in range(EPOCH):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            loss = lossf(Xt[idx] @ W + b, yt[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if (ep + 1) % 10 == 0:
            print("  ep%d loss=%.4f" % (ep + 1, tot / n), flush=True)

    # 验证：逐场景前向 + 评估（不上采样到原始点，直接在场景内所有点上算 IoU）
    scenes = sorted(os.listdir(os.path.join(DATA, "val")))
    inter = np.zeros(NC)
    union = np.zeros(NC)
    print("评估 %d 个 val 场景" % len(scenes), flush=True)
    t0 = time.time()
    with torch.no_grad():
        for i, sc in enumerate(scenes):
            d0 = os.path.join(DATA, "val", sc)
            coord = np.load(os.path.join(d0, "coord.npy")).astype(np.float32)
            color = np.load(os.path.join(d0, "color.npy")).astype(np.float32)
            normal = np.load(os.path.join(d0, "normal.npy")).astype(np.float32)
            seg200 = np.load(os.path.join(d0, "segment200.npy")).reshape(-1)
            gfs = V.convert_to_gfs(seg200)
            f = feat_sonata(model, coord, color, normal)
            f = ((f - mu) / sd).astype(np.float32)
            p = (torch.tensor(f, device=dev) @ W + b).argmax(1).cpu().numpy()
            for c in range(NC):
                tp = ((p == c) & (gfs == c)).sum()
                fp = ((p == c) & (gfs != c)).sum()
                fn = ((p != c) & (gfs == c)).sum()
                inter[c] += tp
                union[c] += tp + fp + fn
            if (i + 1) % 100 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(scenes), time.time() - t0), flush=True)

    iou = inter / np.maximum(union, 1)
    base = iou[:NO].mean()
    novel = iou[NO:].mean()
    hm = 2 * base * novel / max(base + novel, 1e-9)
    print()
    print("=== Sonata 1232d 线性探针（57 类，per_scene=%d）===" % args.per_scene)
    print("Base  (12 类) = %.4f" % base)
    print("Novel (45 类) = %.4f" % novel)
    print("HM            = %.4f" % hm)
    print("mIoU          = %.4f" % iou.mean())
    print()
    print("逐类 IoU:", np.round(iou, 4).tolist())


if __name__ == "__main__":
    main()
