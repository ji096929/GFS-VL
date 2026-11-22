# -*- coding: utf-8 -*-
"""纯 3D 1-shot：Base 与 Novel 放在同一个余弦球面上比较（v3）。

v1（sonata_proto.py）的错误：
  base_logits（尺度 ~±2.4，均值 ≈ 0）与 cos × 51.44（均值 +0.39 → +20）是两套
  不同尺度的分数，只对齐 std 不对齐 mean，45 个 novel 分数整体压过 12 个 base 分数，
  Base IoU 恒为 0。即便如此 Novel 仍有 0.1276，已经说明「线性头过拟合」是真死结。

v3 按用户要求的做法：
  Base  12 类：可学习 L2 归一化余弦分类头  F.normalize(x) @ F.normalize(W) * tau
  Novel 45 类：L2 归一化类原型（1-shot support 均值）
  两者在同一球面上 argmax，只加一个标量 novel_bias，扫 [-0.10, -0.05, 0.00, +0.05]

复用 sonata_proto2.py 建的缓存（base 72 万点 fp16 + 45 类原型），不重复前向。
用法: python sonata_proto3.py
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/data01/jh/code/ps_diag")

import sonata_proto2 as S2  # noqa: E402

CACHE_DIR = "/data01/jh/data/sonata_feat_cache"
NO, NC = 12, 57
TAU = 20.0
EPOCH = 40
LR = 1e-3
WD = 5e-4
BIASES = [-0.10, -0.05, 0.00, 0.05]


def train_cos_head(X, y, dev):
    """可学习 L2 归一化余弦分类头。返回 (mu, sd, Wn)，Wn 已按列归一化 (D, 12)。"""
    torch.manual_seed(0)
    mu = torch.from_numpy(X.mean(0, keepdims=True)).to(dev)
    sd = torch.from_numpy(X.std(0, keepdims=True) + 1e-6).to(dev)
    Xt = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    Xn = F.normalize((Xt - mu) / sd, dim=1)

    D = X.shape[1]
    W = (torch.randn(D, NO, device=dev) * 0.02).requires_grad_(True)
    opt = torch.optim.AdamW([W], lr=LR, weight_decay=WD)
    lossf = torch.nn.CrossEntropyLoss()

    n = len(yt)
    for ep in range(EPOCH):
        perm = torch.randperm(n, device=dev)
        tot, cor = 0.0, 0
        for s in range(0, n, 8192):
            i = perm[s:s + 8192]
            opt.zero_grad()
            logits = Xn[i] @ F.normalize(W, dim=0) * TAU
            loss = lossf(logits, yt[i])
            loss.backward()
            opt.step()
            tot += float(loss) * len(i)
            cor += int((logits.argmax(1) == yt[i]).sum())
        if (ep + 1) % 10 == 0:
            print("    ep%d loss=%.4f acc=%.3f"
                  % (ep + 1, tot / n, cor / n), flush=True)
    return mu, sd, F.normalize(W.detach(), dim=0)


def main():
    dev = "cuda"
    fx = os.path.join(CACHE_DIR, "base_ps50.npy")
    fy = os.path.join(CACHE_DIR, "base_ps50_y.npy")
    fp = os.path.join(CACHE_DIR, "proto_ps50.npy")
    for p in (fx, fy, fp):
        if not os.path.exists(p):
            raise SystemExit("缓存缺失: %s（先跑 sonata_proto2.py）" % p)
    X = np.load(fx).astype(np.float32)
    y = np.load(fy)
    P = np.load(fp).astype(np.float32)
    print("缓存: base %s  proto %s" % (X.shape, P.shape), flush=True)

    print("[余弦头] Base 12 类", flush=True)
    mu, sd, Wn = train_cos_head(X, y, dev)          # Wn (D, 12)
    Pn = F.normalize(torch.from_numpy(P).to(dev), dim=1)   # (45, D)
    P57 = torch.cat([Wn.T, Pn], 0)                  # (57, D)
    del X

    # 训练集上的两个分支分数分布（诊断用）
    model = S2.build_model()

    root = os.path.join(S2.DATA_SHM, "val")
    if not os.path.isdir(root):
        root = os.path.join(S2.DATA_HDD, "val")
    scenes = sorted(os.listdir(root))
    inter = {b: np.zeros(NC) for b in BIASES}
    union = {b: np.zeros(NC) for b in BIASES}
    print("评估 %d 个 val 场景，扫 bias %s" % (len(scenes), BIASES), flush=True)
    t0 = time.time()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i, sc in enumerate(scenes):
            coord, color, normal, seg200 = S2.load_raw(S2.scene_dir("val", sc))
            gfs = S2.V.convert_to_gfs(seg200)
            f = S2.feat_gpu(model, coord, color, normal).float()
            fn = F.normalize((f - mu) / sd, dim=1)
            cos = fn @ P57.T                       # (N, 57)
            valid = gfs >= 0
            vi = torch.from_numpy(np.where(valid)[0]).to(f.device)
            g = torch.from_numpy(gfs[valid].astype(np.int64)).to(f.device)
            for b in BIASES:
                s = cos[vi]
                if b != 0.0:
                    s = s.clone()
                    s[:, NO:] += b
                p = s.argmax(1)
                cm = torch.bincount(g * NC + p, minlength=NC * NC).reshape(NC, NC)
                cm = cm.cpu().numpy().astype(np.float64)
                inter[b] += np.diag(cm)
                union[b] += cm.sum(0) + cm.sum(1) - np.diag(cm)
            if (i + 1) % 100 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(scenes), time.time() - t0),
                      flush=True)

    print()
    print("=== Sonata 1232d · 纯 3D 1-shot · 统一余弦球面 ===")
    for b in BIASES:
        iou = inter[b] / np.maximum(union[b], 1)
        base = iou[:NO].mean()
        novel = iou[NO:].mean()
        hm = 2 * base * novel / max(base + novel, 1e-9)
        print("bias=%+.2f  Base=%.4f  Novel=%.4f  HM=%.4f  mIoU=%.4f"
              % (b, base, novel, hm, iou.mean()))
    print()
    print("bias=0.00 逐类:", np.round(inter[0.0] / np.maximum(union[0.0], 1), 4).tolist())


if __name__ == "__main__":
    main()
