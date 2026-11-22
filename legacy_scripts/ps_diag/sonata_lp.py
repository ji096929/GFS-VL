# -*- coding: utf-8 -*-
"""格 1：Sonata 冻结特征的 57 类线性探针。

与 cut2_variants_lp.py 同口径：
  - 按类均匀采样，每类上限 CAP=2000 点
  - 线性头（无隐藏层），AdamW(lr=1e-3, wd=5e-4)，40 epoch
  - 在 312 val 上算 12 类 Base / 45 类 Novel / HM / mIoU

用法:
  python sonata_lp.py
"""
import glob
import os

import numpy as np
import torch

FEAT = "/data01/jh/data/sonata_feat"
NO, NC = 12, 57
CAP = 2000
EPOCH = 40
LR = 1e-3
WD = 5e-4


def load_split(split, cap=CAP):
    files = sorted(glob.glob(os.path.join(FEAT, "%s_*.npz" % split)))
    print("%s: %d 个文件" % (split, len(files)), flush=True)
    cnt = np.zeros(NC, np.int64)
    feats, gts = [], []
    rng = np.random.RandomState(0)
    for fi, f in enumerate(files):
        z = np.load(f)
        gt = z["gt"].astype(np.int64)
        idx = []
        for c in range(NC):
            w = np.where(gt == c)[0]
            if len(w) == 0:
                continue
            rng.shuffle(w)
            need = cap - cnt[c]
            if need > 0:
                a = w[:need]
                idx.append(a)
                cnt[c] += len(a)
        if not idx:
            continue
        idx = np.concatenate(idx)
        feats.append(z["feat"][idx].astype(np.float32))
        gts.append(gt[idx])
        if (fi + 1) % 200 == 0:
            print("  %d/%d" % (fi + 1, len(files)), flush=True)
    X = np.concatenate(feats, 0)
    y = np.concatenate(gts, 0)
    print("  -> %s: %d 点, %d 维" % (split, len(y), X.shape[1]), flush=True)
    return X, y


def main():
    torch.manual_seed(0)
    dev = "cuda"
    Xtr, ytr = load_split("train")
    # val 不预加载：评估时逐文件读（312 × 95 MB 一次读会 OOM）

    # 标准化（线性探针的标准做法）
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd

    Xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(ytr, device=dev)

    D = Xtr.shape[1]
    W = torch.zeros(D, NC, requires_grad=True, device=dev)
    b = torch.zeros(NC, requires_grad=True, device=dev)
    opt = torch.optim.AdamW([W, b], lr=LR, weight_decay=WD)
    lossf = torch.nn.CrossEntropyLoss()

    n = len(yt)
    bs = 8192
    for ep in range(EPOCH):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            out = Xt[idx] @ W + b
            loss = lossf(out, yt[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if (ep + 1) % 10 == 0:
            print("  ep%d loss=%.4f" % (ep + 1, tot / n), flush=True)

    # 评估：逐文件分批，避免把 51 GB val 特征一次性搬进 GPU
    inter = np.zeros(NC)
    union = np.zeros(NC)
    val_files = sorted(glob.glob(os.path.join(FEAT, "val_*.npz")))
    print("评估 %d 个 val 文件" % len(val_files), flush=True)
    with torch.no_grad():
        for fi, f in enumerate(val_files):
            z = np.load(f)
            gt = z["gt"].astype(np.int64)
            feat = (z["feat"].astype(np.float32) - mu) / sd
            p = (torch.tensor(feat, device=dev) @ W + b).argmax(1).cpu().numpy()
            for c in range(NC):
                tp = ((p == c) & (gt == c)).sum()
                fp = ((p == c) & (gt != c)).sum()
                fn = ((p != c) & (gt == c)).sum()
                inter[c] += tp
                union[c] += tp + fp + fn
            if (fi + 1) % 100 == 0:
                print("  %d/%d" % (fi + 1, len(val_files)), flush=True)
    iou = inter / np.maximum(union, 1)

    base = iou[:NO].mean()
    novel = iou[NO:].mean()
    hm = 2 * base * novel / max(base + novel, 1e-9)
    print()
    print("=== Sonata 冻结线性探针（%d 类）===" % NC)
    print("Base  (12 类) = %.4f" % base)
    print("Novel (45 类) = %.4f" % novel)
    print("HM            = %.4f" % hm)
    print("mIoU          = %.4f" % iou.mean())
    print()
    print("逐类 IoU:", np.round(iou, 4).tolist())


if __name__ == "__main__":
    main()
