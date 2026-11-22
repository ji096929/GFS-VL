# -*- coding: utf-8 -*-
"""格 1：Sonata 1232 维多尺度特征 + 57 类线性探针（性能与评估双修版）。

本版修复：
  [性能 1] bf16 autocast：前向 2.27s → 1.19s
  [性能 2] 绝不把 (N, 1232) 拷回 CPU：
           train 在 GPU 上按 idx 切片，只拷 ~1000 点（2 MB）
           val 在 GPU 上归一化 + 矩阵乘 + argmax，只拷 (N,) 预测
  [性能 3] train/val 都读 /dev/shm（内存盘）
  [评估]   IoU 前过滤 gfs == -1 的 ignore 点（原来把 ignore 全算成 FP，IoU 腰斩）
  [正确]   复用 build_sonata_region 的 1232 维拼接逻辑（沿 pooling_parent 链）
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

import sonata  # noqa: E402
import vlm_common as V  # noqa: E402

CKPT = "/data01/jh/code/sonata/ckpt/sonata.pth"
DATA_SHM = "/dev/shm/sc200_train"          # 内存盘（含 train/val 子目录）
DATA_HDD = "/data01/jh/data/scannet200_gfs"  # 兜底
NO, NC = 12, 57
EPOCH = 40
LR = 1e-3
WD = 5e-4


def build_model():
    cfg = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    m = sonata.load(CKPT, custom_config=cfg).cuda().eval()
    print("[sonata] %.2fM params" % (sum(p.numel() for p in m.parameters()) / 1e6),
          flush=True)
    return m


def feat_gpu(model, coord, color, normal):
    """返回 (N, 1232) 的 GPU 张量，全程不落 CPU。

    沿 pooling_parent 链逐层 cat，最后用 base_inv 映回原始点数。
    """
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
    base_inv = top["inverse"]
    feat = base.feat
    for par, iv in chain:
        feat = torch.cat([par.feat, feat[iv]], dim=-1)
    return feat[base_inv]            # (N, 1232) on GPU


def scene_dir(split, sc):
    p = os.path.join(DATA_SHM, split, sc)
    if os.path.isdir(p):
        return p
    return os.path.join(DATA_HDD, split, sc)


def load_raw(d):
    return (np.load(os.path.join(d, "coord.npy")).astype(np.float32),
            np.load(os.path.join(d, "color.npy")).astype(np.float32),
            np.load(os.path.join(d, "normal.npy")).astype(np.float32),
            np.load(os.path.join(d, "segment200.npy")).reshape(-1))


def collect(model, split, per_scene, limit=0, rng=None):
    root = os.path.join(DATA_SHM, split)
    if not os.path.isdir(root):
        root = os.path.join(DATA_HDD, split)
    scenes = sorted(os.listdir(root))
    if limit > 0:
        scenes = scenes[:limit]
    if rng is None:
        rng = np.random.RandomState(0)

    feats, gts = [], []
    t0 = time.time()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i, sc in enumerate(scenes):
            coord, color, normal, seg200 = load_raw(scene_dir(split, sc))
            gfs = V.convert_to_gfs(seg200)
            f = feat_gpu(model, coord, color, normal)     # GPU (N, 1232)
            assert f.shape[0] == len(gfs), "点数不一致"

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
            sel = torch.from_numpy(idx).to(f.device)
            feats.append(f[sel].float().cpu().half())     # 只拷 ~1000 点
            gts.append(gfs[idx])
            if (i + 1) % 100 == 0:
                print("  %d/%d  %.0fs  累计 %d 点"
                      % (i + 1, len(scenes), time.time() - t0,
                         sum(len(g) for g in gts)), flush=True)

    X = torch.cat(feats, 0).numpy().astype(np.float32)
    y = np.concatenate(gts, 0)
    print("  -> %s: %d 点, %d 维, %.0fs"
          % (split, len(y), X.shape[1], time.time() - t0), flush=True)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scene", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda"
    model = build_model()
    Xtr, ytr = collect(model, "train", args.per_scene, args.limit)

    mu = torch.tensor(Xtr.mean(0, keepdims=True), device=dev, dtype=torch.float32)
    sd = torch.tensor(Xtr.std(0, keepdims=True) + 1e-6, device=dev, dtype=torch.float32)
    Xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(ytr, dtype=torch.long, device=dev)   # convert_to_gfs 返回 int32
    del Xtr

    D = Xt.shape[1]
    W = torch.zeros(D, NC, requires_grad=True, device=dev)
    b = torch.zeros(NC, requires_grad=True, device=dev)
    opt = torch.optim.AdamW([W, b], lr=LR, weight_decay=WD)
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
            loss = lossf((Xt[idx] - mu) / sd @ W + b, yt[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if (ep + 1) % 10 == 0:
            print("  ep%d loss=%.4f" % (ep + 1, tot / n), flush=True)

    # 验证：GPU 上归一化 + 矩阵乘 + argmax，只拷 (N,) 预测；过滤 ignore
    root = os.path.join(DATA_SHM, "val")
    if not os.path.isdir(root):
        root = os.path.join(DATA_HDD, "val")
    scenes = sorted(os.listdir(root))
    if args.val_limit > 0:
        scenes = scenes[:args.val_limit]
    inter = np.zeros(NC)
    union = np.zeros(NC)
    print("评估 %d 个 val 场景（per_scene=%d）" % (len(scenes), args.per_scene), flush=True)
    t0 = time.time()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i, sc in enumerate(scenes):
            coord, color, normal, seg200 = load_raw(scene_dir("val", sc))
            gfs = V.convert_to_gfs(seg200)
            f = feat_gpu(model, coord, color, normal)
            p = ((f.float() - mu) / sd @ W + b).argmax(1).cpu().numpy()
            valid = gfs >= 0                       # ← 关键：过滤 ignore
            p = p[valid]
            g = gfs[valid]
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
    print("=== Sonata 1232d 线性探针（57 类，per_scene=%d，已过滤 ignore）==="
          % args.per_scene)
    print("Base  (12 类) = %.4f" % base)
    print("Novel (45 类) = %.4f" % novel)
    print("HM            = %.4f" % hm)
    print("mIoU          = %.4f" % iou.mean())
    print("逐类 IoU:", np.round(iou, 4).tolist())


if __name__ == "__main__":
    main()
