# -*- coding: utf-8 -*-
"""纯 3D 1-shot 机理诊断 v2：一次前向，三个分类器，带特征缓存。

v1（sonata_proto.py）的 bug：
  base 线性头与 novel 余弦分支只对齐了 std，没对齐 mean。
  min(12) 个 base logit 的 max 系统性低于 max(45) 个 cos → Base IoU = 0.0000，
  但 Novel 0.1276 已经比线性头 0.0930 高 37%，证明「高维无约束拟合过拟合」是真死结。

v2 做四件事：
  [对齐] base 线性头 + novel 余弦，**mean + std 双对齐**（全局标量，val 直接复用）
  [对照] 对称余弦原型：57 类全部用类均值原型 L2 归一化；argmax 对温度不变，零超参
  [对照] lin57：57 类线性头，**去掉 wcls 类别加权**（v1 存在的另一个不稳定源）
  [缓存] base 训练特征落盘 fp16，改分类器不必重跑 1201 场景前向

用法:
  python sonata_proto2.py                 # 全量（首次会建缓存，约 1 h）
  python sonata_proto2.py --limit 10 --val-limit 3   # 干跑
  python sonata_proto2.py --rebuild-cache             # 强制重建缓存
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
CACHE_DIR = "/data01/jh/data/sonata_feat_cache"
NO, NC = 12, 57
EPOCH = 40
LR = 1e-3
WD = 5e-4


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
    f = base.feat
    for par, iv in chain:
        f = torch.cat([par.feat, f[iv]], dim=-1)
    return f[top["inverse"]]


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


def build_cache(model, per_scene, limit, rebuild):
    """base 12 类的训练特征 + novel 45 类原型，落盘 fp16。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = "ps%d%s" % (per_scene, "" if limit == 0 else "_lim%d" % limit)
    fx = os.path.join(CACHE_DIR, "base_%s.npy" % tag)
    fy = os.path.join(CACHE_DIR, "base_%s_y.npy" % tag)
    fp = os.path.join(CACHE_DIR, "proto_%s.npy" % tag)

    if not rebuild and all(os.path.exists(p) for p in (fx, fy, fp)):
        X = np.load(fx)
        y = np.load(fy)
        P = np.load(fp)
        print("[cache] 命中 %s: X%s y%s proto%s" % (tag, X.shape, y.shape, P.shape),
              flush=True)
        return X.astype(np.float32), y, P.astype(np.float32)

    root = os.path.join(DATA_SHM, "train")
    if not os.path.isdir(root):
        root = os.path.join(DATA_HDD, "train")
    scenes = sorted(os.listdir(root))
    if limit > 0:
        scenes = scenes[:limit]
    print("Base 场景 %d（建缓存）" % len(scenes), flush=True)

    rng = np.random.RandomState(0)
    feats, gts = [], []
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
                if len(w) > per_scene:
                    w = rng.choice(w, per_scene, replace=False)
                idx.append(w)
            if not idx:
                continue
            idx = np.concatenate(idx)
            sel = torch.from_numpy(idx).to(f.device)
            feats.append(f[sel].float().cpu().half().numpy())
            gts.append(gfs[idx])
            if (i + 1) % 200 == 0:
                print("  base %d/%d  %.0fs" % (i + 1, len(scenes), time.time() - t0),
                      flush=True)

    X = np.concatenate(feats, 0)
    y = np.concatenate(gts, 0)
    print("  base 特征 %s  %d 点  %.0fs" % (X.shape, len(y), time.time() - t0), flush=True)

    protos = np.zeros((45, X.shape[1]), dtype=np.float32)
    have = np.zeros(45, dtype=bool)
    t1 = time.time()
    for line in open(REGIS):
        cid, rel = line.strip().split("\t")
        gc = cid_to_gfs(int(cid))
        coord, color, normal, seg200 = load_raw(os.path.join(DATA_HDD, rel))
        gfs = V.convert_to_gfs(seg200)
        w = np.where(gfs == gc)[0]
        if len(w) == 0:
            continue
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            f = feat_gpu(model, coord, color, normal)
        protos[gc - NO] = f[torch.from_numpy(w).to(f.device)].float().mean(0).cpu().numpy()
        have[gc - NO] = True
    print("  novel 原型覆盖 %d/45  %.0fs" % (have.sum(), time.time() - t1), flush=True)

    np.save(fx, X)
    np.save(fy, y)
    np.save(fp, protos)
    print("[cache] 已写 %s" % CACHE_DIR, flush=True)
    return X.astype(np.float32), y, protos


def train_linear(X, y, ncls, dev, use_wcls=False, seed=0):
    """线性头。X 已在 CPU(float32)。返回 (mu, sd, W, b)。"""
    torch.manual_seed(seed)
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    Xt = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    mut = torch.from_numpy(mu).to(dev)
    sdt = torch.from_numpy(sd).to(dev)
    D = X.shape[1]
    W = torch.zeros(D, ncls, requires_grad=True, device=dev)
    b = torch.zeros(ncls, requires_grad=True, device=dev)
    opt = torch.optim.AdamW([W, b], lr=LR, weight_decay=WD)
    if use_wcls:
        freq = np.bincount(y, minlength=ncls).astype(np.float64)
        w = np.where(freq > 0, 1.0 / np.sqrt(np.maximum(freq, 1)), 0.0)
        w = w / w.sum() * ncls
        lossf = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32, device=dev))
    else:
        lossf = torch.nn.CrossEntropyLoss()
    n = len(yt)
    for ep in range(EPOCH):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for s in range(0, n, 8192):
            i = perm[s:s + 8192]
            opt.zero_grad()
            loss = lossf((Xt[i] - mut) / sdt @ W + b, yt[i])
            loss.backward()
            opt.step()
            tot += float(loss) * len(i)
        if (ep + 1) % 20 == 0:
            print("    ep%d loss=%.4f" % (ep + 1, tot / n), flush=True)
    return mu, sd, W.detach(), b.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scene", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    dev = "cuda"
    model = build_model()

    Xb, yb, protos = build_cache(model, args.per_scene, args.limit,
                                 args.rebuild_cache)
    Xb = Xb.astype(np.float32)

    # ---------- 分类器 A：base 线性头（12 类，无 wcls） ----------
    print("[A] base 线性头", flush=True)
    mu, sd, W12, b12 = train_linear(Xb, yb, NO, dev, use_wcls=False)

    # 57 类线性头不作为对照：novel 侧只有 1-shot 点，无 wcls 时会被 base 巨量点压死，
    # 有 wcls 时就是 v1 的 0.0930。这里只比「线性 base + 余弦 novel」与「全对称原型」。

    P45 = torch.from_numpy(protos).to(dev)
    P45 = F.normalize(P45, dim=1)

    # ---------- 对称原型：base 12 类原型 = base 点类均值 ----------
    P12 = torch.zeros(NO, Xb.shape[1], device=dev)
    for c in range(NO):
        w = np.where(yb == c)[0]
        if len(w):
            P12[c] = torch.from_numpy(Xb[w].mean(0)).to(dev)
    P12 = F.normalize(P12, dim=1)
    Pall = torch.cat([P12, P45], 0)          # (57, 1232)

    # ---------- A 分支的 mean+std 双对齐标量 ----------
    Xbt = torch.from_numpy(Xb).to(dev)
    mut = torch.from_numpy(mu).to(dev)
    sdt = torch.from_numpy(sd).to(dev)
    with torch.no_grad():
        bl = (Xbt - mut) / sdt @ W12 + b12
        cs = F.normalize(Xbt, dim=1) @ P45.T
        bl_mu, bl_sd = bl.mean(), bl.std()
        cs_mu, cs_sd = cs.mean(), cs.std()
    print("对齐: bl(mean=%.4f std=%.4f)  cos(mean=%.4f std=%.4f)"
          % (float(bl_mu), float(bl_sd), float(cs_mu), float(cs_sd)), flush=True)
    del Xbt

    # ---------- 评估 ----------
    root = os.path.join(DATA_SHM, "val")
    if not os.path.isdir(root):
        root = os.path.join(DATA_HDD, "val")
    scenes = sorted(os.listdir(root))
    if args.val_limit > 0:
        scenes = scenes[:args.val_limit]
    names = ["lin_cos", "proto"]
    inter = {k: np.zeros(NC) for k in names}
    union = {k: np.zeros(NC) for k in names}
    print("评估 %d 个 val 场景" % len(scenes), flush=True)
    t0 = time.time()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i, sc in enumerate(scenes):
            coord, color, normal, seg200 = load_raw(scene_dir("val", sc))
            gfs = V.convert_to_gfs(seg200)
            f = feat_gpu(model, coord, color, normal).float()
            fn = F.normalize(f, dim=1)
            s_lin = torch.cat([(f - mut) / sdt @ W12 + b12,
                               (fn @ P45.T - cs_mu) / cs_sd * bl_sd + bl_mu], 1)
            s_pro = fn @ Pall.T
            valid = gfs >= 0
            vi = torch.from_numpy(np.where(valid)[0]).to(f.device)
            g = torch.from_numpy(gfs[valid].astype(np.int64)).to(f.device)
            for k, s in (("lin_cos", s_lin), ("proto", s_pro)):
                p = s[vi].argmax(1)
                # 一次 bincount 出混淆矩阵，避免 57 类逐个同步
                cm = torch.bincount(g * NC + p, minlength=NC * NC).reshape(NC, NC)
                cm = cm.cpu().numpy().astype(np.float64)
                inter[k] += np.diag(cm)
                union[k] += cm.sum(0) + cm.sum(1) - np.diag(cm)
            if (i + 1) % 100 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(scenes), time.time() - t0),
                      flush=True)

    print()
    for k in names:
        iou = inter[k] / np.maximum(union[k], 1)
        base = iou[:NO].mean()
        novel = iou[NO:].mean()
        hm = 2 * base * novel / max(base + novel, 1e-9)
        print("=== %-8s Base=%.4f  Novel=%.4f  HM=%.4f  mIoU=%.4f ==="
              % (k, base, novel, hm, iou.mean()))
        print("    逐类:", np.round(iou, 4).tolist())


if __name__ == "__main__":
    main()
