# -*- coding: utf-8 -*-
"""格 1 真 1-shot：冻结 Sonata 1232d 特征 + 57 类线性头。

口径（与 GFS-VL 1-shot 严格一致）：
  Base 12 类  : 1201 个训练场景的全部 Base GT（每场景每类最多 PER_SCENE 点）
  Novel 45 类 : 仅 45 个 1-shot support 场景（sc200_regis_1_10.txt），每类只有 1 个场景
  评估       : 312 val 全量，过滤 gfs == -1 的 ignore 点

== 2026-11-21 修正（v2）==
旧实现的 --use-pl 有两个致命 bug：
  1. PL 只在 45 个 support 场景里读，116 个其它训练场景的伪标签完全没用上；
  2. 筛选条件是 (pl == gfs_c) & (gfs == -1)，而在 support 场景里该类点都已标成
     gfs == gfs_c，所以这个条件筛出来的 100% 都是「VLM 把背景误判成该类」的假阳性。
  结果：--use-pl 只往训练集里灌噪声，Novel 从 0.0930 掉到 0.0827 是 bug 产物。

v2 的做法（与 GFS-VL Stage-2 的监督口径对齐）：
  在 **1201 个训练场景的主循环**里读 /dev/shm/jh_pl_k1_s10/{sc}_pl.npy，
  对每个新类 c in [12, 57)，取满足 (pl == c) 且不在 Base GT 区域
  （~(0 <= gfs < 12)）的点，每场景每类最多 PER_SCENE 个加入训练集。
  support 45 个场景的全部 GT 点照旧加入。

用法:
  python sonata_1shot.py                    # 纯 3D 1-shot（只用 support GT）
  python sonata_1shot.py --use-pl           # + 1201 场景 VLM 离线伪标签
  python sonata_1shot.py --limit 10 --val-limit 3   # 干跑
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
from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
    CLASS_LABELS_200)  # noqa: E402

CKPT = "/data01/jh/code/sonata/ckpt/sonata.pth"
DATA_SHM = "/dev/shm/sc200_train"
DATA_HDD = "/data01/jh/data/scannet200_gfs"
REGIS = "/data01/jh/data/scannet200_gfs/sc200_regis_1_10.txt"
PL_DIR = "/dev/shm/jh_pl_k1_s10"
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


def base_label_to_gfs(class_id):
    """原始 200 类 ID -> GFS 57 类索引（必须用 vlm_common.CLASS_NAMES 的顺序）。"""
    return V.CLASS_NAMES.index(CLASS_LABELS_200[class_id])


def read_support():
    sup = []
    for line in open(REGIS):
        cid, rel = line.strip().split("\t")
        sup.append((int(cid), rel))
    return sup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scene", type=int, default=50,
                    help="每个场景每个类最多取多少点（Base 与 PL 共用）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--use-pl", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda"
    model = build_model()
    rng = np.random.RandomState(0)

    root = os.path.join(DATA_SHM, "train")
    if not os.path.isdir(root):
        root = os.path.join(DATA_HDD, "train")
    scenes = sorted(os.listdir(root))
    if args.limit > 0:
        scenes = scenes[:args.limit]
    print("Base 场景 %d  use_pl=%s" % (len(scenes), args.use_pl), flush=True)

    feats, gts, is_novel = [], [], []
    pl_cnt = np.zeros(NC, dtype=np.int64)
    n_pl_scene = 0
    t0 = time.time()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i, sc in enumerate(scenes):
            coord, color, normal, seg200 = load_raw(scene_dir("train", sc))
            gfs = V.convert_to_gfs(seg200)
            f = feat_gpu(model, coord, color, normal)

            idx_b, lab_b = [], []
            for c in range(NO):
                w = np.where(gfs == c)[0]
                if len(w) == 0:
                    continue
                if len(w) > args.per_scene:
                    w = rng.choice(w, args.per_scene, replace=False)
                idx_b.append(w)
                lab_b.append(np.full(len(w), c, dtype=np.int32))

            idx_n, lab_n = [], []
            if args.use_pl:
                p = os.path.join(PL_DIR, sc + "_pl.npy")
                if os.path.exists(p):
                    pl = np.load(p).reshape(-1)
                    if len(pl) == len(gfs):
                        n_pl_scene += 1
                        # 训练时看不到非 support 的 Novel GT，非 Base 区域一律按背景处理
                        nonbase = ~((gfs >= 0) & (gfs < NO))
                        for gc in range(NO, NC):
                            w2 = np.where((pl == gc) & nonbase)[0]
                            if len(w2) == 0:
                                continue
                            if len(w2) > args.per_scene:
                                w2 = rng.choice(w2, args.per_scene, replace=False)
                            idx_n.append(w2)
                            lab_n.append(np.full(len(w2), gc, dtype=np.int32))
                            pl_cnt[gc] += len(w2)

            idx = np.concatenate(idx_b + idx_n) if (idx_b or idx_n) else None
            if idx is None or len(idx) == 0:
                continue
            lab = np.concatenate(lab_b + lab_n)
            sel = torch.from_numpy(idx).to(f.device)
            feats.append(f[sel].float().cpu().half())
            gts.append(lab)
            is_novel.append(lab >= NO)
            if (i + 1) % 200 == 0:
                print("  scene %d/%d  %.0fs  累计 %d 点"
                      % (i + 1, len(scenes), time.time() - t0,
                         sum(len(g) for g in gts)), flush=True)

    print("PL 命中场景 %d/%d，伪标签点数 (novel 类):" % (n_pl_scene, len(scenes)),
          flush=True)
    print("  " + str(pl_cnt[NO:].tolist()), flush=True)

    # ---------- Novel：45 个 1-shot support 场景的 GT ----------
    sup = read_support()
    print("Novel support %d 个场景" % len(sup), flush=True)
    t1 = time.time()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for cid, rel in sup:
            gfs_c = base_label_to_gfs(cid)
            d = os.path.join(DATA_HDD, rel)
            coord, color, normal, seg200 = load_raw(d)
            gfs = V.convert_to_gfs(seg200)
            f = feat_gpu(model, coord, color, normal)
            w = np.where(gfs == gfs_c)[0]
            if len(w) == 0:
                print("  !! class %d (%s) 在 %s 里没有点"
                      % (cid, CLASS_LABELS_200[cid], rel), flush=True)
                continue
            sel = torch.from_numpy(w).to(f.device)
            feats.append(f[sel].float().cpu().half())
            gts.append(np.full(len(w), gfs_c, dtype=np.int32))
            is_novel.append(np.ones(len(w), dtype=bool))
    print("  novel %.0fs" % (time.time() - t1), flush=True)

    X = torch.cat(feats, 0).numpy().astype(np.float32)
    y = np.concatenate(gts, 0).astype(np.int64)
    nv = np.concatenate(is_novel, 0)
    print("训练集: %d 点（Base %d / Novel %d）"
          % (len(y), (~nv).sum(), nv.sum()), flush=True)
    binc = np.bincount(y, minlength=NC)
    print("各类点数 0..11 :", binc[:12].tolist(), flush=True)
    print("各类点数 12..56:", binc[12:].tolist(), flush=True)

    mu = torch.tensor(X.mean(0, keepdims=True), device=dev, dtype=torch.float32)
    sd = torch.tensor(X.std(0, keepdims=True) + 1e-6, device=dev, dtype=torch.float32)
    Xt = torch.tensor(X, device=dev)
    yt = torch.tensor(y, dtype=torch.long, device=dev)
    del X

    freq = np.bincount(y, minlength=NC).astype(np.float64)
    wcls = np.where(freq > 0, 1.0 / np.sqrt(np.maximum(freq, 1)), 0.0)
    wcls = wcls / wcls.sum() * NC
    lossf = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(wcls, dtype=torch.float32, device=dev))

    D = Xt.shape[1]
    W = torch.zeros(D, NC, requires_grad=True, device=dev)
    b = torch.zeros(NC, requires_grad=True, device=dev)
    opt = torch.optim.AdamW([W, b], lr=LR, weight_decay=WD)

    n = len(yt)
    bs = 8192
    for ep in range(EPOCH):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for s in range(0, n, bs):
            i = perm[s:s + bs]
            opt.zero_grad()
            loss = lossf((Xt[i] - mu) / sd @ W + b, yt[i])
            loss.backward()
            opt.step()
            tot += float(loss) * len(i)
        if (ep + 1) % 10 == 0:
            print("  ep%d loss=%.4f" % (ep + 1, tot / n), flush=True)
    del Xt, yt

    # ---------- 评估 312 val ----------
    root = os.path.join(DATA_SHM, "val")
    if not os.path.isdir(root):
        root = os.path.join(DATA_HDD, "val")
    scenes = sorted(os.listdir(root))
    if args.val_limit > 0:
        scenes = scenes[:args.val_limit]
    inter = np.zeros(NC)
    union = np.zeros(NC)
    print("评估 %d 个 val 场景" % len(scenes), flush=True)
    t0 = time.time()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i, sc in enumerate(scenes):
            coord, color, normal, seg200 = load_raw(scene_dir("val", sc))
            gfs = V.convert_to_gfs(seg200)
            f = feat_gpu(model, coord, color, normal)
            p = ((f.float() - mu) / sd @ W + b).argmax(1).cpu().numpy()
            valid = gfs >= 0
            p, g = p[valid], gfs[valid]
            for c in range(NC):
                tp = ((p == c) & (g == c)).sum()
                fp = ((p == c) & (g != c)).sum()
                fn = ((p != c) & (g == c)).sum()
                inter[c] += tp
                union[c] += tp + fp + fn
            if (i + 1) % 100 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(scenes), time.time() - t0),
                      flush=True)

    iou = inter / np.maximum(union, 1)
    base = iou[:NO].mean()
    novel = iou[NO:].mean()
    hm = 2 * base * novel / max(base + novel, 1e-9)
    print()
    print("=== Sonata 1232d 线性探针 · 真 1-shot（%s）==="
          % ("+VLM PL(1201 场景)" if args.use_pl else "纯 3D"))
    print("Base  (12 类) = %.4f" % base)
    print("Novel (45 类) = %.4f" % novel)
    print("HM            = %.4f" % hm)
    print("mIoU          = %.4f" % iou.mean())
    print("逐类 IoU:", np.round(iou, 4).tolist())


if __name__ == "__main__":
    main()
