# -*- coding: utf-8 -*-
"""拆解 1+2（全 GPU 版）：VLM 伪标签三阶段全场景体检 + 512d 特征 Fisher 矩。

相比 CPU 版的关键改动：
  - vlm_forward(return_gpu=True)，特征全程不落 CPU（避免 400MB/场景 的往返）
  - PS 的原型用 index_add_ 一次算完（57 次 O(N) 归约 → 1 次）
  - 混淆矩阵用 torch.bincount（GPU）
  - 矩累积用 index_add_，全点精确，无降采样

三阶段：S1=raw  S2=ps(ps_thresh=0.6)  S3=psai(ai_thresh=0.9)
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/data01/jh/code/GFS-VL")
sys.path.insert(0, "/data01/jh/code/GFS-VL/pointcept/models/PLA")
sys.path.insert(0, "/data01/jh/code/ps_diag")

import vlm_common as V  # noqa: E402

DATA = "/data01/jh/data/scannet200_gfs"
PROTO = "/data01/jh/data/support_protos/sc200_regis_1_10.npy"
NO, NN = 12, 45
PS_THRESH = 0.6
AI_THRESH = 0.9
OUT = "/data01/jh/data/cm_analysis"
NB = NO + NN + 1          # 57 类 + 1 个"背景/ignore"桶


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()

    dev = "cuda"
    pn = F.normalize(torch.from_numpy(np.load(PROTO).astype(np.float32)).to(dev), dim=1)
    cfg, model, _, _, _ = V.build_vlm()
    print("[vlm] ready", flush=True)

    scenes = sorted(os.listdir(os.path.join(DATA, "train")))
    mine = scenes[args.shard::args.shards]
    if args.limit:
        mine = mine[:args.limit]
    print("本片 %d 场景" % len(mine), flush=True)

    names = ["raw", "ps", "psai"]
    agg_cm = {n: torch.zeros(NN, NN, dtype=torch.int64, device=dev) for n in names}
    agg_fp = {n: 0 for n in names}
    agg_np = {n: 0 for n in names}
    agg_tp = {n: 0 for n in names}
    agg_ngt = 0
    # 512d 特征矩（全点精确，无降采样）
    feat_sum = torch.zeros(NB, 512, dtype=torch.float64, device=dev)
    feat_sq = torch.zeros(NB, 512, dtype=torch.float64, device=dev)
    feat_cnt = torch.zeros(NB, dtype=torch.float64, device=dev)
    bg_bucket = NO + NN

    for i, sc in enumerate(mine):
        d = os.path.join(DATA, "train", sc)
        try:
            coord = np.load(os.path.join(d, "coord.npy")).astype(np.float32)
            color = np.load(os.path.join(d, "color.npy"))
            seg200 = np.load(os.path.join(d, "segment200.npy")).reshape(-1)
        except Exception as e:  # noqa: BLE001
            print("SKIP %s %s" % (sc, e), flush=True)
            continue
        gfs_np = V.convert_to_gfs(seg200).astype(np.int64)
        try:
            feats, raw = V.vlm_forward(model, cfg, coord, color, return_gpu=True)
        except Exception as e:  # noqa: BLE001
            print("FAIL vlm %s %s" % (sc, e), flush=True)
            continue

        N = feats.shape[0]
        gfs = torch.from_numpy(gfs_np).to(dev)
        ngt = int(((gfs >= NO) & (gfs < NO + NN)).sum())
        agg_ngt += ngt

        fn = F.normalize(feats, dim=1)
        s1 = raw.to(torch.int64)
        lab = torch.where((gfs >= 0) & (gfs < NO), gfs, torch.full_like(gfs, -1))

        # ---- S2: PS。原型用 index_add_ 一次算完 ----
        uniq, inv = torch.unique(s1, return_inverse=True)
        sums = torch.zeros(len(uniq), feats.shape[1], dtype=torch.float32, device=dev)
        sums.index_add_(0, inv, feats)
        cnts = torch.bincount(inv, minlength=len(uniq)).to(torch.float32)
        proto_n = F.normalize(sums / cnts[:, None], dim=1)
        s2 = s1.clone()
        uv = uniq.tolist()
        for j, c in enumerate(uv):
            m = s1 == c
            if c < NO:
                s2[m] = -1
            elif float(proto_n[j] @ pn[c - NO]) < PS_THRESH:
                s2[m] = -1

        # ---- S3: AI ----
        bg = lab == -1
        cur = {}
        u2, inv2 = torch.unique(s2[bg], return_inverse=True)
        s2bg = s2[bg]
        sums2 = torch.zeros(len(u2), feats.shape[1], dtype=torch.float32, device=dev)
        sums2.index_add_(0, inv2, feats[bg])
        cnts2 = torch.bincount(inv2, minlength=len(u2)).to(torch.float32)
        protos_cur = sums2 / cnts2[:, None]
        for j, c in enumerate(u2.tolist()):
            if c >= NO:
                cur[c] = protos_cur[j]
        protos = torch.empty(NN, feats.shape[1], dtype=torch.float32, device=dev)
        for j in range(NN):
            protos[j] = cur.get(j + NO, pn[j])
        protos = F.normalize(protos, dim=1)

        m_ai = (s2 == -1) & bg
        s3 = s2.clone()
        if m_ai.any():
            sim = fn[m_ai] @ protos.T
            best = sim.max(1).values
            cls = sim.argmax(1) + NO
            s3[m_ai] = torch.where(best >= AI_THRESH, cls, torch.full_like(cls, -1))

        # ---- 三阶段统计 ----
        gt_nov = (gfs >= NO) & (gfs < NO + NN)
        for nm, pl in (("raw", s1), ("ps", s2), ("psai", s3)):
            pred_nov = pl >= NO
            if gt_nov.any():
                g = gfs[gt_nov] - NO
                p = torch.where(pred_nov[gt_nov], pl[gt_nov] - NO,
                                torch.full_like(pl[gt_nov], NN))
                agg_cm[nm] += torch.bincount(
                    g * (NN + 1) + p, minlength=NN * (NN + 1)
                ).reshape(NN, NN + 1)[:, :NN]
            agg_fp[nm] += int((pred_nov & (gfs == -1)).sum())
            agg_np[nm] += int(pred_nov.sum())
            agg_tp[nm] += int(((pl == gfs) & gt_nov).sum())

        # ---- 512d 特征矩（全点）----
        g_idx = torch.where(gfs < 0, torch.full_like(gfs, bg_bucket), gfs)
        fd = feats.to(torch.float64)
        feat_sum.index_add_(0, g_idx, fd)
        feat_sq.index_add_(0, g_idx, fd.square())
        feat_cnt.index_add_(0, g_idx, torch.ones(N, dtype=torch.float64, device=dev))
        del fd

        if (i + 1) % 50 == 0:
            print("  %d/%d  S1 P=%.3f R=%.3f  S3 P=%.3f R=%.3f"
                  % (i + 1, len(mine),
                     agg_tp["raw"] / max(agg_np["raw"], 1),
                     agg_tp["raw"] / max(agg_ngt, 1),
                     agg_tp["psai"] / max(agg_np["psai"], 1),
                     agg_tp["psai"] / max(agg_ngt, 1)), flush=True)

    out = {}
    for nm in names:
        cm = agg_cm[nm].cpu().numpy()
        out[nm + "_cm"] = cm
        out[nm + "_fp_bg"] = np.array([agg_fp[nm]])
        out[nm + "_n_pred"] = np.array([agg_np[nm]])
        out[nm + "_tp"] = np.array([agg_tp[nm]])
    out["n_gt"] = np.array([agg_ngt])
    out["feat_sum"] = feat_sum.cpu().numpy()
    out["feat_sq"] = feat_sq.cpu().numpy()
    out["feat_cnt"] = feat_cnt.cpu().numpy()
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "pl_tristage_audit_s%d.npz" % args.shard)
    np.savez_compressed(p, **out)
    print("-> %s" % p, flush=True)

    print()
    print("%-6s %8s %8s %8s %10s %10s" % ("stage", "Prec", "Rec", "PLmIoU", "FP(bg143)", "n_pred"))
    for nm in names:
        cm = agg_cm[nm].cpu().numpy().astype(np.float64)
        iou = np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1) - np.diag(cm), 1)
        print("%-6s %8.4f %8.4f %8.4f %10d %10d"
              % (nm, agg_tp[nm] / max(agg_np[nm], 1),
                 agg_tp[nm] / max(agg_ngt, 1), iou.mean(),
                 agg_fp[nm], agg_np[nm]))


if __name__ == "__main__":
    main()
