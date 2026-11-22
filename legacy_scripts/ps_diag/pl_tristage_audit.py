# -*- coding: utf-8 -*-
"""拆解 1+2：VLM 伪标签三阶段的全场景体检 + 512d 特征 Fisher 比。

与 dump_online_pl.py 的区别：
  1. 保留 Step 1（VLM 原始 argmax）的统计
  2. 全场景统计（不再只算 Y_b==-1 的区域），把 143 类背景误判全部计入 FP
  3. 同时累积 512d 特征的 per-class 均值/二阶矩，用于算 Fisher 比

三阶段定义：
  S1 = raw        VLM 文本头 argmax
  S2 = ps         pseudo_label_selection（ps_thresh=0.6）后
  S3 = psai       Adaptive Infilling（ai_thresh=0.9）后

用法: python pl_tristage_audit.py [--limit N] [--shard i --shards M]
输出: /data01/jh/data/cm_analysis/pl_tristage_audit.npz
"""
import argparse
import json
import os
import sys

import numpy as np

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


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def stage_stats(pl, gfs):
    """全场景统计：把 PL>=12 的预测与该点 GT 比较；gfs==-1 的点若被预测成 novel 就是 FP。"""
    cm = np.zeros((NN, NN), dtype=np.int64)     # 行=GT novel, 列=预测 novel
    fp_bg = 0                                    # GT 为 -1（143 类背景）却被预测成 novel
    pred_novel = pl >= NO
    n_pred = int(pred_novel.sum())
    # GT novel 区域
    gt_nov = (gfs >= NO) & (gfs <= NO + NN - 1)
    if gt_nov.any():
        g = gfs[gt_nov] - NO
        p = np.where(pred_novel[gt_nov], pl[gt_nov] - NO, NN)  # NN = 未预测为 novel
        cm += np.bincount(g * (NN + 1) + p,
                          minlength=NN * (NN + 1)).reshape(NN, NN + 1)[:, :NN]
    # 143 类背景 / ignore 区域
    bg = gfs == -1
    if bg.any():
        fp_bg = int((pred_novel & bg).sum())
    tp = int(np.diag(cm).sum())
    gt_n = int(gt_nov.sum())
    precision = tp / max(n_pred, 1)
    recall = tp / max(gt_n, 1)
    iou = np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1) - np.diag(cm), 1)
    return dict(cm=cm, tp=tp, n_pred=n_pred, n_gt=gt_n, fp_bg=fp_bg,
                precision=precision, recall=recall, miou=float(iou.mean()),
                per_class_iou=iou)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()

    pn = unit(np.load(PROTO).astype(np.float32))
    cfg, model, _, _, _ = V.build_vlm()
    print("[vlm] ready", flush=True)

    scenes = sorted(os.listdir(os.path.join(DATA, "train")))
    mine = scenes[args.shard::args.shards]
    if args.limit:
        mine = mine[:args.limit]
    print("本片 %d 场景" % len(mine), flush=True)

    names = ["raw", "ps", "psai"]
    agg = {n: dict(cm=np.zeros((NN, NN), dtype=np.int64), tp=0, n_pred=0,
                   n_gt=0, fp_bg=0) for n in names}
    # 512d 特征的一阶/二阶矩（只为 novel 类 + 全背景）
    feat_sum = np.zeros((NO + NN + 1, 512), dtype=np.float64)
    feat_sq = np.zeros((NO + NN + 1, 512), dtype=np.float64)
    feat_cnt = np.zeros(NO + NN + 1, dtype=np.int64)

    for i, sc in enumerate(mine):
        d = os.path.join(DATA, "train", sc)
        try:
            coord = np.load(os.path.join(d, "coord.npy")).astype(np.float32)
            color = np.load(os.path.join(d, "color.npy"))
            seg200 = np.load(os.path.join(d, "segment200.npy")).reshape(-1)
        except Exception as e:  # noqa: BLE001
            print("SKIP %s %s" % (sc, e), flush=True)
            continue
        gfs = V.convert_to_gfs(seg200).astype(np.int64)
        lab = np.where((gfs >= 0) & (gfs < NO), gfs, -1)

        try:
            feats, raw = V.vlm_forward(model, cfg, coord, color)
        except Exception as e:  # noqa: BLE001
            print("FAIL vlm %s %s" % (sc, e), flush=True)
            continue
        fn = unit(feats)

        # ---- S1 ----
        s1 = raw.astype(np.int16).copy()
        # ---- S2: PS ----
        s2 = s1.copy()
        for c in np.unique(s1):
            c = int(c)
            m = s1 == c
            if c < NO:
                s2[m] = -1
                continue
            proto = unit(feats[m].mean(0))
            if float(proto @ pn[c - NO]) < PS_THRESH:
                s2[m] = -1
        # ---- S3: AI ----
        bg = lab == -1
        cur = {}
        for c in np.unique(s2[bg]):
            c = int(c)
            if c < NO:
                continue
            cur[c] = feats[bg][s2[bg] == c].mean(0)
        protos = np.empty((NN, feats.shape[1]), dtype=np.float32)
        for j in range(NN):
            p = cur.get(j + NO)
            protos[j] = p if p is not None else pn[j]
        protos = unit(protos)
        s3 = s2.copy()
        m_ai = (s2 == -1) & bg
        if m_ai.any():
            sim = fn[m_ai] @ protos.T
            best = sim.max(1)
            cls = sim.argmax(1) + NO
            s3[m_ai] = np.where(best >= AI_THRESH, cls, -1).astype(np.int16)

        # ---- 累积三阶段的全场景指标 ----
        for nm, pl in (("raw", s1), ("ps", s2), ("psai", s3)):
            st = stage_stats(pl, gfs)
            a = agg[nm]
            a["cm"] += st["cm"]
            a["tp"] += st["tp"]
            a["n_pred"] += st["n_pred"]
            a["n_gt"] += st["n_gt"]
            a["fp_bg"] += st["fp_bg"]

        # ---- 512d 特征矩（按 GT 的 gfs 分组，-1 归到最后一档）----
        # 降采样：全场景 20 万点 × 512 的两次归约会分配 400MB 临时数组，
        # 是纯 CPU 瓶颈（实测 GPU 0% / CPU 98%）。每场景采 2 万点在统计上足够。
        g_idx = np.where(gfs < 0, NO + NN, gfs)
        _cap = 20000
        if len(g_idx) > _cap:
            _sel = np.random.RandomState(i).choice(len(g_idx), _cap, replace=False)
            _g_sub = g_idx[_sel]
            _f_sub = feats[_sel]
        else:
            _g_sub = g_idx
            _f_sub = feats
        # 一次性分组的矩：避免逐类重复分配临时数组
        _fs = _f_sub.astype(np.float64)
        _sq = (_fs * _fs)
        _order = np.argsort(_g_sub, kind="stable")
        _gs = _g_sub[_order]
        _bounds = np.searchsorted(_gs, np.arange(NO + NN + 2))
        for j in range(NO + NN + 1):
            lo, hi = _bounds[j], _bounds[j + 1]
            if hi <= lo:
                continue
            _blk = _fs[_order[lo:hi]]
            feat_sum[j] += _blk.sum(0)
            feat_sq[j] += (_blk * _blk).sum(0)
            feat_cnt[j] += hi - lo

        if (i + 1) % 50 == 0:
            print("  %d/%d  S1 P=%.3f R=%.3f  S3 P=%.3f R=%.3f"
                  % (i + 1, len(mine),
                     agg["raw"]["tp"] / max(agg["raw"]["n_pred"], 1),
                     agg["raw"]["tp"] / max(agg["raw"]["n_gt"], 1),
                     agg["psai"]["tp"] / max(agg["psai"]["n_pred"], 1),
                     agg["psai"]["tp"] / max(agg["psai"]["n_gt"], 1)),
                  flush=True)

    out = {}
    for nm in names:
        a = agg[nm]
        out[nm + "_cm"] = a["cm"]
        out[nm + "_fp_bg"] = np.array([a["fp_bg"]])
        out[nm + "_n_pred"] = np.array([a["n_pred"]])
        out[nm + "_n_gt"] = np.array([a["n_gt"]])
        out[nm + "_tp"] = np.array([a["tp"]])
    out["feat_sum"] = feat_sum
    out["feat_sq"] = feat_sq
    out["feat_cnt"] = feat_cnt
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "pl_tristage_audit_s%d.npz" % args.shard)
    np.savez_compressed(p, **out)
    print("-> %s" % p, flush=True)

    # 汇总打印
    print()
    print("%-6s %8s %8s %8s %10s %8s" % ("stage", "Prec", "Rec", "PLmIoU", "FP(bg143)", "n_pred"))
    for nm in names:
        a = agg[nm]
        prec = a["tp"] / max(a["n_pred"], 1)
        rec = a["tp"] / max(a["n_gt"], 1)
        cm = a["cm"]
        iou = np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1) - np.diag(cm), 1)
        print("%-6s %8.4f %8.4f %8.4f %10d %8d"
              % (nm, prec, rec, iou.mean(), a["fp_bg"], a["n_pred"]))


if __name__ == "__main__":
    main()
