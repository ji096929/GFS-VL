# -*- coding: utf-8 -*-
"""Track 2 Step C 体检：Base mIoU + dec0 的 eRank（新类点上）。

用法
  python ontr_step_c.py --weight <ckpt> --tag <名字>

口径（与训练 val transform 完全一致）
  * 场景：312 val 全量
  * 预处理：CenterShift -> GridSample(0.02, fnv) -> NormalizeColor -> ToTensor
    （grid_coord 用 fnv hash，与 Stage-1/Stage-2 训练口径相同）
  * segment200 与点一起下采样，保证逐点对齐
  * Base mIoU：12 类
  * eRank：只在真实新类点（GFS 标签 12..56）上统计 dec0 特征
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/data01/jh/code/GFS-VL")
sys.path.insert(0, "/data01/jh/code/GFS-VL/pointcept/models/PLA")
sys.path.insert(0, "/data01/jh/code/ps_diag")

import vlm_common as V  # noqa: E402
from pointcept.datasets import collate_fn  # noqa: E402
from pointcept.datasets.transform import (  # noqa: E402
    CenterShift, Collect, Compose, GridSample, NormalizeColor, ToTensor)
from pointcept.models import build_model  # noqa: E402
from pointcept.models.utils.structure import Point  # noqa: E402
from pointcept.utils.config import Config  # noqa: E402

ROOT = "/data01/jh/code/GFS-VL"
CFG = ROOT + "/configs/scannet200/semseg-pt-v3m1-0-gfspretrain.py"
DATA = "/data01/jh/data/scannet200_gfs"
NB, NN = 12, 45


def erank(x):
    x = x.float()
    x = x - x.mean(0, keepdim=True)
    c = (x.T @ x) / max(x.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(c).clamp_min(1e-12)
    p = ev / ev.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def build_tf():
    return Compose([
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train",
             return_grid_coord=True, return_inverse=True,
             keys=("coord", "color", "normal", "segment200")),
        dict(type="CenterShift", apply_z=False),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(type="Collect",
             keys=("coord", "grid_coord", "segment200", "inverse"),
             feat_keys=("color", "normal")),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--max-scene", type=int, default=0, help="0 表示全量 312")
    args = ap.parse_args()

    cfg = Config.fromfile(CFG)
    cfg.model.backbone.enable_flash = False

    model = build_model(cfg.model)
    ck = torch.load(args.weight, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print("load missing=%d unexpected=%d" % (len(miss), len(unexp)), flush=True)
    model.eval().cuda()

    tf = build_tf()
    scenes = sorted(os.listdir(os.path.join(DATA, "val")))
    if args.max_scene > 0:
        scenes = scenes[: args.max_scene]
    print("评估场景数: %d" % len(scenes), flush=True)

    inter = np.zeros(NB, np.float64)
    union = np.zeros(NB, np.float64)
    novel_feats = []
    all_feats = []          # 全体点采样，用于子空间尺度诊断
    n_novel_pts = 0
    n_base_pts = 0

    with torch.no_grad():
        for i, sc in enumerate(scenes):
            d0 = os.path.join(DATA, "val", sc)
            raw = {
                "coord": np.load(os.path.join(d0, "coord.npy")).astype(np.float32),
                "color": np.load(os.path.join(d0, "color.npy")),
                "normal": np.load(os.path.join(d0, "normal.npy")),
                "segment200": np.load(os.path.join(d0, "segment200.npy")).reshape(-1),
                "name": sc,
            }
            raw["segment200"] = raw["segment200"].astype(np.int64)
            # GridSample 会原地改写 raw 里的字段，必须先留一份原点级 GT
            seg200_orig = raw["segment200"].copy()
            # 固定 voxel 代表点的随机选择，保证可复现
            np.random.seed(0)
            item = tf(raw)
            seg200_v = np.asarray(item["segment200"]).astype(np.int64)   # voxel 级
            inv = np.asarray(item["inverse"]).astype(np.int64)           # (原点数,) -> voxel 索引
            gfs_v = V.convert_to_gfs(seg200_v)                           # voxel 级 GFS 标签
            gfs_full = V.convert_to_gfs(seg200_orig)                     # 原点级 GFS 标签
            base_full = np.where((gfs_full >= 0) & (gfs_full < NB), gfs_full, -1)

            inp = collate_fn([item])
            for k in inp:
                if isinstance(inp[k], torch.Tensor):
                    inp[k] = inp[k].cuda(non_blocking=True)
            out = model.backbone(Point(inp))
            h0 = (out.feat if hasattr(out, "feat") else out).float()
            pred_v = model.seg_head(h0).float().argmax(1).cpu().numpy()
            # 预测散射回原点云，IoU 在原点云上统计（与 SemSegTester 口径一致）
            pred_full = pred_v[inv]

            for c in range(NB):
                g = base_full == c
                p = pred_full == c
                inter[c] += float((p & g).sum())
                union[c] += float((p | g).sum())
            n_base_pts += int((base_full >= 0).sum())

            m = gfs_v >= NB
            if m.any():
                novel_feats.append(h0[torch.from_numpy(m).cuda()].cpu())
                n_novel_pts += int(m.sum())
            # 全体点采样（诊断用），固定 2000 点/场景
            k = min(2000, h0.shape[0])
            ia = torch.randperm(h0.shape[0], device=h0.device)[:k]
            all_feats.append(h0[ia].cpu())

            if (i + 1) % 50 == 0:
                print("  %d/%d" % (i + 1, len(scenes)), flush=True)

    iou = inter / np.maximum(union, 1e-9)
    print()
    print("=== %s ===" % args.tag, flush=True)
    print("Base mIoU (12 类) = %.4f" % iou.mean(), flush=True)
    print("逐类 IoU:", np.round(iou, 4).tolist(), flush=True)
    print("Base 点 %d / 新类点 %d" % (n_base_pts, n_novel_pts), flush=True)

    if novel_feats:
        X = torch.cat(novel_feats, 0)
        if X.shape[0] > 60000:
            X = X[torch.randperm(X.shape[0])[:60000]]
        print("新类点 dec0 eRank = %.2f / 64  (样本 %d 点)"
              % (erank(X), X.shape[0]), flush=True)

    # ---------- 子空间尺度诊断 ----------
    # 总 eRank 会被基类子空间的大奇异值压住，真正该看的是零空间内部的秩。
    if all_feats:
        H = torch.cat(all_feats, 0).float()
        Wb = model.seg_head.weight.detach().float().cpu()
        U, S, _ = torch.linalg.svd(Wb.T, full_matrices=True)
        Qb, Qp = U[:, :NB], U[:, NB:]
        Hb, Hp = H @ Qb, H @ Qp
        rms_b = Hb.pow(2).mean().sqrt().item()
        rms_p = Hp.pow(2).mean().sqrt().item()
        print()
        print("--- 子空间尺度诊断（全体点 %d）---" % H.shape[0], flush=True)
        print("RMS(h0 @ Q_b)     = %.4f" % rms_b, flush=True)
        print("RMS(h0 @ Q_perp)  = %.4f   比值 Q_b/Q_perp = %.2f"
              % (rms_p, rms_b / max(rms_p, 1e-9)), flush=True)
        print("eRank(h0)            = %.2f / 64" % erank(H), flush=True)
        print("eRank(h0 @ Q_perp)   = %.2f / 52   <- 纯零空间秩"
              % erank(Hp), flush=True)
        print("eRank(h0 @ Q_b)      = %.2f / 12" % erank(Hb), flush=True)
        print("eRank(LN(h0))        = %.2f / 64"
              % erank(torch.nn.functional.layer_norm(H, (64,))), flush=True)
        # 零空间内部的秩，按维度归一化后与 dec0 原始对比
        print("零空间秩占满额比     = %.1f%%" % (100 * erank(Hp) / 52), flush=True)


if __name__ == "__main__":
    main()
