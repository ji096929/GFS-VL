# -*- coding: utf-8 -*-
"""90 秒摸底：冻结 VLM_3D 在 312 个验证场景上的真实 Novel mIoU。

目的：区分两种情况
  情况 1（val Novel >= 0.40）：信息在教师里，损耗在 PS+AI 硬标签压缩 + Stage-2
  情况 2（val Novel ≈ 0.33）：训练集 0.4548 主要是记忆，学生已榨干教师
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/data01/jh/code/GFS-VL")
sys.path.insert(0, "/data01/jh/code/GFS-VL/pointcept/models/PLA")
sys.path.insert(0, "/data01/jh/code/ps_diag")

import vlm_common as V  # noqa: E402

DATA = "/data01/jh/data/scannet200_gfs"
NO, NC = 12, 57


def main():
    cfg, model, _, _, _ = V.build_vlm()
    print("[vlm] ready", flush=True)

    scenes = sorted(os.listdir(os.path.join(DATA, "val")))
    print("val 场景 %d" % len(scenes), flush=True)

    inter = np.zeros(NC, dtype=np.int64)
    union = np.zeros(NC, dtype=np.int64)
    pred_cnt = np.zeros(NC, dtype=np.int64)
    gt_cnt = np.zeros(NC, dtype=np.int64)

    for i, sc in enumerate(scenes):
        d = os.path.join(DATA, "val", sc)
        try:
            coord = np.load(os.path.join(d, "coord.npy")).astype(np.float32)
            color = np.load(os.path.join(d, "color.npy"))
            seg200 = np.load(os.path.join(d, "segment200.npy")).reshape(-1)
        except Exception as e:  # noqa: BLE001
            print("SKIP %s %s" % (sc, e), flush=True)
            continue
        gfs = V.convert_to_gfs(seg200).astype(np.int64)
        try:
            _, raw = V.vlm_forward(model, cfg, coord, color, return_gpu=True)
        except Exception as e:  # noqa: BLE001
            print("FAIL %s %s" % (sc, e), flush=True)
            continue
        p = raw.detach().cpu().numpy().astype(np.int64)
        valid = (gfs >= 0) & (gfs < NC)
        p, g = p[valid], gfs[valid]
        if len(g) == 0:
            continue
        cm = np.bincount(g * NC + p, minlength=NC * NC).reshape(NC, NC)
        inter += np.diag(cm)
        union += cm.sum(0) + cm.sum(1) - np.diag(cm)
        pred_cnt += cm.sum(0)
        gt_cnt += cm.sum(1)
        if (i + 1) % 50 == 0:
            iou = inter / np.maximum(union, 1)
            print("  %d/%d  当前 Novel mIoU=%.4f  Base=%.4f"
                  % (i + 1, len(scenes), iou[NO:].mean(), iou[:NO].mean()),
                  flush=True)

    iou = inter / np.maximum(union, 1)
    print()
    print("=" * 70)
    print("冻结 VLM_3D 在 312 val 上的真实结果（raw argmax，无 PS/AI）")
    print("=" * 70)
    print("Base  (12 类) = %.4f" % iou[:NO].mean())
    print("Novel (45 类) = %.4f" % iou[NO:].mean())
    print("HM            = %.4f"
          % (2 * iou[:NO].mean() * iou[NO:].mean()
             / max(iou[:NO].mean() + iou[NO:].mean(), 1e-9)))
    print("mIoU          = %.4f" % iou.mean())
    print()
    print("对比：")
    print("  训练集 PLmIoU (raw)       = 0.4548")
    print("  学生 sym_lovasz Novel     = 0.3249")
    print("  教师 val Novel            = %.4f" % iou[NO:].mean())
    print()
    print("判定：")
    nv = iou[NO:].mean()
    if nv >= 0.40:
        print("  情况 1：信息在教师里，损耗在 PS+AI 硬标签压缩 + Stage-2")
        print("  -> 方向：密集连续 logits / 特征级蒸馏 + 桶 A 边界校准")
    elif nv >= 0.33:
        print("  情况 2：训练集 0.4548 主要是记忆，学生已榨干教师 90%%+")
        print("  -> 方向：学生端 3D 优化（桶 A Base 吞噬）+ 修 PS 误杀")
    else:
        print("  情况 3：教师 val 比学生更低，说明 Stage-2 训练反而提升了 Novel")
        print("  -> 方向：继续加学生端能力（跨层已证有效）")
    print()
    print("逐类 IoU:", np.round(iou, 4).tolist())


if __name__ == "__main__":
    main()
