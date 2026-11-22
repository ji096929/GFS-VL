# -*- coding: utf-8 -*-
"""摸底 0：VLM 原始 seg_scores 的 Top-1/2/3 Recall（只统计 Novel 类）。

目的：验证「Top-K 候选约束」在我们数据上是否成立（Beyond-Argmax 假设）。
      如果 Top-2 相对 Top-1 大涨，方向 2 有物理基础；如果几乎不涨，直接省掉。

口径：
    * 只统计 GT 为 novel 类（segment >= num_bases）的点。
    * 命中定义：VLM seg_scores 的 top-k 预测里包含真实类。
    * 复用训练时的同一套数据增强（训练时 PL 也是在这些增强点上生成的）。

用法：
    python tools/probe_topk_recall.py \
        --config-file configs/scannetpp/semseg-pt-v3m1-0-gfsregistrain_k5.py \
        --num-samples 50 --out /data01/jh/data/probe_topk_scpp.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/data01/jh/code/GFS-VL")

from pointcept.engines.defaults import default_config_parser
from pointcept.datasets import build_dataset
from pointcept.engines.train import GFS_VL_Trainer
from pointcept.utils.logger import get_root_logger


class _Mini(GFS_VL_Trainer):
    """只借用 build_zero_shot_model，不跑 Trainer.__init__（省掉 optimizer/dataloader）。"""

    def __init__(self, cfg, logger, dataset):
        self.cfg = cfg
        self.logger = logger

        class _L:
            pass

        self.train_loader = _L()
        self.train_loader.dataset = dataset
        self.build_zero_shot_model()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--out", default="/data01/jh/data/probe_topk_recall.json")
    a = ap.parse_args()

    cfg = default_config_parser(a.config_file, options=None)
    logger = get_root_logger(log_file=None)
    logger.info("=> building dataset ...")
    dataset = build_dataset(cfg.data.train)
    logger.info("train samples: %d" % len(dataset))

    logger.info("=> building VLM ...")
    mini = _Mini(cfg, logger, dataset)

    nb = int(cfg.data.num_bases)
    nc = int(cfg.data.num_base_novels)
    logger.info("num_bases=%d, num_base_novels=%d" % (nb, nc))

    n = min(a.num_samples, len(dataset))
    hit = np.zeros(3, dtype=np.int64)
    tot = 0
    # raw argmax 口径：预测为 novel 的集合 vs GT novel 集合
    n_raw_pred = 0          # 所有原始 novel 预测点数
    n_raw_correct = 0       # 其中正确的
    n_gt_novel = 0          # GT novel 点数（已被 tot 覆盖，单列便于阅）
    n_recalled = 0          # GT novel 里被正确预测的
    n_stolen_by_base = 0    # GT novel 点的 Top-1 被 12 个 Base 类抢走
    hit_masked = np.zeros(3, dtype=np.int64)   # 屏蔽 base 类后只看 18 个 novel
    per_cls_hit = {}
    per_cls_tot = {}

    for i in range(n):
        d = dataset[i]
        for k, v in list(d.items()):
            if isinstance(v, torch.Tensor):
                d[k] = v.cuda()
        data_dict = mini.construct_3D_vlm_input(d)
        out = mini.VLM_3D(data_dict)

        scores = out.get("seg_scores", None)
        if scores is None:
            scores = mini.VLM_3D.task_head.forward_ret_dict["seg_scores"]
        gt = d["segment"].view(-1)

        valid = (gt >= nb) & (gt < nc)
        if not valid.any():
            continue
        s = scores[valid]
        g = gt[valid]
        topk = s.topk(3, dim=-1).indices            # (M, 3)
        for kk in range(3):
            h = (topk[:, : kk + 1] == g[:, None]).any(dim=1)
            hit[kk] += int(h.sum())
        tot += int(valid.sum())

        # raw argmax 口径（与 target_calibrate 里 PS 之前的输入一致）
        raw = s.argmax(dim=-1)
        rp = (raw >= nb) & (raw < nc)
        n_raw_pred += int(rp.sum())
        n_raw_correct += int((rp & (raw == g)).sum())
        n_gt_novel += int(valid.sum())
        n_recalled += int((raw == g).sum())
        n_stolen_by_base += int((raw < nb).sum())

        # 屏蔽 base 类：只在 seg_scores[:, nb:] 里选
        s_nv = s[:, nb:]
        tk_nv = s_nv.topk(min(3, s_nv.shape[1]), dim=-1).indices + nb
        for kk in range(3):
            h = (tk_nv[:, : kk + 1] == g[:, None]).any(dim=1)
            hit_masked[kk] += int(h.sum())

        for c in g.unique():
            ci = int(c.item())
            m = g == c
            per_cls_tot[ci] = per_cls_tot.get(ci, 0) + int(m.sum())
            for kk in range(3):
                h = (topk[m, : kk + 1] == g[m, None]).any(dim=1)
                per_cls_hit.setdefault(ci, np.zeros(3, dtype=np.int64))
                per_cls_hit[ci][kk] += int(h.sum())
        if (i + 1) % 10 == 0:
            logger.info("[%d/%d] novel pts=%d  top1=%.4f top2=%.4f top3=%.4f"
                        % (i + 1, n, tot, hit[0] / max(tot, 1),
                           hit[1] / max(tot, 1), hit[2] / max(tot, 1)))

    res = dict(
        dataset=os.path.basename(a.config_file),
        num_samples=n, novel_points=tot,
        top1=hit[0] / max(tot, 1),
        top2=hit[1] / max(tot, 1),
        top3=hit[2] / max(tot, 1),
        delta_top2_minus_top1=(hit[1] - hit[0]) / max(tot, 1),
        delta_top3_minus_top1=(hit[2] - hit[0]) / max(tot, 1),
        # raw argmax 口径：PS 之前的真实起点
        raw_pred_points=n_raw_pred,
        raw_novel_precision=n_raw_correct / max(n_raw_pred, 1),
        raw_novel_recall=n_recalled / max(n_gt_novel, 1),
        raw_novel_coverage=n_raw_pred / max(n_gt_novel, 1),
        stolen_by_base_ratio=n_stolen_by_base / max(n_gt_novel, 1),
        masked_top1=hit_masked[0] / max(tot, 1),
        masked_top2=hit_masked[1] / max(tot, 1),
        masked_top3=hit_masked[2] / max(tot, 1),
        masked_delta_top1=hit_masked[0] / max(tot, 1) - hit[0] / max(tot, 1),
        per_class={str(c): dict(
            n=per_cls_tot[c],
            top1=float(per_cls_hit[c][0] / max(per_cls_tot[c], 1)),
            top2=float(per_cls_hit[c][1] / max(per_cls_tot[c], 1)),
            top3=float(per_cls_hit[c][2] / max(per_cls_tot[c], 1)))
            for c in sorted(per_cls_tot)},
    )
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    print("=" * 60)
    print("Novel points: %d" % tot)
    print("Top-1 %.4f | Top-2 %.4f (+%.4f) | Top-3 %.4f (+%.4f)"
          % (res["top1"], res["top2"], res["delta_top2_minus_top1"],
             res["top3"], res["delta_top3_minus_top1"]))
    print("raw argmax: 预测novel %d 点 | precision %.4f | recall %.4f | coverage %.4f"
          % (n_raw_pred, res["raw_novel_precision"], res["raw_novel_recall"],
             res["raw_novel_coverage"]))
    print("Top-1 被 Base 类抢走: %.4f of GT novel" % res["stolen_by_base_ratio"])
    print("屏蔽 base 后: Top-1 %.4f (Δ%+.4f) | Top-2 %.4f | Top-3 %.4f"
          % (res["masked_top1"], res["masked_delta_top1"],
             res["masked_top2"], res["masked_top3"]))
    print("saved:", a.out)


if __name__ == "__main__":
    main()
