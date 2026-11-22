# -*- coding: utf-8 -*-
"""Level 0：零训练验证「EMA 学生仲裁」是否有价值。

设计（按 review 修正）：
    * 测试床：ScanNet++ val 50 场景（完整 GT、学生从没在这些场景上训过 → 零泄漏）
    * 学生：1A0 权重（scpp_base_lovasz_k5/regis1_model_last.pth，本身 HM 0.3140）
    * 仲裁信号：学生的 30 类 softmax 概率（**不用原型相似度**，拆解 3 已证伪 Fisher<1）
    * 评估范围：排除 base 类点（训练时 PL 只写入 label==-1，那里不含 base）

对比三种策略：
    A  屏蔽 base + 余弦熵 Q_ρ
    B1 A + 学生「候选内改判」（Top-K 里选学生最支持的）
    B2 B1 + 学生「否决假阳性」（学生高置信判 base → ignore）
    B3 B2 + 学生「从被删尾部捞回」（学生高置信且落在 Top-K 内）

用法：
    python tools/probe_level0_arbitration.py --num-scenes 50 --topk 2 --rho 0.1
"""
import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/data01/jh/code/GFS-VL")

from pointcept.datasets import build_dataset
from pointcept.engines.defaults import default_config_parser
from pointcept.engines.train import GFS_VL_Trainer
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger

CKPT = "/data01/jh/code/GFS-VL/exp/scannetpp/scpp_base_lovasz_k5/model/regis1_model_last.pth"


class _Mini(GFS_VL_Trainer):
    def __init__(self, cfg, logger, ds):
        self.cfg = cfg
        self.logger = logger

        class _L:
            pass

        self.train_loader = _L()
        self.train_loader.dataset = ds
        self.build_zero_shot_model()


def ent_filter(pred, ent, rho, nb, mask):
    out = pred.clone()
    for c in out[mask].unique():
        ci = int(c)
        if ci < nb:
            continue
        m = (out == c) & mask
        n = int(m.sum())
        if n <= 1:
            continue
        k = max(1, int(n * rho))
        thr = torch.topk(ent[m], k, largest=False).values[-1]
        out[m & (ent > thr)] = -1
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", default="configs/scannetpp/semseg-pt-v3m1-0-gfsregistrain_k5.py")
    ap.add_argument("--num-scenes", type=int, default=50)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--out", default="/data01/jh/data/level0_arbitration.json")
    a = ap.parse_args()

    cfg = default_config_parser(a.config_file, options=None)
    lg = get_root_logger(log_file=None)
    ds = build_dataset(cfg.data.val)
    lg.info("val scenes: %d" % len(ds))
    mini = _Mini(cfg, lg, ds)
    nb = int(cfg.data.num_bases)
    nc = int(cfg.data.num_base_novels)
    W = mini.VLM_3D.task_head.cls_head.weight

    lg.info("building student ...")
    stu = build_model(cfg.model)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    _sd = {k.replace("module.", "", 1): v for k, v in ck["state_dict"].items()}
    miss, unexp = stu.load_state_dict(_sd, strict=False)
    lg.info("student load: missing=%d unexpected=%d" % (len(miss), len(unexp)))
    stu.eval().cuda()

    names = ["A_mask_ent", "B1_arbitrate", "B2_veto_base", "B3_rescue_tail"]
    acc = {k: dict(tp=0, pred=0, gt=0, inter=np.zeros(nc), union=np.zeros(nc)) for k in names}

    n = min(a.num_scenes, len(ds))
    for i in range(n):
        d = ds[i]
        for k, v in list(d.items()):
            if isinstance(v, torch.Tensor):
                d[k] = v.cuda()
        gt = d["segment"].view(-1)
        # 排除 base 类点：训练时 PL 只写 label==-1，那里不含 base
        keep = ~((gt >= 0) & (gt < nb))

        # ---- VLM ----
        vo = mini.VLM_3D(mini.construct_3D_vlm_input(d))
        feats = vo["all_feats"]
        cos_logits = F.normalize(feats.float(), dim=-1) @ W.t()
        cos_nv = cos_logits[:, nb:]
        prob_v = F.softmax(cos_nv, dim=-1)
        ent = -(prob_v * (prob_v + 1e-8).log()).sum(-1)

        # ---- 学生 ----
        so = stu(d)
        s_logits = so["seg_logits"]
        s_prob = F.softmax(s_logits.float(), dim=-1)
        s_conf, s_pred = s_prob.max(dim=-1)

        # ---- A：屏蔽 base + 熵过滤 ----
        predA = cos_nv.argmax(-1) + nb
        predA = ent_filter(predA, ent, a.rho, nb, keep)

        # ---- Top-K 候选 ----
        tp_v, ti_v = prob_v.topk(min(a.topk, prob_v.shape[1]), dim=-1)
        cand = ti_v + nb                                    # (N,K)
        in_cand = (s_pred.unsqueeze(1) == cand).any(dim=1)   # 学生预测落在候选内
        # 学生在候选里的最大概率
        cand_p = s_prob.gather(1, cand)                      # (N,K)
        best_p, best_j = cand_p.max(dim=1)
        cand_best = cand.gather(1, best_j.unsqueeze(1)).squeeze(1)

        strong = s_conf > a.tau

        # B1：候选内改判（保留 A 的裁剪结果，但把 strong & in_cand 的点改成学生预测）
        predB1 = predA.clone()
        take = strong & in_cand
        predB1[take] = s_pred[take]

        # B2：+ 学生高置信判 base → veto
        predB2 = predB1.clone()
        veto = strong & (s_pred < nb)
        predB2[veto] = -1

        # B3：+ 从 A 删掉的尾部捞回（学生强支持某候选且该候选不是 A 的预测）
        predB3 = predB2.clone()
        rescued = (predB3 == -1) & strong & in_cand
        predB3[rescued] = s_pred[rescued]

        for name, pred in (("A_mask_ent", predA), ("B1_arbitrate", predB1),
                           ("B2_veto_base", predB2), ("B3_rescue_tail", predB3)):
            p = pred[keep]
            g = gt[keep]
            sel = p >= nb
            a_ = acc[name]
            a_["tp"] += int((sel & (p == g)).sum())
            a_["pred"] += int(sel.sum())
            a_["gt"] += int(((g >= nb) & (g < nc)).sum())
            for c in range(nb, nc):
                pc = p == c
                gc = g == c
                a_["inter"][c] += int((pc & gc).sum())
                a_["union"][c] += int((pc | gc).sum())

        if (i + 1) % 10 == 0:
            lg.info("[%d/%d] done" % (i + 1, n))

    res = {}
    for name in names:
        a_ = acc[name]
        iou = a_["inter"] / np.maximum(a_["union"], 1)
        res[name] = dict(
            precision=a_["tp"] / max(a_["pred"], 1),
            recall=a_["tp"] / max(a_["gt"], 1),
            novel_miou=float(iou[nb:].mean()),
            pred_points=a_["pred"],
        )
    print("=" * 72)
    print("%-16s %10s %10s %12s %10s" % ("策略", "Precision", "Recall", "Novel mIoU", "预测点数"))
    for name in names:
        r = res[name]
        print("%-16s %10.4f %10.4f %12.4f %10d"
              % (name, r["precision"], r["recall"], r["novel_miou"], r["pred_points"]))
    json.dump(dict(cfg=vars(a), res=res), open(a.out, "w"), ensure_ascii=False, indent=1)
    print("saved:", a.out)


if __name__ == "__main__":
    main()
