#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段一 v2：七信号精确判别力（修复 v1 的量化误差）。

【v1 的两个致命 bug，2026-11-23 复检确认】

Bug 1 — 固定直方图边界吃掉了全部信号
  v1 用 `[-3, 3]` 分 100 格（每格宽 0.06），但实测非零格数：
      S1 负熵        18 个类每类只有 1 格（全挤在 [-2.94, -2.88]）
      S2 文本边际    每类 5-10 格
      S3 原型相似度  每类 5-13 格
      S4 非 base 概率 每类 17 格
  把所有正负样本塞进 1 个格子里算 AUROC，数学上恒等 0.500；塞进 5-17 个
  格子里算，「真实区分力」会被量化噪声大幅稀释。
  **v1 的「四信号全灭」是尺子坏了，不是信号坏了。**

Bug 2 — S3 原型提取漏了标准预处理，索引方式也不对
  v1 手写 np.load 读原始 coord/color（color 还是 0-255 未归一化、coord 未
  CenterShift），跳过 GridSample；且相似度取「18 类 max」而非当前预测类
  `proto[pred]`。正式训练走的是 `extract_3d_vlm_proto(regis_val_loader)`。

【v2 的修法】

1. 不用直方图。每 chunk 每类每极性随机抽 CAP 个点，保留原始 float32 分数，
   merge 后按精确排序算 AUROC（Mann-Whitney U，ties 取平均秩）。
   单 chunk 内随机抽样无偏，合并后 AUROC 估计无偏。
2. S3 直接复用 `GFS_VL_Trainer.extract_3d_vlm_proto`（走标准 dataset 管线）。
3. 补 S4b / S4c。因为「真 smoke detector 贴在天花板上」，S4a
   `1 - P_base` 会把真 smoke 判成 base，必须同时看 3D 学生对 `pred` 的
   置信度本身。

七个信号（统一成「分数越高越可能是真点」）
    S1  -H(softmax(cos_nv))                       1C-a / 1C-e 现用
    S2  cos_nv_max - cos_base_max                 文本边际
    S3a cos(feat, proto[pred])                    预测类原型相似度
    S3b S3a - max_{c!=pred} cos(feat, proto[c])   原型边际
    S4a 1 - max_base P_3D(base|x)                 非 base 概率
    S4b P_3D(pred|x)                              3D 学生认可 pred
    S4c S4b - max_base P_3D(base|x)               双向净边际

两个指标
    AUROC          类内判别力（对应「类内 top-10% 提纯」）
    Precision@10%  取分数最高的 10% 候选点，其中正样本占比
                   （实际过滤场景的纯度；随机基线 = 该类正样本率）

用法
    # 1) 先生成 5-shot 原型缓存（单卡，约 1-2 分钟）
    python tools/probe_signal_auroc_v2.py --build-proto
    # 2) 8 卡分片扫描
    for i in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$i python tools/probe_signal_auroc_v2.py \
            --shard $i --shards 8 &
    done
    # 3) 合并出表
    python tools/probe_signal_auroc_v2.py --merge --shards 8
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/data01/jh/code/GFS-VL")

from pointcept.datasets import build_dataset, collate_fn
from pointcept.engines.defaults import default_config_parser
from pointcept.engines.train import GFS_VL_Trainer
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger

OUT_DIR = "/data01/jh/data/step1_auroc_v2"
PROTO = os.path.join(OUT_DIR, "proto_5shot.npz")
STU_CKPT = (
    "/data01/jh/code/GFS-VL/exp/scannetpp/scpp_base_lovasz_k5"
    "/model/regis1_model_last.pth"
)
CAP = 192  # 每 chunk 每类每极性最多保留的点数
SIGNALS = (
    "S1_negent",
    "S2_margin",
    "S3a_proto_pred",
    "S3b_proto_margin",
    "S4a_nonbase",
    "S4b_pred",
    "S4c_netmargin",
)
NAMES = [
    "wall", "floor", "door", "ceiling", "table", "window", "box",
    "ceiling lamp", "light switch", "cabinet", "chair", "heater",
    "monitor", "whiteboard", "office chair", "bottle", "doorframe",
    "keyboard", "window frame", "mouse", "paper", "blinds", "trash can",
    "telephone", "book", "shelf", "sink", "windowsill", "bag",
    "smoke detector",
]


class _Mini(GFS_VL_Trainer):
    """只借用 VLM_3D 前向与原型提取，不建 optimizer / scheduler。"""

    def __init__(self, cfg, logger, ds):
        self.cfg = cfg
        self.logger = logger

        class _L:
            pass

        self.train_loader = _L()
        self.train_loader.dataset = ds
        self.build_zero_shot_model()


def build_proto(cfg, mini, logger):
    """走标准 dataset 管线提取 5-shot 原型（v1 在这里手写 np.load 读原始数据）。"""
    if os.path.exists(PROTO):
        z = np.load(PROTO)
        logger.info("loaded proto cache %s  shape=%s" % (PROTO, z["proto"].shape))
        return torch.from_numpy(z["proto"]).cuda()

    vds = build_dataset(getattr(cfg.data, "regis1"))
    loader = torch.utils.data.DataLoader(
        vds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=None,
        collate_fn=collate_fn,
    )
    logger.info(
        "extracting 5-shot prototypes from %d support scenes (standard pipeline) ..."
        % len(vds)
    )
    avg = mini.extract_3d_vlm_proto(loader)  # (18, 512)
    proto = F.normalize(avg.float(), dim=-1)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(PROTO, proto=proto.cpu().numpy())
    logger.info(
        "proto saved: %s  |row_norm| min/max = %.4f / %.4f"
        % (PROTO, float(avg.norm(dim=-1).min()), float(avg.norm(dim=-1).max()))
    )
    return proto


def auroc_exact(sp, sn):
    """精确 AUROC（Mann-Whitney U，ties 取 0.5）。sp/sn: 1D float32。"""
    if sp.size == 0 or sn.size == 0:
        return np.nan
    sns = np.sort(sn)
    lt = np.searchsorted(sns, sp, side="left")
    le = np.searchsorted(sns, sp, side="right")
    u = float(lt.sum()) + 0.5 * float((le - lt).sum())
    return u / (float(sp.size) * float(sn.size))


def prec_at_keep(sp, sn, frac=0.10):
    """取分数最高的 frac 比例（正+负混合），返回其中正样本占比与实际保留数。"""
    n_pos, n_neg = sp.size, sn.size
    n = n_pos + n_neg
    if n == 0 or n_pos == 0:
        return np.nan, 0
    k = max(int(round(n * frac)), 1)
    if k >= n:
        return float(n_pos) / float(n), k
    allv = np.concatenate([sp, sn])
    # argpartition 取最大的 k 个
    idx = np.argpartition(-allv, k - 1)[:k]
    hit = int((idx < n_pos).sum())
    return float(hit) / float(k), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config-file",
        default="configs/scannetpp/semseg-pt-v3m1-0-gfsregistrain_k5.py",
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--build-proto", action="store_true")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()

    if a.merge:
        merge(a.shards)
        return

    cfg = default_config_parser(a.config_file, options=None)
    lg = get_root_logger(log_file=None)
    lg.info("config: %s" % a.config_file)

    ds_train = build_dataset(cfg.data.train)
    mini = _Mini(cfg, lg, ds_train)
    proto = build_proto(cfg, mini, lg)
    if a.build_proto:
        return

    nb = int(cfg.data.num_bases)
    nc = int(cfg.data.num_base_novels)
    nov = nc - nb
    W = mini.VLM_3D.task_head.cls_head.weight

    lg.info("building 3D student from %s" % STU_CKPT)
    stu = build_model(cfg.model)
    ck = torch.load(STU_CKPT, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", "", 1): v for k, v in ck["state_dict"].items()}
    miss, unexp = stu.load_state_dict(sd, strict=False)
    lg.info("student: missing=%d unexpected=%d" % (len(miss), len(unexp)))
    stu.eval().cuda()

    # full_gt: 训练 chunk 的 novel GT 不被置 -1
    cfg.data.train.full_gt = True
    ds = build_dataset(cfg.data.train)

    # 采样缓冲：每类每极性一个 list of float32
    buf_pos = {k: [[] for _ in range(nc)] for k in SIGNALS}
    buf_neg = {k: [[] for _ in range(nc)] for k in SIGNALS}
    rng = np.random.default_rng(20261123 + a.shard)  # 固定种子 → 可复现

    idxs = list(range(a.shard, len(ds), a.shards))
    lg.info("shard %d/%d: %d chunks" % (a.shard, a.shards, len(idxs)))

    for n, i in enumerate(idxs):
        d = ds[i]
        for k, v in list(d.items()):
            if isinstance(v, torch.Tensor):
                d[k] = v.to("cuda")
        gt = d["segment"].view(-1)
        is_orig = (
            d["mask"].view(-1) == 1
            if "mask" in d
            else torch.ones_like(gt, dtype=torch.bool)
        )

        # ---- VLM 前向 ----
        vo = mini.VLM_3D(mini.construct_3D_vlm_input(d))
        feats = vo["all_feats"].float()
        fn = F.normalize(feats, dim=-1)
        cos = fn @ W.t()
        cos_nv = cos[:, nb:]
        cos_base = cos[:, :nb]
        pred = cos_nv.argmax(-1) + nb

        p_nv = F.softmax(cos_nv, dim=-1)
        ent = -(p_nv * (p_nv + 1e-8).log()).sum(-1)

        # 原型相似度：pred 类 + 最佳其他类
        psim = fn @ proto.t()  # (N, 18)
        pidx = (pred - nb).unsqueeze(1)
        s3a = psim.gather(1, pidx).squeeze(1)
        psim_other = psim.clone().scatter_(1, pidx, -2.0)
        s3b = s3a - psim_other.max(1).values

        # ---- 3D 学生前向 ----
        with torch.no_grad():
            so = stu(d)
            sp = F.softmax(so["seg_logits"].float(), dim=-1)
        base_max_p = sp[:, :nb].max(1).values
        s4b = sp.gather(1, pidx).squeeze(1)

        scores = {
            "S1_negent": (-ent).cpu().numpy().astype(np.float32),
            "S2_margin": (cos_nv.max(-1).values - cos_base.max(-1).values)
            .cpu().numpy().astype(np.float32),
            "S3a_proto_pred": s3a.cpu().numpy().astype(np.float32),
            "S3b_proto_margin": s3b.cpu().numpy().astype(np.float32),
            "S4a_nonbase": (1.0 - base_max_p).cpu().numpy().astype(np.float32),
            "S4b_pred": s4b.cpu().numpy().astype(np.float32),
            "S4c_netmargin": (s4b - base_max_p).cpu().numpy().astype(np.float32),
        }

        gt_np = gt.cpu().numpy()
        pred_np = pred.cpu().numpy()
        orig_np = is_orig.cpu().numpy()
        novel_gt = (gt_np >= nb) & (gt_np < nc)
        pos = orig_np & novel_gt & (pred_np == gt_np)
        neg = orig_np & (
            (novel_gt & (pred_np != gt_np))
            | ((gt_np == -1) & (pred_np >= nb))
        )

        for k in SIGNALS:
            sc = scores[k]
            for c in range(nb, nc):
                mp = pos & (pred_np == c)
                if mp.any():
                    v = sc[mp]
                    if v.size > CAP:
                        v = v[rng.choice(v.size, CAP, replace=False)]
                    buf_pos[k][c].append(v)
                mn = neg & (pred_np == c)
                if mn.any():
                    v = sc[mn]
                    if v.size > CAP:
                        v = v[rng.choice(v.size, CAP, replace=False)]
                    buf_neg[k][c].append(v)

        if (n + 1) % 10 == 0:
            lg.info("  [%d/%d]" % (n + 1, len(idxs)))

    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {}
    for k in SIGNALS:
        for c in range(nb, nc):
            for pol, bufs in (("pos", buf_pos), ("neg", buf_neg)):
                arr = (
                    np.concatenate(bufs[k][c])
                    if bufs[k][c]
                    else np.zeros(0, np.float32)
                )
                payload["%s__%s__%d" % (k, pol, c)] = arr
    out = os.path.join(OUT_DIR, "shard%02d.npz" % a.shard)
    np.savez_compressed(out, **payload)
    meta = dict(
        shard=a.shard,
        shards=a.shards,
        n_chunks=len(idxs),
        nb=nb,
        nc=nc,
        cap=CAP,
    )
    with open(os.path.join(OUT_DIR, "shard%02d.json" % a.shard), "w") as f:
        json.dump(meta, f)
    lg.info("saved %s" % out)


def merge(shards):
    fs = sorted(glob.glob(os.path.join(OUT_DIR, "shard*.npz")))
    metas = sorted(glob.glob(os.path.join(OUT_DIR, "shard*.json")))
    if not fs:
        print("没有分片可合并")
        return
    print("合并 %d 个分片" % len(fs))
    with open(metas[0]) as f:
        meta = json.load(f)
    nb, nc, sh = meta["nb"], meta["nc"], meta["nc"]
    sh = meta["nc"]

    acc_pos = {k: [[] for _ in range(nc)] for k in SIGNALS}
    acc_neg = {k: [[] for _ in range(nc)] for k in SIGNALS}
    n_chunks = 0
    for f in fs:
        z = np.load(f)
        for k in SIGNALS:
            for c in range(nb, nc):
                v = z["%s__pos__%d" % (k, c)]
                if v.size:
                    acc_pos[k][c].append(v)
                v = z["%s__neg__%d" % (k, c)]
                if v.size:
                    acc_neg[k][c].append(v)
    for f in metas:
        with open(f) as fh:
            n_chunks += json.load(fh)["n_chunks"]

    # ---- 逐类 ----
    print("chunks=%d  CAP=%d/chunk/class/polarity\n" % (n_chunks, meta["cap"]))
    hdr = "%-15s %8s %8s" % ("class", "nPos", "nNeg")
    for k in SIGNALS:
        hdr += " %13s" % k
    print(hdr)
    print("-" * len(hdr))

    aucs = {k: np.full(nc, np.nan) for k in SIGNALS}
    precs = {k: np.full(nc, np.nan) for k in SIGNALS}
    for c in range(nb, nc):
        line = ""
        any_pos = False
        for k in SIGNALS:
            sp = (
                np.concatenate(acc_pos[k][c])
                if acc_pos[k][c]
                else np.zeros(0, np.float32)
            )
            sn = (
                np.concatenate(acc_neg[k][c])
                if acc_neg[k][c]
                else np.zeros(0, np.float32)
            )
            if sp.size:
                any_pos = True
            aucs[k][c] = auroc_exact(sp, sn)
            precs[k][c] = prec_at_keep(sp, sn)[0]
        if not any_pos:
            continue
        # 该行只打印一次样本数（用 S1 的，各信号相同）
        sp1 = (
            np.concatenate(acc_pos["S1_negent"][c])
            if acc_pos["S1_negent"][c]
            else np.zeros(0, np.float32)
        )
        sn1 = (
            np.concatenate(acc_neg["S1_negent"][c])
            if acc_neg["S1_negent"][c]
            else np.zeros(0, np.float32)
        )
        line += "%-15s %8d %8d" % (NAMES[c], sp1.size, sn1.size)
        for k in SIGNALS:
            line += "  %5.3f/%5.3f" % (aucs[k][c], precs[k][c])
        print(line)

    # ---- 均值 ----
    print("\n=== 18 类均值（AUROC / Precision@10%）===")
    print("%-16s %10s %10s %14s %14s" % ("signal", "AUROC均", "AUROC中位", "P@10均", "P@10中位"))
    summary = {}
    for k in SIGNALS:
        v = aucs[k][nb:]
        v = v[~np.isnan(v)]
        p = precs[k][nb:]
        p = p[~np.isnan(p)]
        summary[k] = dict(
            auroc_mean=float(v.mean()) if v.size else float("nan"),
            auroc_median=float(np.median(v)) if v.size else float("nan"),
            auroc_n=int(v.size),
            prec_mean=float(p.mean()) if p.size else float("nan"),
            prec_median=float(np.median(p)) if p.size else float("nan"),
        )
        print(
            "%-16s %10.3f %10.3f %14.3f %14.3f"
            % (
                k,
                summary[k]["auroc_mean"],
                summary[k]["auroc_median"],
                summary[k]["prec_mean"],
                summary[k]["prec_median"],
            )
        )

    # 随机基线
    base_p = []
    for c in range(nb, nc):
        sp = (
            np.concatenate(acc_pos["S1_negent"][c])
            if acc_pos["S1_negent"][c]
            else np.zeros(0, np.float32)
        )
        sn = (
            np.concatenate(acc_neg["S1_negent"][c])
            if acc_neg["S1_negent"][c]
            else np.zeros(0, np.float32)
        )
        if sp.size + sn.size:
            base_p.append(sp.size / (sp.size + sn.size))
    if base_p:
        print(
            "%-16s %10s %10s %14.3f %14.3f"
            % ("[随机命中率]", "0.500", "0.500", np.mean(base_p), np.median(base_p))
        )

    # ---- 重点类 ----
    print("\n=== 重点类明细（AUROC / P@10%）===")
    for c in (29, 7, 9, 17, 26, 25, 14):
        line = "%-15s" % NAMES[c]
        for k in SIGNALS:
            line += "  %5.3f/%5.3f" % (aucs[k][c], precs[k][c])
        print(line)

    # ---- 分数真实分布（证伪量化问题）----
    print("\n=== 分数真实分布（log10 尺度跨度；证明 v1 为何恒等 0.5）===")
    print("%-16s %10s %10s %10s" % ("signal", "p1", "p50", "p99"))
    for k in SIGNALS:
        allv = []
        for c in range(nb, nc):
            if acc_pos[k][c]:
                allv.append(np.concatenate(acc_pos[k][c]))
            if acc_neg[k][c]:
                allv.append(np.concatenate(acc_neg[k][c]))
        if not allv:
            continue
        v = np.concatenate(allv)
        print(
            "%-16s %10.4f %10.4f %10.4f"
            % (
                k,
                float(np.percentile(v, 1)),
                float(np.percentile(v, 50)),
                float(np.percentile(v, 99)),
            )
        )

    np.savez_compressed(
        os.path.join(OUT_DIR, "summary.npz"),
        aucs=np.array([aucs[k] for k in SIGNALS]),
        precs=np.array([precs[k] for k in SIGNALS]),
        signals=np.array(SIGNALS),
        names=np.array(NAMES),
    )
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nsaved %s/summary.{npz,json}" % OUT_DIR)


if __name__ == "__main__":
    main()
