# -*- coding: utf-8 -*-
"""Step 0：4 种伪标签过滤策略的全量离线对比（只读扫描，不训练）。

对比：
    S1  单场景 Q_0.1 子空间熵（1C-e 的做法，会触发缺席类强制配额）
    S2  跨场景全局 Q_0.1 子空间熵
    S3  跨场景全局 Q_0.1 绝对余弦 max(cos_nv)
    S4  支持集锚定绝对余弦（用 support 点对 W_text 的余弦分位 p25/p50/p75）

统计口径：
    * 只作用于未标注区（segment == -1），不含 base GT
    * support 实例（segment >= num_bases）用于算 TP / recall
    * 未标注区的 novel 预测一律计入 FP（未知杂物）
    * 按「该 chunk 的 support 里有没有这个类」拆成 with / empty 两组直方图，
      这样汇总后可以直接算「空房间假阳性倍数」

每类输出：
    n_pred_with / n_pred_empty      预测点数（不过滤）
    ent_hist_with / ent_hist_empty  子空间熵直方图（100 bin）
    sim_hist_with / sim_hist_empty  绝对余弦直方图（100 bin）
    supp_ent_hist / supp_sim_hist   support 上正确预测的点的直方图
    n_supp_gt                       support 的 GT 点数
    n_pred_s1 / tp_s1               S1（逐 chunk Q0.1）的精确统计

用法（8 卡分片）：
    python tools/probe_step0_filter_compare.py --shard 0 --shards 8
汇总：
    python tools/probe_step0_filter_compare.py --merge --shards 8
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

from pointcept.datasets import build_dataset
from pointcept.engines.defaults import default_config_parser
from pointcept.engines.train import GFS_VL_Trainer
from pointcept.utils.logger import get_root_logger

NBIN = 100
ENT_MAX = 3.0          # ln(18) ≈ 2.89，留余量
SIM_MIN, SIM_MAX = -1.0, 1.0
OUT_DIR = "/data01/jh/data/step0"


class _Mini(GFS_VL_Trainer):
    def __init__(self, cfg, logger, ds):
        self.cfg = cfg
        self.logger = logger

        class _L:
            pass

        self.train_loader = _L()
        self.train_loader.dataset = ds
        self.build_zero_shot_model()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", default="configs/scannetpp/semseg-pt-v3m1-0-gfsregistrain_k5.py")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()

    if a.merge:
        merge(a.shards)
        return

    cfg = default_config_parser(a.config_file, options=None)
    lg = get_root_logger(log_file=None)
    ds = build_dataset(cfg.data.train)
    mini = _Mini(cfg, lg, ds)
    nb = int(cfg.data.num_bases)
    nc = int(cfg.data.num_base_novels)
    W = mini.VLM_3D.task_head.cls_head.weight

    ent_edges = np.linspace(0, ENT_MAX, NBIN + 1)
    sim_edges = np.linspace(SIM_MIN, SIM_MAX, NBIN + 1)

    z = lambda: np.zeros(nc, dtype=np.int64)
    st = dict(
        n_pred_with=z(), n_pred_empty=z(),
        n_pred_s1=z(), tp_s1=z(),
        n_supp_gt=z(),
        ent_hist_with=np.zeros((nc, NBIN), dtype=np.int64),
        ent_hist_empty=np.zeros((nc, NBIN), dtype=np.int64),
        sim_hist_with=np.zeros((nc, NBIN), dtype=np.int64),
        sim_hist_empty=np.zeros((nc, NBIN), dtype=np.int64),
        supp_ent_hist=np.zeros((nc, NBIN), dtype=np.int64),
        supp_sim_hist=np.zeros((nc, NBIN), dtype=np.int64),
    )

    idxs = list(range(a.shard, len(ds), a.shards))
    lg.info("shard %d/%d: %d chunks" % (a.shard, a.shards, len(idxs)))

    for n, i in enumerate(idxs):
        d = ds[i]
        for k, v in list(d.items()):
            if isinstance(v, torch.Tensor):
                d[k] = v.to("cuda")
        gt = d["segment"].view(-1)
        bg = gt == -1
        supp = (gt >= nb) & (gt < nc)
        if not bg.any():
            continue

        vo = mini.VLM_3D(mini.construct_3D_vlm_input(d))
        cos = F.normalize(vo["all_feats"].float(), dim=-1) @ W.t()
        cos_nv = cos[:, nb:]
        maxsim = cos_nv.max(dim=-1).values
        p = F.softmax(cos_nv, dim=-1)
        ent = -(p * (p + 1e-8).log()).sum(-1)
        pred = cos_nv.argmax(dim=-1) + nb

        # 该 chunk 的 support 里出现过的类
        has = np.zeros(nc, dtype=bool)
        for c in gt[supp].unique().tolist():
            if nb <= c < nc:
                has[int(c)] = True

        for c in range(nb, nc):
            m = bg & (pred == c)
            n_m = int(m.sum())
            if has[c]:
                st["n_pred_with"][c] += n_m
                if n_m:
                    st["ent_hist_with"][c] += np.histogram(ent[m].cpu().numpy(), ent_edges)[0]
                    st["sim_hist_with"][c] += np.histogram(maxsim[m].cpu().numpy(), sim_edges)[0]
            else:
                st["n_pred_empty"][c] += n_m
                if n_m:
                    st["ent_hist_empty"][c] += np.histogram(ent[m].cpu().numpy(), ent_edges)[0]
                    st["sim_hist_empty"][c] += np.histogram(maxsim[m].cpu().numpy(), sim_edges)[0]

            # S1：逐 chunk 每类保留熵最低 10%
            if n_m > 1:
                k = max(1, int(n_m * 0.1))
                thr = torch.topk(ent[m], k, largest=False).values[-1]
                keep = m & (ent <= thr)
                st["n_pred_s1"][c] += int(keep.sum())
                st["tp_s1"][c] += int((supp & (pred == c) & keep).sum())
            elif n_m == 1:
                st["n_pred_s1"][c] += 1
                st["tp_s1"][c] += int((supp & (pred == c)).sum())

            # support 上的 GT 点数 + 正确预测点的分数分布
            gt_c = supp & (gt == c)
            st["n_supp_gt"][c] += int(gt_c.sum())
            hit = gt_c & (pred == c)
            if int(hit.sum()):
                st["supp_ent_hist"][c] += np.histogram(ent[hit].cpu().numpy(), ent_edges)[0]
                st["supp_sim_hist"][c] += np.histogram(maxsim[hit].cpu().numpy(), sim_edges)[0]

        if (n + 1) % 10 == 0:
            lg.info("  [%d/%d]" % (n + 1, len(idxs)))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "shard%02d.json" % a.shard)
    payload = {k: (v.tolist() if isinstance(v, np.ndarray) else v.tolist()) for k, v in st.items()}
    payload["meta"] = dict(shard=a.shard, shards=a.shards, n_chunks=len(idxs),
                           nb=nb, nc=nc, nbin=NBIN, ent_max=ENT_MAX,
                           sim_min=SIM_MIN, sim_max=SIM_MAX)
    json.dump(payload, open(out, "w"))
    lg.info("saved %s" % out)


def merge(shards):
    fs = sorted(glob.glob(os.path.join(OUT_DIR, "shard*.json")))
    if len(fs) != shards:
        print("!! 只有 %d 个分片，期望 %d" % (len(fs), shards))
    acc = None
    for f in fs:
        d = json.load(open(f))
        if acc is None:
            acc = {k: np.array(v) for k, v in d.items() if k != "meta"}
            meta = d["meta"]
        else:
            for k, v in d.items():
                if k == "meta":
                    continue
                acc[k] = acc[k] + np.array(v)
    acc["meta"] = meta
    out = os.path.join(OUT_DIR, "merged.npz")
    np.savez_compressed(out, **acc)
    print("merged %d shards -> %s" % (len(fs), out))
    print("chunks:", meta["n_chunks"] * len(fs))


if __name__ == "__main__":
    main()
