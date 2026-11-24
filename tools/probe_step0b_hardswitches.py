# -*- coding: utf-8 -*-
"""Step 0b：硬开关 1（基类拒识边际）+ 特征范数诊断。

对比 4 种「候选生成」方式，看 smoke detector 的 1,047 万点基数掉到多少：

    M0  屏蔽 12 个基类（当前做法，1C 系列）
    M1  30 类全空间 argmax（基类可吸收）
    M2  边际门控 delta=0.00：cos_nv_max - cos_base_max > 0 才允许进入新类
    M3  边际门控 delta=0.05
    M4  边际门控 delta=0.10

同时诊断：
    * 每类 support 点（segment>=nb）的 feats 范数中位数
    * 每类未标注预测点的 feats 范数中位数
    —— 用来验证「smoke 的 254.4 范数是不是背景区域专属畸变」

用法（8 卡分片）：
    python tools/probe_step0b_hardswitches.py --shard 0 --shards 8
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

OUT_DIR = "/data01/jh/data/step0b"
DELTAS = (0.0, 0.05, 0.10)


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

    keys = ["M0_mask", "M1_30cls"] + ["M2_d%.2f" % d for d in DELTAS]
    st = {k: dict(n_pred_with=np.zeros(nc, np.int64), n_pred_empty=np.zeros(nc, np.int64))
          for k in keys}
    # 范数诊断
    norm_supp_sum = np.zeros(nc, np.float64)   # support 点范数求和
    norm_supp_cnt = np.zeros(nc, np.int64)
    norm_pred_sum = np.zeros(nc, np.float64)   # 未标注预测点范数求和
    norm_pred_cnt = np.zeros(nc, np.int64)

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
        feats = vo["all_feats"].float()
        fn = feats.norm(dim=-1)
        cos = F.normalize(feats, dim=-1) @ W.t()
        cos_base = cos[:, :nb]
        cos_nv = cos[:, nb:]

        has = np.zeros(nc, bool)
        for c in gt[supp].unique().tolist():
            if nb <= c < nc:
                has[int(c)] = True

        # ---- 各策略预测 ----
        preds = {}
        preds["M0_mask"] = cos_nv.argmax(-1) + nb
        p1 = cos.argmax(-1)
        preds["M1_30cls"] = torch.where(p1 >= nb, p1, torch.full_like(p1, -1))
        base_max = cos_base.max(-1).values
        nv_max, nv_idx = cos_nv.max(-1)
        margin = nv_max - base_max
        for dl in DELTAS:
            pd = nv_idx + nb
            preds["M2_d%.2f" % dl] = torch.where(margin > dl, pd, torch.full_like(pd, -1))

        for k, pred in preds.items():
            for c in range(nb, nc):
                m = bg & (pred == c)
                nm = int(m.sum())
                if has[c]:
                    st[k]["n_pred_with"][c] += nm
                else:
                    st[k]["n_pred_empty"][c] += nm

        # ---- 范数诊断 ----
        for c in range(nb, nc):
            gm = supp & (gt == c)
            if int(gm.sum()):
                norm_supp_sum[c] += float(fn[gm].double().sum())
                norm_supp_cnt[c] += int(gm.sum())
            pm = bg & (preds["M0_mask"] == c)
            if int(pm.sum()):
                norm_pred_sum[c] += float(fn[pm].double().sum())
                norm_pred_cnt[c] += int(pm.sum())

        if (n + 1) % 20 == 0:
            lg.info("  [%d/%d]" % (n + 1, len(idxs)))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "shard%02d.json" % a.shard)
    payload = dict(meta=dict(shard=a.shard, shards=a.shards, n_chunks=len(idxs), nb=nb, nc=nc),
                   keys=keys,
                   stats={k: {m: v[m].tolist() for m in ("n_pred_with", "n_pred_empty")}
                          for k, v in st.items()},
                   norm_supp_sum=norm_supp_sum.tolist(), norm_supp_cnt=norm_supp_cnt.tolist(),
                   norm_pred_sum=norm_pred_sum.tolist(), norm_pred_cnt=norm_pred_cnt.tolist())
    json.dump(payload, open(out, "w"))
    lg.info("saved %s" % out)


def merge(shards):
    fs = sorted(glob.glob(os.path.join(OUT_DIR, "shard*.json")))
    print("合并 %d 个分片" % len(fs))
    acc = None
    for f in fs:
        d = json.load(open(f))
        if acc is None:
            acc = d
            acc["stats"] = {k: {m: np.array(v[m]) for m in v} for k, v in d["stats"].items()}
            for m in ("norm_supp_sum", "norm_supp_cnt", "norm_pred_sum", "norm_pred_cnt"):
                acc[m] = np.array(d[m])
        else:
            for k, v in d["stats"].items():
                for m in v:
                    acc["stats"][k][m] = acc["stats"][k][m] + np.array(v[m])
            for m in ("norm_supp_sum", "norm_supp_cnt", "norm_pred_sum", "norm_pred_cnt"):
                acc[m] = acc[m] + np.array(d[m])
            acc["meta"]["n_chunks"] += d["meta"]["n_chunks"]

    nb, nc = acc["meta"]["nb"], acc["meta"]["nc"]
    names = ['wall', 'floor', 'door', 'ceiling', 'table', 'window', 'box',
             'ceiling lamp', 'light switch', 'cabinet', 'chair', 'heater',
             'monitor', 'whiteboard', 'office chair', 'bottle', 'doorframe',
             'keyboard', 'window frame', 'mouse', 'paper', 'blinds', 'trash can',
             'telephone', 'book', 'shelf', 'sink', 'windowsill', 'bag',
             'smoke detector']
    print("chunks=%d\n" % acc["meta"]["n_chunks"])

    keys = acc["keys"]
    print("=== 每类在未标注区的预测点数（M0=屏蔽base / M1=30类 / M2=边际门控）===")
    hdr = "%-15s" % "class"
    for k in keys:
        hdr += " %12s" % k
    print(hdr)
    for c in range(nb, nc):
        line = "%-15s" % names[c]
        for k in keys:
            s = acc["stats"][k]
            line += " %12d" % (s["n_pred_with"][c] + s["n_pred_empty"][c])
        print(line)

    print("\n=== 范数诊断：support 点 vs 未标注预测点 ===")
    print("%-15s %10s %10s %8s" % ("class", "supp范数", "pred范数", "倍率"))
    for c in range(nb, nc):
        ns = acc["norm_supp_sum"][c] / max(acc["norm_supp_cnt"][c], 1)
        npr = acc["norm_pred_sum"][c] / max(acc["norm_pred_cnt"][c], 1)
        if acc["norm_supp_cnt"][c] == 0:
            print("%-15s %10s %10.1f" % (names[c], "—", npr))
        else:
            print("%-15s %10.1f %10.1f %7.2fx" % (names[c], ns, npr, npr / max(ns, 1e-6)))

    # 只打印 smoke
    c = nc - 1
    print("\n=== smoke detector 在 5 种候选生成下的点数（重点）===")
    for k in keys:
        s = acc["stats"][k]
        print("  %-12s with=%-9d empty=%-9d 合计=%d" % (
            k, s["n_pred_with"][c], s["n_pred_empty"][c],
            s["n_pred_with"][c] + s["n_pred_empty"][c]))


if __name__ == "__main__":
    main()
