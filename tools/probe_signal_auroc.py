# -*- coding: utf-8 -*-
"""阶段一：四信号判别力 PK（零训练，只读）。

在 ScanNet++ 全部 712 个训练 chunk 的「原场景」上，对 M0（屏蔽 Base）的
18 类 Top-1 预测，比较四个信号把「真点」和「假点」分开的能力：

    正样本 P：原场景 & GT 是 novel & pred == GT（留住它们 = U2 = 0.4120）
    负样本 N：原场景 & (GT 是 novel 但 pred != GT)  错标
                      ∪ (GT 是背景但 pred 是 novel) 背景杂物

四个信号（统一成「分数越高越可能是真点」）：
    S1  文本余弦熵的负数      -H(softmax(cos_nv))
    S2  文本边际              cos_nv_max - cos_base_max
    S3  5-shot 视觉原型相似度  max_c cos(feat, proto_c)
    S4  3D 学生的「非 base」  1 - max_{b<12} P_3D(b|x)

输出：每类的 AUROC + 相同保留率下的 Precision/Recall + smoke 假阳性残留。

用法（8 卡分片）：
    python tools/probe_signal_auroc.py --shard 0 --shards 8
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
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger

OUT_DIR = "/data01/jh/data/step1_auroc"
NBIN = 100
STU_CKPT = "/data01/jh/code/GFS-VL/exp/scannetpp/scpp_base_lovasz_k5/model/regis1_model_last.pth"
PROTO = "/data01/jh/data/step1_auroc/proto_5shot.npz"


class _Mini(GFS_VL_Trainer):
    def __init__(self, cfg, logger, ds):
        self.cfg = cfg
        self.logger = logger

        class _L:
            pass

        self.train_loader = _L()
        self.train_loader.dataset = ds
        self.build_zero_shot_model()


def build_prototypes(cfg, mini, logger):
    """从 scpp_regis_5_10.txt 的 18 个 support 场景算每类的 VLM 视觉原型。"""
    if os.path.exists(PROTO):
        d = np.load(PROTO)
        logger.info("loaded prototype cache %s" % PROTO)
        return torch.from_numpy(d["proto"]).cuda(), torch.from_numpy(d["valid"]).cuda()

    nb = int(cfg.data.num_bases)
    nc = int(cfg.data.num_base_novels)
    root = cfg.data_root
    f = os.path.join(root, "scpp_regis_5_10.txt")
    cls2scans = {}
    for line in open(f):
        cid, rel = line.strip().split("\t")
        cls2scans.setdefault(int(cid), []).append(os.path.join(root, rel))

    # 原始 100 类 -> 0-29 的映射
    from pointcept.datasets.preprocessing.scannetpp.metadata.semantic_benchmark.top100 import (
        CLASS_LABELS_100, CLASS_LABELS_BASE, CLASS_LABELS_BASE_NOVEL)
    base = list(CLASS_LABELS_BASE)
    novel = [x for x in CLASS_LABELS_BASE_NOVEL if x not in CLASS_LABELS_BASE]
    name2idx = {n: i for i, n in enumerate(base + novel)}

    sums = torch.zeros(nc, 512, device="cuda")
    cnts = torch.zeros(nc, device="cuda")
    W = mini.VLM_3D.task_head.cls_head.weight

    for raw_cid, scans in cls2scans.items():
        name = CLASS_LABELS_100[raw_cid]
        if name not in name2idx:
            continue
        ci = name2idx[name]
        if ci < nb:
            continue
        for sp in scans:
            try:
                coord = np.load(os.path.join(sp, "coord.npy")).astype(np.float32)
                color = np.load(os.path.join(sp, "color.npy")).astype(np.float32)
                seg = np.load(os.path.join(sp, "segment.npy"))
            except OSError:
                continue
            m = seg[:, 0] == raw_cid
            if m.sum() < 10:
                continue
            d = {"coord": torch.from_numpy(coord).cuda(),
                 "color": torch.from_numpy(color).cuda(),
                 "offset": torch.tensor([coord.shape[0]]).cuda()}
            vo = mini.VLM_3D(mini.construct_3D_vlm_input(d))
            feats = vo["all_feats"].float()
            mv = torch.from_numpy(m).cuda()
            sums[ci] += feats[mv].sum(0)
            cnts[ci] += mv.sum()

    proto = F.normalize(sums, dim=-1)
    valid = cnts > 0
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(PROTO, proto=proto.cpu().numpy(), valid=valid.cpu().numpy())
    logger.info("prototype built: %d/%d valid" % (int(valid.sum()), nc - nb))
    return proto, valid


def hist_auroc(hp, hn):
    """hp/hn: (..., NBIN) 正/负样本直方图，返回逐类 AUROC。"""
    npos = hp.sum(-1)
    nneg = hn.sum(-1)
    out = np.full(hp.shape[0], np.nan)
    for c in range(hp.shape[0]):
        if npos[c] == 0 or nneg[c] == 0:
            continue
        cum = np.cumsum(hn[c])
        auc = 0.0
        for b in range(hp.shape[1]):
            below = cum[b - 1] if b > 0 else 0
            auc += hp[c, b] * (below + 0.5 * hn[c, b])
        out[c] = auc / (npos[c] * nneg[c])
    return out


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
    cfg.data.train.full_gt = True
    ds = build_dataset(cfg.data.train)
    mini = _Mini(cfg, lg, ds)
    nb = int(cfg.data.num_bases)
    nc = int(cfg.data.num_base_novels)
    W = mini.VLM_3D.task_head.cls_head.weight

    proto, proto_valid = build_prototypes(cfg, mini, lg)

    lg.info("building 3D student ...")
    stu = build_model(cfg.model)
    ck = torch.load(STU_CKPT, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", "", 1): v for k, v in ck["state_dict"].items()}
    miss, unexp = stu.load_state_dict(sd, strict=False)
    lg.info("student: missing=%d unexpected=%d" % (len(miss), len(unexp)))
    stu.eval().cuda()

    keys = ["S1_negent", "S2_margin", "S3_proto", "S4_nonbase"]
    # 每类的正/负直方图：分数区间 [-2, 2]（各信号都会落进来）
    hp = {k: np.zeros((nc, NBIN), np.int64) for k in keys}
    hn = {k: np.zeros((nc, NBIN), np.int64) for k in keys}
    # 熵取负后范围 [-2.9, 0]，必须覆盖到 -3；其他信号也在 [-1,2] 内
    edges = np.linspace(-3.0, 3.0, NBIN + 1)

    idxs = list(range(a.shard, len(ds), a.shards))
    lg.info("shard %d/%d: %d chunks" % (a.shard, a.shards, len(idxs)))

    for n, i in enumerate(idxs):
        d = ds[i]
        for k, v in list(d.items()):
            if isinstance(v, torch.Tensor):
                d[k] = v.to("cuda")
        gt = d["segment"].view(-1)
        is_orig = d["mask"].view(-1) == 1 if "mask" in d else torch.ones_like(gt, dtype=torch.bool)
        pred = None

        # VLM
        vo = mini.VLM_3D(mini.construct_3D_vlm_input(d))
        feats = vo["all_feats"].float()
        cos = F.normalize(feats, dim=-1) @ W.t()
        cos_nv = cos[:, nb:]
        pred = cos_nv.argmax(-1) + nb
        p = F.softmax(cos_nv, dim=-1)
        ent = -(p * (p + 1e-8).log()).sum(-1)
        nv_max = cos_nv.max(-1).values
        base_max = cos[:, :nb].max(-1).values
        margin = nv_max - base_max
        # 5-shot 原型
        proto_sim = (F.normalize(feats, dim=-1) @ proto.t()).max(-1).values

        # 3D 学生
        with torch.no_grad():
            so = stu(d)
            sl = so["seg_logits"].float()
            sp = F.softmax(sl, dim=-1)
            base_prob = sp[:, :nb].max(-1).values

        scores = {
            "S1_negent": (-ent).cpu().numpy(),
            "S2_margin": margin.cpu().numpy(),
            "S3_proto": proto_sim.cpu().numpy(),
            "S4_nonbase": (1.0 - base_prob).cpu().numpy(),
        }
        gt_np = gt.cpu().numpy()
        pred_np = pred.cpu().numpy()
        orig_np = is_orig.cpu().numpy()

        novel_gt = (gt_np >= nb) & (gt_np < nc)
        pos = orig_np & novel_gt & (pred_np == gt_np)
        neg = orig_np & ((novel_gt & (pred_np != gt_np)) | ((gt_np == -1) & (pred_np >= nb)))

        for k in keys:
            sc = scores[k]
            for c in range(nb, nc):
                mp = pos & (pred_np == c)
                mn = neg & (pred_np == c)
                if mp.any():
                    hp[k][c] += np.histogram(sc[mp], edges)[0]
                if mn.any():
                    hn[k][c] += np.histogram(sc[mn], edges)[0]

        if (n + 1) % 10 == 0:
            lg.info("  [%d/%d]" % (n + 1, len(idxs)))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "shard%02d.npz" % a.shard)
    np.savez_compressed(out, meta=dict(shard=a.shard, shards=a.shards,
                                       n_chunks=len(idxs), nb=nb, nc=nc, nbin=NBIN,
                                       lo=float(edges[0]), hi=float(edges[-1])),
                        **{("hp_" + k): hp[k] for k in keys},
                        **{("hn_" + k): hn[k] for k in keys})
    lg.info("saved %s" % out)


def merge(shards):
    fs = sorted(glob.glob(os.path.join(OUT_DIR, "shard*.npz")))
    print("合并 %d 个分片" % len(fs))
    acc = {}
    meta = None
    for f in fs:
        z = np.load(f, allow_pickle=True)
        if meta is None:
            meta = z["meta"].item()
        for k in z.files:
            if k == "meta":
                continue
            acc[k] = acc.get(k, 0) + z[k]
    edges = np.linspace(meta["lo"], meta["hi"], meta["nbin"] + 1)
    names = ['wall', 'floor', 'door', 'ceiling', 'table', 'window', 'box',
             'ceiling lamp', 'light switch', 'cabinet', 'chair', 'heater',
             'monitor', 'whiteboard', 'office chair', 'bottle', 'doorframe',
             'keyboard', 'window frame', 'mouse', 'paper', 'blinds', 'trash can',
             'telephone', 'book', 'shelf', 'sink', 'windowsill', 'bag',
             'smoke detector']
    nb, nc = meta["nb"], meta["nc"]
    keys = ["S1_negent", "S2_margin", "S3_proto", "S4_nonbase"]
    print("chunks=%d\n" % meta["n_chunks"])
    print("%-15s" % "class" + "".join(" %12s" % k for k in keys))
    aucs = {}
    for k in keys:
        aucs[k] = hist_auroc(acc["hp_" + k], acc["hn_" + k])
    for c in range(nb, nc):
        print("%-15s" % names[c] + "".join(" %12.3f" % aucs[k][c] for k in keys))
    print("\n=== 18 类均值 / 中位数 ===")
    for k in keys:
        v = aucs[k][nb:]
        v = v[~np.isnan(v)]
        print("  %-10s mean %.3f  median %.3f  n=%d" % (k, v.mean(), np.median(v), len(v)))
    print("\n=== 重点类 ===")
    for c in (29, 7, 9, 17, 21, 14, 26):
        print("  %-15s" % names[c] + "".join(" %12.3f" % aucs[k][c] for k in keys))


if __name__ == "__main__":
    main()
