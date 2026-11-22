"""整场景下三种 voxel_max 采样策略对 baseline IoU 的影响。

v101    官方 data_prepare_v101：全场景随机采样（缓存目录名里的 rdmsp）
sphere  球状裁剪 + 中心化（diag_bc.data_prepare_seg，block 口径用的那个）
none    不采样，保留 voxel 0.02 后的全部点

只算 Baseline，用来判定 0.21 的绝对水平是"模型真不行"还是"采样把密度打散了"。
"""
import argparse
import os
import pickle
import random

import numpy as np
import torch

import diag_bc as D
from ocsb_core import (NEW_IDS, build_proto, feats_one, iou_counts, load_model,
                       predict_agg, seed_everything)
from util.data_util import data_prepare_v101

SC = "/data01/jh/data/ScanNet/scenes/data"


def prep(name, cfg, cls, mode):
    r = np.load(os.path.join(SC, name + ".npy"))
    coord = r[:, 0:3].astype(np.float32)
    feat = r[:, 3:6].astype(np.float32)
    label = r[:, 6].astype(np.int64)
    if mode == "v101":
        return data_prepare_v101(coord, feat, label, "val", cfg.voxel_size,
                                 cfg.voxel_max, None, False, cls)
    if mode == "sphere":
        seg = np.zeros(len(coord), dtype=np.int64)
        c, f, l, _ = D.data_prepare_seg(coord, feat, label, seg,
                                        cfg.voxel_size, cfg.voxel_max)
        return c, f, l
    if mode == "none":
        return data_prepare_v101(coord, feat, label, "val", cfg.voxel_size,
                                 None, None, False, cls)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["v101", "sphere", "none"])
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--out", default="/data01/jh/data/ocsb_cache/sampler_probe_%s.json")
    args = ap.parse_args()
    out = args.out % args.mode

    c2s = pickle.load(open(os.path.join(os.path.dirname(SC), "class2scans.pkl"), "rb"))
    model, cfg = load_model()
    dev = torch.device("cuda:0")

    rng = random.Random(0)
    eps = []
    for c in NEW_IDS:
        pool = sorted(c2s[c])
        for _ in range(args.per_class):
            if len(pool) >= 2:
                s, q = rng.sample(pool, 2)
                eps.append((c, s, q))

    rows = []
    for ei, (c, sn, qn) in enumerate(eps):
        seed_everything(10000 + ei)
        sc, sf_, sl = prep(sn, cfg, c, args.mode)
        qc, qf_, ql = prep(qn, cfg, c, args.mode)
        s_x = torch.cat([sc, sf_], 1)
        q_x = torch.cat([qc, qf_], 1)
        s_bin = (sl == c).long()
        q_map = (ql == c).long()
        sf = feats_one(model, s_x, s_bin, dev)
        qf = feats_one(model, q_x, q_map, dev)
        fg_s = sf["yl"].bool()
        sparse_s = torch.cat([build_proto(model, sf, ~fg_s),
                              build_proto(model, sf, fg_s)], 0)
        if int(q_map.sum()) == 0:
            rows.append(dict(cls=c, iou=0.0, tp=0, fp=0, fn=0, n_q=len(q_map),
                             q_fg=0, n_keep=len(s_x)))
            print("  %2d cls %2d  npts %6d  q_fg 0 (目标被裁光)" % (
                ei + 1, c, len(s_x)), flush=True)
            continue
        pred, _ = predict_agg(model, qf, [sparse_s], 0.0, dev)
        tp, fp, fn, iou = iou_counts(pred, q_map)
        p1 = float((pred.argmax(0) == 1).float().mean())
        rows.append(dict(cls=c, iou=iou, tp=tp, fp=fp, fn=fn, n_q=len(q_map),
                         q_fg=int(q_map.sum()), n_keep=len(s_x), pred_fg=p1))
        print("  %2d cls %2d  npts %6d  q_fg %6d (%.2f%%)  pred_fg %.2f%%  iou %.4f" % (
            ei + 1, c, len(s_x), int(q_map.sum()),
            100 * float(q_map.float().mean()), 100 * p1, iou), flush=True)

    import json as _json
    per = []
    for c in NEW_IDS:
        rs = [r for r in rows if r["cls"] == c]
        tp = sum(r["tp"] for r in rs)
        fp = sum(r["fp"] for r in rs)
        fn = sum(r["fn"] for r in rs)
        per.append(tp / max(tp + fp + fn, 1))
    print("\nmode=%s  Cumul %.4f  Arith %.4f  mean_npts %.0f  mean_q_fg %.0f" % (
        args.mode, float(np.mean(per)), float(np.mean([r["iou"] for r in rows])),
        np.mean([r["n_keep"] for r in rows]), np.mean([r["q_fg"] for r in rows])))
    _json.dump(dict(mode=args.mode, rows=rows, per=per), open(out, "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
