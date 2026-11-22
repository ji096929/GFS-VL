"""只做前景比例校准能把整场景 IoU 提到多少。

对 baseline 的概率 p1 取 top-k 分位作为预测，k 分别用：
  s_ratio   support 的前景占比（无标签可得）
  oracle    query 的真值前景占比（用来看上限，实际不可得）
  固定值    0.05 / 0.10 / 0.20
和 argmax(p>0.5) 的原始预测对比。
"""
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as Fs

from ocsb_core import (NEW_IDS, build_proto, feats_one, iou_counts, load_model,
                       predict_agg, seed_everything)
from util.data_util import data_prepare_v101

SC = "/data01/jh/data/ScanNet/scenes/data"
KS = [("s_ratio", None), ("oracle", None), ("k0.05", 0.05),
      ("k0.10", 0.10), ("k0.20", 0.20)]


def prep(name, cfg, cls):
    r = np.load(os.path.join(SC, name + ".npy"))
    return data_prepare_v101(r[:, 0:3].astype(np.float32),
                             r[:, 3:6].astype(np.float32),
                             r[:, 6].astype(np.int64), "val",
                             cfg.voxel_size, cfg.voxel_max, None, False, cls)


def bin_iou(pred01, gt):
    tp = int(((pred01 == 1) & (gt == 1)).sum())
    fp = int(((pred01 == 1) & (gt == 0)).sum())
    fn = int(((pred01 == 0) & (gt == 1)).sum())
    return tp / max(tp + fp + fn, 1), tp, fp, fn


def main():
    c2s = pickle.load(open(os.path.join(os.path.dirname(SC), "class2scans.pkl"), "rb"))
    model, cfg = load_model()
    dev = torch.device("cuda:0")
    rng = random.Random(0)
    eps = []
    for c in NEW_IDS:
        pool = sorted(c2s[c])
        for _ in range(2):
            if len(pool) >= 2:
                s, q = rng.sample(pool, 2)
                eps.append((c, s, q))

    acc = {k: [] for k, _ in KS}
    acc["argmax"] = []
    for ei, (c, sn, qn) in enumerate(eps):
        seed_everything(10000 + ei)
        sc, sfe, sl = prep(sn, cfg, c)
        qc, qfe, ql = prep(qn, cfg, c)
        s_bin = (sl == c).long()
        q_map = (ql == c).long()
        if int(q_map.sum()) == 0:
            continue
        sf = feats_one(model, torch.cat([sc, sfe], 1), s_bin, dev)
        qf = feats_one(model, torch.cat([qc, qfe], 1), q_map, dev)
        fg_s = sf["yl"].bool()
        sparse_s = torch.cat([build_proto(model, sf, ~fg_s),
                              build_proto(model, sf, fg_s)], 0)
        pred, _ = predict_agg(model, qf, [sparse_s], 0.0, dev)
        p1 = Fs.softmax(pred, dim=0)[1]

        tp, fp, fn, iou_am = iou_counts(pred, q_map)
        acc["argmax"].append(iou_am)
        s_r = float(s_bin.float().mean())
        q_r = float(q_map.float().mean())
        for name, kfix in KS:
            k = s_r if name == "s_ratio" else (q_r if name == "oracle" else kfix)
            k = min(max(k, 1e-4), 0.95)
            thr = torch.quantile(p1.float(), 1.0 - k)
            pred01 = (p1 >= thr).long()
            iou, _, _, _ = bin_iou(pred01.cpu(), q_map)
            acc[name].append(iou)
        print("  %2d cls %2d  gt %.2f%%  argmax %.3f  s_ratio %.3f  oracle %.3f" % (
            ei + 1, c, 100 * q_r, acc["argmax"][-1],
            acc["s_ratio"][-1], acc["oracle"][-1]), flush=True)

    print("\n%-9s %8s" % ("method", "mean IoU"))
    for k in ["argmax"] + [n for n, _ in KS]:
        print("%-9s %8.4f" % (k, float(np.mean(acc[k]))))


if __name__ == "__main__":
    main()
