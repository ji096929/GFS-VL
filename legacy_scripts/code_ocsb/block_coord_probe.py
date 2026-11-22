"""隔离坐标平移：同样 1m block、同样 voxel、同样原型流程，只改最后一步坐标处理。

center  coord -= (min+max)/2   ← prepare_block 现行做法，范围 [-0.5,0.5]
min0    coord -= min           ← 官方 data_prepare_v101 做法，范围 [0,1]（训练时用的）
"""
import argparse
import os
import sys

sys.path.insert(0, "/data01/jh/code")
sys.path.insert(0, "/data01/jh/code/COSeg")
import fake_tp  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ocsb_core import (NEW_IDS, build_episodes, build_proto, episode_tensors,  # noqa: E402
                       feats_one, iou_counts, load_model, predict_agg,
                       seed_everything)
from util.data_util import voxelize  # noqa: E402

DATA = "/data01/jh/data/ScanNet/blocks_bs1_s1/data"


def load_block(name, cfg, mode):
    r = np.load(os.path.join(DATA, name + ".npy"))
    coord = r[:, 0:3].astype(np.float32).copy()
    feat = r[:, 3:6].astype(np.float32)
    label = r[:, 6].astype(np.int64)
    if cfg.voxel_size:
        coord -= np.min(coord, 0)
        uq = voxelize(coord, cfg.voxel_size)
        coord, feat, label = coord[uq], feat[uq], label[uq]
    if cfg.voxel_max and label.shape[0] > cfg.voxel_max:
        init = label.shape[0] // 2
        crop = np.argsort(np.sum(np.square(coord - coord[init]), 1))[:cfg.voxel_max]
        coord, feat, label = coord[crop], feat[crop], label[crop]
    if mode == "center":
        cmin, cmax = np.min(coord, 0), np.max(coord, 0)
        coord = coord - (cmin + cmax) / 2.0
    else:
        coord = coord - np.min(coord, 0)
    return {"coord": torch.FloatTensor(coord),
            "feat": torch.FloatTensor(feat) / 255.0,
            "label": torch.LongTensor(label)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--out", default="/data01/jh/data/ocsb_cache/coord_probe.json")
    args = ap.parse_args()

    model, cfg = load_model()
    dev = torch.device("cuda:0")
    print("voxel_size=%s voxel_max=%s" % (cfg.voxel_size, cfg.voxel_max))
    eps = build_episodes(args.per_class)
    rows = []
    for ei, m in enumerate(eps):
        seed_everything(10000 + ei)
        c = m["class"]
        rec = {"cls": c, "ep": ei}
        for mode in ("center", "min0"):
            s = load_block(m["support"], cfg, mode)
            q = load_block(m["query"], cfg, mode)
            s_x, s_bin = episode_tensors(s, c)
            q_x, q_map = episode_tensors(q, c)
            sf = feats_one(model, s_x, s_bin, dev)
            qf = feats_one(model, q_x, q_map, dev)
            fg_s = sf["yl"].bool()
            sparse_s = torch.cat([build_proto(model, sf, ~fg_s),
                                  build_proto(model, sf, fg_s)], 0)
            pred, _ = predict_agg(model, qf, [sparse_s], 0.0, dev)
            tp, fp, fn, iou = iou_counts(pred, q_map)
            pf = float((pred.argmax(0) == 1).float().mean())
            rec[mode] = dict(iou=iou, tp=tp, fp=fp, fn=fn,
                             pred_fg=pf, n=len(q_map),
                             gt=float(q_map.float().mean()))
        rows.append(rec)
        print("  %2d cls %2d  npts %5d gt %.2f%%  center iou %.4f (fg %.1f%%)  "
              "min0 iou %.4f (fg %.1f%%)" % (
                  ei + 1, c, rec["center"]["n"], 100 * rec["center"]["gt"],
                  rec["center"]["iou"], 100 * rec["center"]["pred_fg"],
                  rec["min0"]["iou"], 100 * rec["min0"]["pred_fg"]), flush=True)

    import json
    print("\n%-8s %9s %9s" % ("mode", "Cumul", "Arith"))
    for mode in ("center", "min0"):
        tp = {c: 0 for c in NEW_IDS}; fp = {c: 0 for c in NEW_IDS}
        fn = {c: 0 for c in NEW_IDS}
        for r in rows:
            tp[r["cls"]] += r[mode]["tp"]; fp[r["cls"]] += r[mode]["fp"]
            fn[r["cls"]] += r[mode]["fn"]
        per = [tp[c] / max(tp[c] + fp[c] + fn[c], 1) for c in NEW_IDS]
        print("%-8s %9.4f %9.4f" % (mode, float(np.mean(per)),
                                    float(np.mean([r[mode]["iou"] for r in rows]))))
    json.dump(rows, open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
