"""合并 scene_eval.py 的多分片结果，重算 Cumul / Arith / 逐类 / U 命中率。"""
import json
import sys

import numpy as np

NEW = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
names = [l.strip() for l in open(
    "/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt")]

rows = []
for p in sys.argv[1:]:
    rows += json.load(open(p))["rows"]
print("episodes %d, rows %d" % (len({r["ep"] for r in rows}), len(rows)))

cases = []
for r in rows:
    if r["case"] not in cases:
        cases.append(r["case"])

print("\n%-20s %9s %9s %9s %7s" % ("case", "Cumul", "Arith", "dCumul", "N"))
base = None
for k in cases:
    tp = {c: 0 for c in NEW}
    fp = {c: 0 for c in NEW}
    fn = {c: 0 for c in NEW}
    ious = []
    for r in rows:
        if r["case"] != k:
            continue
        tp[r["cls"]] += r["tp"]
        fp[r["cls"]] += r["fp"]
        fn[r["cls"]] += r["fn"]
        ious.append(r["iou"])
    per = [tp[c] / max(tp[c] + fp[c] + fn[c], 1) for c in NEW]
    cu = float(np.mean(per))
    if k == "Baseline":
        base = cu
    d = "" if base is None else "%+9.4f" % (cu - base)
    print("%-20s %9.4f %9.4f %9s %7d" % (k, cu, float(np.mean(ious)), d, len(ious)))

print("\n逐类（Baseline）")
tp = {c: 0 for c in NEW}; fp = {c: 0 for c in NEW}; fn = {c: 0 for c in NEW}
for r in rows:
    if r["case"] != "Baseline":
        continue
    tp[r["cls"]] += r["tp"]; fp[r["cls"]] += r["fp"]; fn[r["cls"]] += r["fn"]
for c in NEW:
    print("  %2d %-14s iou %.4f  tp %7d fp %7d fn %6d  FP/TP %.1f" % (
        c, names[c], tp[c] / max(tp[c] + fp[c] + fn[c], 1),
        tp[c], fp[c], fn[c], fp[c] / max(tp[c], 1)))

hit = np.mean([r["n_u_hit"] for r in rows if r["case"] == "Baseline"])
k_u = max(r["n_u_hit"] for r in rows if r["case"] == "Baseline")
print("\n盲采 U 命中率 %.1f%%" % (100 * hit / max(k_u, 1)))
