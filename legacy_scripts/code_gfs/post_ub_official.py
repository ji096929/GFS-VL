"""合并官方口径（min_gt=0.05，435 集）的四路上限结果。

与 post_ub560.py 的区别：rows 里的 ep 已经是全局编号（upper_bound.py 的
--min-gt 分支直接存 g_ep），不需要按 NSHARDS 还原。
"""
import glob
import json
import os

import numpy as np
from scipy import stats

CACHE = "/data01/jh/data/ocsb_cache"
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
NAMES = [l.strip() for l in open(
    "/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt")]

files = sorted(glob.glob(os.path.join(CACHE, "ub_official_s*.json")))
files = [f for f in files if ".partial" not in f]
rows = []
for f in files:
    rows += json.load(open(f))["rows"]
eps = sorted({r["ep"] for r in rows})
print("shards %d   episodes %d   rows %d" % (len(files), len(eps), len(rows)))

cases = []
for r in rows:
    if r["case"] not in cases:
        cases.append(r["case"])

per_ep = {}
for r in rows:
    per_ep.setdefault(r["case"], {})[r["ep"]] = r["iou"]

print("\n%-18s %9s %9s %9s %9s" % ("case", "Cumul", "Arith", "dCumul", "dArith"))
base_c = base_a = None
out = {}
for k in cases:
    tp = {c: 0 for c in NEW_IDS}
    fp = {c: 0 for c in NEW_IDS}
    fn = {c: 0 for c in NEW_IDS}
    ious = []
    for r in rows:
        if r["case"] != k:
            continue
        tp[r["cls"]] += r["tp"]
        fp[r["cls"]] += r["fp"]
        fn[r["cls"]] += r["fn"]
        ious.append(r["iou"])
    per = [tp[c] / max(tp[c] + fp[c] + fn[c], 1) for c in NEW_IDS]
    cu = float(np.mean(per))
    ar = float(np.mean(ious))
    if k == "Baseline":
        base_c, base_a = cu, ar
    print("%-18s %9.4f %9.4f %+9.4f %+9.4f" % (
        k, cu, ar, cu - base_c, ar - base_a))
    out[k] = dict(arith=ar, cumul=cu, per_class=per)

print("\n逐类 Cumul（Baseline vs 三个 λ=0.3）")
show = ["Baseline", "Naive-B_l0.3", "Oracle-B_l0.3", "Oracle-B+C_l0.3"]
show = [k for k in show if k in out]
print("%-14s" % "class" + "".join("%18s" % k for k in show))
for i, c in enumerate(NEW_IDS):
    print("%-14s" % NAMES[c] + "".join("%18.4f" % out[k]["per_class"][i]
                                       for k in show))

print("\n配对 t 检验（vs Baseline，逐集 IoU）")
for base in ["Naive-B", "Oracle-B", "Oracle-B+C"]:
    for lam in [0.2, 0.3, 0.5, 0.7, 0.9]:
        k = "%s_l%.1f" % (base, lam)
        if k not in per_ep:
            continue
        a = np.array([per_ep["Baseline"][e] for e in eps])
        b = np.array([per_ep[k][e] for e in eps])
        t, p = stats.ttest_rel(b, a)
        print("  %-14s d=%+.4f  t=%+6.2f  p=%.4g" % (
            k, float((b - a).mean()), t, p))

json.dump(dict(out=out, n_ep=len(eps)), open(
    os.path.join(CACHE, "ub_official_summary.json"), "w"), indent=1)
print("\n->", os.path.join(CACHE, "ub_official_summary.json"))
