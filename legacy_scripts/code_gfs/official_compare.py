"""官方 5% 口径终极对比（纯后处理，不占 GPU）。

把四个来源的逐集 TP/FP/FN 统一到全局 episode 编号，再按官方
get_class2scans 的阈值过滤，重算所有方案的 Cumul / Arith。

来源与编号约定
--------------
ub560 / ub560x   4 分片，rows.ep 是分片内局部序号 -> g = shard + 4*ep
v3 (P3)          4 分片，rows.ep 是分片内局部序号 -> g = shard + 4*ep
v3g              4 分片，recs.g_ep 已全局；pred 是 logits，需现算 TP/FP/FN
v3i              8 分片，recs.g_ep 已全局，已存 *_tp/_fp/_fn

两种过滤
--------
A  query >= 5%
B  query >= 5% 且 support >= 5%   （严格对齐官方：support 与 query 都从同一过滤池抽）
"""
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, "/data01/jh/code")
sys.path.insert(0, "/data01/jh/code/COSeg")

from ocsb_core import build_episodes  # noqa: E402

CACHE = "/data01/jh/data/ocsb_cache"
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
NAMES = [l.strip() for l in open(
    "/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt")]

eps = build_episodes(56)
print("episodes:", len(eps))

# ---------- 1. 目标占比 ----------
gtr = json.load(open(os.path.join(CACHE, "val_query_gt.json")))
sp_cache = os.path.join(CACHE, "val_support_gt.json")
if os.path.exists(sp_cache):
    sgr = json.load(open(sp_cache))
else:
    from ring4_isolated import DATA
    sgr = {}
    for m in eps:
        k = "%s|%d" % (m["support"], m["class"])
        if k in sgr:
            continue
        lab = np.asarray(np.load(os.path.join(DATA, m["support"] + ".npy"),
                                 mmap_mode="r")[:, 6]).astype(int)
        sgr[k] = float((lab == m["class"]).mean())
    json.dump(sgr, open(sp_cache, "w"))

qk = lambda m: gtr["%s|%d" % (m["query"], m["class"])]
sk = lambda m: sgr["%s|%d" % (m["support"], m["class"])]
fA = [i for i, m in enumerate(eps) if qk(m) >= 0.05]
fB = [i for i, m in enumerate(eps) if qk(m) >= 0.05 and sk(m) >= 0.05]
print("539 集 -> query>=5%%: %d | query>=5%% & support>=5%%: %d" % (len(fA), len(fB)))


def _add(dst, g, key, tp, fp, fn, iou, cls):
    dst[(g, key)] = (int(tp), int(fp), int(fn), float(iou), int(cls))


# ---------- 2. ub560 / ub560x ----------
ub = {}
for pref, nsh in (("ub560", 4), ("ub560x", 4)):
    files = [f for f in sorted(glob.glob(os.path.join(CACHE, pref + "_s[0-9].json")))
             if ".partial" not in f]
    for f in files:
        i = int(re.search(r"_s(\d+)\.json$", f).group(1))
        for r in json.load(open(f))["rows"]:
            _add(ub, i + nsh * r["ep"], r["case"], r["tp"], r["fp"], r["fn"],
                 r["iou"], r["cls"])
print("ub560/x: %d 条记录" % len(ub))

# ---------- 3. v3 (P3, 零训练) ----------
p3 = {}
files = [f for f in sorted(glob.glob(os.path.join(CACHE, "v3_s[0-9].json")))
         if ".partial" not in f and "smoke" not in f]
for f in files:
    i = int(re.search(r"_s(\d+)\.json$", f).group(1))
    for r in json.load(open(f))["rows"]:
        _add(p3, i + 4 * r["ep"], r["step"], r["tp"], r["fp"], r["fn"],
             r["iou"], r["cls"])
print("v3: %d 条记录, %d 个 step" % (len(p3), len({k[1] for k in p3})))

# ---------- 4. v3g (折外训练, 现算) ----------
g3 = {}
for f in sorted(glob.glob(os.path.join(CACHE, "v3g_s*.pt"))):
    if "smoke" in f:
        continue
    for r in torch.load(f):
        g = int(r["g_ep"])
        gt = np.asarray(r["gt"]).astype(np.int64)
        for key, name in (("pred_s", "base"), ("pred_u_l0.3", "u_l03")):
            p = np.asarray(r[key]).argmax(0)
            tp = int(((p == 1) & (gt == 1)).sum())
            fp = int(((p == 1) & (gt == 0)).sum())
            fn = int(((p == 0) & (gt == 1)).sum())
            _add(g3, g, name, tp, fp, fn, tp / max(tp + fp + fn, 1), r["cls"])
print("v3g: %d 条记录" % len(g3))

# ---------- 5. v3i (自桥接) ----------
i3 = {}
for f in sorted(glob.glob(os.path.join(CACHE, "v3i_s*.pt"))):
    if "smoke" in f:
        continue
    for r in torch.load(f):
        g = int(r["g_ep"])
        for key, name in (("pred_s", "base"), ("pred_q_l03", "q_l03")):
            _add(i3, g, name, r[key + "_tp"], r[key + "_fp"], r[key + "_fn"],
                 r[key], r["cls"])
print("v3i: %d 条记录" % len(i3))

# ---------- 6. 对齐自检 ----------
print("\nBaseline 一致性自检（前 5 个 kept episode）")
for g in fA[:5]:
    vals = [("%s=%.6f" % (n, d[(g, k)][3])) for n, d, k in (
        ("ub", ub, "Baseline"), ("p3", p3, "Baseline"),
        ("v3g", g3, "base"), ("v3i", i3, "base")) if (g, k) in d]
    print("  g_ep %3d  %s" % (g, "  ".join(vals)))

ROWS = [
    ("Baseline", ub, "Baseline"),
    ("Naive-B l0.3 (跨房间)", ub, "Naive-B_l0.3"),
    ("P3 零训练 l0.7@0.65", p3, "P3_l0.7@0.65"),
    ("折外训练 u_l0.3", g3, "u_l03"),
    ("自桥接 q_l0.3", i3, "q_l03"),
    ("Oracle-B l0.5", ub, "Oracle-B_l0.5"),
    ("Oracle-B+C l0.5", ub, "Oracle-B+C_l0.5"),
    ("Oracle-B l0.9", ub, "Oracle-B_l0.9"),
    ("Oracle-B+C l0.9", ub, "Oracle-B+C_l0.9"),
    ("Naive-B l0.9", ub, "Naive-B_l0.9"),
]


def agg(src, key, subset):
    tp = {c: 0 for c in NEW_IDS}
    fp = {c: 0 for c in NEW_IDS}
    fn = {c: 0 for c in NEW_IDS}
    ious = []
    for g in subset:
        if (g, key) not in src:
            continue
        t, f, n, i, c = src[(g, key)]
        tp[c] += t
        fp[c] += f
        fn[c] += n
        ious.append(i)
    if not ious:
        return None
    per = [tp[c] / max(tp[c] + fp[c] + fn[c], 1) for c in NEW_IDS]
    return float(np.mean(per)), float(np.mean(ious)), len(ious), per


for tag, subset in (("A: query>=5%% (%d 集)" % len(fA), fA),
                    ("B: query>=5%% & support>=5%% (%d 集)" % len(fB), fB)):
    print("\n" + "=" * 78)
    print("过滤 " + tag)
    print("%-28s %5s %9s %9s %9s %9s" % (
        "方案", "N", "Cumul", "Arith", "dCumul", "dArith"))
    base = None
    for name, src, key in ROWS:
        r = agg(src, key, subset)
        if r is None:
            print("%-28s %5s  (无数据)" % (name, "-"))
            continue
        cu, ar, n, _ = r
        if name == "Baseline":
            base = (cu, ar)
        print("%-28s %5d %9.4f %9.4f %+9.4f %+9.4f" % (
            name, n, cu, ar, cu - base[0], ar - base[1]))

print("\n逐类 Cumul（过滤 B）")
show = [("Baseline", ub, "Baseline"), ("P3 零训练", p3, "P3_l0.7@0.65"),
        ("折外训练", g3, "u_l03"), ("自桥接", i3, "q_l03"),
        ("Oracle-B+C l0.9", ub, "Oracle-B+C_l0.9")]
print("%-14s" % "class" + "".join("%16s" % n for n, _, _ in show))
res = [(n, agg(s, k, fB)) for n, s, k in show]
for i, c in enumerate(NEW_IDS):
    line = "%-14s" % NAMES[c]
    for n, r in res:
        line += "%16s" % ("%.4f" % r[3][i] if r else "-")
    print(line)
