"""按 query 目标占比阈值重算 539 集 baseline / 自桥接（用已有缓存，不重跑）。"""
import os
import sys

sys.path.insert(0, "/data01/jh/code")
sys.path.insert(0, "/data01/jh/code/COSeg")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ocsb_core import NEW_IDS, build_episodes  # noqa: E402

D = "/data01/jh/data/ScanNet/blocks_bs1_s1/data"
CACHE = "/data01/jh/data/ocsb_cache"

eps = build_episodes(56)
print("episodes", len(eps))

ratios = []
for m in eps:
    r = np.load(os.path.join(D, m["query"] + ".npy"), mmap_mode="r")
    lab = np.asarray(r[:, 6]).astype(int)
    ratios.append(float((lab == m["class"]).mean()))

recs = []
for i in range(8):
    recs += torch.load(os.path.join(CACHE, "v3i_s%d.pt" % i))
recs.sort(key=lambda r: r["g_ep"])
assert len(recs) == len(eps), (len(recs), len(eps))
assert all(r["g_ep"] == i for i, r in enumerate(recs)), "g_ep 不连续"

print("\n%-6s %5s  %-11s %8s %8s %8s" % ("thr", "n", "case", "Cumul", "Arith", "dCumul"))
base_by_thr = {}
for thr in [0.0, 0.01, 0.03, 0.05, 0.10, 0.20]:
    keep = [i for i in range(len(eps)) if ratios[i] >= thr]
    for key, name in [("pred_s", "Baseline"), ("pred_q_l03", "SelfBridge")]:
        tp = {c: 0 for c in NEW_IDS}
        fp = {c: 0 for c in NEW_IDS}
        fn = {c: 0 for c in NEW_IDS}
        ious = []
        for i in keep:
            r = recs[i]
            c = r["cls"]
            tp[c] += r[key + "_tp"]
            fp[c] += r[key + "_fp"]
            fn[c] += r[key + "_fn"]
            ious.append(r[key])
        per = [tp[c] / max(tp[c] + fp[c] + fn[c], 1) for c in NEW_IDS]
        cu = float(np.mean(per))
        if name == "Baseline":
            base_by_thr[thr] = cu
            print("%-6.2f %5d  %-11s %8.4f %8.4f %8s" % (
                thr, len(keep), name, cu, np.mean(ious), "-"))
        else:
            print("%-6.2f %5d  %-11s %8.4f %8.4f %+8.4f" % (
                thr, len(keep), name, cu, np.mean(ious), cu - base_by_thr[thr]))
