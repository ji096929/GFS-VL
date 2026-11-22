"""合并 eRank 分片并计算有效秩。"""
import glob
import sys

import numpy as np


def erank(X):
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    w = np.linalg.eigvalsh(C.astype(np.float64))
    w = np.maximum(w, 0.0)
    p = w / w.sum()
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


d = sys.argv[1] if len(sys.argv) > 1 else "/data01/jh/afford_diag/xl_runs"
pat = sys.argv[2] if len(sys.argv) > 2 else "erank_s42_s*.npz"
files = sorted(glob.glob(d + "/" + pat))
print("shards: %d" % len(files))
acc = {}
for f in files:
    z = np.load(f)
    for k in z.files:
        if k == "gt":
            continue
        acc.setdefault(k, []).append(z[k])
for k in sorted(acc):
    X = np.concatenate(acc[k], 0)
    print("  %-8s D=%3d N=%d eRank=%.2f"
          % (k, X.shape[1], len(X), erank(X)))
