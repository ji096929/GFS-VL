"""按官方 get_class2scans 的逻辑生成 class2scans.pkl。

官方阈值：某类点数 > max(该文件总点数 * 5%, 100) 才把这个文件计入该类。
scene 级和 block 级都用同一段逻辑，只是 data_root 不同。

用法
----
  python make_class2scans.py --data-root /data01/jh/data/ScanNet/scenes/data
  python make_class2scans.py --data-root /data01/jh/data/scannet_train_v2/blocks_bs1.0_s1.0/data

默认输出到 <dirname(data_root)>/class2scans.pkl，与官方加载路径一致。
"""
import argparse
import glob
import os
import pickle

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--out", default=None,
                help="默认 <dirname(data_root)>/class2scans.pkl")
ap.add_argument("--class-count", type=int, default=21)
ap.add_argument("--min-ratio", type=float, default=0.05)
ap.add_argument("--min-pts", type=int, default=100)
args = ap.parse_args()

root = args.data_root.rstrip("/")
out = args.out or os.path.join(os.path.dirname(root), "class2scans.pkl")
files = sorted(glob.glob(os.path.join(root, "*.npy")))
print("files:", len(files), "-> ", out, flush=True)

c2s = {k: [] for k in range(args.class_count)}
for i, f in enumerate(files):
    lab = np.load(f, mmap_mode="r")[:, 6].astype(np.int64)
    thr = max(int(lab.shape[0] * args.min_ratio), args.min_pts)
    name = os.path.basename(f)[:-4]
    for c in np.unique(lab):
        if int((lab == c).sum()) > thr:
            c2s[int(c)].append(name)
    if (i + 1) % 2000 == 0:
        print("  %d / %d" % (i + 1, len(files)), flush=True)

with open(out, "wb") as fh:
    pickle.dump(c2s, fh, pickle.HIGHEST_PROTOCOL)

print("\nclass | n")
for c in range(args.class_count):
    print("%5d | %d" % (c, len(c2s[c])))
print("->", out)
