"""合并 sonata_b_auc.py 的 8 个分片，输出全局 AUC 与头部精度。"""
import glob

import numpy as np
from scipy.stats import mannwhitneyu

files = sorted(glob.glob("/data01/jh/data/sonata_dump/auc_s*.npz"))
print("shards:", len(files))
sc, lb, tk = [], [], []
for f in files:
    d = np.load(f)
    sc.append(d["score"]); lb.append(d["label"]); tk.append(d["task"])
sc = np.concatenate(sc); lb = np.concatenate(lb); tk = np.concatenate(tk)
pos, neg = sc[lb == 1], sc[lb == 0]
u, p = mannwhitneyu(pos, neg, alternative="greater")
auc = u / (len(pos) * len(neg))
print("\n=== Sonata(small) 独立特征判别 B_pos / B_neg ===")
print("总 B 点数 %d  (pos %d / neg %d)  正例率 %.1f%%"
      % (len(sc), len(pos), len(neg), 100 * len(pos) / len(sc)))
print("AUC = %.4f   p = %.3g" % (auc, p))
order = np.argsort(-sc)
print("\n按 score 排序的头部精度")
for frac in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
    k = max(int(len(sc) * frac), 1)
    print("  Top %4.1f%% (n=%6d): precision = %.4f"
          % (100 * frac, k, lb[order[:k]].mean()))
print("\n对照：COSeg 冻结特征的上限是 0.625 ~ 0.662")
