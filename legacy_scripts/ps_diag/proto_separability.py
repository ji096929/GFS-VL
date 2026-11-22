# -*- coding: utf-8 -*-
"""拆解 3：1-shot 的 dec1/dec2 原型在 Top-10 混淆对上到底能不能分开？

直接证伪或证实「1-shot 原型方差大、会误杀同类真值」这个风险。

指标：
  1. 每个 novel 类的 support 点数、类内平均半径（到原型距离）
  2. 对 Top-K 混淆对 (c_i, c_j)：
     - 原型余弦相似度
     - Fisher 判别比 = 2*||mu_i - mu_j|| / (r_i + r_j)，> 1 才可能线性分开
     - 用「最近原型」分类，看 c_i 的点有多少被判成 c_j（自测口径）
  3. 三个层（dec0/dec1/dec2）并排，看高秩层是否真的更可分
"""
import sys

import numpy as np

SUP = "/data01/jh/data/cut2_feat/S1/support_s0.npz"
NO = 12


def fisher_and_confusion(X, gt, pairs, classes):
    """返回每层的原型、半径、对每对的 Fisher 比与自测混淆率。"""
    protos, radii, counts = {}, {}, {}
    for c in classes:
        w = gt == c
        if w.sum() < 2:
            continue
        P = X[w]
        mu = P.mean(0)
        protos[c] = mu
        radii[c] = np.linalg.norm(P - mu, axis=1).mean()
        counts[c] = int(w.sum())

    # 全部 novel 类做最近原型分类（自测）
    cls_list = sorted(protos.keys())
    Pm = np.stack([protos[c] for c in cls_list])          # (K, D)
    Pn = Pm / np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-9)
    res = {}
    for a, b in pairs:
        if a not in protos or b not in protos:
            res[(a, b)] = None
            continue
        d = np.linalg.norm(protos[a] - protos[b])
        fish = 2 * d / max(radii[a] + radii[b], 1e-9)
        cos = float(np.dot(Pn[cls_list.index(a)], Pn[cls_list.index(b)]))
        # a 的点被判成 b 的比例（自测）
        wa = gt == a
        Xa = X[wa]
        Xn = Xa / np.maximum(np.linalg.norm(Xa, axis=1, keepdims=True), 1e-9)
        pred = np.argmax(Xn @ Pn.T, axis=1)
        to_b = float((pred == cls_list.index(b)).mean())
        res[(a, b)] = dict(dist=float(d), fisher=float(fish), cos=cos,
                           to_b=to_b, n_a=counts[a], n_b=counts[b],
                           r_a=float(radii[a]), r_b=float(radii[b]))
    return res, protos, radii, counts


def main():
    z = np.load(SUP)
    gt = z["gt"].astype(np.int64)
    classes = [c for c in range(NO, 57)]

    # Top-10 混淆对（来自正式 val 混淆矩阵，按绝对量）
    pairs = [(39, 12), (21, 12), (30, 48), (16, 28), (22, 20),
             (52, 17), (49, 20), (17, 53), (56, 51), (37, 40)]

    print("=" * 92)
    print("拆解 3：1-shot support 原型的可分性（来源 %s）" % SUP.split("/")[-1])
    print("=" * 92)
    print("novel 类中有 support 点的: %d / 45" % len(set(gt[gt >= NO].tolist())))
    print()

    layers = ["f_dec0", "f_dec1", "f_dec2"]
    all_res = {}
    for L in layers:
        X = z[L].astype(np.float32)
        # 先做 LN（和模型里的 dec0_norm / cross_norms 对齐）
        mu = X.mean(0, keepdims=True)
        sd = X.std(0, keepdims=True) + 1e-6
        X = (X - mu) / sd
        res, protos, radii, counts = fisher_and_confusion(X, gt, pairs, classes)
        all_res[L] = (res, protos, radii, counts)
        print("%s: 覆盖 %d 类  support 点 %d" % (L, len(protos), int(X.shape[0])))

    print()
    print("-" * 92)
    print("Top-10 混淆对：Fisher 判别比（>1 才可能被线性分开）与自测串类率")
    print("-" * 92)
    hdr = "%-10s" % "pair"
    for L in layers:
        hdr += " | %-22s" % L.replace("f_", "")
    print(hdr)
    for a, b in pairs:
        line = "%-10s" % ("%d->%d" % (a, b))
        for L in layers:
            r = all_res[L][0].get((a, b))
            if r is None:
                line += " | %-22s" % "缺类"
            else:
                line += " | F=%5.2f 混%4.1f%%  " % (r["fisher"], 100 * r["to_b"])
        print(line)

    print()
    print("-" * 92)
    print("汇总：三层的 Fisher 比与串类率（Top-10 平均）")
    print("-" * 92)
    print("%-8s %12s %14s %12s" % ("layer", "Fisher均值", "自测串类率均值", "有效类数"))
    for L in layers:
        res = all_res[L][0]
        fs = [v["fisher"] for v in res.values() if v]
        ts = [v["to_b"] for v in res.values() if v]
        print("%-8s %12.2f %13.1f%% %12d"
              % (L.replace("f_", ""), np.mean(fs), 100 * np.mean(ts), len(res)))

    # 判定
    print()
    fd1 = np.mean([v["fisher"] for v in all_res["f_dec1"][0].values() if v])
    fd2 = np.mean([v["fisher"] for v in all_res["f_dec2"][0].values() if v])
    td1 = np.mean([v["to_b"] for v in all_res["f_dec1"][0].values() if v])
    td2 = np.mean([v["to_b"] for v in all_res["f_dec2"][0].values() if v])
    print("【判定】")
    print("  dec1 Fisher %.2f / dec2 Fisher %.2f  -> %s"
          % (fd1, fd2,
             "高秩层确实更可分" if fd2 > fd1 * 1.1 else "高秩层不比低秩更可分"))
    print("  dec2 平均串类率 %.1f%%  -> %s"
          % (100 * td2,
             "1-shot 原型本身就能分开" if td2 < 0.20 else "1-shot 原型本身就糊，原型分支收益有限"))


if __name__ == "__main__":
    main()
