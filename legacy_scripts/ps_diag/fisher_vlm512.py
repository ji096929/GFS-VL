# -*- coding: utf-8 -*-
"""拆解 2 第一把尺子：512d VLM 特征的 Fisher 判别比 + 三阶段逐类明细。

输入 pl_tristage_audit_s0.npz（含 feat_sum/feat_sq/feat_cnt 与三阶段混淆矩阵）

Fisher 比定义（与拆解 3 同口径）：
  F(i,j) = 2 * ||mu_i - mu_j|| / (sigma_i + sigma_j)
  sigma_c = sqrt( sum_d Var(x_d | class c) )   —— 各向同性类内半径
  F > 1 才可能被线性分开
"""
import sys

import numpy as np

NO, NN = 12, 45
NC = NO + NN
P = "/data01/jh/data/cm_analysis/pl_tristage_audit_s0.npz"


def main():
    z = np.load(P)
    fs, fq, fc = z["feat_sum"], z["feat_sq"], z["feat_cnt"]
    print("feature moments: sum %s  cnt %s" % (fs.shape, fc.shape))

    mu = np.zeros((NC, fs.shape[1]), dtype=np.float64)
    sig = np.zeros(NC, dtype=np.float64)
    for c in range(NC):
        n = fc[c]
        if n < 100:
            continue
        m = fs[c] / n
        var_d = np.maximum(fq[c] / n - m * m, 0.0)
        mu[c] = m
        sig[c] = np.sqrt(var_d.sum())

    ok = [c for c in range(NC) if fc[c] >= 100]
    print("有效类（>=100 点）: %d / %d" % (len(ok), NC))

    # ---- Top-10 混淆对（来自正式 val 混淆矩阵）----
    pairs = [(39, 12), (21, 12), (30, 48), (16, 28), (22, 20),
             (52, 17), (49, 20), (17, 53), (56, 51), (37, 40)]
    print()
    print("【Fisher 比】512d VLM 特征上的 Top-10 混淆对")
    print("%-10s %10s %10s %8s  %s" % ("pair", "类间距离", "sigma_i+s_j", "Fisher", "判定"))
    fs_list = []
    for a, b in pairs:
        if fc[a] < 100 or fc[b] < 100:
            print("%-10s  缺类" % ("%d->%d" % (a, b)))
            continue
        d = np.linalg.norm(mu[a] - mu[b])
        s = sig[a] + sig[b]
        f = 2 * d / max(s, 1e-9)
        fs_list.append(f)
        print("%-10s %10.3f %10.3f %8.2f  %s"
              % ("%d->%d" % (a, b), d, s, f,
                 "可分" if f > 1 else "不可分"))
    print("-" * 60)
    print("Top-10 Fisher 均值 %.2f  中位数 %.2f"
          % (np.mean(fs_list), np.median(fs_list)))

    # ---- 6 个漏标类 vs Base / 背景 ----
    blind = [15, 30, 42, 47]
    bg_bucket = NC     # feat_cnt 的最后一档是 gfs==-1
    print()
    print("【漏标类 vs 12 类 Base / 背景】512d 特征上的 Fisher 比")
    print("%-8s %-12s %10s  %s" % ("cls", "对比对象", "Fisher", "判定"))
    for c in blind:
        if fc[c] < 100:
            print("%-8d 样本不足（%d 点）" % (c, int(fc[c])))
            continue
        # 与最近的 Base 类
        best_b, best_f = None, -1
        for b in range(NO):
            if fc[b] < 100:
                continue
            d = np.linalg.norm(mu[c] - mu[b])
            f = 2 * d / max(sig[c] + sig[b], 1e-9)
            if f > best_f:
                best_f, best_b = f, b
        print("%-8d Base=%d（最近）%12.2f  %s"
              % (c, best_b, best_f, "可分" if best_f > 1 else "不可分"))
        # 与背景桶
        if fc[bg_bucket] >= 100:
            n_bg = fc[bg_bucket]
            m_bg = fs[bg_bucket] / n_bg
            var_bg = np.maximum(fq[bg_bucket] / n_bg - m_bg * m_bg, 0.0)
            s_bg = np.sqrt(var_bg.sum())
            d = np.linalg.norm(mu[c] - m_bg)
            f = 2 * d / max(sig[c] + s_bg, 1e-9)
            print("%-8s 背景桶 %12.2f  %s"
                  % ("", f, "可分" if f > 1 else "不可分"))

    # ---- 三阶段逐类明细 ----
    print()
    print("=" * 84)
    print("【三阶段逐类】PL 判为该类的点数 / 该类 GT 点数（覆盖率），以及串出率")
    print("=" * 84)
    gt = z["n_gt"][0]
    # 从 agg 的 cm 反推每类 GT 量（cm 行和）
    cm_raw = z["raw_cm"]
    cm_ps = z["ps_cm"]
    cm_psai = z["psai_cm"]
    gt_per = cm_raw.sum(1).astype(np.float64)
    print("%-6s %9s | %s" % ("cls", "GT点", "  raw覆盖   ps覆盖  psai覆盖 | raw串出  ps串出 psai串出"))
    rows = []
    for j in range(NN):
        c = NO + j
        g = gt_per[j]
        if g < 100:
            continue
        r_cov = cm_raw[j].sum() / g
        p_cov = cm_ps[j].sum() / g
        a_cov = cm_psai[j].sum() / g
        r_out = 1 - cm_raw[j, j] / max(g, 1)
        p_out = 1 - cm_ps[j, j] / max(g, 1)
        a_out = 1 - cm_psai[j, j] / max(g, 1)
        rows.append((g, c, r_cov, p_cov, a_cov, r_out, p_out, a_out))
    rows.sort(reverse=True)
    for g, c, rc, pc, ac, ro, po, ao in rows[:25]:
        print("%-6d %9.0f | %7.1f%% %7.1f%% %7.1f%%   | %6.1f%% %6.1f%% %7.1f%%"
              % (c, g, 100 * rc, 100 * pc, 100 * ac,
                 100 * ro, 100 * po, 100 * ao))

    print()
    print("【汇总】覆盖率与串出率的三阶段变化（45 类平均）")
    for name, idx in (("raw", 2), ("ps", 3), ("psai", 4)):
        pass
    cov_r = np.array([r[2] for r in rows])
    cov_p = np.array([r[3] for r in rows])
    cov_a = np.array([r[4] for r in rows])
    out_r = np.array([r[5] for r in rows])
    out_p = np.array([r[6] for r in rows])
    out_a = np.array([r[7] for r in rows])
    print("  覆盖率:  raw %.1f%%  ps %.1f%%  psai %.1f%%"
          % (100 * cov_r.mean(), 100 * cov_p.mean(), 100 * cov_a.mean()))
    print("  串出率:  raw %.1f%%  ps %.1f%%  psai %.1f%%"
          % (100 * out_r.mean(), 100 * out_p.mean(), 100 * out_a.mean()))


if __name__ == "__main__":
    main()
