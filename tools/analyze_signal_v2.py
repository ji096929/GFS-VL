#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v2 七信号正式分析：精确 AUROC + 加权 Precision@10% + 3D 否决精确率。

【为什么必须加权】

v2 采样时每 chunk 每类每极性上限 CAP=192。正负样本**都会被抽稀**：
    正样本采样率约 49%（95 万 / 195.6 万）
    负样本采样率约 29%（489 万 / 1689 万）
原因是少数大 chunk 里某类有几千个点，被 CAP 截断。所以正负比例从真实的
1:8.6 变成采样后的 1:5.1，朴素 Precision@10% 会高估纯度。

所以：
    * AUROC 不受影响（Mann-Whitney 只看排序，各自无偏采样不改变期望值）
    * Precision@10% 必须加权，且权重是「负相对正」的比值：
          w_neg = (n_neg_true / n_neg_sampled) / (n_pos_true / n_pos_sampled)

真实计数从 v1 的直方图累计拿（sum 不受 bin 量化影响）。

【额外输出：Exp B 的 3D 否决到底安不安全】

Exp B 用的是 `base_p > 0.7 且 pred 是 novel → 置 -1`。这个规则不需要 AUROC，
只需要「否决精确率」。对每类算：

    pos_kill_rate   真 novel 点被误杀的比例            ← 召回损失
    neg_kill_rate   假点被清除的比例                    ← 收益
    veto_precision  被否决的点里真正是假点的比例         ← 否决准不准
    purity_after    否决后伪标签纯度                    ← 训练信号质量

并对 tau ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95} 扫一遍。

用法
    python tools/analyze_signal_v2.py
"""
import glob
import json
import os

import numpy as np

V2_DIR = "/data01/jh/data/step1_auroc_v2"
V1_DIR = "/data01/jh/data/step1_auroc"
SIGNALS = (
    "S1_negent",
    "S2_margin",
    "S3a_proto_pred",
    "S3b_proto_margin",
    "S4a_nonbase",
    "S4b_pred",
    "S4c_netmargin",
)
NAMES = [
    "wall", "floor", "door", "ceiling", "table", "window", "box",
    "ceiling lamp", "light switch", "cabinet", "chair", "heater",
    "monitor", "whiteboard", "office chair", "bottle", "doorframe",
    "keyboard", "window frame", "mouse", "paper", "blinds", "trash can",
    "telephone", "book", "shelf", "sink", "windowsill", "bag",
    "smoke detector",
]
# 重点类必须全部落在 novel 区间 [12, 30)：base 是 0-11（wall~heater）
# smoke=29 office chair=14 keyboard=17 sink=26 shelf=25 trash can=22 doorframe=16
FOCUS = [29, 14, 17, 26, 25, 22, 16]


def auroc_exact(sp, sn):
    """精确 AUROC（ties 取 0.5）。"""
    if sp.size == 0 or sn.size == 0:
        return np.nan
    sns = np.sort(sn)
    lt = np.searchsorted(sns, sp, side="left")
    le = np.searchsorted(sns, sp, side="right")
    u = float(lt.sum()) + 0.5 * float((le - lt).sum())
    return u / (float(sp.size) * float(sn.size))


def wprec_at_keep(sp, sn, w_neg, frac=0.10):
    """加权 Precision@保留 frac 总权重（按分数从高到低累积）。

    w_neg：每个负样本代表多少个真实负样本（= n_neg_true / n_neg_sampled）。
    返回 (加权纯度, 该阈值对应的 S4a 分数)。
    """
    n_pos, n_neg = sp.size, sn.size
    if n_pos + n_neg == 0 or n_pos == 0:
        return np.nan, np.nan
    allv = np.concatenate([sp, sn])
    y = np.concatenate(
        [np.ones(n_pos, np.float64), np.zeros(n_neg, np.float64)]
    )
    w = np.concatenate(
        [np.ones(n_pos, np.float64), np.full(n_neg, float(w_neg), np.float64)]
    )
    order = np.argsort(-allv)
    cw = np.cumsum(w[order])
    total = cw[-1]
    k = int(np.searchsorted(cw, frac * total)) + 1
    k = min(k, allv.size)
    pure = float((y[order][:k] * w[order][:k]).sum() / cw[k - 1])
    return pure, float(allv[order][k - 1])


def veto_at(sp, sn, w_neg, n_pos_true, n_neg_true, tau):
    """给定 base_p > tau 的否决规则，返回误杀/清除/精确率/否决后纯度。"""
    thr = 1.0 - tau  # S4a = 1 - base_p
    kp = float((sp < thr).mean()) if sp.size else np.nan
    kn = float((sn < thr).mean()) if sn.size else np.nan
    if np.isnan(kp) or np.isnan(kn):
        return None
    n_pk = n_pos_true * kp
    n_nk = n_neg_true * kn
    den = n_pk + n_nk
    prec = n_nk / den if den > 0 else np.nan
    # 否决后仍保留的点（按真实点数算）
    kept_p = n_pos_true * (1.0 - kp)
    kept_n = n_neg_true * (1.0 - kn)
    purity = kept_p / (kept_p + kept_n) if (kept_p + kept_n) > 0 else np.nan
    return dict(
        pos_kill_rate=kp,
        neg_kill_rate=kn,
        n_pos_kill=n_pk,
        n_neg_kill=n_nk,
        veto_precision=prec,
        kept_pos=kept_p,
        kept_neg=kept_n,
        purity_after=purity,
        pos_recall=1.0 - kp,
        neg_removed=kn,
    )


def main():
    fs2 = sorted(glob.glob(os.path.join(V2_DIR, "shard*.npz")))
    if not fs2:
        print("没有 v2 分片")
        return
    with open(os.path.join(V2_DIR, "shard00.json")) as f:
        meta = json.load(f)
    nb, nc = meta["nb"], meta["nc"]
    cap = meta["cap"]

    # ---- 1) v1 真实计数（每类 n_pos / n_neg）----
    true_pos = np.zeros(nc, np.int64)
    true_neg = np.zeros(nc, np.int64)
    for f in sorted(glob.glob(os.path.join(V1_DIR, "shard*.npz"))):
        z = np.load(f)
        true_pos += z["hp_S1_negent"].sum(1)
        true_neg += z["hn_S1_negent"].sum(1)

    # ---- 2) v2 采样 ----
    samp_pos = {k: [[] for _ in range(nc)] for k in SIGNALS}
    samp_neg = {k: [[] for _ in range(nc)] for k in SIGNALS}
    for f in fs2:
        z = np.load(f)
        for k in SIGNALS:
            for c in range(nb, nc):
                v = z["%s__pos__%d" % (k, c)]
                if v.size:
                    samp_pos[k][c].append(v)
                v = z["%s__neg__%d" % (k, c)]
                if v.size:
                    samp_neg[k][c].append(v)

    P = {k: {} for k in SIGNALS}
    N = {k: {} for k in SIGNALS}
    for k in SIGNALS:
        for c in range(nb, nc):
            P[k][c] = (
                np.concatenate(samp_pos[k][c])
                if samp_pos[k][c]
                else np.zeros(0, np.float32)
            )
            N[k][c] = (
                np.concatenate(samp_neg[k][c])
                if samp_neg[k][c]
                else np.zeros(0, np.float32)
            )

    print("=" * 100)
    print("v2 七信号精确判别力（CAP=%d/chunk/class/polarity）" % cap)
    print("=" * 100)

    # ---- 3) 逐类 AUROC 与加权 P@10% ----
    auc = {k: np.full(nc, np.nan) for k in SIGNALS}
    wp = {k: np.full(nc, np.nan) for k in SIGNALS}
    # 正负样本都被 per-chunk CAP 抽稀（某些大 chunk 该类有几千点），
    # 所以权重必须是「负相对正」的比值，而不是负样本的绝对代表数。
    wneg = np.ones(nc)
    supp_ratio = np.zeros(nc)
    for c in range(nb, nc):
        nsp = P["S1_negent"][c].size
        nsn = N["S1_negent"][c].size
        rp = true_pos[c] / nsp if nsp > 0 else np.nan
        rn = true_neg[c] / nsn if nsn > 0 else np.nan
        if np.isfinite(rp) and rp > 0 and np.isfinite(rn) and rn > 0:
            wneg[c] = rn / rp
        if true_pos[c] + true_neg[c] > 0 and nsp + nsn > 0:
            supp_ratio[c] = (nsp + nsn) / float(true_pos[c] + true_neg[c])
    for k in SIGNALS:
        for c in range(nb, nc):
            sp, sn = P[k][c], N[k][c]
            auc[k][c] = auroc_exact(sp, sn)
            wp[k][c], _ = wprec_at_keep(sp, sn, wneg[c], 0.10)

    print("\n%s" % "=" * 100)
    print("[表 1] 真实样本量 vs 采样量（v1 全量计数，不受 v2 采样影响）")
    print(
        "%-2s %-15s %10s %12s %10s %9s %9s %8s"
        % ("id", "class", "nPos真实", "nNeg真实", "正样本率", "正采样率", "负采样率", "wNeg")
    )
    for c in range(nb, nc):
        tot = true_pos[c] + true_neg[c]
        if tot == 0:
            continue
        nsp = P["S1_negent"][c].size
        nsn = N["S1_negent"][c].size
        rs_p = nsp / true_pos[c] if true_pos[c] else np.nan
        rs_n = nsn / true_neg[c] if true_neg[c] else np.nan
        print(
            "%-2d %-15s %10d %12d %9.4f%% %8.3f %8.3f %8.2f"
            % (
                c,
                NAMES[c],
                true_pos[c],
                true_neg[c],
                100.0 * true_pos[c] / tot,
                rs_p,
                rs_n,
                wneg[c],
            )
        )
    tp_s = true_pos[nb:].sum()
    tn_s = true_neg[nb:].sum()
    print(
        "%-2s %-15s %10d %12d %9.4f%%"
        % ("", "合计", tp_s, tn_s, 100.0 * tp_s / max(tp_s + tn_s, 1))
    )

    print("\n%s" % "=" * 100)
    print("[表 2] 逐类 AUROC / 加权P@10%%（格式 AUROC / P@10）")
    hdr = "%-15s %9s %9s" % ("class", "nPosSamp", "nNegSamp")
    for k in SIGNALS:
        hdr += " %14s" % k
    print(hdr)
    print("-" * len(hdr))
    for c in range(nb, nc):
        if true_pos[c] == 0 and true_neg[c] == 0:
            continue
        line = "%-15s %9d %9d" % (
            NAMES[c],
            P["S1_negent"][c].size,
            N["S1_negent"][c].size,
        )
        for k in SIGNALS:
            line += "  %6.3f/%6.3f" % (auc[k][c], wp[k][c])
        print(line)

    print("\n%s" % "=" * 100)
    print("[表 3] 18 类汇总")
    print(
        "%-18s %10s %10s %12s %12s"
        % ("signal", "AUROC均", "AUROC中位", "P@10均", "P@10中位")
    )
    for k in SIGNALS:
        v = auc[k][nb:]
        v = v[~np.isnan(v)]
        p = wp[k][nb:]
        p = p[~np.isnan(p)]
        print(
            "%-18s %10.3f %10.3f %12.3f %12.3f"
            % (
                k,
                v.mean() if v.size else np.nan,
                np.median(v) if v.size else np.nan,
                p.mean() if p.size else np.nan,
                np.median(p) if p.size else np.nan,
            )
        )
    # 随机基线：正样本率
    rp = []
    for c in range(nb, nc):
        tot = true_pos[c] + true_neg[c]
        if tot:
            rp.append(true_pos[c] / tot)
    if rp:
        print(
            "%-18s %10s %10s %12.3f %12.3f"
            % ("[随机基线]", "0.500", "0.500", np.mean(rp), np.median(rp))
        )

    print("\n%s" % "=" * 100)
    print("[表 4] 重点类（AUROC / 加权P@10）")
    hdr = "%-15s" % "class"
    for k in SIGNALS:
        hdr += " %14s" % k
    print(hdr)
    for c in FOCUS:
        line = "%-15s" % NAMES[c]
        for k in SIGNALS:
            line += "  %6.3f/%6.3f" % (auc[k][c], wp[k][c])
        print(line)

    # ---- 4) 3D 否决分析 ----
    print("\n%s" % "=" * 100)
    print("[表 5] Exp B 的 3D 否决：tau 扫描（S4a = 1 - max_base P_3D）")
    print("  pos_kill=真 novel 被误杀率  neg_kill=假点清除率  prec=否决精确率  purity=否决后纯度")
    taus = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    print(
        "\n%-15s %5s %9s %9s %9s %9s %9s"
        % ("class", "tau", "pos_kill", "neg_kill", "prec", "purity", "posRecall")
    )
    for c in FOCUS:
        sp, sn = P["S4a_nonbase"][c], N["S4a_nonbase"][c]
        for tau in taus:
            r = veto_at(sp, sn, wneg[c], true_pos[c], true_neg[c], tau)
            if r is None:
                continue
            print(
                "%-15s %5.2f %8.3f%% %8.3f%% %9.3f %9.3f %8.3f%%"
                % (
                    NAMES[c] if tau == taus[0] else "",
                    tau,
                    100 * r["pos_kill_rate"],
                    100 * r["neg_kill_rate"],
                    r["veto_precision"],
                    r["purity_after"],
                    100 * r["pos_recall"],
                )
            )
        print()

    print("\n[表 6] 全 18 类汇总的 tau 扫描")
    print(
        "%-5s %11s %11s %9s %9s %9s %13s"
        % ("tau", "pos_kill", "neg_kill", "prec", "purity", "posRecall", "剩余真实点数")
    )
    for tau in taus:
        kp_l, kn_l, pk_l, nk_l, kp_n, kn_n = [], [], 0.0, 0.0, 0, 0
        for c in range(nb, nc):
            sp, sn = P["S4a_nonbase"][c], N["S4a_nonbase"][c]
            r = veto_at(sp, sn, wneg[c], true_pos[c], true_neg[c], tau)
            if r is None:
                continue
            kp_l.append(r["pos_kill_rate"])
            kn_l.append(r["neg_kill_rate"])
            pk_l += r["n_pos_kill"]
            nk_l += r["n_neg_kill"]
            kp_n += true_pos[c]
            kn_n += true_neg[c]
        if not kp_l:
            continue
        den = pk_l + nk_l
        kept_p = kp_n - pk_l
        kept_n = kn_n - nk_l
        pur = kept_p / (kept_p + kept_n) if (kept_p + kept_n) > 0 else np.nan
        print(
            "%-5.2f %10.3f%% %10.3f%% %9.3f %9.3f %8.3f%% %13d"
            % (
                tau,
                100 * float(np.mean(kp_l)),
                100 * float(np.mean(kn_l)),
                nk_l / den if den else np.nan,
                pur,
                100 * kept_p / max(kp_n, 1),
                kept_p,
            )
        )

    # ---- 5) 分数分布 ----
    print("\n%s" % "=" * 100)
    print("[表 7] 分数真实分布（证明 v1 的 [-3,3]/100bin 为何逐类恒等 0.5）")
    print("%-18s %10s %10s %10s %12s" % ("signal", "p1", "p50", "p99", "跨度p99-p1"))
    for k in SIGNALS:
        v = []
        for c in range(nb, nc):
            v.append(P[k][c])
            v.append(N[k][c])
        v = np.concatenate([x for x in v if x.size]) if any(
            x.size for x in v
        ) else None
        if v is None:
            continue
        p1, p50, p99 = np.percentile(v, [1, 50, 99])
        print(
            "%-18s %10.4f %10.4f %10.4f %12.4f"
            % (k, p1, p50, p99, p99 - p1)
        )

    # ---- 存盘 ----
    out = os.path.join(V2_DIR, "analysis.npz")
    np.savez_compressed(
        out,
        auc=np.array([auc[k] for k in SIGNALS]),
        wprec=np.array([wp[k] for k in SIGNALS]),
        signals=np.array(SIGNALS),
        names=np.array(NAMES),
        true_pos=true_pos,
        true_neg=true_neg,
        wneg=wneg,
    )
    print("\nsaved %s" % out)


if __name__ == "__main__":
    main()
