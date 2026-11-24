# -*- coding: utf-8 -*-
"""Step 0 汇总：4 种过滤策略的每类 Precision / Recall / 空房间假阳性倍数。

输入：/data01/jh/data/step0/merged.npz（由 probe_step0_filter_compare.py --merge 生成）

策略：
    S1  单场景 Q_0.1 子空间熵（逐 chunk，精确统计；空房间 FP 用保留率近似）
    S2  跨场景全局 Q_0.1 子空间熵
    S3  跨场景全局 Q_0.1 绝对余弦 max(cos_nv)
    S4  支持集锚定绝对余弦（support 点的 sim 分位 p25/p50/p75）

指标：
    Precision = TP / 保留点数        （bg 区域的预测一律算 FP）
    Recall    = TP / support GT 点数
    empty×    = 空房间（support 里没有该类）的保留点数 / GT 规模
"""
import numpy as np

MERGED = "/data01/jh/data/step0/merged.npz"
NAMES = ['wall', 'floor', 'door', 'ceiling', 'table', 'window', 'box',
         'ceiling lamp', 'light switch', 'cabinet', 'chair', 'heater',
         'monitor', 'whiteboard', 'office chair', 'bottle', 'doorframe',
         'keyboard', 'window frame', 'mouse', 'paper', 'blinds', 'trash can',
         'telephone', 'book', 'shelf', 'sink', 'windowsill', 'bag',
         'smoke detector']


def cum(hist, edges, thr):
    """thr 及以下的累积点数（逐类）。"""
    idx = np.searchsorted(edges, thr, side="right") - 1
    idx = np.clip(idx, 0, hist.shape[1] - 1)
    out = np.zeros(hist.shape[0], dtype=np.float64)
    for c in range(hist.shape[0]):
        out[c] = hist[c, : idx[c] + 1].sum()
    return out


def quantile(hist, edges, q):
    """从直方图算分位阈值（逐类）。"""
    cumsum = np.cumsum(hist, axis=1)
    total = cumsum[:, -1]
    out = np.full(hist.shape[0], np.nan)
    for c in range(hist.shape[0]):
        if total[c] < 50:      # 样本太少，不给阈值
            continue
        target = total[c] * q
        idx = int(np.searchsorted(cumsum[c], target))
        out[c] = edges[min(idx, len(edges) - 1)]
    return out


def main():
    d = np.load(MERGED, allow_pickle=True)
    meta = d["meta"].item()
    nc, nbin = meta["nc"], meta["nbin"]
    ent_edges = np.linspace(0, meta["ent_max"], nbin + 1)
    sim_edges = np.linspace(meta["sim_min"], meta["sim_max"], nbin + 1)

    get = lambda k: d[k].astype(np.float64)
    n_supp = get("n_supp_gt")
    n_pred_with = get("n_pred_with")
    n_pred_empty = get("n_pred_empty")
    n_pred_all = n_pred_with + n_pred_empty

    res = {}

    # ---- S1：逐 chunk Q0.1（精确 tp / 保留数；空房间按保留率近似）----
    pred_s1 = get("n_pred_s1")
    tp_s1 = get("tp_s1")
    ratio_s1 = np.where(n_pred_all > 0, pred_s1 / np.maximum(n_pred_all, 1), 0.0)
    res["S1"] = dict(keep=pred_s1, tp=tp_s1,
                     keep_empty=n_pred_empty * ratio_s1,
                     thr=np.full(nc, np.nan))

    # ---- S2：全局熵 Q0.1 ----
    ent_all = get("ent_hist_with") + get("ent_hist_empty")
    thr2 = quantile(ent_all, ent_edges, 0.1)
    res["S2"] = dict(
        keep=cum(ent_all, ent_edges, thr2),
        tp=cum(get("supp_ent_hist"), ent_edges, thr2),
        keep_empty=cum(get("ent_hist_empty"), ent_edges, thr2),
        thr=thr2)

    # ---- S3：全局绝对余弦 Q0.1（越小越不确定 → 保留 sim 高的？这里测「保留下分位」）----
    # 注意：绝对余弦越大越可信，所以「Q0.1」应理解为保留 sim 最高的 10%，
    # 即阈值取 p90；这里同时给出 p90（保留前 10%）的版本。
    sim_all = get("sim_hist_with") + get("sim_hist_empty")
    thr3 = quantile(sim_all, sim_edges, 0.9)
    res["S3"] = dict(
        keep=sim_all.sum(1) - cum(sim_all, sim_edges, thr3),
        tp=get("supp_sim_hist").sum(1) - cum(get("supp_sim_hist"), sim_edges, thr3),
        keep_empty=get("sim_hist_empty").sum(1) - cum(get("sim_hist_empty"), sim_edges, thr3),
        thr=thr3)

    # ---- S4：support 锚定（support 正确点的 sim 分位）----
    for p in (25, 50, 75):
        thr4 = quantile(get("supp_sim_hist"), sim_edges, p / 100.0)
        res["S4p%d" % p] = dict(
            keep=sim_all.sum(1) - cum(sim_all, sim_edges, thr4),
            tp=get("supp_sim_hist").sum(1) - cum(get("supp_sim_hist"), sim_edges, thr4),
            keep_empty=get("sim_hist_empty").sum(1) - cum(get("sim_hist_empty"), sim_edges, thr4),
            thr=thr4)

    # ---- 打印 ----
    keys = ["S1", "S2", "S3", "S4p25", "S4p50", "S4p75"]
    print("chunks=%d  novel 类=%d\n" % (meta["n_chunks"] * meta["shards"], nc - meta["nb"]))
    hdr = "%-2s %-15s %7s %7s" % ("id", "class", "supGT", "predAll")
    for k in keys:
        hdr += " | %-18s" % k
    print(hdr)
    print("-" * len(hdr))

    for c in range(meta["nb"], nc):
        line = "%-2d %-15s %7d %7d" % (c, NAMES[c], n_supp[c], n_pred_all[c])
        for k in keys:
            r = res[k]
            prec = r["tp"][c] / max(r["keep"][c], 1)
            rec = r["tp"][c] / max(n_supp[c], 1)
            emp = r["keep_empty"][c] / max(n_supp[c], 1)
            line += " | P%.3f R%.3f %5.1fx" % (prec, rec, emp)
        print(line)

    print("\n=== 汇总（18 个 novel 类均值）===")
    print("%-8s %10s %10s %14s %14s" % ("策略", "Precision", "Recall", "保留点数", "空房间FP"))
    for k in keys:
        r = res[k]
        p = r["tp"][meta["nb"]:].sum() / max(r["keep"][meta["nb"]:].sum(), 1)
        rc = r["tp"][meta["nb"]:].sum() / max(n_supp[meta["nb"]:].sum(), 1)
        emp = r["keep_empty"][meta["nb"]:].sum() / max(n_supp[meta["nb"]:].sum(), 1)
        print("%-8s %10.4f %10.4f %14d %13.1fx" % (
            k, p, rc, int(r["keep"][meta["nb"]:].sum()), emp))

    print("\n=== 重点类（smoke detector / ceiling lamp / ceiling / sink / shelf）===")
    for c in (29, 7, 3, 26, 25):
        print("\n-- %s (supGT=%d) --" % (NAMES[c], n_supp[c]))
        for k in keys:
            r = res[k]
            print("   %-6s keep=%-9d tp=%-8d P=%.3f R=%.3f empty=%d (%.1fx)" % (
                k, r["keep"][c], r["tp"][c],
                r["tp"][c] / max(r["keep"][c], 1), r["tp"][c] / max(n_supp[c], 1),
                r["keep_empty"][c], r["keep_empty"][c] / max(n_supp[c], 1)))

    np.savez_compressed("/data01/jh/data/step0/summary.npz",
                        **{k + "_" + m: v[m] for k, v in res.items() for m in ("keep", "tp", "keep_empty", "thr")})
    print("\nsaved /data01/jh/data/step0/summary.npz")


if __name__ == "__main__":
    main()
