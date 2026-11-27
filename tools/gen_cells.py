# -*- coding: utf-8 -*-
"""按「结构 × 数据集 × shot × 伪标签模式 × 损失」列出可用的三 seed 格子。

**结构从权重文件的 key 判定，不从 config.py 判定。**

原因（2026-11-27 实测）：
`exp/scannet200/sc200_k1_sym_s42/config.py` 的 mtime 是 11-26，`train.log` 是 11-20——
config 被后来的操作覆盖过，内容与训练时不符。权重文件的 key 才是真身。

判据（只读 zip 内的 data.pkl，0.02 秒/文件，不加载张量）：

| 权重里有 | 结构 |
|---|---|
| `novel_cross_head` | **M5**（dec2-only + 两个零初始化残差）|
| `dec0_norm`（且无 novel_cross_head） | **sym**（192 维拼接 + 新类 dec0 过 LN）|
| `base_cross_head`（且无上面两个） | **xl**（只有基类接跨层）|
| 都没有 | **base** |

用法: python tools/gen_cells.py
"""
import os
import re
import zipfile
from collections import defaultdict

ROOT = "/data01/jh/code/GFS-VL/exp"
TOPLEVEL = ("seed", "k_shot")
FIELDS = ("lam_cross", "decorr_weight", "freeze_backbone", "cross_grad_scale",
          "k_shot", "seed", "use_lovasz", "pl_mode")


def cfg_get(src, key):
    pat = r"^%s[ \t]*=[ \t]*([^\n]+)" % key if key in TOPLEVEL else \
          r"^[ \t]*%s[ \t]*=[ \t]*([^\n]+)" % key
    m = re.search(pat, src, re.M)
    return m.group(1).strip().rstrip(",").strip() if m else None


def struct_from_ckpt(path):
    """只读 zip 里的 data.pkl，看 key 名。不加载任何张量。"""
    try:
        with zipfile.ZipFile(path) as z:
            name = [n for n in z.namelist() if n.endswith("data.pkl")]
            if not name:
                return None
            b = z.read(name[0])
    except (OSError, zipfile.BadZipFile):
        return None
    if b"novel_cross_head" in b:
        return "M5"
    if b"dec0_norm" in b:
        return "sym"
    if b"base_cross_head" in b:
        return "xl"
    return "base"


def collect():
    rows = []
    for ds in ("scannetpp", "scannet200"):
        dp = os.path.join(ROOT, ds)
        if not os.path.isdir(dp):
            continue
        for name in sorted(os.listdir(dp)):
            d = os.path.join(dp, name)
            cfgp = os.path.join(d, "config.py")
            if not os.path.exists(cfgp):
                continue
            src = open(cfgp, encoding="utf-8", errors="ignore").read()
            c = {k: cfg_get(src, k) for k in FIELDS}
            st = struct_from_ckpt(os.path.join(d, "model", "regis1_model_last.pth"))

            hm, online = None, None
            lg = os.path.join(d, "train.log")
            if os.path.exists(lg):
                t = open(lg, encoding="utf-8", errors="ignore").read()
                online = "在线" if "sparseunet32_636.pth to GPU" in t else "离线"
                hits = re.findall(r"hm_mIoU: ([\d.]+)", t)
                if hits:
                    hm = float(hits[-1])

            excl = []
            if c.get("freeze_backbone") == "True" or "frozen" in name:
                excl.append("frozen")
            if "oracle" in name:
                excl.append("oracle")
            if c.get("lam_cross") not in (None, "0", "0.0"):
                excl.append("lcross")
            if c.get("decorr_weight") not in (None, "0", "0.0"):
                excl.append("decorr")
            if c.get("cross_grad_scale") not in (None, "1.0", "1"):
                excl.append("gradscale")

            rows.append(dict(ds=ds, name=name, st=st, k=c.get("k_shot"),
                             seed=c.get("seed"), online=online,
                             lov=c.get("use_lovasz") == "True", hm=hm,
                             plm=c.get("pl_mode") or "ps_ai", excl=excl))
    return rows


def main():
    rows = collect()
    no_w = [r["name"] for r in rows if r["st"] is None]
    print("# 无权重、结构无法判定（%d 个）" % len(no_w))
    if no_w:
        print("  " + ", ".join(no_w[:12]) + ("..." if len(no_w) > 12 else ""))
    print()

    # 结构分布
    cnt = defaultdict(int)
    for r in rows:
        cnt[r["st"]] += 1
    print("# 结构分布（按权重判定）")
    for k in sorted(cnt, key=lambda x: -cnt[x]):
        print("  %-6s %d" % (k, cnt[k]))
    print()

    cell = defaultdict(dict)
    for r in rows:
        if r["excl"] or r["st"] is None or r["seed"] not in ("42", "7", "13") or not r["hm"]:
            continue
        key = (r["ds"], r["k"], r["st"], r["online"], r["lov"], r["plm"])
        cur = cell[key].get(r["seed"])
        if cur is None or len(r["name"]) < len(cur[1]):
            cell[key][r["seed"]] = (r["hm"], r["name"])

    print("# 三 seed 齐全的格子")
    print("%-6s %-3s %-6s %-4s %-3s %-14s   s42     s7      s13     mean" % (
        "数据集", "shot", "结构", "模式", "lov", "PL方案"))
    print("-" * 100)
    for key in sorted(cell):
        d = cell[key]
        if set(d) != {"42", "7", "13"}:
            continue
        ds = key[0].replace("scannetpp", "Sc++").replace("scannet200", "Sc200")
        m = (d["42"][0] + d["7"][0] + d["13"][0]) / 3
        print("%-6s %-3s %-6s %-4s %-3s %-14s   %.4f  %.4f  %.4f  %.4f" % (
            ds, key[1], key[2], key[3] or "-", "Y" if key[4] else "N", key[5],
            d["42"][0], d["7"][0], d["13"][0], m))
        print("        runs: %s / %s / %s" % (d["42"][1], d["7"][1], d["13"][1]))


if __name__ == "__main__":
    main()
