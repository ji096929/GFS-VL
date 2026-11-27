# -*- coding: utf-8 -*-
"""生成「结构 vs 配方」全表。

结构代号按 config.py 的参数指纹判定，**不看目录名**。
用法: python tools/gen_structure_map.py > docs/00_ledger/STRUCTURE_MAP.md
"""
import os
import re
from collections import defaultdict

ROOT = "/data01/jh/code/GFS-VL/exp"
DATASETS = ("scannetpp", "scannet200")
FIELDS = ("cross_layers", "cross_dims", "cross_zero_head", "base_cross",
          "cross_direct", "rep_dec0", "depth_gate", "mlp_hidden", "ln_only",
          "k_shot", "seed", "use_lovasz", "pl_mode")

STRUCT_DEF = [
    ("base", "无 `cross_layers`", "只吃 dec0"),
    ("xl", "有跨层 + `base_cross=False`", "只有新类头接跨层"),
    ("sym", "`cross_layers=(1,2)` + `base_cross=True`", "新类拼接 192 + 基类残差 128（新类 dec0 过 LN）"),
    ("M5", "`(2,)` + `base_cross=True` + `zero_head=True`", "两路零初始化残差，都用原始 dec0"),
    ("m9", "同 M5 但 `cross_direct=True`", "dec2 直连，不经过投影层"),
    ("dec2only", "`(2,)` + `base_cross=True` + 无 zero_head", "dec2 拼进新类头"),
    ("rep3", "`rep_dec0=True`", "dec0 复制两份冒充跨层（对照）"),
    ("gated", "`depth_gate=True`", "逐点门控"),
    ("mlp_dec0", "`mlp_hidden>0`", "dec0-only 两层 MLP"),
    ("lnonly", "`ln_only=True`", "只加 LayerNorm"),
]
ORDER = [x[0] for x in STRUCT_DEF]


def _get(src, key):
    m = re.search(r"^[ \t]*%s[ \t]*=[ \t]*([^\n]+)" % key, src, re.M)
    return m.group(1).strip().rstrip(",").strip() if m else None


def struct_of(c):
    L = c.get("cross_layers")
    bc = c.get("base_cross") == "True"
    zh = c.get("cross_zero_head") == "True"
    dr = c.get("cross_direct") == "True"
    rp = c.get("rep_dec0") == "True"
    gt = c.get("depth_gate") == "True"
    mh = c.get("mlp_hidden", "0")
    lo = c.get("ln_only") == "True"
    if lo:
        return "lnonly"
    if mh not in ("0", None):
        return "mlp_dec0"
    if not L or L == "-":
        return "base"
    if rp:
        return "rep3"
    if gt:
        return "gated"
    if L.startswith("(2") and bc and zh and dr:
        return "m9"
    if L.startswith("(2") and bc and zh:
        return "M5"
    if L.startswith("(1") and bc:
        return "sym"
    if L.startswith("(1"):
        return "xl"
    return "dec2only"


def collect():
    rows = []
    for ds in DATASETS:
        dp = os.path.join(ROOT, ds)
        if not os.path.isdir(dp):
            continue
        for name in sorted(os.listdir(dp)):
            cfgp = os.path.join(dp, name, "config.py")
            if not os.path.exists(cfgp):
                continue
            src = open(cfgp, encoding="utf-8", errors="ignore").read()
            c = {k: _get(src, k) for k in FIELDS}
            m = re.search(r"point_max=(\d+)", src)
            online, hm = None, None
            logp = os.path.join(dp, name, "train.log")
            if os.path.exists(logp):
                t = open(logp, encoding="utf-8", errors="ignore").read()
                online = "在线" if "sparseunet32_636.pth to GPU" in t else "离线"
                hits = re.findall(r"hm_mIoU: ([\d.]+)", t)
                if hits:
                    hm = hits[-1]
            rows.append(dict(
                ds=ds, name=name, st=struct_of(c),
                k=c.get("k_shot"), seed=c.get("seed"), online=online,
                lov=c.get("use_lovasz") == "True", plm=c.get("pl_mode") or "ps_ai",
                hm=hm,
            ))
    return rows


def render(rows):
    out = []
    out.append("# 结构与配方全表（2026-11-27 自动生成）")
    out.append("")
    out.append("> 由每个 run 的 `config.py` 逐字段提取。**结构代号按参数指纹判定，不看目录名。**")
    out.append("> 重新生成：`python tools/gen_structure_map.py > docs/00_ledger/STRUCTURE_MAP.md`")
    out.append("")
    out.append("## 结构代号定义")
    out.append("")
    out.append("| 代号 | 判据 | 含义 |")
    out.append("|---|---|---|")
    for tag, crit, mean in STRUCT_DEF:
        out.append("| `%s` | %s | %s |" % (tag, crit, mean))
    out.append("")
    out.append("## 命名与结构不符的 run")
    out.append("")
    sus = []
    for r in rows:
        n, st = r["name"], r["st"]
        if "sym" in n and st in ("M5", "m9"):
            sus.append((n, st, "名字带 sym，结构是 %s" % st))
        elif "m5" in n and st not in ("M5", "m9"):
            sus.append((n, st, "名字带 m5，结构是 %s" % st))
        elif "xl" in n and st not in ("xl", "sym"):
            sus.append((n, st, "名字带 xl，结构是 %s" % st))
    if sus:
        out.append("| 目录 | 真实结构 | 问题 |")
        out.append("|---|---|---|")
        for n, st, why in sus:
            out.append("| `%s` | **%s** | %s |" % (n, st, why))
    else:
        out.append("无。")
    out.append("")
    g = defaultdict(list)
    for r in rows:
        g[r["st"]].append(r)
    for st in ORDER:
        rs = g.get(st, [])
        if not rs:
            continue
        out.append("## %s（%d 个）" % (st, len(rs)))
        out.append("")
        out.append("| 目录 | 数据集 | shot | PL模式 | PL方案 | Lovász | seed | HM |")
        out.append("|---|---|---|---|---|---|---|---|")
        for r in rs:
            ds = r["ds"].replace("scannetpp", "Sc++").replace("scannet200", "Sc200")
            out.append("| `%s` | %s | %s | %s | %s | %s | %s | %s |" % (
                r["name"], ds, r["k"], r["online"] or "-", r["plm"],
                "是" if r["lov"] else "否", r["seed"], r["hm"] or "-"))
        out.append("")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    print(render(collect()))
