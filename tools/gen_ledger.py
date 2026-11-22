# -*- coding: utf-8 -*-
"""生成 GFS-VL 实验台账（只读，不改任何实验目录）。

用法：
    python tools/gen_ledger.py                      # 打印 Markdown 到 stdout
    python tools/gen_ledger.py --md out.md --json out.json

口径说明（重要）：
    抖动口径  jitter —— train.log 里最后一次 val。shuffle_orders 在训练/旧评估时随机，
                        同一权重多次评估会差 0.5-0.8，不能与确定性口径混比。
    确定性口径 det   —— /data01/jh/data/cm_analysis/det_eval/<run>/run.log。
                        shuffle_orders 已修（只在训练生效），两次评估 0.0000 误差。

状态判定：
    done        有 val 结果
    incomplete  有目录/配置但没有任何 val 行
    failed      日志里有 Traceback / OutOfMemory 且无 val
    invalid     已知实现 bug（目录名带 BUG，或 support_* 的 PL 位置 bug）
"""
import argparse
import glob
import json
import os
import re

EXP_ROOT = "/data01/jh/code/GFS-VL/exp"
DET_ROOT = "/data01/jh/data/cm_analysis/det_eval"

# 只扫真正的数据集目录；exp/default 是 Pointcept 默认输出，不是实验
DATASETS = ("scannet200", "scannetpp", "scannet")

VAL_RE = re.compile(
    r"Val result: mIoU/Base/Novel/HM/mAcc/allAcc "
    r"([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)"
)

# config.py 抓取规则：字段名 -> (正则, 类型)
CFG_RULES = {
    "model_type": (r"^model = dict\(\s*\n\s*type=['\"](\w+)['\"]", str),
    "backbone_type": (r"backbone=dict\(\s*\n\s*type=['\"]([\w\-]+)['\"]", str),
    "num_base_classes": (r"num_base_classes=(\d+)", int),
    "use_lovasz": (r"^use_lovasz\s*=\s*(\w+)", str),
    "epoch": (r"^\s{4}epoch\s*=\s*(\d+)", int),
    "batch_size": (r"^\s{4}batch_size\s*=\s*(\d+)", int),
    "num_bases": (r"^\s{4}num_bases=(\d+)", int),
    "num_base_novels": (r"^\s{4}num_base_novels=(\d+)", int),
    "point_max": (r"point_max=(\d+)", int),
    "enc_mode": (r"enc_mode=(\w+)", str),
    "norm_all_ln": (r"norm_all_ln=(\w+)", str),
    "embedding_type": (r"embedding_type=['\"](\w+)['\"]", str),
    "freeze_enc": (r"freeze_enc=(\w+)", str),
    "freeze_backbone": (r"freeze_backbone=(\w+)", str),
    "regis_train_list": (r"regis_train_list=\[([^\]]*)\]", str),
    "data_root": (r"^data_root\s*=\s*['\"]([^'\"]+)['\"]", str),
    # 复现必需：种子、续训起点、伪标签阈值、logit 缩放
    "seed": (r"^seed\s*=\s*(\d+)", int),
    "init_weight": (r"^weight\s*=\s*['\"]([^'\"]+)['\"]", str),
    "backbone_weight": (r"^backbone_weight\s*=\s*['\"]([^'\"]+)['\"]", str),
    "ps_thresh": (r"^ps_thresh\s*=\s*([\d.]+)", float),
    "ai_thresh": (r"^ai_thresh\s*=\s*([\d.]+)", float),
    "novel_logit_scale": (r"^novel_logit_scale\s*=\s*([\d.]+)", float),
    "pl_mode": (r"^pl_mode\s*=\s*['\"](\w+)['\"]", str),
    "warmup_epochs": (r'^warmup_epochs\s*=\s*(\d+)', int),
    "stu_tau": (r'^stu_tau\s*=\s*([\d.]+)', float),
    "scheduler": (r"scheduler = dict\(\s*\n?\s*type=['\"](\w+)['\"]", str),
    "optimizer": (r"^optimizer = dict\(type=['\"](\w+)['\"]", str),
}

# 注意：GFS_OFFLINE_PL 等口径只存在于启动命令，config 与日志均不记录。台账无法还原。

ERR_PAT = re.compile(r"Traceback|OutOfMemory|CUDA out of memory|did not receive grad")


def parse_config(path):
    try:
        src = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return {}
    out = {}
    for key, (pat, typ) in CFG_RULES.items():
        m = re.search(pat, src, re.M)
        if not m:
            continue
        v = m.group(1)
        if typ is int:
            try:
                v = int(v)
            except ValueError:
                continue
        elif typ is float:
            try:
                v = float(v)
            except ValueError:
                continue
        out[key] = v
    return out


def parse_log(path):
    """返回 (指标列表, 是否有异常堆栈)。"""
    vals, err = [], False
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = VAL_RE.search(line)
                if m:
                    vals.append([float(x) for x in m.groups()])
                elif ERR_PAT.search(line):
                    err = True
    except OSError:
        pass
    return vals, err


def parse_ckpt(model_dir):
    pths = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
    if not pths:
        return dict(n=0, size_gb=0.0, mtime=None, has_last=False, has_best=False)
    total = sum(os.path.getsize(p) for p in pths)
    mt = max(os.path.getmtime(p) for p in pths)
    names = [os.path.basename(p) for p in pths]
    return dict(
        n=len(pths),
        size_gb=round(total / 1024 ** 3, 2),
        mtime=__import__("datetime").datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M"),
        has_last=any("last" in n for n in names),
        has_best=any("best" in n for n in names),
    )


def tags_of(name, cfg):
    """从目录名 + 配置提取标签，用于分组与筛选。"""
    n = name.lower()
    t = []
    # shot / 点数
    if "_k5" in n:
        t.append("k5")
    elif "_k1" in n:
        t.append("k1")
    if "102400" in n:
        t.append("pts102400")
    # 结构
    struct = []
    if "sym_lcross01" in n:
        struct.append("sym+lcross01")
    elif "lcross_cos" in n:
        struct.append("lcross+cos")
    elif "lcross" in n:
        struct.append("lcross")
    elif "sym_xl" in n:
        struct.append("sym_xl")
    elif n.startswith("sym_") or "_sym_" in n:
        struct.append("sym")
    elif n.startswith("base_"):
        struct.append("base")
    if "gated" in n:
        struct.append("gated")
    if "ortho" in n:
        struct.append("ortho")
    if "mlp_dec0" in n:
        struct.append("mlp_dec0")
    if "rep3" in n:
        struct.append("rep3")
    if "detach" in n:
        struct.append("detach")
    if n.startswith("cos_") or "_cos_" in n:
        struct.append("cos")
    if "lnonly" in n:
        struct.append("lnonly")
    if n == "xl_s13" or n.startswith("xl_"):
        struct.append("xl")
    t += struct
    # 损失
    if "lovasz" in n:
        t.append("lovasz")
    elif "_la_" in n:
        t.append("la")
    elif cfg.get("use_lovasz") == "True":
        t.append("lovasz")
    else:
        t.append("ce")
    # 伪标签来源
    for k, v in (("online", "pl:online"), ("offline", "pl:offline"),
                 ("_sw2", "pl:sw2"), ("_sw3", "pl:sw3"),
                 ("_a2_", "pl:a2"), ("xlu", "pl:xlu"), ("baseu", "pl:baseu"),
                 ("regis1", "pl:official")):
        if k in n:
            t.append(v)
    # 骨干
    if "sonata" in n:
        t.append("bb:sonata")
    elif "spunet" in n:
        t.append("bb:spunet")
    else:
        t.append("bb:ptv3")
    # 冻结 / ONTR
    if n.startswith("frozen_"):
        t.append("frozen")
    if "ontr" in n:
        t.append("ontr")
    if n.startswith("support_"):
        t.append("support-ft")
    return t


def workstream_of(name, dataset=""):
    """主分节。优先级从高到低。"""
    n = name.lower()
    if dataset == "scannetpp":
        return "08_scannetpp"
    if n.startswith(("ptv3_bk", "spunet_bk", "_guard", "ontr_eval")):
        return "00_infra"
    if "sonata" in n:
        return "05_sonata"
    if n.startswith(("frozen_", "support_", "ontr_")):
        return "04_ontr_frozen"
    if n.startswith("sc200_k1") or "offline" in n or "online" in n:
        return "03_pseudo_label"
    if "_la_" in n and "lovasz" not in n:
        return "02_loss"
    if "_k5" in n or "_k1" in n:
        return "07_shot_protocol"
    return "01_cross_layer"


WS_TITLES = {
    "01_cross_layer": "01 跨层读出主干与结构变体",
    "02_loss": "02 损失函数",
    "03_pseudo_label": "03 伪标签来源与 1-shot 口径",
    "04_ontr_frozen": "04 ONTR 与冻结骨干",
    "05_sonata": "05 Sonata 基模替换",
    "06_diagnostics": "06 机理诊断（产物见 ASSETS）",
    "07_shot_protocol": "07 shot 数与点数上限",
    "08_scannetpp": "08 ScanNet++（跨数据集验证）",
    "00_infra": "00 基础设施与备份",
}


def collect(root):
    rows = []
    for ds in sorted(os.listdir(root)):
        ds_dir = os.path.join(root, ds)
        if ds not in DATASETS or not os.path.isdir(ds_dir):
            continue
        for name in sorted(os.listdir(ds_dir)):
            d = os.path.join(ds_dir, name)
            if not os.path.isdir(d):
                continue
            cfg = parse_config(os.path.join(d, "config.py"))
            tr, tr_err = parse_log(os.path.join(d, "train.log"))
            te, te_err = parse_log(os.path.join(d, "test.log"))
            allv = te or tr          # test.log 优先（离线重评）
            jit_last = allv[-1] if allv else None
            jit_best = max(allv, key=lambda v: v[3]) if allv else None
            n_val = len(tr) + len(te)

            dv, dv_err = parse_log(os.path.join(DET_ROOT, name, "run.log"))
            det = dv[-1] if dv else None

            ck = parse_ckpt(os.path.join(d, "model"))
            ws = workstream_of(name, ds)

            if ws == "00_infra":
                status = "infra"
            elif "bug" in name.lower():
                status = "invalid"
            elif name.startswith("support_") and not name.endswith("_fixed"):
                status = "invalid"   # PL 只在 45 个 support 场景读 + gfs==-1 假阳性
            elif jit_last is not None:
                status = "done"
            elif tr_err or te_err:
                status = "failed"
            elif ck["n"]:
                status = "unevaluated"   # 训完了但没跑过 val
            else:
                status = "incomplete"

            rows.append(dict(
                dataset=ds, name=name,
                cfg=cfg, tags=tags_of(name, cfg),
                ws=ws,
                n_val=n_val,
                jitter_last=jit_last, jitter_best=jit_best,
                det=det, ckpt=ck, status=status,
                has_err=bool(tr_err or te_err),
            ))
    return rows


def spec_note(cfg):
    """只在偏离默认值时记一笔，避免整列全是噪声。默认：Adam + MultiStepLR + PS0.6/AI0.9 + α=1。"""
    notes = []
    s = cfg.get("scheduler")
    if s and s != "MultiStepLR":
        notes.append("余弦退火" if s == "CosineAnnealingLR" else s)
    o = cfg.get("optimizer")
    if o and o != "Adam":
        notes.append(o)
    nls = cfg.get("novel_logit_scale")
    if nls is not None and abs(nls - 1.0) > 1e-9:
        notes.append("α=%.2f" % nls)
    # PS/AI 只在原流程（ps_ai）下生效；其他 pl_mode 下这两个字段是死参数
    _pm_now = cfg.get("pl_mode", "ps_ai")
    if _pm_now == "ps_ai":
        ps, ai = cfg.get("ps_thresh"), cfg.get("ai_thresh")
        if (ps is not None and abs(ps - 0.6) > 1e-9) or (ai is not None and abs(ai - 0.9) > 1e-9):
            notes.append("PS%.2f/AI%.2f" % (ps if ps is not None else 0.6,
                                            ai if ai is not None else 0.9))
    pm = cfg.get("pl_mode")
    if pm and pm != "ps_ai":
        notes.append("PL:" + pm)
    if cfg.get("warmup_epochs") is not None:
        notes.append("warm%d" % cfg["warmup_epochs"])
    if cfg.get("stu_tau") is not None:
        notes.append("tau%.2f" % cfg["stu_tau"])
    w = cfg.get("init_weight")
    if w:
        parts = w.replace("\\", "/").split("/")
        notes.append("续训:" + (parts[-3] if len(parts) >= 3 else w))
    bw = cfg.get("backbone_weight")
    if bw:
        notes.append("主干:" + os.path.basename(bw))
    return ",".join(notes) or "—"


def fmt(v):
    if v is None:
        return "—"
    return "%.4f/%.4f/%.4f" % (v[1], v[2], v[3])


def render_md(rows):
    out = []
    out.append("# GFS-VL 实验台账")
    out.append("")
    out.append("> 自动生成，勿手工编辑。重新生成：`python tools/gen_ledger.py --md docs/00_ledger/LEDGER.md`")
    out.append(">")
    out.append("> **口径**：`抖动` = train.log 最后一次 val（`shuffle_orders` 随机，同权重两次可差 0.5–0.8）；")
    out.append("> `确定性` = `det_eval/` 重评（已修 `shuffle_orders`，两次评估 0.0000 误差）。**跨实验比较只用确定性口径。**")
    out.append("")
    from collections import Counter
    cnt = Counter(r["status"] for r in rows)
    out.append("实验目录 **%d** 个：%s。" % (
        len(rows),
        " / ".join("%s %d" % (k, cnt[k]) for k in
                   ("done", "unevaluated", "incomplete", "failed", "invalid", "infra")
                   if cnt[k])))
    out.append("")
    out.append("确定性命中 %d 个（`det_eval/`）。" % sum(1 for r in rows if r["det"]))
    out.append("")
    for st, label in (("invalid", "实现有 bug，结果不可用"),
                      ("failed", "训练/评估崩溃"),
                      ("unevaluated", "有权重但从未评估"),
                      ("incomplete", "空壳或未跑完"),
                      ("infra", "基础设施/备份，非实验")):
        names = [r["name"] for r in rows if r["status"] == st]
        if names:
            out.append("- **%s**（%s，%d）：%s" %
                       (st, label, len(names), "、".join("`%s`" % x for x in names)))
    out.append("")

    for ws in sorted(WS_TITLES):
        sub = [r for r in rows if r["ws"] == ws]
        if not sub:
            continue
        out.append("## %s（%d）" % (WS_TITLES[ws], len(sub)))
        out.append("")
        out.append("| 目录 | 数据集 | 结构/损失 | 伪标签 | 抖动 Base/Novel/HM | 确定性 Base/Novel/HM | 种子 | 口径备注 | 权重 | 状态 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in sub:
            tags = r["tags"]
            struct = "/".join(t for t in tags if t in (
                "base", "sym", "sym_xl", "sym+lcross01", "lcross", "lcross+cos",
                "xl", "gated", "ortho", "mlp_dec0", "rep3", "detach", "cos", "lnonly"))
            loss = "/".join(t for t in tags if t in ("lovasz", "la", "ce"))
            pl = "/".join(t.replace("pl:", "") for t in tags if t.startswith("pl:"))
            ck = "%d个/%.1fG" % (r["ckpt"]["n"], r["ckpt"]["size_gb"]) if r["ckpt"]["n"] else "—"
            out.append("| `%s` | %s | %s / %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["name"], r["dataset"], struct or "—", loss or "—", pl or "—",
                fmt(r["jitter_last"]), fmt(r["det"]),
                r["cfg"].get("seed", "—"), spec_note(r["cfg"]), ck, r["status"]))
        out.append("")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=EXP_ROOT)
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = collect(a.root)
    md = render_md(rows)
    if a.md:
        os.makedirs(os.path.dirname(a.md), exist_ok=True)
        open(a.md, "w", encoding="utf-8").write(md)
        print("written: %s (%d runs)" % (a.md, len(rows)))
    else:
        print(md)
    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        json.dump(rows, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("written: %s" % a.json)


if __name__ == "__main__":
    main()
