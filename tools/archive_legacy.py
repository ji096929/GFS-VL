# -*- coding: utf-8 -*-
"""归档散落在非仓库目录的一次性脚本（只读复制，绝不移动原文件）。

背景：真正的 git 仓库是 /data01/jh/code/GFS-VL/，但 2026-11 的实验脚本
都写在它的父目录 /data01/jh/code/ 和 /data01/jh/afford_diag/ 下，
git 从第一天起就看不到它们。本脚本把它们复制进仓库并生成索引。

原处文件保留不动 —— 脚本里全是硬编码绝对路径，挪动即失效。

用法：python legacy_scripts/archive_legacy.py
"""
import datetime
import glob
import io
import os
import re
import shutil

CODE = "/data01/jh/code"
AFF = "/data01/jh/afford_diag"
OUT = "/data01/jh/code/GFS-VL/legacy_scripts"
DATA = "/data01/jh/data"

# OCSB 主线（3D 可供性 / 桥接 / 环实验 / block 级评估），不属于 GFS-VL
# 注意：v3*/v3g/v3h/v3i 是桥接实验的第三轮（BridgeScorer、跨房间 U 块），不是 GFS-VL 的 v3
OCSB_RE = re.compile(
    r"ocsb|ring[0-9]|bridge|coseg|clip_vit|phase[0-9]|local_geometry|same_scene|"
    r"run_s2|run_s34|build_train2k|scene_eval|scene_merge|export_counter|"
    r"export_room|export_ply|make_class2scans|prep_|train_sweep|sweep_step2|"
    r"^check\.py|^v3_build|^v3g_|^v3h_|^v3i_|block|manifest|orth_feats|"
    r"stats_blocks|compare_u_mode|collect_train|scan_block|rebuild_meta|"
    r"summarize_full|scene_sampler|death_test"
)

# 功能分节：按顺序匹配，先中者优先
CATS = [
    ("01_config_gen", "配置生成器（生成 100 个实验配置的源头）", re.compile(
        r"^add_|^gen_|patch_|append_cut2|decide_s13|mem_budget|extract_results|"
        r"gen_ab_configs|group_experiment|^fix_|update_report")),
    ("02_upper_bound", "上限估计与 Oracle", re.compile(
        r"ub[0-9]|upper|oracle|topk|or_gate|post_ub")),
    ("03_signal_gate", "门控与信号探测", re.compile(
        r"gate|signal|combo|adaptive|query_gate")),
    ("04_pseudo_label", "伪标签生成与筛选", re.compile(
        r"_pl_|genpl|purity|post_v3|recompute_filtered|weak_support")),
    ("05_sonata", "Sonata 迁移探针", re.compile(r"sonata")),
    ("06_diagnosis", "诊断与归因", re.compile(
        r"diag_|probe_|analyze_|audit_|perclass|control_u|death_test|fake_tp|"
        r"u_block|cohens|^pr_|^quick_pr|^xl_|^task1_|norm")),
    ("07_reproduce", "校准与复现", re.compile(
        r"calib|sweep|verify|repro|selfcheck|pair|check_|same_")),
    ("08_train_eval", "训练与评估入口", re.compile(
        r"^train_|^run_|^eval_|official_compare|assemble_|^build_|scene_|export_")),
]

REPORT_MAP = {
    "01_config_gen": "docs/00_ledger/LEDGER.md（每个实验的 config 来源）",
    "02_upper_bound": "docs/06_diagnostics/DIAGNOSIS_REPORT_2026-11-22.md §3 4桶上界",
    "03_signal_gate": "docs/04_ontr_frozen/DAYTIME_ONTR_REPORT*.md",
    "04_pseudo_label": "docs/03_pseudo_label/margin_filter_experiment.md",
    "05_sonata": "docs/05_sonata/SONATA_SMALL_INTEGRATION.md",
    "06_diagnosis": "docs/06_diagnostics/DIAGNOSIS_REPORT_2026-11-22.md",
    "07_reproduce": "docs/legacy/SESSION_STATE.md",
    "08_train_eval": "docs/07_shot_protocol/OVERNIGHT_ICCV_REPORT.md",
    "90_other": "—",
}


def docstring(path, limit=110):
    try:
        src = io.open(path, encoding="utf-8", errors="ignore").read(2000)
    except OSError:
        return "(无法读取)"
    m = re.search(r'^(?:#!.*\n)?(?:#.*\n)*\s*"""(.+?)"""', src, re.S)
    if m:
        line = " ".join(x.strip() for x in m.group(1).strip().split("\n") if x.strip())
        return line[:limit] or "(空说明)"
    # 没有 docstring：退而抓第一条实质注释，避免整张表一半是「无说明」
    for ln in src.split("\n")[:25]:
        s = ln.strip()
        if not s.startswith("#"):
            continue
        s = s.lstrip("#").strip()
        if len(s) > 4 and not s.startswith("-*-"):
            return s[:limit] + "（据注释）"
    return "(无说明)"


def classify(name, path, mtime):
    """三重判定：路径 > 日期 > 关键词。

    1. afford_diag/ 下全部是 GFS-VL（11-19~11-21 的配置生成与报告补丁）
    2. code/ 下 2026-11-14 及以前全部是 OCSB（可供性/桥接/环实验）
    3. 其余按关键词分功能节
    """
    if path.startswith(AFF):
        return _cat_by_keyword(name)
    if mtime <= "2026-11-14":
        return "98_ocsb"
    if OCSB_RE.search(name):
        return "98_ocsb"
    return _cat_by_keyword(name)


def _cat_by_keyword(name):
    for key, _, pat in CATS:
        if pat.search(name):
            return key
    return "90_other"


def scan(base):
    out = []
    for p in sorted(glob.glob(os.path.join(base, "*.py"))):
        n = os.path.basename(p)
        out.append(dict(
            name=n, path=p,
            size=os.path.getsize(p),
            mtime=datetime.date.fromtimestamp(os.path.getmtime(p)).isoformat(),
            note=docstring(p),
            cat=classify(n, p, datetime.date.fromtimestamp(os.path.getmtime(p)).isoformat()),
        ))
    return out


def log_index():
    rows = []
    for base in (CODE, DATA):
        for p in sorted(glob.glob(os.path.join(base, "*.log"))):
            rows.append(dict(
                dir=base.replace("/data01/jh/", ""),
                name=os.path.basename(p),
                size=os.path.getsize(p),
                mtime=datetime.date.fromtimestamp(os.path.getmtime(p)).isoformat(),
            ))
    return rows


def main():
    gfs = scan(CODE) + scan(AFF)
    # 只复制 GFS-VL 的；OCSB 的 113 个已分流到 experiment/3D_OCSB/legacy_scripts/code_ocsb/
    for sub in ("code_gfs", "afford_diag"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    n_cp = 0
    for r in gfs:
        if r["cat"] == "98_ocsb":
            continue
        if r["path"].startswith(AFF):
            dst = os.path.join(OUT, "afford_diag", r["name"])
        else:
            dst = os.path.join(OUT, "code_gfs", r["name"])
        shutil.copy2(r["path"], dst)
        n_cp += 1

    logs = log_index()

    lines = []
    lines.append("# 散落脚本归档索引")
    lines.append("")
    lines.append("> 归档日期 **2026-11-22**。源目录 `/data01/jh/code/`（158 个 .py）与")
    lines.append("> `/data01/jh/afford_diag/`（54 个 .py）**都不是 git 仓库**，真正的仓库是它们的子目录")
    lines.append("> `code/GFS-VL/`。所以这些脚本从第一天起就在版本控制之外。")
    lines.append(">")
    lines.append("> **本目录是只读副本，原处文件保留不动**（脚本内硬编码绝对路径，挪动即失效）。")
    lines.append("> 重新归档：`python legacy_scripts/archive_legacy.py`")
    lines.append("")
    lines.append("## 归档统计")
    lines.append("")
    by = {}
    for r in gfs:
        by.setdefault(r["cat"], []).append(r)
    for key, title, _ in CATS:
        lines.append("- `%s` %s：**%d** 个" % (key, title, len(by.get(key, []))))
    lines.append("- `90_other` 未归类：**%d** 个" % len(by.get("90_other", [])))
    lines.append("- `98_ocsb` 属 OCSB 主线：**%d** 个，已分流到 `3D_OCSB/legacy_scripts/code_ocsb/`（本仓库不存副本）" % len(by.get("98_ocsb", [])))
    lines.append("")
    lines.append("日志：`%d` 个（`code/` %d + `data/` %d），**只建索引不复制**（见文末）。" % (
        len(logs), sum(1 for x in logs if x["dir"] == "code"),
        sum(1 for x in logs if x["dir"] == "data")))
    lines.append("")
    lines.append("---")
    lines.append("")

    order = [k for k, _, _ in CATS] + ["90_other"]
    titles = {k: t for k, t, _ in CATS}
    titles["90_other"] = "未归类"
    for key in order:
        rows = by.get(key, [])
        if not rows:
            continue
        lines.append("## %s %s（%d）" % (key, titles[key], len(rows)))
        lines.append("")
        lines.append("对应报告：%s" % REPORT_MAP.get(key, "—"))
        lines.append("")
        lines.append("| 脚本 | 修改日 | 说明 |")
        lines.append("|---|---|---|")
        for r in rows:
            lines.append("| `%s` | %s | %s |" % (
                r["name"], r["mtime"], r["note"].replace("|", "\\|")))
        lines.append("")

    # OCSB 脚本不在这里存副本，只留清单（物理副本在 3D_OCSB 仓库）
    ocsb = by.get("98_ocsb", [])
    if ocsb:
        lines.append("## 98_ocsb OCSB 主线（%d，已分流到 3D_OCSB 仓库）" % len(ocsb))
        lines.append("")
        lines.append("这 %d 个脚本属于 3D 可供性 / 桥接 / 环实验主线，**不在本仓库保存副本**。" % len(ocsb))
        lines.append("物理副本：`experiment/3D_OCSB/legacy_scripts/code_ocsb/`（源目录 `/data01/jh/code/`）。")
        lines.append("")
        lines.append("、".join("`%s`" % r["name"] for r in ocsb))
        lines.append("")
        io.open(os.path.join(OUT, "OCSB_SCRIPTS.md"), "w", encoding="utf-8").write(
            "# OCSB 主线散落脚本清单\n\n"
            "> 从 `/data01/jh/code/` 归档（2026-11-22）。物理副本在 "
            "`experiment/3D_OCSB/legacy_scripts/code_ocsb/`。\n\n"
            "| 脚本 | 修改日 | 说明 |\n|---|---|---|\n" +
            "\n".join("| `%s` | %s | %s |" % (
                r["name"], r["mtime"], r["note"].replace("|", "\\|")) for r in ocsb) + "\n")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 日志索引（不复制内容）")
    lines.append("")
    lines.append("| 目录 | 文件 | 大小 | 修改日 |")
    lines.append("|---|---|---|---|")
    for x in logs:
        lines.append("| `%s` | `%s` | %d K | %s |" % (
            x["dir"], x["name"], max(1, x["size"] // 1024), x["mtime"]))
    lines.append("")

    md = "\n".join(lines) + "\n"
    io.open(os.path.join(OUT, "INDEX.md"), "w", encoding="utf-8").write(md)
    print("复制 %d 个脚本，日志索引 %d 条" % (n_cp, len(logs)))
    print("索引: %s/INDEX.md (%d 行)" % (OUT, md.count("\n")))


if __name__ == "__main__":
    main()
