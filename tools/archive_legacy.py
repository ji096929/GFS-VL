# -*- coding: utf-8 -*-
"""归档散落在非仓库目录的一次性脚本（只读复制，绝不移动原文件）。

背景：真正的 git 仓库是 /data01/jh/code/GFS-VL/，但实验脚本写在三个兄弟位置：
    /data01/jh/code/ps_diag/     117 个 GFS-VL 核心诊断脚本（此前被整个漏掉）
    /data01/jh/afford_diag/       54 个 GFS-VL 配置生成与报告补丁
    /data01/jh/code/*.py         158 个，GFS-VL 与 OCSB 混杂
这三个位置都不是 git 仓库，所以从未进入版本管理。

归属判据（按可靠性排序）：
    1. 路径：ps_diag/ 与 afford_diag/ 下全部是 GFS-VL
    2. 数据路径：脚本读 /data01/jh/data/<dir>，命中哪个项目的数据集
    3. 强关键词：Cumul/Arith/B_pos/episode 是 OCSB 专有；eRank/dec0/sym_lovasz 是 GFS-VL 专有
    4. 日期：只在前三项打平时兜底（OCSB 一直做到 11-20，与 GFS-VL 重叠，单靠日期会错）

用法：python legacy_scripts/archive_legacy.py
"""
import collections
import datetime
import glob
import io
import os
import re
import shutil

CODE = "/data01/jh/code"
PSDIAG = "/data01/jh/code/ps_diag"
AFF = "/data01/jh/afford_diag"
OUT = "/data01/jh/code/GFS-VL/legacy_scripts"
DATA = "/data01/jh/data"

DATAPAT = re.compile(r"/data01/jh/data/([A-Za-z0-9_]+)")

GFS_DATA = {
    "scannet200_gfs", "pl_k1_s10", "pl_k1_s20", "pl_k1_s30", "pl_k1_s40", "pl_k1_s50",
    "online_pl", "online_pl_w", "online_pl_w_uniform", "offline_pl", "val_logits",
    "val_logits_s7", "val_logits_s13", "cut2_feat", "sonata_feat_cache",
    "val_sonata_cos", "pp_score", "cm_analysis", "empty_pl",
}
OCSB_DATA = {
    "ocsb_train2k", "ocsb_cache", "scannet_train_v2", "ScanNet", "region_cache",
    "base_cache", "region_index", "base_index", "support_protos", "retrieval_cache",
    "orth_cache", "sonata_dump", "manifest_fold0",
}

# GFS-VL 专有（跨层读出、伪标签口径、配置命名）
GFS_STRONG = re.compile(
    r"scannet200_gfs|GFS_OFFLINE_PL|base_novel|novel_seg_head|sym_lovasz|eRank|"
    r"dec0|dec1|dec2|RegisTrainSegmentor|num_base_classes|"
    r"configs/scannet200|exp/scannet200|novel_seg_head|base_cross")
# OCSB 专有（桥接 B/C 池、块级 episode、Cumul/Arith 指标）
OCSB_STRONG = re.compile(
    r"ocsb|B_Q|C_Q|B_pos|B_neg|episode|Cumul|Arith|coseg|superpoint|bridge|"
    r"affordance|可供性|U\s*块|U块|B\s*池|proto_B|manifest_fold|357|560")

CATS = [
    ("01_config_gen", "配置生成器（生成 100 个实验配置的源头）", re.compile(
        r"^add_|^gen_|^make_.*_cfg|patch_|append_cut2|decide_s13|mem_budget|"
        r"extract_results|gen_ab_configs|group_experiment|^fix_|update_report")),
    ("02_upper_bound", "上限估计与 Oracle", re.compile(
        r"ub[0-9]|upper|oracle|topk|or_gate|post_ub|conflict_oracle|instance_bound")),
    ("05_sonata", "Sonata 接入与探针", re.compile(r"sonata")),
    ("03_signal_gate", "门控与信号探测", re.compile(
        r"gate|signal|combo|adaptive|query_gate")),
    ("04_pseudo_label", "伪标签生成与筛选", re.compile(
        r"_pl_|genpl|purity|post_v3|recompute_filtered|weak_support|dump_online|"
        r"build_pl|build_w|build_pp|build_miss|build_region|build_base|dump_region|"
        r"dump_base|gen_offline|gen_sw|genB|full_pipeline|pseudo_miou|per_class_pts")),
    ("06_diagnosis", "诊断与归因", re.compile(
        r"diag_|probe_|analyze_|audit_|perclass|control_u|death_test|fake_tp|"
        r"u_block|cohens|^pr_|^quick_pr|^xl_|^task1_|norm|fisher|proto_separability|"
        r"teacher_vs_student|val_cm|val_cls|val_dump|obj_diag|cm_bucket|arbiter|"
        r"cut2_|ontr_step|ontr_|^ai_auc|^base_experiment|^miss_experiment|^task2|"
        r"^vlm_|^cmp_|^perpoint|^pl_tristage|^ps_diag|^bridge_|^b_pool|^bd_transfer")),
    ("07_reproduce", "校准与复现", re.compile(
        r"calib|sweep|verify|repro|selfcheck|pair|check_|same_|flash_diff|sonata_rand|"
        r"patch_det|patch_head|patch_save|patch_vlm|patch_dsv2|patch_freeze|patch_pl|"
        r"convert_sonata|gpu_watch|retry_err")),
    ("08_train_eval", "训练与评估入口", re.compile(
        r"^train_|^run_|^eval_|official_compare|assemble_|^build_|scene_|export_")),
    ("09_launchers", "启动与运维脚本（含 PL 口径证据）", re.compile(
        r"\.sh$|^guard_|^overnight|^pl_pipeline|^reeval|^report_watcher|^resume_|"
        r"^retry_|^start_|^watch_|^dl_|^install_deps")),
]

REPORT_MAP = {
    "01_config_gen": "docs/00_ledger/LEDGER.md",
    "02_upper_bound": "docs/06_diagnostics/DIAGNOSIS_REPORT_2026-11-22.md §3",
    "03_signal_gate": "docs/04_ontr_frozen/DAYTIME_ONTR_REPORT*.md",
    "04_pseudo_label": "docs/03_pseudo_label/margin_filter_experiment.md",
    "05_sonata": "docs/05_sonata/SONATA_SMALL_INTEGRATION.md",
    "06_diagnosis": "docs/06_diagnostics/DIAGNOSIS_REPORT_2026-11-22.md",
    "07_reproduce": "docs/legacy/SESSION_STATE.md",
    "08_train_eval": "docs/07_shot_protocol/OVERNIGHT_ICCV_REPORT.md",
    "09_launchers": "docs/legacy/SESSION_STATE.md（**含 GFS_OFFLINE_PL 口径证据**）",
    "90_other": "—",
}

BIN_EXT = {".sh"}


def docstring(path, limit=110):
    try:
        src = io.open(path, encoding="utf-8", errors="ignore").read(2000)
    except OSError:
        return "(无法读取)"
    m = re.search(r'^(?:#!.*\n)?(?:#.*\n)*\s*"""(.+?)"""', src, re.S)
    if m:
        line = " ".join(x.strip() for x in m.group(1).strip().split("\n") if x.strip())
        return line[:limit] or "(空说明)"
    for ln in src.split("\n")[:25]:
        s = ln.strip()
        if not s.startswith("#"):
            continue
        s = s.lstrip("#").strip()
        if len(s) > 4 and not s.startswith("-*-"):
            return s[:limit] + "（据注释）"
    return "(无说明)"


def cat_by_keyword(name):
    for key, _, pat in CATS:
        if pat.search(name):
            return key
    return "90_other"


# 人工覆写：这几个靠关键词与日期都判不准，已逐个读过内容确认
MANUAL = {
    "repro_probe.py": "OCSB",         # 定位 COSeg Baseline 的随机源
    "repro_probe2.py": "OCSB",        # per-episode 种子锁死，COSeg 任务集
    "run_official_eval.py": "OCSB",   # 用 fake_tp 垫片跑官方 main_fs.py
    "diag_sonata2.py": "OCSB",        # 读 sonata_dump/ep%03d.npz（357 任务集）
    "sonata_probe2.py": "OCSB",
    "sonata_probe3.py": "OCSB",
    "dump_for_sonata.py": "OCSB",     # 357 个官方口径任务 support/query 落盘
}


def judge(name, path, mtime, src):
    """返回 ('GFS'|'OCSB', 功能分节)。"""
    if name in MANUAL:
        o = MANUAL[name]
        return o, (cat_by_keyword(name) if o == "GFS" else None)
    if path.startswith(PSDIAG) or path.startswith(AFF):
        return "GFS", cat_by_keyword(name)
    d = collections.Counter(DATAPAT.findall(src))
    g = 3 * sum(v for k, v in d.items() if k in GFS_DATA) + 2 * len(GFS_STRONG.findall(src))
    o = 3 * sum(v for k, v in d.items() if k in OCSB_DATA) + 2 * len(OCSB_STRONG.findall(src))
    if g > o:
        return "GFS", cat_by_keyword(name)
    if o > g:
        return "OCSB", None
    return ("OCSB", None) if mtime <= "2026-11-14" else ("GFS", cat_by_keyword(name))


def scan(base, exts=(".py", ".sh")):
    out = []
    for ext in exts:
        for p in sorted(glob.glob(os.path.join(base, "*" + ext))):
            n = os.path.basename(p)
            src = io.open(p, encoding="utf-8", errors="ignore").read()
            mt = datetime.date.fromtimestamp(os.path.getmtime(p)).isoformat()
            owner, cat = judge(n, p, mt, src)
            out.append(dict(name=n, path=p, size=os.path.getsize(p), mtime=mt,
                            note=docstring(p), owner=owner, cat=cat))
    return out


def log_index():
    rows = []
    for base in (CODE, DATA):
        for p in sorted(glob.glob(os.path.join(base, "*.log"))):
            rows.append(dict(dir=base.replace("/data01/jh/", ""),
                             name=os.path.basename(p), size=os.path.getsize(p),
                             mtime=datetime.date.fromtimestamp(os.path.getmtime(p)).isoformat()))
    return rows


def main():
    rows = scan(PSDIAG) + scan(AFF) + scan(CODE)
    for sub in ("ps_diag", "afford_diag", "code_gfs", "code_unknown"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    n_cp = {"GFS": 0, "OCSB": 0}
    for r in rows:
        if r["owner"] == "OCSB":
            n_cp["OCSB"] += 1
            continue
        if r["path"].startswith(PSDIAG):
            dst = os.path.join(OUT, "ps_diag", r["name"])
        elif r["path"].startswith(AFF):
            dst = os.path.join(OUT, "afford_diag", r["name"])
        else:
            dst = os.path.join(OUT, "code_gfs", r["name"])
        shutil.copy2(r["path"], dst)
        n_cp["GFS"] += 1

    logs = log_index()
    gfs = [r for r in rows if r["owner"] == "GFS"]
    ocsb = [r for r in rows if r["owner"] == "OCSB"]
    by = collections.defaultdict(list)
    for r in gfs:
        by[r["cat"]].append(r)

    L = []
    L.append("# 散落脚本归档索引")
    L.append("")
    L.append("> 归档日期 **2026-11-22**。源目录三个，**都不是 git 仓库**（真正的仓库是它们的兄弟目录")
    L.append("> `code/GFS-VL/`，所以这些脚本从第一天起就在版本控制之外）：")
    L.append(">")
    L.append("> | 源目录 | 脚本数 | 归属 |")
    L.append("> |---|---|---|")
    L.append("> | `/data01/jh/code/ps_diag/` | %d | **全部 GFS-VL**（拆解 1/2/3、4 桶、教师vs学生、ONTR、Sonata 接入） |" % len(glob.glob(PSDIAG + "/*.py")))
    L.append("> | `/data01/jh/afford_diag/` | %d | **全部 GFS-VL**（配置生成器 + 报告补丁） |" % len(glob.glob(AFF + "/*.py")))
    L.append("> | `/data01/jh/code/`（顶层） | %d | GFS-VL 与 OCSB 混杂，逐个判定 |" % len(glob.glob(CODE + "/*.py")))
    L.append(">")
    L.append("> **本目录是只读副本，原处文件保留不动**（脚本内硬编码绝对路径，挪动即失效）。")
    L.append("> 重新归档：`python legacy_scripts/archive_legacy.py`")
    L.append("")
    L.append("## 归档统计")
    L.append("")
    L.append("共扫描 **%d** 个脚本：GFS-VL **%d** 个（已复制），OCSB **%d** 个（只在清单里，物理副本在 `3D_OCSB/legacy_scripts/`）。" % (len(rows), len(gfs), len(ocsb)))
    L.append("")
    for key, title, _ in CATS:
        L.append("- `%s` %s：**%d** 个" % (key, title, len(by.get(key, []))))
    L.append("- `90_other` 未归类：**%d** 个" % len(by.get("90_other", [])))
    L.append("")
    L.append("日志索引：`%d` 个（`code/` %d + `data/` %d），**只建索引不复制**。" % (
        len(logs), sum(1 for x in logs if x["dir"] == "code"),
        sum(1 for x in logs if x["dir"] == "data")))
    L.append("")
    L.append("---")
    L.append("")

    order = [k for k, _, _ in CATS] + ["90_other"]
    titles = {k: t for k, t, _ in CATS}
    titles["90_other"] = "未归类"
    for key in order:
        sub = by.get(key, [])
        if not sub:
            continue
        L.append("## %s %s（%d）" % (key, titles[key], len(sub)))
        L.append("")
        L.append("对应报告：%s" % REPORT_MAP.get(key, "—"))
        L.append("")
        L.append("| 脚本 | 修改日 | 说明 |")
        L.append("|---|---|---|")
        for r in sub:
            L.append("| `%s` | %s | %s |" % (r["name"], r["mtime"], r["note"].replace("|", "\\|")))
        L.append("")

    if ocsb:
        L.append("## OCSB 主线（%d，已分流到 3D_OCSB 仓库）" % len(ocsb))
        L.append("")
        L.append("这些脚本属于 3D 可供性 / 桥接 / 环实验主线，**不在本仓库保存副本**。")
        L.append("物理副本：`experiment/3D_OCSB/legacy_scripts/code_ocsb/`。")
        L.append("")
        L.append("、".join("`%s`" % r["name"] for r in ocsb))
        L.append("")
        io.open(os.path.join(OUT, "OCSB_SCRIPTS.md"), "w", encoding="utf-8").write(
            "# OCSB 主线散落脚本清单\n\n"
            "> 归档自 `/data01/jh/code/`（2026-11-22）。物理副本在 "
            "`experiment/3D_OCSB/legacy_scripts/code_ocsb/`。\n\n"
            "| 脚本 | 修改日 | 说明 |\n|---|---|---|\n" +
            "\n".join("| `%s` | %s | %s |" % (r["name"], r["mtime"], r["note"].replace("|", "\\|"))
                      for r in ocsb) + "\n")

    L.append("---")
    L.append("")
    L.append("## 日志索引（不复制内容）")
    L.append("")
    L.append("| 目录 | 文件 | 大小 | 修改日 |")
    L.append("|---|---|---|---|")
    for x in logs:
        L.append("| `%s` | `%s` | %d K | %s |" % (x["dir"], x["name"], max(1, x["size"] // 1024), x["mtime"]))
    L.append("")

    md = "\n".join(L) + "\n"
    io.open(os.path.join(OUT, "INDEX.md"), "w", encoding="utf-8").write(md)
    print("扫描 %d 个：GFS-VL %d（复制 %d），OCSB %d（仅清单）" % (
        len(rows), len(gfs), n_cp["GFS"], len(ocsb)))
    print("索引: %s/INDEX.md (%d 行)" % (OUT, md.count("\n")))


if __name__ == "__main__":
    main()
