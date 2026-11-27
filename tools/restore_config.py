# -*- coding: utf-8 -*-
"""从 train.log 的 config dump 恢复被覆盖的 config.py。

背景（2026-11-27）：
`exp/<ds>/<run>/config.py` 在训练时由 `train.py` 写出，但之后被某次操作覆盖过
（实测 `sc200_k1_sym_s42` 的 config mtime 11-26，train.log 11-20，内容一个是 M5 一个是 sym）。
`train.log` 第 4 行起是训练时的完整 config dump，可以据此恢复。

用法:
    python tools/restore_config.py --dry-run     # 只报告哪些需要恢复
    python tools/restore_config.py --apply       # 实际写回（先备份为 config.py.bak）
"""
import argparse
import os
import re
import shutil

ROOT = "/data01/jh/code/GFS-VL/exp"
DATASETS = ("scannetpp", "scannet200")
TS = re.compile(r"^\[\d{4}-\d\d-\d\d [\d:,]+ INFO [^\]]+\]\s?")


def extract(log_path):
    """从 train.log 提取 config 文本。返回 str 或 None。"""
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    starts = [i for i, l in enumerate(lines) if l.rstrip().endswith("Config:")]
    if not starts:
        return None
    start = starts[0] + 1
    end = None
    for i in range(start, len(lines)):
        if "=> Reserve VRAM" in lines[i] or "=> Building model" in lines[i]:
            end = i
            break
    if end is None:
        return None
    body = []
    for l in lines[start:end]:
        body.append(TS.sub("", l))
    text = "".join(x if x.endswith("\n") else x + "\n" for x in body).rstrip() + "\n"
    return text if "model = dict(" in text else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (a.apply or a.dry_run):
        ap.error("指定 --dry-run 或 --apply")

    report = []
    for ds in DATASETS:
        dp = os.path.join(ROOT, ds)
        if not os.path.isdir(dp):
            continue
        for name in sorted(os.listdir(dp)):
            d = os.path.join(dp, name)
            cfgp, logp = os.path.join(d, "config.py"), os.path.join(d, "train.log")
            if not (os.path.exists(cfgp) and os.path.exists(logp)):
                continue
            # 只有 config 比 log 新才处理
            if os.path.getmtime(cfgp) <= os.path.getmtime(logp):
                continue
            text = extract(logp)
            if text is None:
                report.append((ds, name, "日志里没有 config dump", 0))
                continue
            # save_path 更新为当前目录名
            text = re.sub(r"^save_path = .*$",
                          "save_path = 'exp/%s/%s'" % (ds, name),
                          text, count=1, flags=re.M)
            cur = open(cfgp, encoding="utf-8", errors="ignore").read()
            same = cur.strip() == text.strip()
            report.append((ds, name, "已一致" if same else "需恢复", len(text)))
            if a.apply and not same:
                shutil.copy2(cfgp, cfgp + ".bak")
                open(cfgp, "w", encoding="utf-8").write(text)

    print("%-10s %-28s %-22s %s" % ("数据集", "run", "状态", "字节"))
    print("-" * 80)
    n_need = 0
    for ds, name, st, sz in report:
        if st == "需恢复":
            n_need += 1
        print("%-10s %-28s %-22s %d" % (ds, name, st, sz))
    print("\n共 %d 个 config 比 train.log 新，其中 %d 个需要恢复" % (len(report), n_need))
    if a.dry_run and n_need:
        print("（dry-run，未写入。加 --apply 执行）")


if __name__ == "__main__":
    main()
