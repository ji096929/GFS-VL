"""解析 4 个 run 的逐类 IoU，输出 Base/Novel/HM 与逐类 delta。

用法: python xl_compare.py [log_dir]
"""
import os
import re
import sys

NB = 12
NC = 57
NAMES = ["base_s42", "xl_s42", "base_s7", "xl_s7"]
FILE = {
    "base_s42": "base_s42",
    "xl_s42": "sc200_k1_XLu_s42_resume",
    "base_s7": "sc200_k1_BASEu_s7",
    "xl_s7": "xl_s7_eval",
}


def parse(path):
    ious = {}
    for line in open(path):
        m = re.search(r"Class_(\d+) - (.+?) Final Result: iou ([\d.]+)", line)
        if m:
            ious[int(m.group(1))] = (m.group(2), float(m.group(3)))
    if len(ious) != NC:
        return None
    v = [ious[i][1] for i in range(NC)]
    b = sum(v[:NB]) / NB
    n = sum(v[NB:]) / (NC - NB)
    return dict(name=[ious[i][0] for i in range(NC)], iou=v, B=b, N=n,
                HM=2 * b * n / (b + n), M=sum(v) / NC)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/data01/jh/afford_diag/xl_runs"
    res = {}
    for r in NAMES:
        p = os.path.join(d, FILE[r] + ".log")
        if not os.path.exists(p):
            print("%-9s 日志缺失" % r)
            continue
        x = parse(p)
        if x is None:
            print("%-9s 评估未完成" % r)
            continue
        res[r] = x
        print("%-9s Base %.4f  Novel %.4f  HM %.4f  mIoU %.4f"
              % (r, x["B"], x["N"], x["HM"], x["M"]))
    for s in ("s42", "s7"):
        a, b = res.get("base_" + s), res.get("xl_" + s)
        if not a or not b:
            continue
        print("\n== %s: xl - base ==" % s)
        print("  Base %+.2fpp  Novel %+.2fpp  HM %+.2fpp"
              % ((b["B"] - a["B"]) * 100, (b["N"] - a["N"]) * 100,
                 (b["HM"] - a["HM"]) * 100))
        ds = [(b["name"][i], (b["iou"][i] - a["iou"][i]) * 100,
               a["iou"][i], b["iou"][i]) for i in range(NC)]
        ds.sort(key=lambda z: -z[1])
        print("  涨最多: " + " | ".join(
            "%s %+.2f (%.3f->%.3f)" % z for z in ds[:10]))
        print("  跌最多: " + " | ".join(
            "%s %+.2f (%.3f->%.3f)" % z for z in ds[-10:]))
        pos_b = sum(1 for i in range(NB) if b["iou"][i] > a["iou"][i])
        pos_n = sum(1 for i in range(NB, NC) if b["iou"][i] > a["iou"][i])
        print("  正类数 %d/%d（Base %d/12, Novel %d/45）"
              % (pos_b + pos_n, NC, pos_b, pos_n))
    if all(r in res for r in NAMES):
        mb = [(res["base_s42"]["B"] + res["base_s7"]["B"]) / 2,
              (res["base_s42"]["N"] + res["base_s7"]["N"]) / 2]
        mx = [(res["xl_s42"]["B"] + res["xl_s7"]["B"]) / 2,
              (res["xl_s42"]["N"] + res["xl_s7"]["N"]) / 2]
        hb = 2 * mb[0] * mb[1] / (mb[0] + mb[1])
        hx = 2 * mx[0] * mx[1] / (mx[0] + mx[1])
        print("\n== 双 seed 平均 ==")
        print("  base: Base %.4f Novel %.4f HM %.4f" % (mb[0], mb[1], hb))
        print("  xl  : Base %.4f Novel %.4f HM %.4f" % (mx[0], mx[1], hx))
        print("  delta: Base %+.2fpp Novel %+.2fpp HM %+.2fpp"
              % ((mx[0] - mb[0]) * 100, (mx[1] - mb[1]) * 100,
                 (hx - hb) * 100))


if __name__ == "__main__":
    main()
