# -*- coding: utf-8 -*-
"""关键分叉实验：Top-20 Novel<->Novel 混淆对，老师教错了 vs 学生没学会。

思路：
  学生矩阵 = 验证集上的 3D 预测 vs GT（已有 val_cm.npy）
  老师矩阵 = 训练集上的 VLM 伪标签 vs GT（本脚本现算）

  对每一对 (c_i, c_j)：
    学生混淆率 = cm_val[c_i, c_j] / cm_val[c_i].sum()
    老师混淆率 = cm_train[c_i, c_j] / cm_train[c_i].sum()

  若老师混淆率也高 => 源头就把 c_i 整块标成 c_j（老师教错了）
  若老师混淆率低而学生高 => VLM 分得开，3D 特征糊了（学生没学会）

用法: python teacher_vs_student.py <val_cm.npy>
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/data01/jh/code/GFS-VL")
sys.path.insert(0, "/data01/jh/code/GFS-VL/pointcept/models/PLA")

import vlm_common as V  # noqa: E402

PL_DIR = "/dev/shm/jh_pl_k1_s10"
GT_ROOT = "/data01/jh/data/scannet200_gfs/train"
NO, NC = 12, 57


def build_teacher_cm():
    """遍历 1201 训练场景，用 VLM 伪标签 vs GT 累积 57x57 矩阵。"""
    cm = np.zeros((NC, NC), dtype=np.int64)
    scenes = sorted(os.listdir(GT_ROOT))
    n_ok = n_skip = 0
    for i, sc in enumerate(scenes):
        p = os.path.join(PL_DIR, sc + "_pl.npy")
        g = os.path.join(GT_ROOT, sc, "segment200.npy")
        if not (os.path.exists(p) and os.path.exists(g)):
            n_skip += 1
            continue
        pl = np.load(p).reshape(-1).astype(np.int64)
        gfs = V.convert_to_gfs(np.load(g).reshape(-1)).astype(np.int64)
        if len(pl) != len(gfs):
            n_skip += 1
            continue
        # PL 的 -1 是背景，GT 的 -1 是 ignore；两边都要求 GT 有效
        v = (gfs >= 0) & (gfs < NC)          # 只看 GT 有效的点
        if not v.any():
            n_skip += 1
            continue
        pv = pl[v]
        gv = gfs[v]
        # PL == -1 视为"未标"，单列到 NC 号桶，便于统计漏标
        pv2 = np.where(pv < 0, NC, pv)
        cm += np.bincount(gv * (NC + 1) + pv2,
                          minlength=NC * (NC + 1)).reshape(NC, NC + 1)[:, :NC]
        n_ok += 1
        if (i + 1) % 300 == 0:
            print("  老师矩阵 %d/%d  %.0fs" % (i + 1, len(scenes), 0), flush=True)
    print("老师矩阵完成: 用 %d 场景，跳过 %d" % (n_ok, n_skip), flush=True)
    return cm


def main():
    val_cm = np.load(sys.argv[1])
    print("学生矩阵 %s" % sys.argv[1], flush=True)
    teacher = build_teacher_cm()
    np.save("/data01/jh/data/cm_analysis/teacher_train_cm.npy", teacher)

    # 严格剔除 ignore：上面已只取 GT 有效点
    print()
    print("=" * 78)
    print("老师（训练集 VLM PL vs GT）  学生（验证集 3D pred vs GT）")
    print("=" * 78)
    tg = teacher.sum(axis=1).astype(np.float64)
    sg = val_cm.sum(axis=1).astype(np.float64)
    print("训练集 GT 点 %d   验证集 GT 点 %d" % (int(tg.sum()), int(sg.sum())))

    # Top-20 Novel<->Novel 混淆对（按学生矩阵的绝对量排序）
    pairs = []
    for i in range(NO, NC):
        for j in range(NO, NC):
            if i == j:
                continue
            pairs.append((int(val_cm[i, j]), i, j))
    pairs.sort(reverse=True)
    top = pairs[:20]

    print()
    print("%-6s %-6s %10s %10s %9s %9s %9s  %s"
          % ("c_i", "c_j", "学生误判", "老师误标", "学生率", "老师率", "差值", "判定"))
    print("-" * 78)
    n_teacher = n_student = 0
    for cnt, i, j in top:
        s_rate = val_cm[i, j] / max(sg[i], 1)
        t_rate = teacher[i, j] / max(tg[i], 1)
        d = s_rate - t_rate
        if t_rate > 0.15 and d < 0.10:
            verdict = "老师教错了"
            n_teacher += 1
        elif d > 0.15:
            verdict = "学生没学会"
            n_student += 1
        else:
            verdict = "两者都有"
        print("%-6d %-6d %10d %10d %8.1f%% %8.1f%% %+8.1f%%  %s"
              % (i, j, cnt, int(teacher[i, j]), 100 * s_rate, 100 * t_rate,
                 100 * d, verdict))

    print("-" * 78)
    print("Top-20 判定: 老师教错 %d 对 / 学生没学会 %d 对 / 混合 %d 对"
          % (n_teacher, n_student, 20 - n_teacher - n_student))

    # 整体：Novel 类的老师覆盖率（PL 有没有给出该类的点）
    print()
    print("【盲区诊断】每个 novel 类的训练集 VLM 覆盖率 = PL 判为该类的总点数 / 该类 GT 点数")
    print("  （< 100%% 的部分说明 PL 漏标或串到了别的类；> 100%% 说明 PL 幻觉多标）")
    cov = []
    for c in range(NO, NC):
        gt_c = tg[c]                       # GT == c 的点数
        pl_c = teacher[:, c].sum()         # PL 判为 c 的总点数（任意 GT）
        cov.append((c, int(gt_c), int(pl_c), pl_c / max(gt_c, 1)))
    cov.sort(key=lambda x: x[3])
    print("  覆盖率最低的 10 类:")
    print("  %-6s %10s %12s %8s" % ("cls", "GT点", "PL判为该类", "覆盖率"))
    for c, gt_c, pl_c, r in cov[:10]:
        print("  %-6d %10d %12d %7.1f%%" % (c, gt_c, pl_c, 100 * r))
    zero = [c for c, _, _, r in cov if r < 0.05]
    print("  覆盖率 < 5%% 的类: %s（真正盲区候选）" % zero)
    over = [c for c, gt_c, pl_c, r in cov if r > 1.5]
    print("  覆盖率 > 150%% 的类: %s（PL 对这些类有幻觉）" % over)


if __name__ == "__main__":
    main()
