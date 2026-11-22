# -*- coding: utf-8 -*-
"""把 Novel mIoU 的丢分拆成可归因的桶。

用法: python cm_bucket_analysis.py <cm.npy> [nr_base=12]

注意桶之间的关系（它们不是同一层面的互斥量）：
  总 error = FN + FP
    FN = A（Novel -> Base） + B（Novel -> 其他 Novel） + cm[n,n] 之外未分类
    FP = D（Base/其他 -> Novel）
  C 不是"去向"，而是"FN 里有多少来自 recall≈0 的类"（与 A/B 重叠）。

所以本脚本输出三层：
  第 1 层：FN 的去向分解 A / B（互斥，加起来 = FN）
  第 2 层：C（盲区类的 FN 贡献）—— 说明 FN 里多少是"根本没召回"
  第 3 层：D（FP 侧）—— 幻觉量
并给出"如果消除某一桶，Novel mIoU 能回到多少"的上界估计。
"""
import sys

import numpy as np

NC_BASE = 12


def main():
    path = sys.argv[1]
    nr_base = int(sys.argv[2]) if len(sys.argv) > 2 else NC_BASE
    cm = np.load(path)
    NC = cm.shape[0]
    novel = list(range(nr_base, NC))
    base = list(range(nr_base))

    # 只保留 GT 非 ignore 的部分（cm 已在采集时过滤）
    gt_novel = cm[novel].sum(axis=1).astype(np.float64)      # 每个 novel 类的真值点数
    pred_novel = cm[:, novel].sum(axis=0).astype(np.float64)  # 每个 novel 类被预测出的点数
    diag_novel = np.diag(cm)[novel].astype(np.float64)

    total_gt_novel = gt_novel.sum()
    total_pred_novel = pred_novel.sum()

    # 第 1 层：FN 的去向（互斥）
    to_base = cm[np.ix_(novel, base)].sum(axis=1).astype(np.float64)   # Novel -> Base
    to_novel = (gt_novel - diag_novel - to_base)                       # Novel -> 其他 Novel
    fn_total = gt_novel - diag_novel

    # 第 2 层：recall≈0 的类贡献的 FN
    recall = diag_novel / np.maximum(gt_novel, 1)
    blind = recall < 0.05
    blind_fn = fn_total[blind].sum()
    blind_gt = gt_novel[blind].sum()

    # 第 3 层：FP（其他类 -> Novel）
    fp_novel = (pred_novel - diag_novel).sum()
    fp_from_base = cm[np.ix_(base, novel)].sum()
    fp_from_novel = cm[np.ix_(novel, novel)].sum() - diag_novel.sum()
    fp_from_other = fp_novel - fp_from_base - fp_from_novel

    # IoU / mIoU
    union = gt_novel + pred_novel - diag_novel
    iou = diag_novel / np.maximum(union, 1)
    miou0 = iou.mean()

    def miou_if(gain_diag):
        """假设额外拿回 gain_diag[i] 个正确 TP，mIoU 变成多少。

        约束：diag 不能超过 min(gt, pred)——否则 union 会变负，IoU 突破 1。
        这是一个保守上界：假定预测总量不变，只是把错判改成对判。
        """
        cap = np.minimum(gt_novel, pred_novel)
        room = np.maximum(cap - diag_novel, 0)
        g = np.minimum(gain_diag, room)
        d = diag_novel + g
        u = np.maximum(gt_novel + pred_novel - d, 1e-9)
        return (d / u).mean()

    print("=" * 70)
    print("混淆矩阵 %s   shape %s" % (path.split("/")[-1], cm.shape))
    print("Novel 类 %d（下标 %d..%d）  Base 类 %d" % (len(novel), nr_base, NC - 1, nr_base))
    print("=" * 70)
    print("Novel 真值总点数 %d   预测为 Novel 的总点数 %d" % (total_gt_novel, total_pred_novel))
    print("当前 Novel mIoU  %.4f   整体准确率 %.4f"
          % (miou0, diag_novel.sum() / max(total_gt_novel, 1)))
    print()

    print("【第 1 层】FN 的去向（互斥，合计 = 全部漏检）")
    print("  TP（判对）              %9d  %5.1f%% of GT" % (diag_novel.sum(),
                                                           100 * diag_novel.sum() / total_gt_novel))
    print("  A: Novel -> Base        %9d  %5.1f%% of GT   %5.1f%% of FN"
          % (to_base.sum(), 100 * to_base.sum() / total_gt_novel,
             100 * to_base.sum() / max(fn_total.sum(), 1)))
    print("  B: Novel -> 其他 Novel  %9d  %5.1f%% of GT   %5.1f%% of FN"
          % (to_novel.sum(), 100 * to_novel.sum() / total_gt_novel,
             100 * to_novel.sum() / max(fn_total.sum(), 1)))
    print("  FN 合计                 %9d  %5.1f%% of GT"
          % (fn_total.sum(), 100 * fn_total.sum() / total_gt_novel))
    print()

    print("【第 2 层】C: 盲区（recall < 5%% 的类）")
    print("  盲区类数 %d / %d   覆盖 GT 点数 %d（%5.1f%% of Novel GT）"
          % (int(blind.sum()), len(novel), int(blind_gt), 100 * blind_gt / total_gt_novel))
    print("  盲区贡献的 FN %d（占全部 FN 的 %5.1f%%）"
          % (int(blind_fn), 100 * blind_fn / max(fn_total.sum(), 1)))
    if blind.any():
        idx = np.array(novel)[blind]
        print("  盲区的类: %s" % idx.tolist())
        print("  它们的 recall: %s" % np.round(recall[blind], 4).tolist())
    print()

    print("【第 3 层】D: FP（其他类被误判成 Novel）")
    print("  FP 合计          %9d   来自 Base %d / 来自其他 Novel %d / 其他 %d"
          % (int(fp_novel), int(fp_from_base), int(fp_from_novel), int(fp_from_other)))
    print("  FP / 预测为 Novel 的总数 = %5.1f%%   （= 1 - Novel Precision）"
          % (100 * fp_novel / max(total_pred_novel, 1)))
    print()

    print("【上界估计】单独消除某一桶，Novel mIoU 能到多少")
    gains = {
        "消除 A（Novel->Base 全部判对）": to_base,
        "消除 B（Novel->Novel 全部判对）": to_novel,
        "消除 C（盲区类全部判对）": np.where(blind, fn_total, 0).astype(np.float64),
        "消除 A+B（全部 FN 判对）": fn_total,
    }
    for name, g in gains.items():
        print("  %-34s mIoU %.4f  (+%.4f)" % (name, miou_if(g), miou_if(g) - miou0))
    print()

    # 每类明细（按 FN 占比排序）
    order = np.argsort(-fn_total)[:15]
    print("【逐类】FN 最多的 15 个 Novel 类")
    print("  %-6s %-9s %-8s %-8s %-8s %-8s" % ("cls", "GT", "recall", "->Base%", "->Novel%", "IoU"))
    for j in order:
        n = novel[j]
        gt = max(gt_novel[j], 1)
        print("  %-6d %-9d %-8.3f %-8.3f %-8.3f %-8.4f"
              % (n, int(gt_novel[j]), recall[j],
                 100 * to_base[j] / gt, 100 * to_novel[j] / gt, iou[j]))


if __name__ == "__main__":
    main()
