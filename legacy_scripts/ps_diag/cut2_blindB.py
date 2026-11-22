"""盲区 B：Cut 1 父子类吞噬体检 + 全点无约束 Top-k Oracle。

用法: python cut2_blindB.py [pred_dir]

pred_dir 默认 /data01/jh/data/val_logits（312 val，ONLINE_s42）。
"""
import glob
import os
import sys

import numpy as np

NO, NC = 12, 57
VAL = sys.argv[1] if len(sys.argv) > 1 else '/data01/jh/data/val_logits'

BASE = ['refrigerator', 'desk', 'curtain', 'bookshelf', 'bed', 'table',
        'window', 'cabinet', 'door', 'chair', 'floor', 'wall']
BN = ['floor', 'wall', 'door', 'chair', 'window', 'trash can', 'table',
      'ceiling', 'doorframe', 'cabinet', 'object', 'shelf', 'desk', 'sink',
      'picture', 'backpack', 'couch', 'box', 'pillow', 'bed', 'radiator',
      'mirror', 'whiteboard', 'lamp', 'curtain', 'toilet', 'book', 'monitor',
      'bookshelf', 'towel', 'tv', 'clothes', 'coffee table', 'office chair',
      'nightstand', 'bag', 'dresser', 'toilet paper', 'recycling bin',
      'kitchen cabinet', 'refrigerator', 'bathtub', 'telephone', 'plant',
      'stool', 'keyboard', 'shoe', 'jacket', 'shower curtain', 'armchair',
      'microwave', 'computer tower', 'bathroom vanity', 'kitchen counter',
      'shower wall', 'paper towel dispenser', 'file cabinet']
NAMES = BASE + [n for n in BN if n not in BASE]
assert len(NAMES) == 57

# 明显的几何近义父子对（父类名, 子类名）
PAIRS = [
    ('chair', 'office chair'), ('chair', 'armchair'), ('chair', 'stool'),
    ('cabinet', 'kitchen cabinet'), ('cabinet', 'file cabinet'),
    ('cabinet', 'nightstand'), ('cabinet', 'dresser'),
    ('cabinet', 'bathroom vanity'),
    ('table', 'coffee table'), ('table', 'kitchen counter'),
    ('desk', 'coffee table'), ('desk', 'kitchen counter'),
    ('door', 'doorframe'),
    ('curtain', 'shower curtain'),
    ('wall', 'shower wall'),
    ('bookshelf', 'shelf'),
    ('window', 'curtain'),
]
N2I = {n: i for i, n in enumerate(NAMES)}
PAIR_IDX = [(N2I[a], N2I[b]) for a, b in PAIRS if a in N2I and b in N2I]


def metrics(cm):
    d = np.diag(cm).astype(np.float64)
    r = cm.sum(1).astype(np.float64)
    c = cm.sum(0).astype(np.float64)
    iou = d / np.maximum(r + c - d, 1)
    b = float(iou[:NO].mean())
    n = float(iou[NO:].mean())
    return b, n, 2 * b * n / max(b + n, 1e-9)


def main():
    cm = np.zeros((NC, NC), np.int64)
    n_top2 = n_top5 = n_err = 0
    cat = {'base->novel': [0, 0, 0], 'novel->base': [0, 0, 0],
           'novel->novel': [0, 0, 0], 'base->base': [0, 0, 0]}
    for f in sorted(glob.glob(VAL + '/*.npz')):
        z = np.load(f)
        p = z['top5p'].astype(np.float32)
        i = z['top5i'].astype(np.int64)
        gt = z['gt'].astype(np.int64)
        pred = z['pred'].astype(np.int64)
        ok = gt >= 0
        gt, pred, i = gt[ok], pred[ok], i[ok]
        cm += np.bincount(gt * NC + pred, minlength=NC * NC).reshape(NC, NC)
        wrong = gt != pred
        n_err += int(wrong.sum())
        in2 = (i[:, :2] == gt[:, None]).any(1)
        in5 = (i == gt[:, None]).any(1)
        n_top2 += int((wrong & in2).sum())
        n_top5 += int((wrong & in5).sum())
        gb = gt < NO
        pb = pred < NO
        kinds = np.where(gb & ~pb, 'base->novel',
                         np.where(~gb & pb, 'novel->base',
                                  np.where(~gb & ~pb, 'novel->novel',
                                           'base->base')))
        for k in cat:
            m = wrong & (kinds == k)
            cat[k][0] += int(m.sum())
            cat[k][1] += int((m & in2).sum())
            cat[k][2] += int((m & in5).sum())

    b, n, h = metrics(cm)
    np.save('/data01/jh/data/cut2_feat/blindB_cm.npy', cm)
    print('=== ONLINE_s42 基线（重算）Base %.4f Novel %.4f HM %.4f ==='
          % (b, n, h))
    print('有效点错误 %d | Top-2 可救 %d (%.1f%%) | Top-5 可救 %d (%.1f%%)'
          % (n_err, n_top2, 100 * n_top2 / max(n_err, 1),
             n_top5, 100 * n_top5 / max(n_err, 1)))

    print('\n=== 错误类型分解（总量 / Top-2 可救 / Top-5 可救）===')
    for k, v in cat.items():
        print('  %-12s %9d  %9d (%5.1f%%)  %9d (%5.1f%%)'
              % (k, v[0], v[1], 100 * v[1] / max(v[0], 1),
                 v[2], 100 * v[2] / max(v[0], 1)))

    # ---- 父子对吞噬体检 ----
    pair_set = set()
    for a, c in PAIR_IDX:
        pair_set.add((a, c))
        pair_set.add((c, a))
    tot_bn = nv_bn = tot_nb = nv_nb = 0
    for g in range(NC):
        for pr in range(NC):
            v = int(cm[g, pr])
            if v == 0 or g == pr:
                continue
            if g < NO <= pr:
                tot_bn += v
                if (g, pr) in pair_set:
                    nv_bn += v
            elif pr < NO <= g:
                tot_nb += v
                if (g, pr) in pair_set:
                    nv_nb += v
    print('\n=== 父子对吞噬（Cut 1 体检）===')
    print('  Base→Novel 误差 %8d，落在父子对 %8d (%.1f%%)'
          % (tot_bn, nv_bn, 100 * nv_bn / max(tot_bn, 1)))
    print('  Novel→Base 误差 %8d，落在父子对 %8d (%.1f%%)'
          % (tot_nb, nv_nb, 100 * nv_nb / max(tot_nb, 1)))

    print('\n=== 每个新类的 Top-1 误判去向 ===')
    for g in range(NO, NC):
        row = cm[g].copy()
        row[g] = 0
        if row.sum() == 0:
            continue
        top = np.argsort(-row)[:3]
        top = [t for t in top if row[t] > 0]
        sv = float(np.diag(cm)[g]) / max(cm[g].sum(), 1)
        print('  %-22s 召回 %.3f | 误判去向: %s'
              % (NAMES[g], sv,
                 ', '.join('%s %d' % (NAMES[t], row[t]) for t in top)))

    # ---- 12 个基类各自作为「吞噬端」接收了多少新类点 ----
    print('\n=== 12 个基类作为接收端的吞噬量（按 base 类汇总）===')
    recv = cm[NO:, :NO].sum(0)          # 每列 = novel GT 被该 base 接收
    to_base = recv.sum()
    order = np.argsort(-recv)
    for b in order:
        if recv[b] == 0:
            continue
        print('  %-14s 接收 %9d (占 Novel→Base 误差 %.1f%%)'
              % (NAMES[b], recv[b], 100 * recv[b] / max(to_base, 1)))
    print('  Novel→Base 总误差 %d' % to_base)

    # ---- 严格父子对 vs 背景类吞噬 ----
    bg = {N2I[x] for x in ('wall', 'floor', 'ceiling', 'window')}
    tot_nb2 = nv_nb2 = bg_nb = 0
    for g in range(NO, NC):
        for pr in range(NO):
            v = int(cm[g, pr])
            if v == 0:
                continue
            tot_nb2 += v
            if (g, pr) in pair_set:
                nv_nb2 += v
            elif pr in bg:
                bg_nb += v
    print('\n=== Novel→Base 误差再分解 ===')
    print('  总数 %d' % tot_nb2)
    print('  严格几何父子对      %8d (%.1f%%)' %
          (nv_nb2, 100 * nv_nb2 / max(tot_nb2, 1)))
    print('  被 wall/floor/ceiling/window 吞 %8d (%.1f%%)'
          % (bg_nb, 100 * bg_nb / max(tot_nb2, 1)))
    print('  两者合计            %8d (%.1f%%)'
          % (nv_nb2 + bg_nb, 100 * (nv_nb2 + bg_nb) / max(tot_nb2, 1)))

    # ---- Base→Novel：被哪些 novel 接收 ----
    recv_n = cm[:NO, NO:].sum(0)
    order_n = np.argsort(-recv_n)
    print('\n=== Base→Novel 误差里，接收量最大的 10 个新类 ===')
    for k in order_n[:10]:
        if recv_n[k] == 0:
            continue
        print('  %-22s 接收 %9d (%.1f%%)'
              % (NAMES[NO + k], recv_n[k],
                 100 * recv_n[k] / max(recv_n.sum(), 1)))


if __name__ == '__main__':
    main()
