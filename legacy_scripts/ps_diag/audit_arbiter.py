"""逐分量审计：Sonata argmax 在「可改判分量」上到底有多准。

对 τ=0.2、novel–novel 冲突：
  correct_now   = 分量当前多数预测 == GT 多数类
  switchable    = 其他（当前是错的）
在 switchable 上统计：
  fix  = sonata argmax == GT 多数类
  same = sonata argmax == 当前多数预测
  brk  = 其他（改成另一个错类）
并给出「全改判」与「仅在 sonata 与当前不同时改判」的净收益上界。
"""
import glob
import os
import sys

import numpy as np

NO, NC = 12, 57
VAL = '/data01/jh/data/val_logits'
COORD = '/data01/jh/data/scannet200_gfs/val'
COS = '/data01/jh/data/val_sonata_cos'
RADIUS = 0.05


def comps(mask, coord, radius=RADIUS):
    from scipy import ndimage
    idx = np.where(mask)[0]
    if len(idx) < 20:
        return []
    pts = coord[idx]
    mn = pts.min(0)
    g = np.floor((pts - mn) / radius).astype(np.int64)
    grid = np.zeros(tuple(g.max(0) + 1), dtype=bool)
    grid[g[:, 0], g[:, 1], g[:, 2]] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=int))
    labels = lab[g[:, 0], g[:, 1], g[:, 2]]
    return [idx[labels == k] for k in range(1, n + 1)]


def main():
    tau = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n_ok = n_sw = fix = same = brk = 0
    oracle_fix = oracle_brk = 0
    for fi, f in enumerate(sorted(glob.glob(VAL + '/*.npz'))):
        sc = os.path.basename(f)[:-4]
        z = np.load(f)
        p = z['top5p'].astype(np.float32)
        i = z['top5i'].astype(np.int64)
        pred = z['pred'].astype(np.int64)
        gt = z['gt'].astype(np.int64)
        cosp = np.load(os.path.join(COS, sc + '_cos.npy')).astype(np.float32)
        coord = np.load(os.path.join(COORD, sc, 'coord.npy')).astype(np.float32)
        margin = p[:, 0] - p[:, 1]
        c1, c2 = i[:, 0], i[:, 1]
        conf = (margin < tau) & (c1 >= NO) & (c2 >= NO) & (c1 != c2)
        for comp in comps(conf, coord):
            if len(comp) < 20:
                continue
            g = gt[comp]
            g = g[g >= 0]
            if len(g) < 0.5 * len(comp):
                continue
            cand = np.unique(i[comp][:, :k].reshape(-1))
            cand = cand[cand >= NO]
            if len(cand) < 2:
                continue
            cnt_all = np.bincount(g, minlength=NC)
            cur = int(cnt_all[cand].argmax())          # 当前多数（限候选内）
            maj = int(cnt_all.argmax())                # 真值多数（任意类）
            # 当前预测的多数类（不限候选，反映 pred 实际状态）
            pv = np.bincount(pred[comp], minlength=NC)
            cur_pred = int(pv.argmax())
            cs = np.array([cosp[comp, c - NO].mean() for c in cand])
            best = int(cand[cs.argmax()])
            if cur_pred == maj:
                n_ok += 1
                continue
            n_sw += 1
            if best == maj:
                fix += 1
            elif best == cur_pred:
                same += 1
            else:
                brk += 1
            # oracle：候选里挑 GT 最多的（等价上限）
            if int(cnt_all[cand].argmax()) == maj:
                oracle_fix += 1
            else:
                oracle_brk += 1
        if (fi + 1) % 100 == 0:
            print('  %d' % (fi + 1), flush=True)

    print('\n=== tau=%.2f topk=%d ===' % (tau, k))
    print('分量总数 %d，当前已对 %d (%.1f%%)，可改判 %d'
          % (n_ok + n_sw, n_ok, 100 * n_ok / max(n_ok + n_sw, 1), n_sw))
    print('在可改判分量上（%d 个）:' % n_sw)
    print('  Sonata 选对（=GT多数）  %4d  %.1f%%' % (fix, 100 * fix / max(n_sw, 1)))
    print('  Sonata 维持原判         %4d  %.1f%%' % (same, 100 * same / max(n_sw, 1)))
    print('  Sonata 改成另一个错类   %4d  %.1f%%' % (brk, 100 * brk / max(n_sw, 1)))
    print('  Oracle（候选内 GT 多数）%4d  %.1f%%' % (oracle_fix, 100 * oracle_fix / max(n_sw, 1)))


if __name__ == '__main__':
    main()
