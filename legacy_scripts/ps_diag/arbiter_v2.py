"""仲裁器 v2（novel–novel）：候选 top-k，多种聚合方式，一次扫完。

对每个冲突分量（margin<tau，top1/top2 同为不同新类）：
  候选类 = 分量内 top-k 出现过的类
  分数 = alpha * 类概率均值 + (1-alpha) * Sonata 余弦均值
  仅当最优-次优 > sw 时整体改判。

用法: python arbiter_v2.py <tau> <topk> <alphas> <sws>
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


def metrics(cm):
    d = np.diag(cm).astype(np.float64)
    r = cm.sum(1).astype(np.float64)
    c = cm.sum(0).astype(np.float64)
    iou = d / np.maximum(r + c - d, 1)
    b = float(iou[:NO].mean())
    n = float(iou[NO:].mean())
    return b, n, 2 * b * n / max(b + n, 1e-9)


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
    tau = float(sys.argv[1])
    topk = int(sys.argv[2])
    alphas = [float(x) for x in sys.argv[3].split(',')]
    sws = [float(x) for x in sys.argv[4].split(',')]
    keys = [(a, s) for a in alphas for s in sws]
    acc = {k: np.zeros((NC, NC), np.int64) for k in keys}
    n_sw = {k: 0 for k in keys}
    cm0 = np.zeros((NC, NC), np.int64)
    k = min(topk, 5)

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
        ok = gt >= 0
        cm0 += np.bincount(gt[ok] * NC + pred[ok], minlength=NC * NC).reshape(NC, NC)

        for comp in comps(conf, coord):
            if len(comp) < 20:
                continue
            cand = np.unique(i[comp][:, :k].reshape(-1))
            cand = cand[cand >= NO]
            if len(cand) < 2:
                continue
            pr = np.stack([np.where(i[comp] == c, p[comp], 0.0).sum(1) for c in cand])
            lg = pr.mean(1)
            cs = np.array([cosp[comp, c - NO].mean() for c in cand])
            okc = gt[comp] >= 0
            g = gt[comp][okc]
            pc = pred[comp][okc]
            base_cell = np.bincount(g * NC + pc, minlength=NC * NC).reshape(NC, NC)
            for a in alphas:
                s_ = a * lg + (1 - a) * cs
                order = np.argsort(-s_)
                best = cand[order[0]]
                gap = s_[order[0]] - s_[order[1]]
                delta = np.bincount(g * NC + best, minlength=NC * NC).reshape(NC, NC) - base_cell
                for sw in sws:
                    if gap > sw:
                        acc[(a, sw)] += delta
                        n_sw[(a, sw)] += 1
        if (fi + 1) % 100 == 0:
            print('  %d' % (fi + 1), flush=True)

    b0, n0, h0 = metrics(cm0)
    print('\n=== tau=%.2f topk=%d 基线 N %.4f H %.4f ===' % (tau, k, n0, h0))
    for kk in keys:
        b, n, h = metrics(cm0 + acc[kk])
        print('a=%.1f sw=%.2f 改判 %6d  N %.4f H %.4f (ΔN %+.2fpp, ΔH %+.2fpp)'
              % (kk[0], kk[1], n_sw[kk], n, h, 100 * (n - n0), 100 * (h - h0)))


if __name__ == '__main__':
    main()
