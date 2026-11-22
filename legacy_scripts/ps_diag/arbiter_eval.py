"""Step 3：纯 3D 局部仲裁器（novel–novel），一次扫多组 alpha/margin。

对每个冲突连通分量（margin<tau 且 top1/top2 都是新类且不同）：
  候选类 = 分量内 top-1/top-2 出现过的类
  每类分数 = alpha * 分量内该类概率均值 + (1-alpha) * 分量内该类 Sonata 余弦均值
  仅当最优-次优 > sw 时整体改判，否则保持逐点预测。

用法: python arbiter_eval.py <tau> <alpha列表> <sw列表>
例:   python arbiter_eval.py 0.2 0.0,0.3,0.5,0.7,1.0 0.0,0.02,0.05
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
    base = float(iou[:NO].mean())
    nov = float(iou[NO:].mean())
    return base, nov, 2 * base * nov / max(base + nov, 1e-9)


def components(mask, coord, radius=RADIUS):
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
    alphas = [float(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2
                                 else ['0.0', '0.3', '0.5', '0.7', '1.0'])]
    sws = [float(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3
                              else ['0.0', '0.02', '0.05'])]
    keys = [(a, s) for a in alphas for s in sws]
    acc = {k: np.zeros((NC, NC), np.int64) for k in keys}
    cm0 = np.zeros((NC, NC), np.int64)
    n_comp = n_sw = {k: 0 for k in keys}

    for fi, f in enumerate(sorted(glob.glob(VAL + '/*.npz'))):
        sc = os.path.basename(f)[:-4]
        z = np.load(f)
        p = z['top5p'].astype(np.float32)
        i = z['top5i'].astype(np.int64)
        pred = z['pred'].astype(np.int64)
        gt = z['gt'].astype(np.int64)
        cosp = np.load(os.path.join(COS, sc + '_cos.npy')).astype(np.float32)
        coord = np.load(os.path.join(COORD, sc, 'coord.npy')).astype(np.float32)
        if len(pred) != len(coord) or len(cosp) != len(pred):
            continue
        margin = p[:, 0] - p[:, 1]
        c1, c2 = i[:, 0], i[:, 1]
        conf = (margin < tau) & (c1 >= NO) & (c2 >= NO) & (c1 != c2)
        ok = gt >= 0
        cm0 += np.bincount(gt[ok] * NC + pred[ok], minlength=NC * NC).reshape(NC, NC)

        for comp in components(conf, coord):
            if len(comp) < 20:
                continue
            cand = np.unique(np.concatenate([i[comp, 0], i[comp, 1]]))
            cand = cand[cand >= NO]
            if len(cand) < 2:
                continue
            pr = np.stack([np.where(i[comp] == c, p[comp], 0.0).sum(1)
                           for c in cand])           # (k, n)
            lg = pr.mean(1)
            cs = np.array([cosp[comp, c - NO].mean() for c in cand])
            for a in alphas:
                s = a * lg + (1 - a) * cs
                order = np.argsort(-s)
                best = cand[order[0]]
                gap = s[order[0]] - s[order[1]]
                for sw in sws:
                    k = (a, sw)
                    n_comp[k] += 1
                    if gap > sw:
                        out = pred.copy()
                        out[comp] = best
                        n_sw[k] += 1
                        gg = gt[ok] * NC + out[ok]
                        acc[k] += np.bincount(gg, minlength=NC * NC).reshape(NC, NC) \
                            - np.bincount(gt[ok] * NC + pred[ok],
                                          minlength=NC * NC).reshape(NC, NC)
        if (fi + 1) % 80 == 0:
            print('  %d' % (fi + 1), flush=True)

    b0, n0, h0 = metrics(cm0)
    print('\n=== tau=%.2f 基线 Base %.4f Novel %.4f HM %.4f ===' % (tau, b0, n0, h0))
    print('%-6s %-6s %8s %8s %8s %8s' % ('alpha', 'sw', '改判数', 'Base', 'Novel', 'HM'))
    for k in keys:
        b, n, h = metrics(cm0 + acc[k])
        print('%-6.1f %-6.2f %8d %8.4f %8.4f %8.4f  (ΔN %+.2fpp)'
              % (k[0], k[1], n_sw[k], b, n, h, 100 * (n - n0)))


if __name__ == '__main__':
    main()
