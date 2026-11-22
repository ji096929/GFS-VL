"""候选集受限的 Oracle：仲裁器只能在 top-k 候选里选时，上限是多少。

对比三种：
  oracle_all   任意类（此前的 +4.5pp 就是这个）
  oracle_top2  仅 top-1/top-2 出现过的类
  oracle_top5  仅 top-5 出现过的类
"""
import glob
import os
import sys

import numpy as np

NO, NC = 12, 57
VAL = '/data01/jh/data/val_logits'
COORD = '/data01/jh/data/scannet200_gfs/val'
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
    tau = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    cm0 = np.zeros((NC, NC), np.int64)
    add = {k: np.zeros((NC, NC), np.int64) for k in ('all', 'top2', 'top5')}

    for fi, f in enumerate(sorted(glob.glob(VAL + '/*.npz'))):
        sc = os.path.basename(f)[:-4]
        z = np.load(f)
        p = z['top5p'].astype(np.float32)
        i = z['top5i'].astype(np.int64)
        pred = z['pred'].astype(np.int64)
        gt = z['gt'].astype(np.int64)
        coord = np.load(os.path.join(COORD, sc, 'coord.npy')).astype(np.float32)
        margin = p[:, 0] - p[:, 1]
        c1, c2 = i[:, 0], i[:, 1]
        conf = (margin < tau) & (c1 >= NO) & (c2 >= NO) & (c1 != c2)
        ok = gt >= 0
        base_cell = np.bincount(gt[ok] * NC + pred[ok], minlength=NC * NC).reshape(NC, NC)
        cm0 += base_cell
        for comp in comps(conf, coord):
            if len(comp) < 20:
                continue
            g = gt[comp]
            g = g[g >= 0]
            if len(g) < 0.5 * len(comp):
                continue
            cnt = np.bincount(g, minlength=NC)
            cands = {
                'all': np.arange(NC),
                'top2': np.unique(np.concatenate([i[comp, 0], i[comp, 1]])),
                'top5': np.unique(i[comp].reshape(-1)),
            }
            okc = gt[comp] >= 0
            gg = gt[comp][okc]
            pc = pred[comp][okc]
            bc = np.bincount(gg * NC + pc, minlength=NC * NC).reshape(NC, NC)
            for k, cand in cands.items():
                sub = cnt[cand]
                if sub.sum() < 0.5 * len(g):
                    continue
                best = int(cand[sub.argmax()])
                ncell = np.bincount(gg * NC + best, minlength=NC * NC).reshape(NC, NC)
                add[k] += ncell - bc
        if (fi + 1) % 100 == 0:
            print('  %d' % (fi + 1), flush=True)

    b0, n0, h0 = metrics(cm0)
    print('\n=== tau=%.2f 基线 Base %.4f Novel %.4f HM %.4f ===' % (tau, b0, n0, h0))
    for k in ('all', 'top2', 'top5'):
        b, n, h = metrics(cm0 + add[k])
        print('oracle_%-5s Base %.4f Novel %.4f HM %.4f  (ΔN %+.2fpp, ΔH %+.2fpp)'
              % (k, b, n, h, 100 * (n - n0), 100 * (h - h0)))


if __name__ == '__main__':
    main()
