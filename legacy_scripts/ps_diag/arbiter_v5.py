"""仲裁器 v4：跨类比较前先做【按类 z-score 校准】。

之前的 argmax 直接比原始 cos，被类级偏置主导（某些类原型天生余弦高）。
这里先用全部 val 点统计每类的 mean/std，把 cos 变成类内 z，再跨类比较。

用法: python arbiter_v4.py <tau> <topk> <alphas> <granularity: pt|comp>
"""
import glob
import os
import sys

import numpy as np

NO, NC = 12, 57
VAL = '/data01/jh/data/val_logits'
COORD = '/data01/jh/data/scannet200_gfs/val'
COS = '/data01/jh/data/val_sonata_cos'
STATS = '/data01/jh/data/val_cos_stats.npy'
RADIUS = 0.05


def cos_stats():
    if os.environ.get('GTSTATS') == '1':
        p = '/data01/jh/data/val_cos_stats_gt.npy'
        if os.path.exists(p):
            d = np.load(p); return d[0].astype(np.float32), d[1].astype(np.float32)
        vals = [[] for _ in range(45)]
        for f in sorted(glob.glob(VAL + '/*.npz')):
            sc = os.path.basename(f)[:-4]
            z = np.load(f); gt = z['gt'].astype(np.int64)
            c = np.load(os.path.join(COS, sc + '_cos.npy')).astype(np.float32)
            for k in range(45):
                m = gt == (NO + k)
                if m.any():
                    vals[k].append(c[m, k])
        m = np.array([np.concatenate(v).mean() if v else 0.0 for v in vals], np.float32)
        s = np.array([np.concatenate(v).std() if v else 1.0 for v in vals], np.float32)
        s = np.maximum(s, 1e-6)
        np.save(p, np.stack([m, s]))
        return m, s
    if os.path.exists(STATS):
        d = np.load(STATS)
        return d[0].astype(np.float32), d[1].astype(np.float32)
    n = np.zeros(45)
    s = np.zeros(45)
    ss = np.zeros(45)
    for f in sorted(glob.glob(COS + '/*_cos.npy')):
        c = np.load(f).astype(np.float32)
        n += 1
        s += c.mean(0)
        ss += (c ** 2).mean(0)
    m = s / n
    v = np.maximum(ss / n - m ** 2, 1e-6)
    out = np.stack([m, np.sqrt(v)])
    np.save(STATS, out)
    return m, np.sqrt(v)


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
    k = min(int(sys.argv[2]), 5)
    alphas = [float(x) for x in sys.argv[3].split(',')]
    gran = sys.argv[4] if len(sys.argv) > 4 else 'comp'
    m_, s_ = cos_stats()
    cms = {a: np.zeros((NC, NC), np.int64) for a in alphas}
    cm0 = np.zeros((NC, NC), np.int64)
    n_chg = {a: 0 for a in alphas}

    for fi, f in enumerate(sorted(glob.glob(VAL + '/*.npz'))):
        sc = os.path.basename(f)[:-4]
        z = np.load(f)
        p = z['top5p'].astype(np.float32)
        i = z['top5i'].astype(np.int64)
        pred = z['pred'].astype(np.int64)
        gt = z['gt'].astype(np.int64)
        cosp = np.load(os.path.join(COS, sc + '_cos.npy')).astype(np.float32)
        zc = (cosp - m_) / s_          # 类内 z
        margin = p[:, 0] - p[:, 1]
        c1, c2 = i[:, 0], i[:, 1]
        conf = (margin < tau) & (c1 >= NO) & (c2 >= NO) & (c1 != c2)
        ok = gt >= 0
        cm0 += np.bincount(gt[ok] * NC + pred[ok], minlength=NC * NC).reshape(NC, NC)

        if gran == 'pt':
            idx = np.where(conf)[0]
            if len(idx) == 0:
                continue
            cand = i[idx, :k]
            prob = p[idx, :k]
            cosv = np.take_along_axis(zc[idx], np.clip(cand - NO, 0, 44), axis=1)
            for a in alphas:
                s = a * prob + (1 - a) * cosv
                pick = cand[np.arange(len(idx)), s.argmax(1)]
                out = pred.copy(); out[idx] = pick
                cms[a] += np.bincount(gt[ok] * NC + out[ok],
                                      minlength=NC * NC).reshape(NC, NC)
                n_chg[a] += int((pick != pred[idx]).sum())
        else:
            coord = np.load(os.path.join(COORD, sc, 'coord.npy')).astype(np.float32)
            for comp in comps(conf, coord):
                if len(comp) < 20:
                    continue
                cand = np.unique(i[comp][:, :k].reshape(-1))
                cand = cand[cand >= NO]
                if len(cand) < 2:
                    continue
                lg = np.stack([np.where(i[comp] == c, p[comp], 0.0).sum(1)
                               for c in cand]).mean(1)
                cs = np.array([zc[comp, c - NO].mean() for c in cand])
                okc = gt[comp] >= 0
                g = gt[comp][okc]; pc = pred[comp][okc]
                bc = np.bincount(g * NC + pc, minlength=NC * NC).reshape(NC, NC)
                for a in alphas:
                    sc_ = a * lg + (1 - a) * cs
                    best = int(cand[sc_.argmax()])
                    delta = np.bincount(g * NC + best,
                                        minlength=NC * NC).reshape(NC, NC) - bc
                    cms[a] += delta if best != int(np.bincount(pc, minlength=NC).argmax()) else 0
                    n_chg[a] += 1 if best != int(np.bincount(pc, minlength=NC).argmax()) else 0
        if (fi + 1) % 100 == 0:
            print('  %d' % (fi + 1), flush=True)

    b0, n0, h0 = metrics(cm0)
    print('\n=== tau=%.2f topk=%d gran=%s 基线 N %.4f H %.4f ===' % (tau, k, gran, n0, h0))
    for a in alphas:
        b, n, h = metrics(cm0 + cms[a])
        print('alpha=%.1f 改 %7d  N %.4f H %.4f (ΔN %+.2fpp, ΔH %+.2fpp)'
              % (a, n_chg[a], n, h, 100 * (n - n0), 100 * (h - h0)))


if __name__ == '__main__':
    main()
