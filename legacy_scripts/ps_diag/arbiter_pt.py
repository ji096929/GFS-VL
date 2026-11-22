"""逐点仲裁：在冲突点上，用「逐点」候选分数选类，而不是分量均值。

对每个点 p（属于 novel–novel 冲突分量）：
  候选 = 该点 top-k 的类
  score_c(p) = alpha * prob_c(p) + (1-alpha) * sonata_cos[p, c]
  out[p] = argmax score_c(p)（仅当该点原本判错方向时才有变化）
用法: python arbiter_pt.py <tau> <topk> <alphas>
"""
import glob
import os
import sys

import numpy as np

NO, NC = 12, 57
VAL = '/data01/jh/data/val_logits'
COORD = '/data01/jh/data/scannet200_gfs/val'
COS = '/data01/jh/data/val_sonata_cos'


def metrics(cm):
    d = np.diag(cm).astype(np.float64)
    r = cm.sum(1).astype(np.float64)
    c = cm.sum(0).astype(np.float64)
    iou = d / np.maximum(r + c - d, 1)
    b = float(iou[:NO].mean())
    n = float(iou[NO:].mean())
    return b, n, 2 * b * n / max(b + n, 1e-9)


def main():
    tau = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    alphas = [float(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3
                                 else ['0.0', '0.3', '0.5', '0.8', '1.0'])]
    k = min(k, 5)
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
        margin = p[:, 0] - p[:, 1]
        c1, c2 = i[:, 0], i[:, 1]
        conf = (margin < tau) & (c1 >= NO) & (c2 >= NO) & (c1 != c2)
        ok = gt >= 0
        cm0 += np.bincount(gt[ok] * NC + pred[ok], minlength=NC * NC).reshape(NC, NC)
        if not conf.any():
            for a in alphas:
                cms[a] += 0
            continue
        idx = np.where(conf)[0]
        cand = i[idx, :k]                      # (n, k) 每点自己的候选
        prob = p[idx, :k]
        cosv = np.take_along_axis(cosp[idx], np.clip(cand - NO, 0, 44), axis=1)
        for a in alphas:
            s = a * prob + (1 - a) * cosv
            pick = cand[np.arange(len(idx)), s.argmax(1)]
            out = pred.copy()
            out[idx] = pick
            cms[a] += np.bincount(gt[ok] * NC + out[ok],
                                  minlength=NC * NC).reshape(NC, NC)
            n_chg[a] += int((pick != pred[idx]).sum())
        if (fi + 1) % 100 == 0:
            print('  %d' % (fi + 1), flush=True)

    b0, n0, h0 = metrics(cm0)
    print('\n=== tau=%.2f topk=%d 基线 N %.4f H %.4f ===' % (tau, k, n0, h0))
    for a in alphas:
        b, n, h = metrics(cms[a])
        print('alpha=%.1f 改点 %7d  N %.4f H %.4f (ΔN %+.2fpp, ΔH %+.2fpp)'
              % (a, n_chg[a], n, h, 100 * (n - n0), 100 * (h - h0)))


if __name__ == '__main__':
    main()
