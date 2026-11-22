'''合并任务 1 分片：eRank、base_cross_head 范数、频次校准扫描。'''
import glob
import sys

import numpy as np

sys.path.insert(0, '/data01/jh/code/GFS-VL')
from pointcept.datasets.preprocessing.scannet.meta_data import (  # noqa: E402
    scannet200_constants as const)

D = '/data01/jh/afford_diag/xl_runs'
TAUS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]
NC = 57
BASE = list(const.CLASS_LABELS_BASE)


def metrics(cm):
    d = np.diag(cm).astype(float)
    r = cm.sum(1).astype(float)
    c = cm.sum(0).astype(float)
    iou = d / np.maximum(r + c - d, 1)
    b = iou[:12].mean()
    n = iou[12:].mean()
    return b, n, 2 * b * n / max(b + n, 1e-9)


def erank(X):
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    w = np.linalg.eigvalsh(C.astype(np.float64))
    w = np.maximum(w, 0.0)
    p = w / w.sum()
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


for seed in ('42', '7'):
    fs = sorted(glob.glob(D + '/task1_sym_s%s_shard*.npz' % seed))
    if not fs:
        print('seed %s: no shards' % seed)
        continue
    cms = {t: np.zeros((NC, NC), np.int64) for t in TAUS}
    feats = []
    norms = None
    for f in fs:
        z = np.load(f)
        for t in TAUS:
            cms[t] += z['cm_%s' % t]
        feats.append(z['feat'])
        norms = z['base_norms']
    X = np.concatenate(feats, 0)
    print('=== sym_xl_s%s  (feat %s, cat192 eRank %.2f) ==='
          % (seed, X.shape, erank(X)))
    for t in TAUS:
        b, n, h = metrics(cms[t])
        print('  tau %+.1f  Base %.4f  Novel %.4f  HM %.4f' % (t, b, n, h))
    order = np.argsort(-norms)
    print('  base_cross_head 范数 top6: %s'
          % [(BASE[i], round(float(norms[i]), 3)) for i in order[:6]])
    print('  bookshelf=%.3f cabinet=%.3f'
          % (norms[BASE.index('bookshelf')], norms[BASE.index('cabinet')]))
