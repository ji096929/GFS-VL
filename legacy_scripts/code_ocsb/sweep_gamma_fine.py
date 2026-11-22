"""Step 4 的 gamma 细化扫描（1.0-4.0），找最优点。"""
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ocsb_faithful import OCSB  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
GAMMAS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0]


def to_t(a):
    if a.dtype == np.float16:
        return torch.from_numpy(a.astype(np.float32)).cuda()
    if a.dtype.kind in 'iu':
        return torch.from_numpy(a.astype(np.int64)).cuda()
    return torch.from_numpy(a.astype(np.float32)).cuda()


def interp(cf, cd, val, k=3):
    d, idx = cKDTree(cd).query(cf, k=k)
    w = 1.0 / (d + 1e-8)
    w /= w.sum(1, keepdims=True)
    return (val[idx] * w).sum(1)


def iou_of(ell_np, ep):
    e = interp(ep['q_coord_full'].cpu().numpy().astype(np.float64),
               ep['q_coord'].cpu().numpy().astype(np.float64), ell_np)
    y = ep['q_y_full'].cpu().numpy()
    p = e > 0
    tp = int((p & (y > 0)).sum())
    fp = int((p & (y == 0)).sum())
    fn = int((~p & (y > 0)).sum())
    return tp / max(tp + fp + fn, 1)


def main():
    files = sorted(glob.glob(os.path.join(CACHE, 'val_*.npz')))
    eps = []
    for f in files:
        d = np.load(f)
        eps.append({k: to_t(d[k]) for k in d.files})
    dim = int(eps[0]['s_h'].shape[1])
    base = float(np.mean([iou_of(ep['q_margin'].cpu().numpy().astype(
        np.float64), ep) for ep in eps]))
    print('baseline %.4f' % base)

    m = OCSB(dim, tau=0.1, tau_b=0.6, fusion_mode='A', gamma_fuse=1.0).cuda()
    deltas = []
    with torch.no_grad():
        for ep in eps:
            out = m(ep, enable_csr=True)
            deltas.append(out['delta_mem'].detach())

    best = None
    rows = []
    for g in GAMMAS:
        ious = [iou_of((ep['q_margin'] + g * d).cpu().numpy().astype(np.float64), ep)
                for ep, d in zip(eps, deltas)]
        v = float(np.mean(ious))
        se = float(np.std(np.array(ious) - np.array(
            [iou_of(ep['q_margin'].cpu().numpy().astype(np.float64), ep)
             for ep in eps]), ddof=1) / np.sqrt(len(ious)))
        rows.append({'gamma': g, 'iou': v, 'delta': v - base, 'pse': se})
        print('gamma %-5.2f IoU %.4f  d %+.4f  pSE %.4f' % (g, v, v - base, se),
              flush=True)
        if best is None or v > best[1]:
            best = (g, v)
    print('best gamma %.2f -> %.4f' % best)
    json.dump(rows, open(os.path.join(CACHE, 'sweep_gamma_fine.json'), 'w'),
              indent=1)


if __name__ == '__main__':
    main()
