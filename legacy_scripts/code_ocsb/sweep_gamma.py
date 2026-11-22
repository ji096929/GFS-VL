"""扫 gamma：融合强度从 0 到 0.2，找不伤 IoU 的临界点。

gamma -> 0 时式(15) 严格退回 COSeg。这条曲线能证明掉点来自融合项本身，
而不是 CSR 选错了桥。
"""
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
    print('episodes:', len(eps))

    rows = []
    for tau in (0.1, 0.5):
        for g in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
            m = OCSB(dim, tau=tau, tau_b=0.6, gamma0=max(g, 1e-6)).cuda()
            ious = []
            with torch.no_grad():
                for ep in eps:
                    out = m(ep)
                    ious.append(iou_of(out['ell'].detach().cpu().numpy()
                                       .astype(np.float64), ep))
            v = float(np.mean(ious))
            rows.append({'tau': tau, 'gamma': g, 'iou': v})
            print('tau %.2f  gamma %.3f  IoU %.4f' % (tau, g, v), flush=True)

    json.dump(rows, open(os.path.join(CACHE, 'sweep_gamma.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
