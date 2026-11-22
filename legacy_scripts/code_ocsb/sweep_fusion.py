"""扫描式(15)修正后的 fusion_mode 与 gamma。

每个 (step, mode) 只前向一次拿 delta_mem，然后在同一增量上扫 gamma。
delta_mem 已乘回 tau，与 margin 同量纲；预期量级 0.05-0.20。
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
GAMMAS = [0.5, 1.0, 2.0, 5.0, 10.0]


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

    base = float(np.mean([iou_of(ep['q_margin'].cpu().numpy().astype(
        np.float64), ep) for ep in eps]))
    print('Step1 baseline %.4f' % base)

    rows = []
    for mode in ('A', 'B'):
        for step in ('step2', 'step3', 'step4'):
            m = OCSB(dim, tau=0.1, tau_b=0.6, fusion_mode=mode,
                     gamma_fuse=1.0).cuda()
            deltas = []
            with torch.no_grad():
                for ep in eps:
                    out = m(ep, enable_csr=(step == 'step4'))
                    if step == 'step2':
                        B = out['B_idx']
                        zq = m.project(ep['q_h'])
                        s0_q = (zq @ out['p_fg']) / m.tau
                        d = m.memory_delta(s0_q, out['z_R'][B],
                                           out['q_reg'][B], zq)
                    else:
                        d = out['delta_mem']
                    if mode == 'B':
                        d = torch.relu(d)
                    deltas.append(d.detach())
            # 增量分布与正比例
            all_d = torch.cat([d.reshape(-1) for d in deltas]).cpu().numpy()
            pos = float((all_d > 0).mean())
            p10, p50, p90, p99 = np.percentile(all_d, [10, 50, 90, 99])
            print('\n%s %s  delta p10 %+.3f p50 %+.3f p90 %+.3f p99 %+.3f  pos %.2f'
                  % (mode, step, p10, p50, p90, p99, pos), flush=True)
            for g in GAMMAS:
                ious = []
                for ep, d in zip(eps, deltas):
                    ell = ep['q_margin'] + g * d
                    ious.append(iou_of(ell.cpu().numpy().astype(np.float64), ep))
                v = float(np.mean(ious))
                rows.append({'mode': mode, 'step': step, 'gamma': g,
                             'iou': v, 'base': base, 'delta': v - base})
                print('  g=%-4.1f IoU %.4f  d %+.4f' % (g, v, v - base),
                      flush=True)

    json.dump(rows, open(os.path.join(CACHE, 'sweep_fusion.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
