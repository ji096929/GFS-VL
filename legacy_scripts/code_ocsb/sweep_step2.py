"""Step 2 掉点诊断：扫描 tau 与 Top-K，确认式(15)的量纲问题。

对每个 tau 只前向一次，然后在同一 out 上换 Top-K 与融合方式，比较
Query IoU。同时测一个「margin 校准」变体：把 sim 的分布对齐到
q_margin 的标准差，验证掉点确实来自量纲不匹配。
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
from ocsb_faithful import OCSB, direct_expansion, fuse_query_score  # noqa

CACHE = '/data01/jh/data/ocsb_cache'
TAUS = [0.05, 0.1, 0.2, 0.3, 0.5]
KS = [5, 10, 20, 50, None]


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
    print('episodes:', len(files))
    eps = []
    for f in files:
        d = np.load(f)
        eps.append({k: to_t(d[k]) for k in d.files})
    dim = int(eps[0]['s_h'].shape[1])

    rows = []
    for tau in TAUS:
        m = OCSB(dim, tau=tau, tau_b=0.6, gamma0=0.2).cuda()
        outs = []
        with torch.no_grad():
            for ep in eps:
                outs.append(m(ep))
        # 基线：原始 margin
        base = float(np.mean([iou_of(ep['q_margin'].cpu().numpy().astype(
            np.float64), ep) for ep in eps]))
        print('\ntau=%.2f  Step1 %.4f' % (tau, base), flush=True)
        for k in KS:
            ious = []
            with torch.no_grad():
                for ep, out in zip(eps, outs):
                    ell = direct_expansion(out, ep, m, topk=k)
                    ious.append(iou_of(ell.cpu().numpy().astype(np.float64), ep))
            v = float(np.mean(ious))
            rows.append({'tau': tau, 'k': k if k else -1, 'iou': v,
                         'base': base, 'delta': v - base})
            print('  K=%-4s IoU %.4f  d %+.4f'
                  % (k if k else 'all', v, v - base), flush=True)

        # margin 校准：把 sim 的均值/方差对齐到 q_margin，再按 γ=1 融合
        cal = []
        with torch.no_grad():
            for ep, out in zip(eps, outs):
                B = out['B_idx']
                s_s = ep['q_margin']
                if B.numel() == 0:
                    cal.append(iou_of(s_s.cpu().numpy().astype(np.float64), ep))
                    continue
                zq = m.project(ep['q_h'])
                sim = (out['z_R'][B] @ zq.t()) / tau        # [M, Nq]
                sim_c = (sim - sim.mean()) / (sim.std() + 1e-6) \
                    * s_s.std() + s_s.mean()
                w = out['q_reg'][B]
                agg = torch.logsumexp(torch.log(w)[None, :] + sim_c.t(), dim=1)
                ell = torch.logaddexp(s_s, agg) - torch.log1p(w.sum())
                cal.append(iou_of(ell.cpu().numpy().astype(np.float64), ep))
        v = float(np.mean(cal))
        rows.append({'tau': tau, 'k': -2, 'iou': v, 'base': base,
                     'delta': v - base})
        print('  calibrated  IoU %.4f  d %+.4f' % (v, v - base), flush=True)

    json.dump(rows, open(os.path.join(CACHE, 'sweep_step2.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
