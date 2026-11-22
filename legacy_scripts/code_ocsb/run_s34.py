"""S3 / S4（减背景得分）的 Logit 注入评测。

S3：Σ ŵ_R relu(z_R·z_q - c_bg(i))
S4：模块一/二基准也换成前景-背景裕度后，再算 S3
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ocsb_cmc import OCSBCMC  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
CKPT = '/data01/jh/weights/sc0_1w1s.pth'
GAMMAS = [0.5, 1.0, 2.0, 5.0]


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


def tp_union(ell, ep):
    e = interp(ep['q_coord_full'].cpu().numpy().astype(np.float64),
               ep['q_coord'].cpu().numpy().astype(np.float64),
               ell.detach().cpu().numpy().astype(np.float64))
    y = ep['q_y_full'].cpu().numpy()
    p = e > 0
    return int((p & (y > 0)).sum()), int((p | (y > 0)).sum())


def main():
    table = torch.load(CKPT, map_location='cpu')['state_dict'][
        'module.base_prototypes'].float().cuda()
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))[:100]
    eps = []
    cmc = None
    for m in metas:
        d = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
        ep = {k: to_t(d[k]) for k in d.files}
        ep['base_protos'] = table
        ep['cls'] = m['class']
        ep['y_full'] = d['q_y_full'].astype(np.int64)
        if cmc is None:
            cmc = OCSBCMC(int(ep['s_h'].shape[1]), tau=0.1, tau_b=0.6,
                          m_anchor=3).cuda()
        eps.append(ep)

    base_ious = [tp_union(ep['q_margin'], ep)[0] /
                 max(tp_union(ep['q_margin'], ep)[1], 1) for ep in eps]
    print('baseline arith %.4f' % np.mean(base_ious))

    for tag, bg in (('S3', False), ('S4', True)):
        S = []
        for ep in eps:
            with torch.no_grad():
                out = cmc(ep, use_csr=True, bg_baseline=bg)
            z_q, M, w = out['z_q'], out['M_feat'], out['M_w']
            if M.shape[0] == 0:
                S.append(torch.zeros(z_q.shape[0], device=z_q.device))
                continue
            wh = w / (w.sum() + 1e-6)
            sim = z_q @ M.t()
            c_bg = (z_q @ out['a_bg'].t()).max(1).values
            if ep['base_protos'] is not None and ep['base_protos'].numel():
                bp = F.normalize(ep['base_protos'].to(z_q.dtype), dim=-1)
                c_bg = torch.maximum(c_bg, (z_q @ bp.t()).max(1).values)
            s = (wh.unsqueeze(0) * F.relu(sim - c_bg.unsqueeze(1))).sum(1)
            S.append((s - s.mean()).detach())
        alls = torch.cat(S)
        print('\n%s centered: mean %.4f std %.4f frac>0 %.2f'
              % (tag, float(alls.mean()), float(alls.std()),
                 float((alls > 0).float().mean())))
        for g in GAMMAS:
            tp_c, un_c, ious = {}, {}, []
            for ep, s in zip(eps, S):
                tp, un = tp_union(ep['q_margin'] + g * s, ep)
                tp_c[ep['cls']] = tp_c.get(ep['cls'], 0) + tp
                un_c[ep['cls']] = un_c.get(ep['cls'], 0) + un
                ious.append(tp / max(un, 1))
            d = np.array(ious) - np.array(base_ious)
            pse = d.std(ddof=1) / np.sqrt(len(d))
            cum = np.mean([tp_c[c] / max(un_c[c], 1) for c in tp_c])
            print('  g=%-4.1f arith %.4f (%+.4f) pSE %.4f t %.2f cum %.4f'
                  % (g, np.mean(ious), d.mean(), pse,
                     d.mean() / pse if pse else 0, cum))


if __name__ == '__main__':
    main()
