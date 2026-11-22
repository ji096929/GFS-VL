"""S2（纯加权相似度）的端到端评测。

S2 有正确的四分组排序（FN > TP > FP > TN），但整体为正常数，必须先中心化
才能避免全局阈值平移。这里测：
  centered  : ell = q_margin + gamma * (S2 - mean(S2))
  raw       : ell = q_margin + gamma * S2            （含全局偏移）
  shuffle   : 打乱 S2 中心化后的顺序
  meanconst : 用 S2 的任务均值（中心化后恒为 0，等价 baseline）
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
GAMMAS = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


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


def tp_union(ell_down, ep):
    e = interp(ep['q_coord_full'].cpu().numpy().astype(np.float64),
               ep['q_coord'].cpu().numpy().astype(np.float64),
               ell_down.detach().cpu().numpy().astype(np.float64))
    y = ep['q_y_full'].cpu().numpy()
    pred = e > 0
    return int((pred & (y > 0)).sum()), int((pred | (y > 0)).sum())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--out', default='s2_eval.json')
    args = ap.parse_args()
    table = torch.load(CKPT, map_location='cpu')['state_dict'][
        'module.base_prototypes'].float().cuda()
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))[:args.n]
    eps, S2s = [], []
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
        with torch.no_grad():
            out = cmc(ep, use_csr=True)
        z_q, M, w = out['z_q'], out['M_feat'], out['M_w']
        if M.shape[0] == 0:
            s2 = torch.zeros(z_q.shape[0], device=z_q.device)
        else:
            wh = w / (w.sum() + 1e-6)
            s2 = (wh.unsqueeze(0) * (z_q @ M.t())).sum(1)
        eps.append(ep)
        S2s.append(s2.detach())

    base_ious = []

    def evaluate(fn, tag, gamma=None, ref=None):
        tp_c, un_c, ious = {}, {}, []
        for ep, s2 in zip(eps, S2s):
            ell = fn(ep, s2)
            tp, un = tp_union(ell, ep)
            tp_c[ep['cls']] = tp_c.get(ep['cls'], 0) + tp
            un_c[ep['cls']] = un_c.get(ep['cls'], 0) + un
            ious.append(tp / max(un, 1))
        cum = float(np.mean([tp_c[c] / max(un_c[c], 1) for c in tp_c]))
        pse = float('nan')
        if ref is not None:
            d = np.array(ious) - np.array(ref)
            pse = float(d.std(ddof=1) / np.sqrt(len(d)))
        print('%-22s gamma %-5s arith %.4f  cum %.4f  pSE %.4f'
              % (tag, gamma, float(np.mean(ious)), cum, pse), flush=True)
        return {'arith': float(np.mean(ious)), 'cum': cum,
                'pse': pse, 'ious': ious}

    r = {}
    r['baseline'] = evaluate(lambda ep, s2: ep['q_margin'], 'baseline', '-')
    base_ious = r['baseline']['ious']
    rng = np.random.RandomState(0)
    for g in GAMMAS:
        r['centered_%g' % g] = evaluate(
            lambda ep, s2: ep['q_margin'] + g * (s2 - s2.mean()),
            'centered', g, base_ious)
        r['raw_%g' % g] = evaluate(
            lambda ep, s2: ep['q_margin'] + g * s2, 'raw', g, base_ious)
    # 打乱对照（固定 gamma=1）
    def shuf(ep, s2):
        n = s2.cpu().numpy()
        return ep['q_margin'] + torch.from_numpy(
            n[rng.permutation(len(n))]).cuda().to(s2.dtype)
    r['shuffle_1'] = evaluate(shuf, 'shuffle-centered', 1.0, base_ious)

    json.dump(r, open(os.path.join(CACHE, args.out), 'w'), indent=1)


if __name__ == '__main__':
    main()
