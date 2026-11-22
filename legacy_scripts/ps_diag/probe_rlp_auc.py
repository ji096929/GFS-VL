# -*- coding: utf-8 -*-
"""RegionPLC baseline AUC for the omission-recovery probe.

Same B/C/Neg split as probe_omission.py, but similarity computed in
RegionPLC's own 512-d feature space (the original AI's representation).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/GFS-VL')
sys.path.insert(0, '/data01/jh/code/GFS-VL/pointcept/models/PLA')
sys.path.insert(0, '/data01/jh/code/ps_diag')

import vlm_common as V  # noqa: E402

N_SCENES = 50
DATA = '/data01/jh/data/scannet200_gfs/train'
PLDIR = '/data01/jh/data/online_pl_5shot_dump'
NO, NN = 12, 45


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    cfg, model, _, _, _ = V.build_vlm()
    scenes = sorted(os.listdir(DATA))[:N_SCENES]
    print('scenes:', len(scenes), flush=True)

    pos_s, neg_s = [], []
    pos_r, neg_r = [], []          # with base rejection
    for i, sc in enumerate(scenes):
        d = os.path.join(DATA, sc)
        pl = os.path.join(PLDIR, sc + '.npz')
        if not os.path.exists(pl):
            continue
        coord = np.load(os.path.join(d, 'coord.npy')).astype(np.float32)
        color = np.load(os.path.join(d, 'color.npy'))
        seg200 = np.load(os.path.join(d, 'segment200.npy')).reshape(-1)
        gfs = V.convert_to_gfs(seg200)
        ps = np.load(pl)['ps']
        if len(ps) != len(gfs):
            continue
        feats, _ = V.vlm_forward(model, cfg, coord, color)
        fn = unit(feats)
        bg = ps == -1

        # base prototypes (12) from GT
        bp = []
        for b in range(NO):
            m = (gfs == b) & (ps >= 0)          # use labelled base points
            if m.sum() >= 50:
                bp.append(unit(fn[m].mean(0)))
        bp = np.stack(bp) if bp else None

        for c in range(NO, NO + NN):
            B = ps == c
            if B.sum() < 20:
                continue
            C = bg & (gfs == c)
            Ng = bg & (gfs != c) & (gfs >= 0)
            if C.sum() == 0 or Ng.sum() < 20:
                continue
            proto = unit(fn[B].mean(0))
            sC = fn[C] @ proto
            sN = fn[Ng] @ proto
            pos_s.append(sC)
            neg_s.append(sN)
            if bp is not None:
                pos_r.append(sC - (fn[C] @ bp.T).max(1))
                neg_r.append(sN - (fn[Ng] @ bp.T).max(1))
        if (i + 1) % 10 == 0:
            print('  %d/%d' % (i + 1, len(scenes)), flush=True)

    P = np.concatenate(pos_s)
    N = np.concatenate(neg_s)
    print()
    print('RegionPLC feature space:')
    print('  AUC            %.4f  (nC=%d)' % (auc(P, N), len(P)))
    if pos_r:
        PR = np.concatenate(pos_r)
        NR = np.concatenate(neg_r)
        print('  AUC + BaseRej  %.4f' % auc(PR, NR))


if __name__ == '__main__':
    main()
