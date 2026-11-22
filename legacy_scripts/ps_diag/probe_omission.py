# -*- coding: utf-8 -*-
"""Omission Recovery probe.

Question: for points that RegionPLC's PS missed (ps == -1) but whose true GT
is a novel class, can PTv3 features retrieve them?

  B   = {ps == c}                          anchor points already labelled
  C   = {gt == c} & {ps == -1}             missed positives
  Neg = {ps == -1} & {gt != c}             negatives

Prototype = mean feature of B. Similarity = cosine. Metric = AUC(C vs Neg).

Layers compared: dec0 (collapsed), dec1 / dec2 (high-rank), and a Base-reject
variant that subtracts the max similarity to the 12 base prototypes.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/GFS-VL')
sys.path.insert(0, '/data01/jh/code/GFS-VL/pointcept/models/PLA')
sys.path.insert(0, '/data01/jh/code/ps_diag')

from cut2_common import build_model_and_cfg, make_ds, Probe, LAYERS, NC, NO  # noqa: E402
from pointcept.datasets import collate_fn  # noqa: E402

N_SCENES = 50
DATA = '/data01/jh/data/scannet200_gfs/train'
PLDIR = '/data01/jh/data/online_pl_5shot_dump'
DIM = {'dec0': 64, 'dec1': 64, 'dec2': 128, 'dec3': 256, 'enc4': 512}


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
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def main():
    model, cfg = build_model_and_cfg('S1')
    pb = Probe()
    pb.attach(model.backbone)
    ds = make_ds(cfg, 'train')
    n_total = min(N_SCENES, len(ds.data_list)) if hasattr(ds, 'data_list') else N_SCENES
    print('scenes to probe:', n_total, flush=True)

    # results[layer] = list of (pos_sims, neg_sims, pos_sims_baserej, neg_sims_baserej)
    store = {L: [] for L in ['dec0', 'dec1', 'dec2']}
    # multi-class competition: per scene, store (pred_by_thr, gt_bg) is too big;
    # instead aggregate confusion counts per threshold.
    THRS = (0.5, 0.7, 0.8, 0.9, 0.95)
    mc = {L: {t: {'tp': 0, 'pred': 0, 'gt': 0} for t in THRS}
          for L in ['dec0', 'dec1', 'dec2']}
    for si in range(n_total):
        name = ds.get_data_name(si)
        pl_path = os.path.join(PLDIR, name + '.npz')
        if not os.path.exists(pl_path):
            continue
        z = np.load(pl_path)
        ps = z['ps']
        d = ds[si]
        frags = d.pop('fragment_list')
        seg = np.asarray(d.pop('segment')).astype(np.int64)
        if len(ps) != len(seg):
            continue

        # original-order feature buffers
        feat = {L: np.zeros((len(seg), DIM[L]), np.float32) for L in store}
        seen = np.zeros(len(seg), dtype=bool)
        for frag in frags:
            inp = collate_fn([frag])
            for k in inp:
                if isinstance(inp[k], torch.Tensor):
                    inp[k] = inp[k].cuda(non_blocking=True)
            ip = inp['index'].detach().cpu().numpy()
            with torch.no_grad():
                model(inp)
            fs = pb.layers()
            for L in store:
                feat[L][ip] = fs[L].float().cpu().numpy()
            seen[ip] = True

        bg = ps == -1
        base_mask = (seg >= 0) & (seg < NO)
        # base prototypes from GT (12 classes) -- cheap, no leak into novel C
        base_protos = {}
        for b in range(NO):
            m = base_mask & (seg == b)
            if m.sum() >= 50:
                base_protos[b] = unit(feat['dec1'][m].mean(0))  # placeholder dim
        for L in store:
            bp = {}
            for b in range(NO):
                m = base_mask & (seg == b)
                if m.sum() >= 50:
                    bp[b] = unit(feat[L][m].mean(0))

            pos_all, neg_all, pos_rej, neg_rej = [], [], [], []
            for c in range(NO, NC):
                B = ps == c
                if B.sum() < 20:
                    continue
                C = bg & (seg == c)
                N = bg & (seg != c) & (seg >= 0)
                if C.sum() == 0 or N.sum() < 20:
                    continue
                proto = unit(feat[L][B].mean(0))
                sC = feat[L][C] @ proto
                sN = feat[L][N] @ proto
                pos_all.append(sC)
                neg_all.append(sN)
                if bp:
                    base_max_C = np.max(np.stack([feat[L][C] @ p for p in bp.values()]), 0)
                    base_max_N = np.max(np.stack([feat[L][N] @ p for p in bp.values()]), 0)
                    pos_rej.append(sC - base_max_C)
                    neg_rej.append(sN - base_max_N)

            if pos_all:
                store[L].append((
                    np.concatenate(pos_all), np.concatenate(neg_all),
                    np.concatenate(pos_rej) if pos_rej else np.zeros(0),
                    np.concatenate(neg_rej) if neg_rej else np.zeros(0),
                ))

        # ---- multi-class competition (faithful to GFS-VL AI) ----
        for L in store:
            protos, pcls = [], []
            for c in range(NO, NC):
                B = ps == c
                if B.sum() < 20:
                    continue
                protos.append(unit(feat[L][B].mean(0)))
                pcls.append(c)
            if not protos:
                continue
            proto_m = np.stack(protos)                       # (K, D)
            pred_cls = np.full(len(seg), -1, dtype=np.int64)
            pred_sim = np.zeros(len(seg), dtype=np.float32)
            sims = feat[L][bg] @ proto_m.T                   # (M, K)
            mx = sims.max(1)
            am = sims.argmax(1)
            pred_cls[bg] = np.asarray(pcls)[am]
            pred_sim[bg] = mx
            gt_bg = seg[bg]
            pred_bg = pred_cls[bg]
            sim_bg = pred_sim[bg]
            for t in THRS:
                sel = (sim_bg >= t) & (pred_bg >= 0)
                mc[L][t]['tp'] += int((sel & (pred_bg == gt_bg)).sum())
                mc[L][t]['pred'] += int(sel.sum())
                mc[L][t]['gt'] += int((gt_bg >= NO).sum())
        if (si + 1) % 10 == 0:
            print('  %d/%d scenes' % (si + 1, n_total), flush=True)

    print()
    print('%-8s %10s %10s %10s' % ('layer', 'AUC', 'AUC+baserej', 'nC_total'))
    print('-' * 44)
    for L in ['dec0', 'dec1', 'dec2']:
        if not store[L]:
            continue
        P = np.concatenate([x[0] for x in store[L]])
        N = np.concatenate([x[1] for x in store[L]])
        PR = np.concatenate([x[2] for x in store[L]]) if store[L][0][2].size else np.zeros(0)
        NR = np.concatenate([x[3] for x in store[L]]) if store[L][0][3].size else np.zeros(0)
        print('%-8s %10.4f %10.4f %10d' % (
            L, auc(P, N), auc(PR, NR) if PR.size else float('nan'), len(P)))

    # threshold sweep: P/R/F1 on the to-be-filled region
    print()
    print('%-8s %7s %10s %10s %8s' % ('layer', 'thr', 'Precision', 'Recall', 'F1'))
    print('-' * 48)
    for L in ['dec0', 'dec1', 'dec2']:
        if not store[L]:
            continue
        P = [x[0] for x in store[L]]
        N = [x[1] for x in store[L]]
        for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
            tp = float(sum((p >= t).sum() for p in P))
            fp = float(sum((n >= t).sum() for n in N))
            fn = float(sum((p < t).sum() for p in P))
            prec = tp / max(tp + fp, 1.0)
            rec = tp / max(tp + fn, 1.0)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            print('%-8s %7.2f %10.4f %10.4f %8.4f' % (L, t, prec, rec, f1))
        print()

    # same sweep but with base rejection
    # multi-class competition (faithful to GFS-VL AI): argmax over 45 protos
    print()
    print('%-8s %7s %10s %10s %8s %10s' % ('layer', 'thr', 'Precision', 'Recall', 'F1', 'n_pred'))
    print('-' * 60)
    for L in ['dec0', 'dec1', 'dec2']:
        P = [x[0] for x in store[L]]
        N = [x[1] for x in store[L]]
        # 单类二分类的 sC/sN 无法做多类 argmax，这里用近似：
        # 对每个阈值，把 C 视为"该类 argmax"、N 视为"非该类 argmax"的代理
        for t in (0.5, 0.7, 0.8, 0.9, 0.95):
            tp = float(sum((p >= t).sum() for p in P))
            fp = float(sum((n >= t).sum() for n in N))
            fn = float(sum((p < t).sum() for p in P))
            prec = tp / max(tp + fp, 1.0)
            rec = tp / max(tp + fn, 1.0)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            print('%-8s %7.2f %10.4f %10.4f %8.4f %10.0f' % (L, t, prec, rec, f1, tp + fp))
        print()

    # ---- report multi-class competition ----
    print()
    print('=== multi-class argmax (faithful to GFS-VL AI) ===')
    print('%-8s %7s %10s %10s %8s %12s' % ('layer', 'thr', 'Precision', 'Recall', 'F1', 'n_pred'))
    print('-' * 62)
    for L in ['dec0', 'dec1', 'dec2']:
        for t in THRS:
            r = mc[L][t]
            prec = r['tp'] / max(r['pred'], 1)
            rec = r['tp'] / max(r['gt'], 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            print('%-8s %7.2f %10.4f %10.4f %8.4f %12d' % (L, t, prec, rec, f1, r['pred']))
        print()

    print()
    print('%-8s %7s %10s %10s %8s' % ('layer+rej', 'thr', 'Precision', 'Recall', 'F1'))
    print('-' * 48)
    for L in ['dec0', 'dec1', 'dec2']:
        if not store[L] or store[L][0][2].size == 0:
            continue
        P = [x[2] for x in store[L]]
        N = [x[3] for x in store[L]]
        for t in (0.0, 0.05, 0.1, 0.15, 0.2):
            tp = float(sum((p >= t).sum() for p in P))
            fp = float(sum((n >= t).sum() for n in N))
            fn = float(sum((p < t).sum() for p in P))
            prec = tp / max(tp + fp, 1.0)
            rec = tp / max(tp + fn, 1.0)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            print('%-8s %7.2f %10.4f %10.4f %8.4f' % (L + '+rej', t, prec, rec, f1))
        print()


if __name__ == '__main__':
    main()
