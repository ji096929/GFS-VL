"""or_gate v2：B_max 与 oracle 在 val_560 上的 Arith + Cumul。"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ocsb_faithful import sp_mean_norm, sp_soft_labels  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from phase2_v2 import adjacent, load_eps  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
CKPT = '/data01/jh/weights/sc0_1w1s.pth'
TAUS = (0.5, 1.0)
KINDS = ('B_max', 'oracle')
TOPS = (0.1, 0.2, 0.3, 0.5)
DMAX = 0.10


def iou_of(pred, y):
    tp = int((pred & (y > 0)).sum())
    un = int((pred | (y > 0)).sum())
    return tp / max(un, 1)


def cum_of(preds, eps):
    tp_c, un_c = {}, {}
    for pred, e in zip(preds, eps):
        c = e['cls']
        tp_c[c] = tp_c.get(c, 0) + int((pred & (e['y'] > 0)).sum())
        un_c[c] = un_c.get(c, 0) + int((pred | (e['y'] > 0)).sum())
    return float(np.mean([tp_c[c] / max(un_c[c], 1) for c in tp_c]))


def score(e, adj, B, kind):
    zc = e['z_sp'][adj]
    zb = e['z_sp'][B]
    if kind == 'B_max':
        qB = torch.sigmoid(torch.from_numpy(e['m'][B]).to(zc.device))
        return (qB.unsqueeze(0) * (zc @ zb.t())).max(1).values
    u_seg = torch.from_numpy(e['u_seg']).to(zc.device).long()
    n_u = int(u_seg.max().item()) + 1
    z_u = F.normalize(torch.from_numpy(e['u_h']).to(zc.device).float(), dim=-1)
    z_uR = sp_mean_norm(z_u, u_seg, n_u)
    yR = sp_soft_labels(
        torch.from_numpy(e['u_y']).to(zc.device).float(), u_seg, n_u, 2)
    fgR = yR[:, 1] > 0.5
    if int(fgR.sum()) == 0:
        return torch.zeros(len(adj), device=zc.device)
    return (zc @ z_uR[fgR].t()).max(1).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=560)
    ap.add_argument('--out', default=os.path.join(CACHE, 'or_gate_v2.json'))
    args = ap.parse_args()
    torch.manual_seed(0)
    table = torch.load(CKPT, map_location='cpu')['state_dict'][
        'module.base_prototypes'].float().cuda()
    eps = load_eps(args.n, table)
    N = len(eps)
    print('episodes:', N, flush=True)

    base_ious = np.array([iou_of(e['m'][e['inv']] > 0.5, e['y']) for e in eps])
    base_cum = cum_of([e['m'][e['inv']] > 0.5 for e in eps], eps)
    print('baseline arith %.4f cum %.4f' % (base_ious.mean(), base_cum),
          flush=True)

    out = []
    for tau in TAUS:
        cache = []
        for e in eps:
            B = np.where(e['m'] >= tau)[0]
            C = np.where((e['m'] < 0.5) & (e['m'] > -3.0))[0]
            cache.append((B, adjacent(e, B, C, DMAX)))
        for kind in KINDS:
            for top in TOPS:
                ious, preds, n_rec = [], [], 0
                for ei, e in enumerate(eps):
                    B, adj = cache[ei]
                    ell = e['m'][e['inv']].copy()
                    if len(adj) and len(B):
                        s = score(e, adj, B, kind)
                        kk = max(int(len(adj) * top), 1)
                        sel = adj[torch.topk(s, kk).indices.cpu().numpy()]
                        ell[np.isin(e['inv'], sel)] = 1.0
                        n_rec += len(sel)
                    pred = ell > 0.5
                    ious.append(iou_of(pred, e['y']))
                    preds.append(pred)
                cum = cum_of(preds, eps)
                d = np.array(ious) - base_ious
                pse = d.std(ddof=1) / np.sqrt(N) if N > 1 else float('nan')
                out.append({'tau': tau, 'kind': kind, 'top': top,
                            'arith': float(np.mean(ious)), 'cum': cum,
                            'delta': float(d.mean()), 'cum_delta': cum - base_cum,
                            'pse': float(pse),
                            't': float(d.mean() / pse) if pse else 0.0,
                            'n_rec': n_rec})
                print('%-6s tau %.1f top %2.0f%%  arith %.4f (d %+.4f)  '
                      'cum %.4f (d %+.4f)  t %+.2f  rec=%d'
                      % (kind, tau, 100 * top, np.mean(ious), d.mean(), cum,
                         cum - base_cum, d.mean() / pse if pse else 0, n_rec),
                      flush=True)
    json.dump({'baseline': {'arith': float(base_ious.mean()), 'cum': base_cum},
               'rows': out}, open(args.out, 'w'), indent=1)
    print('->', args.out)


if __name__ == '__main__':
    main()
