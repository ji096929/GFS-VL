"""补齐 Phase 4 后两行的逐类对比：Phase 2 CV 盲测 + 去污染（同上一行）。"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase2_v2 import adjacent, load_eps, score_adj  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
CKPT = '/data01/jh/weights/sc0_1w1s.pth'
DMAX = 0.10


def iou_of(pred, y):
    tp = int((pred & (y > 0)).sum())
    un = int((pred | (y > 0)).sum())
    return tp / max(un, 1)


def main():
    torch.manual_seed(0)
    table = torch.load(CKPT, map_location='cpu')['state_dict'][
        'module.base_prototypes'].float().cuda()
    eps = load_eps(560, table)
    N = len(eps)
    print('episodes:', N, flush=True)

    p2 = json.load(open(os.path.join(CACHE, 'phase2_v2.json')))
    picks = p2['picks']
    print('picks:', picks, flush=True)

    half = N // 2
    folds = [list(range(half)), list(range(half, N))]

    base_preds = [e['m'][e['inv']] > 0.5 for e in eps]
    preds_blind = [None] * N
    for f in (0, 1):
        p = picks[f]
        te = folds[1 - f]
        for ei in te:
            e = eps[ei]
            B = np.where(e['m'] >= p['tau'])[0]
            C = np.where((e['m'] < p['tau']) & (e['m'] > -3.0))[0]
            adj = adjacent(e, B, C, DMAX)
            ell = e['m'][e['inv']].copy()
            if len(adj) and len(B):
                s = score_adj(e, adj, B, p['score'])
                if p['score'] == 'vq':
                    keep = s > 0
                    adj2 = adj[keep.cpu().numpy()]
                    s2 = s[keep]
                else:
                    adj2, s2 = adj, s
                if len(adj2):
                    kk = max(int(len(adj2) * p['top']), 1)
                    sel = adj2[torch.topk(s2, kk).indices.cpu().numpy()]
                    ell[np.isin(e['inv'], sel)] = 1.0
            preds_blind[ei] = ell > 0.5

    def per_class(preds):
        tp_c, un_c = {}, {}
        for pred, e in zip(preds, eps):
            c = e['cls']
            tp_c[c] = tp_c.get(c, 0) + int((pred & (e['y'] > 0)).sum())
            un_c[c] = un_c.get(c, 0) + int((pred | (e['y'] > 0)).sum())
        return {int(c): float(tp_c[c] / max(un_c[c], 1)) for c in sorted(tp_c)}

    pb = per_class(base_preds)
    pp = per_class(preds_blind)
    cls_ids = sorted(pb)
    print('\n%-10s %s' % ('version', ' '.join('%6d' % c for c in cls_ids)))
    print('%-10s %s' % ('baseline', ' '.join('%6.3f' % pb[c] for c in cls_ids)))
    print('%-10s %s' % ('+ SP+adj', ' '.join('%6.3f' % pp[c] for c in cls_ids)))
    print('%-10s %s' % ('delta',
                        ' '.join('%+6.3f' % (pp[c] - pb[c]) for c in cls_ids)))
    print('\nnote: support decontamination has no interface to this pipeline,'
          ' so its per-class row equals the SP+adj row.')

    json.dump({'baseline': pb, 'sp_adj': pp,
               'picks': picks}, open(os.path.join(
                   CACHE, 'phase4_p2_perclass.json'), 'w'), indent=1)
    print('->', os.path.join(CACHE, 'phase4_p2_perclass.json'))


if __name__ == "__main__":
    main()
