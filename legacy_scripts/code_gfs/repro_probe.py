"""定位 Baseline 的随机源：同进程连跑 3 次，打印各阶段校验和。"""
import json
import sys

import torch

sys.path.insert(0, '/data01/jh/code/COSeg')
sys.path.insert(0, '/data01/jh/code')
from ring4_isolated import prepare_block  # noqa: E402
import diag_bc as D  # noqa: E402
from upper_bound_v2 import CACHE, NEW_IDS, feats_one, proto, predict_agg  # noqa: E402


def main():
    metas = json.load(open(CACHE + '/val_meta_all.json'))
    by = {}
    for m in metas:
        by.setdefault(m['class'], []).append(m)
    eps = []
    for i in range(56):
        for c in NEW_IDS:
            if i < len(by[c]):
                eps.append(by[c][i])
    print('total eps %d' % len(eps), flush=True)

    model, cfg = D.build_model()
    dev = torch.device('cuda:0')

    for g in [0, 2, 7, 20]:
        m = eps[g]
        c = m['class']
        s = prepare_block(m['support'], cfg)
        q = prepare_block(m['query'], cfg)
        s_bin = (s['label'] == c).long()
        q_map = (q['label'] == c).long()
        s_x = torch.cat([s['coord'], s['feat']], 1)
        q_x = torch.cat([q['coord'], q['feat']], 1)
        print('\n--- g_ep %d cls %d  npts s=%d q=%d  seg_s=%d' % (
            g, c, len(s_x), len(q_x), int(s['seg'].max())), flush=True)
        for t in range(3):
            sf = feats_one(model, s_x, s_bin, dev)
            qf = feats_one(model, q_x, q_map, dev)
            fg = sf['yl'].bool()
            sp = torch.cat([proto(model, sf, ~fg), proto(model, sf, fg)], 0)
            pr, _ = predict_agg(model, qf, [sp], 0.0, dev)
            p = pr.argmax(0).cpu()
            tp = int(((p == 1) & (q_map == 1)).sum())
            fp = int(((p == 1) & (q_map == 0)).sum())
            fn = int(((p == 0) & (q_map == 1)).sum())
            print('  t%d iou=%.6f | coord_s=%.6f fd_s=%.6f fd_abs=%.4f '
                  'yl_n=%d sp_abs=%.4f logit_sum=%.4f npos=%d' % (
                      t, tp / max(tp + fp + fn, 1),
                      float(sf['cd'].sum()), float(sf['fd'].sum()),
                      float(sf['fd'].abs().sum()), int(fg.sum()),
                      float(sp.abs().sum()), float(pr.sum()), int(p.sum())),
                  flush=True)


if __name__ == '__main__':
    main()
