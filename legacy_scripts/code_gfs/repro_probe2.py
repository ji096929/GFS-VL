"""验证 per-episode 种子锁死后，跨进程 Baseline 是否一致。

用法：起两个独立进程（不同卡），比较同一 --gps 的输出。
"""
import argparse
import json
import sys

import torch

sys.path.insert(0, '/data01/jh/code/COSeg')
sys.path.insert(0, '/data01/jh/code')
from ring4_isolated import prepare_block  # noqa: E402
import diag_bc as D  # noqa: E402
from upper_bound_v2 import (CACHE, NEW_IDS, feats_one, proto,  # noqa: E402
                            predict_agg, seed_everything)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gps', default='0,2,7,20')
    ap.add_argument('--tag', default='A')
    args = ap.parse_args()

    metas = json.load(open(CACHE + '/val_meta_all.json'))
    by = {}
    for m in metas:
        by.setdefault(m['class'], []).append(m)
    eps = []
    for i in range(56):
        for c in NEW_IDS:
            if i < len(by[c]):
                eps.append(by[c][i])

    seed_everything(0)
    model, cfg = D.build_model()
    dev = torch.device('cuda:0')

    for g in [int(x) for x in args.gps.split(',')]:
        m = eps[g]
        c = m['class']
        seed_everything(10000 + g)
        s = prepare_block(m['support'], cfg)
        q = prepare_block(m['query'], cfg)
        s_bin = (s['label'] == c).long()
        q_map = (q['label'] == c).long()
        s_x = torch.cat([s['coord'], s['feat']], 1)
        q_x = torch.cat([q['coord'], q['feat']], 1)
        sf = feats_one(model, s_x, s_bin, dev)
        qf = feats_one(model, q_x, q_map, dev)
        fg = sf['yl'].bool()
        sp = torch.cat([proto(model, sf, ~fg), proto(model, sf, fg)], 0)
        pr, _ = predict_agg(model, qf, [sp], 0.0, dev)
        p = pr.argmax(0).cpu()
        tp = int(((p == 1) & (q_map == 1)).sum())
        fp = int(((p == 1) & (q_map == 0)).sum())
        fn = int(((p == 0) & (q_map == 1)).sum())
        print('%s g_ep=%3d cls=%2d iou=%.6f npts_s=%d npts_q=%d '
              'fd_abs=%.4f npos=%d' % (
                  args.tag, g, c, tp / max(tp + fp + fn, 1),
                  len(s_x), len(q_x), float(sf['fd'].abs().sum()),
                  int(p.sum())), flush=True)


if __name__ == '__main__':
    main()
