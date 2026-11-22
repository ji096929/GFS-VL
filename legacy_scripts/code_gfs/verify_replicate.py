"""验证 predict_agg 是否精确复刻 COSeg 原生 forward 的后半段。

原生：model(support, query, epoch=100) -> final_pred [1, 2, N_raw]
复刻：feats_one + proto + predict_agg       -> [N_raw, 2]

若两者 logits 最大绝对差 < 1e-4，则后续 560 集里
Baseline(R=仅 S) 与各 U 行处于同一数值尺度，可直接比较。
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/COSeg')
sys.path.insert(0, '/data01/jh/code')
from ring4_isolated import prepare_block  # noqa: E402
import diag_bc as D  # noqa: E402
from upper_bound_v2 import CACHE, NEW_IDS, feats_one, proto, predict_agg  # noqa: E402


def iou_of(p, gt):
    tp = int(((p == 1) & (gt == 1)).sum())
    fp = int(((p == 1) & (gt == 0)).sum())
    fn = int(((p == 0) & (gt == 1)).sum())
    return tp / max(tp + fp + fn, 1)


def main():
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))
    by_class = {}
    for m in metas:
        by_class.setdefault(m['class'], []).append(m)
    # 每类取 1 个 episode，共 10 个
    eps = [by_class[c][0] for c in NEW_IDS if c in by_class]

    model, cfg = D.build_model()
    dev = torch.device('cuda:0')
    model.k_shot = 1

    print('%-4s %-14s %8s %8s %10s %10s' % (
        'cls', 'name', 'iou_nat', 'iou_rep', 'd_iou', 'max|dlogit|'))
    worst = 0.0
    for m in eps:
        c = m['class']
        s = prepare_block(m['support'], cfg)
        q = prepare_block(m['query'], cfg)
        s_bin = (s['label'] == c).long()
        q_map = (q['label'] == c).long()
        s_x = torch.cat([s['coord'], s['feat']], 1)
        q_x = torch.cat([q['coord'], q['feat']], 1)

        with torch.no_grad():
            nat, _ = model(
                torch.tensor([len(s_x)], dtype=torch.int32).cuda(),
                s_x.cuda(), s_bin.cuda(),
                torch.tensor([len(q_x)], dtype=torch.int32).cuda(),
                q_x.cuda(), q_map.cuda(),
                100, None, None, np.array([c]))
            nat = nat[0]                      # [2, N_raw]

            sf = feats_one(model, s_x, s_bin, dev)
            qf = feats_one(model, q_x, q_map, dev)
        fg = sf['yl'].bool()
        sp = torch.cat([proto(model, sf, ~fg), proto(model, sf, fg)], 0)
        rep, _ = predict_agg(model, qf, [sp], 0.0, dev)   # [N_raw, 2]

        d = float((nat - rep).abs().max())
        p_nat = nat.argmax(0).cpu()
        p_rep = rep.argmax(0).cpu()
        in_, ir = iou_of(p_nat, q_map), iou_of(p_rep, q_map)
        worst = max(worst, d)
        print('%-4d %-14s %8.4f %8.4f %10.2e %10.2e' % (
            c, m.get('name', '-')[:14], in_, ir, abs(in_ - ir), d))

    print('\nmax|dlogit| over all = %.3e  -> %s' % (
        worst, 'PASS' if worst < 1e-4 else 'FAIL'))


if __name__ == '__main__':
    main()
