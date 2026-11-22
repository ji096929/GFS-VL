"""第一步：用 COSeg 原生多样本接口测 B_pos / B_pos∪C_pos 的真实上限。

四行对照（全部零训练，只改支持集）：
  Baseline    仅 1-shot S（k_shot=1，100 个原型）
  Naive-B     S + U 中初始预测为前景的点（含假阳性，k_shot=2，各 50 个原型）
  Oracle-B    S + U 中「预测为前景 且 真值是前景」的点（真桥梁）
  Oracle-B+C  S + U 中「真值是前景」的全部点（真桥梁 + 真遗漏）

U 的初始预测用同一个模型的 S 原型在 U 上跑 forward 得到（与原生逻辑等价）。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data01/jh/code/COSeg')
sys.path.insert(0, '/data01/jh/code')
from ring4_isolated import prepare_block  # noqa: E402
import diag_bc as D  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
NAMES_PATH = '/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]


def fwd(model, k_shot, s_x, s_y, s_off, q_x, q_y, q_off, c):
    model.k_shot = k_shot
    pred, _ = model(
        torch.tensor(s_off, dtype=torch.int32).cuda(),
        s_x.cuda(), s_y.cuda(),
        torch.tensor(q_off, dtype=torch.int32).cuda(),
        q_x.cuda(), q_y.cuda(),
        100, None, None, np.array([c]),
    )
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='0 = 全量 560')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(CACHE, 'upper_bound.json'))
    args = ap.parse_args()

    names = [l.strip() for l in open(NAMES_PATH)]
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))
    model, cfg = D.build_model()
    dev = torch.device('cuda:0')

    cases = ['Baseline', 'Baseline-2s', 'Naive-B', 'Oracle-B', 'Oracle-B+C']
    ious = {k: [] for k in cases}
    cm = {k: {c: [0, 0, 0] for c in NEW_IDS} for k in cases}
    n_bsp = {k: [] for k in cases}

    eps = metas if args.limit <= 0 else metas[:args.limit]
    if args.shards > 1:
        eps = eps[args.shard::args.shards]
    for ei, m in enumerate(eps):
        c = m['class']
        s = prepare_block(m['support'], cfg)
        q = prepare_block(m['query'], cfg)
        us = [prepare_block(nm, cfg) for nm in m['u']]

        s_bin = (s['label'] == c).long()
        q_map = (q['label'] == c).long()
        u_lab = torch.cat([u['label'] for u in us], 0)
        u_bin = (u_lab == c).long()
        s_x = torch.cat([s['coord'], s['feat']], 1)
        q_x = torch.cat([q['coord'], q['feat']], 1)
        u_x = torch.cat([torch.cat([u['coord'], u['feat']], 1) for u in us], 0)
        q_off = [len(q_x)]
        u_zero = torch.zeros(len(u_x), dtype=torch.long)

        with torch.no_grad():
            # U 的初始预测：S 原型 -> U（原生 forward，U 当 query）
            u_pred = fwd(model, 1, s_x, s_bin, [len(s_x)], u_x, u_zero,
                         [len(u_x)], c)
            u_prob = F.softmax(u_pred[0], dim=0)[1].cpu()
            u_pos = (u_prob > 0.5).long()

            variants = {
                'Baseline': (1, s_x, s_bin, [len(s_x)]),
                # 对照：S 复制两遍，同样是 50+50 个原型，分离「S 原型减半」的影响
                'Baseline-2s': (2, torch.cat([s_x, s_x], 0),
                                torch.cat([s_bin, s_bin], 0),
                                [len(s_x), 2 * len(s_x)]),
                'Naive-B': (2, torch.cat([s_x, u_x], 0),
                            torch.cat([s_bin, u_pos], 0),
                            [len(s_x), len(s_x) + len(u_x)]),
                'Oracle-B': (2, torch.cat([s_x, u_x], 0),
                             torch.cat([s_bin, u_pos * u_bin], 0),
                             [len(s_x), len(s_x) + len(u_x)]),
                'Oracle-B+C': (2, torch.cat([s_x, u_x], 0),
                               torch.cat([s_bin, u_bin], 0),
                               [len(s_x), len(s_x) + len(u_x)]),
            }
            n_bsp.setdefault('Baseline-2s', []).append(0)
            n_bsp['Naive-B'].append(int(u_pos.sum()))
            n_bsp['Oracle-B'].append(int((u_pos * u_bin).sum()))
            n_bsp['Oracle-B+C'].append(int(u_bin.sum()))

            for name, (k, sx, sy, soff) in variants.items():
                pred = fwd(model, k, sx, sy, soff, q_x, q_map, q_off, c)
                p = pred[0].argmax(0).cpu()
                tp = int(((p == 1) & (q_map == 1)).sum())
                fp = int(((p == 1) & (q_map == 0)).sum())
                fn = int(((p == 0) & (q_map == 1)).sum())
                ious[name].append(tp / max(tp + fp + fn, 1))
                cm[name][c][0] += tp
                cm[name][c][1] += fp
                cm[name][c][2] += fn

        if (ei + 1) % 20 == 0:
            print('  %d / %d' % (ei + 1, len(eps)), flush=True)
            json.dump(dict(cases=cases, ious=ious,
                           cm={k: {str(c2): v for c2, v in d.items()}
                               for k, d in cm.items()},
                           n_bsp=n_bsp, done=ei + 1),
                      open(args.out + '.partial', 'w'))

    out = {}
    print('\n%-12s %10s %10s %12s' % ('case', 'Arith', 'Cumul', 'mean n_B'))
    for k in cases:
        arith = float(np.mean(ious[k]))
        per = [cm[k][c][0] / max(cm[k][c][0] + cm[k][c][1] + cm[k][c][2], 1)
               for c in NEW_IDS]
        cumul = float(np.mean(per))
        nb = float(np.mean(n_bsp[k])) if n_bsp[k] else 0
        out[k] = dict(arith=arith, cumul=cumul, per_class=per,
                      mean_n_b=nb, n_ep=len(ious[k]))
        print('%-12s %10.4f %10.4f %12.0f' % (k, arith, cumul, nb))

    print('\n逐类 Cumul IoU')
    print('%-14s' % 'class' + ''.join('%12s' % k for k in cases))
    for i, c in enumerate(NEW_IDS):
        print('%-14s' % names[c] + ''.join(
            '%12.4f' % out[k]['per_class'][i] for k in cases))

    json.dump(out, open(args.out, 'w'), indent=1)
    print('\n->', args.out)


if __name__ == '__main__':
    main()
