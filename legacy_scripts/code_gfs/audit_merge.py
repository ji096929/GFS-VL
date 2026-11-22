"""审计汇总：逐 ckpt 指标 + 两折盲选轮次。"""
import glob
import json
import os

import numpy as np

ST2 = '/data01/jh/afford_diag/stage2'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
NAMES = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter',
         'curtain', 'desk', 'door', 'floor']


def load():
    res = {}
    for f in sorted(glob.glob(os.path.join(ST2, 'audit_s*.json'))):
        d = json.load(open(f))
        for k, rows in d.items():
            res.setdefault(k, {})
            for g, r in rows.items():
                res[k][int(g)] = r
    return res


def cumul(rows):
    tp = {c: 0 for c in NEW_IDS}
    fp = {c: 0 for c in NEW_IDS}
    fn = {c: 0 for c in NEW_IDS}
    ious = []
    for r in rows.values():
        c = r['cls']
        tp[c] += r['tp']; fp[c] += r['fp']; fn[c] += r['fn']
        ious.append(r['iou'])
    per = [tp[c] / max(tp[c] + fp[c] + fn[c], 1) for c in NEW_IDS]
    return float(np.mean(per)), float(np.mean(ious)), per


def agg(rows):
    tp = sum(r['tp'] for r in rows.values())
    fp = sum(r['fp'] for r in rows.values())
    fn = sum(r['fn'] for r in rows.values())
    gold_n = sum(r['gold_n'] for r in rows.values())
    gold_hit = sum(r['gold_n'] * max(r['gold_pure'], 0) for r in rows.values())
    gnew_n = sum(r['gnew_n'] for r in rows.values())
    gnew_hit = sum(r['gnew_n'] * max(r['gnew_pure'], 0) for r in rows.values())
    va = [r['oldauc'] for r in rows.values() if r['oldauc'] >= 0]
    na = [r['newauc'] for r in rows.values() if r['newauc'] >= 0]
    return dict(tp=tp, fp=fp, fn=fn,
                predfg=float(np.mean([r['predfg'] for r in rows.values()])),
                gold_n=gold_n, gold_pure=gold_hit / max(gold_n, 1),
                gnew_n=gnew_n, gnew_pure=gnew_hit / max(gnew_n, 1),
                oldauc=float(np.mean(va)) if va else -1,
                newauc=float(np.mean(na)) if na else -1)


def main():
    res = load()
    order = ['base'] + sorted([k for k in res if k.startswith('exp0_lowlr')],
                              key=lambda x: int(x.split('ep')[-1])) + \
        sorted([k for k in res if k.startswith('exp1_lowlr')],
               key=lambda x: int(x.split('ep')[-1]))

    print('=== 逐 ckpt 指标（357 集）===')
    print('%-16s %7s %7s %8s %8s %8s %8s %7s %7s'
          % ('ckpt', 'Cumul', 'Arith', 'TP', 'FP', 'FN',
             'predFG', 'gold纯', 'goldAUC'))
    for name in order:
        rows = res[name]
        cum, arith, _ = cumul(rows)
        a = agg(rows)
        print('%-16s %.4f %.4f %8d %8d %8d %7.3f %7.4f %7.4f'
              % (name, cum, arith, a['tp'], a['fp'], a['fn'],
                 a['predfg'], a['gold_pure'], a['oldauc']))

    print('\n=== 两折盲选 epoch（候选：exp0_lowlr / exp1_lowlr 的 ep1..9）===')
    gids = sorted(res['base'])
    half = len(gids) // 2
    folds = [gids[:half], gids[half:]]
    for tag in ('exp0_lowlr', 'exp1_lowlr'):
        cands = [k for k in res if k.startswith(tag + '_ep')]
        picks = []
        for si, ti in ((0, 1), (1, 0)):
            best, bv = None, -1
            for k in cands:
                v = cumul({g: res[k][g] for g in folds[si]})[0]
                if v > bv:
                    bv, best = v, k
            test = cumul({g: res[best][g] for g in folds[ti]})[0]
            picks.append((best, bv, test))
            print('%-12s  fold%d 选 %s (内部 %.4f) -> 另一折 %.4f'
                  % (tag, si, best, bv, test))
        print('  => %s 两折盲选平均 Cumul %.4f'
              % (tag, float(np.mean([p[2] for p in picks]))))

    # 基线在两折上的 Cumul（参考）
    for i, f in enumerate(folds):
        print('base fold%d Cumul %.4f'
              % (i, cumul({g: res['base'][g] for g in f})[0]))

    # 训练曲线
    print('\n=== 训练曲线（loss / eval）===')
    for tag in ('exp0_lowlr', 'exp1_lowlr', 'exp0_s', 'exp1_su_l03'):
        p = os.path.join(ST2, tag, 'hist.json')
        if not os.path.exists(p):
            continue
        hist = json.load(open(p))
        print(tag, 'loss:', ' '.join('%.3f' % h['loss'] for h in hist))
        print(tag, 'eval:', ' '.join(
            '%d:%.4f' % (h['ep'], h['cumul'])
            for h in hist if 'cumul' in h))


if __name__ == '__main__':
    main()
