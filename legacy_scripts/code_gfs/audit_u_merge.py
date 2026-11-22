"""U 推理消融汇总：4 种推理模式的 Cumul/Arith + U 块上的 B 池质量。"""
import glob
import json
import os

import numpy as np

ST2 = '/data01/jh/afford_diag/stage2'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
NAMES = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter',
         'curtain', 'desk', 'door', 'floor']
MODES = ['official', 'su_l0', 'su_l03', 'su_noise']


def load():
    res = {}
    for f in sorted(glob.glob(os.path.join(ST2, 'uinfer_s*.json'))):
        d = json.load(open(f))
        for g, r in d.items():
            res[int(g)] = r
    return res


def main():
    res = load()
    print('tasks', len(res))

    print('\n=== 推理模式对比（Exp1-low ep1，357 集）===')
    print('%-10s %8s %8s %9s %9s %9s %8s'
          % ('mode', 'Cumul', 'Arith', 'TP', 'FP', 'FN', 'predFG'))
    for m in MODES:
        tp = {c: 0 for c in NEW_IDS}
        fp = {c: 0 for c in NEW_IDS}
        fn = {c: 0 for c in NEW_IDS}
        ious, pf = [], []
        for r in res.values():
            rr = r[m]
            c = r['cls']
            tp[c] += rr['tp']; fp[c] += rr['fp']; fn[c] += rr['fn']
            ious.append(rr['iou']); pf.append(rr['predfg'])
        per = [tp[c] / max(tp[c] + fp[c] + fn[c], 1) for c in NEW_IDS]
        print('%-10s %.4f %.4f %9d %9d %9d %8.3f'
              % (m, float(np.mean(per)), float(np.mean(ious)),
                 sum(tp.values()), sum(fp.values()), sum(fn.values()),
                 float(np.mean(pf))))

    print('\n=== U 块上的 B={p>0.7} 质量（仅统计含目标类的 U 块）===')
    for tag in ('u_base', 'u_exp1'):
        rows = [r[tag] for r in res.values()
                if r[tag]['Bn'] > 0 and r[tag]['pure'] >= 0
                and r[tag]['auc'] >= 0]
        if not rows:
            print(tag, 'no valid U blocks')
            continue
        print('%-8s n_blocks %3d  mean Bn %7.1f  mean pure %.4f  mean AUC %.4f'
              % (tag, len(rows), float(np.mean([x['Bn'] for x in rows])),
                 float(np.mean([x['pure'] for x in rows])),
                 float(np.mean([x['auc'] for x in rows]))))

    # 逐类 Cumul（official vs su_l03）
    print('\n逐类 IoU (official vs su_l03)')
    print('%-10s %10s %10s' % ('class', 'official', 'su_l03'))
    for c, nm in zip(NEW_IDS, NAMES):
        out = []
        for m in ('official', 'su_l03'):
            tp = fp = fn = 0
            for r in res.values():
                if r['cls'] != c:
                    continue
                rr = r[m]
                tp += rr['tp']; fp += rr['fp']; fn += rr['fn']
            out.append(tp / max(tp + fp + fn, 1))
        print('%-10s %10.4f %10.4f' % (nm, out[0], out[1]))


if __name__ == '__main__':
    main()
