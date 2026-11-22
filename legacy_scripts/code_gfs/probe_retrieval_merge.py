"""检索对照汇总：命中率 + Cumul mIoU。"""
import glob
import json
import os

import numpy as np

ST2 = '/data01/jh/afford_diag/stage2'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
NAMES = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter',
         'curtain', 'desk', 'door', 'floor']


def load():
    rows = {}
    for f in sorted(glob.glob(os.path.join(ST2, 'retrieval_s*.json'))):
        d = json.load(open(f))
        for g, r in d.items():
            rows[int(g)] = r
    return rows


def main():
    rows = load()
    print('tasks', len(rows))

    print('\n=== 块级命中率（3 块中至少一块）===')
    print('%-10s %4s %9s %9s %10s %10s'
          % ('class', 'n', 'rand命中', 'retr命中', 'rand>=5%', 'retr>=5%'))
    for c, nm in zip(NEW_IDS, NAMES):
        rs = [r for r in rows.values() if r['cls'] == c]
        if not rs:
            continue
        print('%-10s %4d %9.3f %9.3f %10.3f %10.3f'
              % (nm, len(rs),
                 np.mean([r['rand_hit'] for r in rs]),
                 np.mean([r['retr_hit'] for r in rs]),
                 np.mean([r['rand_hit5'] for r in rs]),
                 np.mean([r['retr_hit5'] for r in rs])))
    print('%-10s %4d %9.3f %9.3f %10.3f %10.3f'
          % ('ALL', len(rows),
             np.mean([r['rand_hit'] for r in rows.values()]),
             np.mean([r['retr_hit'] for r in rows.values()]),
             np.mean([r['rand_hit5'] for r in rows.values()]),
             np.mean([r['retr_hit5'] for r in rows.values()])))

    print('\n=== Cumul mIoU ===')
    print('%-10s %8s %8s %8s' % ('config', 'Cumul', 'Arith', 'Δvs base'))
    base_per = None
    for m in ('base', 'rand', 'retr'):
        tp = {c: 0 for c in NEW_IDS}
        fp = {c: 0 for c in NEW_IDS}
        fn = {c: 0 for c in NEW_IDS}
        ious = []
        for r in rows.values():
            c = r['cls']
            t, f, n = r[m]
            tp[c] += t; fp[c] += f; fn[c] += n
            ious.append(t / max(t + f + n, 1))
        per = [tp[c] / max(tp[c] + fp[c] + fn[c], 1) for c in NEW_IDS]
        cum = float(np.mean(per))
        if base_per is None:
            base_per = cum
        print('%-10s %.4f %.4f %+8.4f'
              % (m, cum, float(np.mean(ious)), cum - base_per))

    print('\n=== 逐类 Cumul（base / rand / retr）===')
    print('%-10s %8s %8s %8s' % ('class', 'base', 'rand', 'retr'))
    for c, nm in zip(NEW_IDS, NAMES):
        out = []
        for m in ('base', 'rand', 'retr'):
            tp = fp = fn = 0
            for r in rows.values():
                if r['cls'] != c:
                    continue
                t, f, n = r[m]
                tp += t; fp += f; fn += n
            out.append(tp / max(tp + fp + fn, 1))
        print('%-10s %8.4f %8.4f %8.4f' % (nm, out[0], out[1], out[2]))

    # 检索池的实际命中构成
    print('\n=== 检索 Top-3 的池命中明细（floor 单列）===')
    for c, nm in zip(NEW_IDS, NAMES):
        rs = [r for r in rows.values() if r['cls'] == c]
        if not rs:
            continue
        print('%-10s retr命中 %5.1f%%  (n=%d)'
              % (nm, 100 * np.mean([r['retr_hit'] for r in rs]), len(rs)))


if __name__ == '__main__':
    main()
