"""第四环出表：读 ring4_full.json，输出主表、纯度分组、Tier3 归因。"""
import json
import os
import sys

import numpy as np

CACHE = '/data01/jh/data/ocsb_cache'
NAMES_PATH = '/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt'

ORDER = ['S', 'onlyB_T1', 'onlyB_T2a', 'onlyB_T2b', 'onlyB_T3',
         'SB_T1', 'SB_T2a', 'SB_T2b', 'SB_T3',
         'FUSE_T1_l0.2', 'FUSE_T1_l0.3', 'FUSE_T2b_l0.2', 'FUSE_T2b_l0.3',
         'FUSE_T3_l0.2', 'FUSE_T3_l0.3']
SHORT = {'S': 'S', 'onlyB_T1': 'oB_T1', 'onlyB_T2a': 'oB_T2a',
         'onlyB_T2b': 'oB_T2b', 'onlyB_T3': 'oB_T3',
         'SB_T1': 'SB_T1', 'SB_T2a': 'SB_T2a', 'SB_T2b': 'SB_T2b',
         'SB_T3': 'SB_T3',
         'FUSE_T1_l0.2': 'F_T1_.2', 'FUSE_T1_l0.3': 'F_T1_.3',
         'FUSE_T2b_l0.2': 'F_T2b_.2', 'FUSE_T2b_l0.3': 'F_T2b_.3',
         'FUSE_T3_l0.2': 'F_T3_.2', 'FUSE_T3_l0.3': 'F_T3_.3'}

HIGH = ['curtain', 'bookshelf', 'chair']
LOW = ['counter', 'desk', 'bathtub', 'cabinet']

# ring2c 的 B 精确率
PREC = {'counter': 0.154, 'desk': 0.228, 'bathtub': 0.230, 'floor': 0.236,
        'cabinet': 0.278, 'bed': 0.335, 'door': 0.362, 'chair': 0.478,
        'bookshelf': 0.553, 'curtain': 0.555}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        CACHE, 'ring4_full.json')
    rows = json.load(open(path))
    got = [v for v in ORDER if v in rows[0]['iou']]

    print('=== 1) 主表（10 类，每类 %d eps）===' % rows[0]['n'])
    print('%-12s %7s %8s' % ('class', 'n_B', 'B@5cm') +
          ''.join('%9s' % SHORT[v] for v in got))
    for r in rows:
        print('%-12s %7.0f %8.3f'
              % (r['class'], r['n_B']['onlyB_T1'], r['ov']['b5']) +
              ''.join('%9.4f' % r['iou'][v] for v in got))
    mean = {v: float(np.mean([r['iou'][v] for r in rows])) for v in got}
    print('%-12s %7s %8.3f' % ('Arith', '', np.mean([r['ov']['b5'] for r in rows])) +
          ''.join('%9.4f' % mean[v] for v in got))

    print('\n=== 2) Δ vs S（pp）===')
    print('%-12s' % 'class' + ''.join('%9s' % SHORT[v] for v in got[1:]))
    deltas = {}
    for r in rows:
        ds = {v: 100 * (r['iou'][v] - r['iou']['S']) for v in got[1:]}
        deltas[r['class']] = ds
        print('%-12s' % r['class'] + ''.join('%+9.2f' % ds[v] for v in got[1:]))
    print('%-12s' % 'MEAN' +
          ''.join('%+9.2f' % (100 * (mean[v] - mean['S'])) for v in got[1:]))

    print('\n=== 3) 纯度分组 Δ（pp）===')
    print('%-10s %-12s' % ('group', 'classes') +
          ''.join('%9s' % SHORT[v] for v in got[1:]))
    for gname, glist in (('HIGH(3)', HIGH), ('LOW(4)', LOW)):
        sub = [deltas[c] for c in glist]
        print('%-10s %-12s' % (gname, ','.join(c[:5] for c in glist)) +
              ''.join('%+9.2f' % np.mean([s[v] for s in sub]) for v in got[1:]))

    print('\n=== 4) Tier3 归因：T3 相对 T1 的变化（pp）与 B 精确率 ===')
    print('%-12s %8s %10s %10s %10s %10s'
          % ('class', 'prec', 'T1', 'T3', 'T3-T1', 'nB_T3'))
    t31 = []
    for r in rows:
        d = 100 * (r['iou']['onlyB_T3'] - r['iou']['onlyB_T1'])
        t31.append(d)
        print('%-12s %8.3f %10.4f %10.4f %+10.2f %10.0f'
              % (r['class'], PREC.get(r['class'], -1), r['iou']['onlyB_T1'],
                 r['iou']['onlyB_T3'], d, r['n_B']['onlyB_T3']))
    precs = [PREC.get(r['class'], 0) for r in rows]
    if len(set(precs)) > 2:
        from scipy.stats import spearmanr, pearsonr
        s1 = spearmanr(precs, [deltas[r['class']]['onlyB_T1'] for r in rows])
        s2 = spearmanr(precs, t31)
        print('\nSpearman(纯度, ΔonlyB_T1) = %+.3f  p=%.3f' % (s1.correlation, s1.pvalue))
        print('Spearman(纯度, T3-T1)     = %+.3f  p=%.3f' % (s2.correlation, s2.pvalue))

    print('\n=== 5) 胜出统计（相对 S 为正的类别数 / 10）===')
    for v in got[1:]:
        npos = sum(1 for r in rows if r['iou'][v] > r['iou']['S'])
        best = max(rows, key=lambda r: r['iou'][v] - r['iou']['S'])
        print('%-12s %2d/10   最好 %-11s %+.2fpp'
              % (SHORT[v], npos, best['class'],
                 100 * (best['iou'][v] - best['iou']['S'])))

    print('\n=== 6) S 前景点数 / B 点数比 与 onlyB 增益 ===')
    SFG = {'counter': 84, 'bathtub': 157, 'chair': 200, 'desk': 213,
           'floor': 224, 'bed': 263, 'cabinet': 366, 'door': 401,
           'curtain': 457, 'bookshelf': 548}
    print('%-12s %8s %8s %8s %10s %10s'
          % ('class', 'S_fg', 'B_pts', 'B/S', 'dOnlyB_T1', 'dFUSE_T1_.2'))
    sf, bs, dt1, dft = [], [], [], []
    for r in rows:
        sfg = SFG.get(r['class'], 0)
        bpt = r['n_B']['onlyB_T1']
        ratio = bpt / max(sfg, 1)
        d1 = 100 * (r['iou']['onlyB_T1'] - r['iou']['S'])
        df = 100 * (r['iou'].get('FUSE_T1_l0.2', r['iou']['S']) - r['iou']['S'])
        sf.append(sfg); bs.append(ratio); dt1.append(d1); dft.append(df)
        print('%-12s %8.0f %8.0f %8.1f %+10.2f %+10.2f'
              % (r['class'], sfg, bpt, ratio, d1, df))
    from scipy.stats import spearmanr
    a = spearmanr(bs, dt1)
    b = spearmanr(sf, dft)
    print('Spearman(B/S, ΔonlyB_T1)  = %+.3f  p=%.3f'
          % (a.correlation, a.pvalue))
    print('Spearman(S_fg, ΔFUSE)     = %+.3f  p=%.3f'
          % (b.correlation, b.pvalue))


if __name__ == '__main__':
    main()
