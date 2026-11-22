"""Cut 2 汇总：合并各分片 CM，打印分层指标总表。

用法: python cut2_analyze.py [tag ...]
"""
import glob
import json
import os
import sys

import numpy as np

FEAT = '/data01/jh/data/cut2_feat'
NO, NC = 12, 57
LAYERS = ['enc4', 'dec3', 'dec2', 'dec1', 'dec0', 'multi']
DIM = {'enc4': 512, 'dec3': 256, 'dec2': 128, 'dec1': 64, 'dec0': 64,
       'multi': 512}


def metrics(cm):
    d = np.diag(cm).astype(np.float64)
    r = cm.sum(1).astype(np.float64)
    c = cm.sum(0).astype(np.float64)
    iou = d / np.maximum(r + c - d, 1)
    b = float(iou[:NO].mean())
    n = float(iou[NO:].mean())
    return b, n, 2 * b * n / max(b + n, 1e-9), iou


def merge(tag, prefix, only=None):
    fs = sorted(glob.glob(os.path.join(FEAT, tag, prefix + '_s*.npz')))
    if not fs:
        return None
    tot = {}
    for f in fs:
        z = np.load(f, allow_pickle=True)
        for k in z.files:
            if only and not k.startswith(only):
                continue
            tot[k] = tot.get(k, 0) + z[k]
    return tot


def main():
    tags = sys.argv[1:] or ['S1', 'S2']
    for tag in tags:
        print('\n' + '=' * 78)
        print('== %s' % tag)
        print('=' * 78, flush=True)
        rep_p = os.path.join(FEAT, tag, 'report.json')
        if os.path.exists(rep_p):
            rep = json.load(open(rep_p))
            print('eRank（每类 %d 点）' % rep.get('n_points', 0))
            print('  %-6s %4s  %-22s %-22s' %
                  ('层', 'D', 'Base eRank(占比)', 'Novel eRank(占比)'))
            for L in LAYERS:
                if L not in rep.get('erank', {}):
                    continue
                e = rep['erank'][L]
                print('  %-6s %4d  %7.2f (%5.1f%%)        %7.2f (%5.1f%%)'
                      % (L, e['dim'], e['base'], 100 * e['base_ratio'],
                         e['novel'], 100 * e['novel_ratio']))
            if 'subspace' in rep:
                s = rep['subspace']
                print('  dec0 Base 子空间能量: Novel %.1f%% | Base %.1f%% '
                      '| 随机基线 %.1f%%'
                      % (100 * s['novel'], 100 * s['base'],
                         100 * s['random_baseline']))

        cm = merge(tag, 'cm')
        if cm:
            print('\nLinear Probe（全监督 57 类，312 val 全量）')
            print('  %-6s %8s %8s %8s   %s' %
                  ('层', 'mIoU-B', 'mIoU-N', 'HM', '相对 dec0'))
            b0 = n0 = None
            for L in LAYERS:
                if L not in cm:
                    continue
                b, n, h, _ = metrics(cm[L])
                if L == 'dec0':
                    b0, n0 = b, n
                print('  %-6s %8.4f %8.4f %8.4f'
                      % (L, b, n, h), end='')
                if L in ('enc4', 'dec3', 'dec2', 'dec1') and n0 is not None:
                    print('   ΔN %+.2fpp' % (100 * (n - n0)), end='')
                print()
            if b0 is not None:
                print('  （ΔN 以 dec0 为基准）')

        os_ = {L: cm['os_' + L] for L in LAYERS if cm and 'os_' + L in cm}
        if os_:
            print('\n1-shot 支持原型（45-way 闭集，312 val 新类真值点）')
            print('  %-6s %10s %10s' % ('层', 'acc', 'mIoU(45way)'))
            for L in LAYERS:
                if L not in os_:
                    continue
                cmx = os_[L]
                d = np.diag(cmx).astype(np.float64)
                r = cmx.sum(1).astype(np.float64)
                c = cmx.sum(0).astype(np.float64)
                iou = d / np.maximum(r + c - d, 1)
                print('  %-6s %10.4f %10.4f'
                      % (L, d.sum() / max(cmx.sum(), 1), iou.mean()))


if __name__ == '__main__':
    main()
