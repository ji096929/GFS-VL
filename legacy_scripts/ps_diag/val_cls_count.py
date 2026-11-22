"""统计 312 val 场景里每个 novel 类的 GT 点数，并和逐类 IoU 对照。

目的：判断 A/B 相对基线的跌幅是集中在「点极少的小类」（IoU 不稳定/噪声）
还是「点不少的类」（真实退化）。
"""
import os
import re
import sys

import numpy as np

ROOT = '/data01/jh/data/scannet200_gfs/val'
sys.path.insert(0, '/data01/jh/code/GFS-VL/pointcept/datasets/preprocessing/scannet/meta_data')
from scannet200_constants import CLASS_LABELS_200, CLASS_LABELS_BASE_NOVEL  # noqa: E402


def parse(p):
    d = {}
    for line in open(p):
        m = re.search(r'Class_(\d+) - (.*?) Final Result: iou ([0-9.]+)', line)
        if m:
            d[m.group(2)] = float(m.group(3))
    return d


def main():
    names = [CLASS_LABELS_200[i] for i in range(len(CLASS_LABELS_200))]
    cnt = np.zeros(len(names), dtype=np.int64)
    scenes = sorted(os.listdir(ROOT))
    for i, sc in enumerate(scenes):
        p = os.path.join(ROOT, sc, 'segment200.npy')
        if not os.path.exists(p):
            continue
        seg = np.load(p)
        v = seg[seg >= 0]
        cnt += np.bincount(v, minlength=len(names))
        if (i + 1) % 100 == 0:
            print('  %d/%d' % (i + 1, len(scenes)), flush=True)
    tot = int(cnt.sum())
    print('val 有效 GT 点 %.1f M（%d 场景）' % (tot / 1e6, len(scenes)), flush=True)

    base = parse('/data01/jh/afford_diag/gfs_build/test_regis1_v3.log')
    b = parse('/data01/jh/afford_diag/gfs_build/test_B.log')
    a = parse('/data01/jh/afford_diag/gfs_build/test_A.log')

    novel = [n for n in CLASS_LABELS_BASE_NOVEL if n not in CLASS_LABELS_BASE_NOVEL[:12]]
    rows = []
    for n in novel:
        c = int(cnt[names.index(n)])
        rows.append((n, c, base.get(n, 0.0), b.get(n, 0.0), a.get(n, 0.0)))
    print('\n%-24s %10s %7s %7s %7s %8s %8s'
          % ('class', 'val GT pts', 'base', 'B', 'A', 'A-B', 'A-base'))
    for r in sorted(rows, key=lambda x: -x[1]):
        print('%-24s %10d %7.3f %7.3f %7.3f %+8.3f %+8.3f'
              % (r[0], r[1], r[2], r[3], r[4], r[4] - r[3], r[4] - r[2]))

    # 按点数分档看平均跌幅
    print('\n按 GT 点数分档（A 相对 base 的 IoU 差）:')
    bins = [(0, 2000), (2000, 10000), (10000, 50000), (50000, 200000), (200000, 10**9)]
    for lo, hi in bins:
        g = [r for r in rows if lo <= r[1] < hi]
        if not g:
            continue
        print('  %7d-%-8d  %2d 类  平均 base %.3f  A %.3f  Δ %+0.3f'
              % (lo, hi, len(g), np.mean([x[2] for x in g]),
                 np.mean([x[4] for x in g]),
                 np.mean([x[4] - x[2] for x in g])))


if __name__ == '__main__':
    main()
