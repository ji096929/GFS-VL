"""盲区 A：δ 反向扫描（给新类 logits 加 δ，δ<0 = 压新类/抬基类）。

用 val_logits 的 top-5：跨类型 argmax 只需比较「最高的 base」与「最高的 novel」，
这两者必在 top-5 内（同类型内部顺序不受统一 δ 影响），所以按 top-5 重算是精确的。

用法: python cut2_blindA.py [pred_dir]
"""
import glob
import sys

import numpy as np

NO, NC = 12, 57
VAL = sys.argv[1] if len(sys.argv) > 1 else '/data01/jh/data/val_logits'
DELTAS = [-3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]


def metrics(cm):
    d = np.diag(cm).astype(np.float64)
    r = cm.sum(1).astype(np.float64)
    c = cm.sum(0).astype(np.float64)
    iou = d / np.maximum(r + c - d, 1)
    b = float(iou[:NO].mean())
    n = float(iou[NO:].mean())
    return b, n, 2 * b * n / max(b + n, 1e-9)


def main():
    cms = {d: np.zeros((NC, NC), np.int64) for d in DELTAS}
    for f in sorted(glob.glob(VAL + '/*.npz')):
        z = np.load(f)
        p = z['top5p'].astype(np.float64)
        i = z['top5i'].astype(np.int64)
        gt = z['gt'].astype(np.int64)
        lp = np.log(np.maximum(p, 1e-12))
        isb = i < NO
        BIG = -1e9
        bv = np.where(isb, lp, BIG)
        nv = np.where(~isb, lp, BIG)
        bidx = bv.argmax(1)
        nidx = nv.argmax(1)
        bmax = bv[np.arange(len(i)), bidx]
        nmax = nv[np.arange(len(i)), nidx]
        row = np.arange(len(i))
        base_cls = i[row, bidx]
        novel_cls = i[row, nidx]
        ok = gt >= 0
        for d in DELTAS:
            pick = np.where(nmax + d > bmax, novel_cls, base_cls)
            cms[d] += np.bincount(gt[ok] * NC + pick[ok],
                                  minlength=NC * NC).reshape(NC, NC)

    print('=== δ 扫描（δ 加在新类 logits 上；δ<0 = 抬基类/压新类）===')
    print('  %6s %8s %8s %8s   %s' % ('δ', 'Base', 'Novel', 'HM', '变化点数(相对δ=0)'))
    b0, n0, h0 = metrics(cms[0.0])
    for d in DELTAS:
        b, n, h = metrics(cms[d])
        print('  %6.2f %8.4f %8.4f %8.4f   ΔN %+.2fpp  ΔH %+.2fpp'
              % (d, b, n, h, 100 * (n - n0), 100 * (h - h0)))


if __name__ == '__main__':
    main()
