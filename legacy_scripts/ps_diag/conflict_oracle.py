"""Step 2：局部 3D 冲突区「完美仲裁」的 Oracle 上限。

对 312 val：
  1. 用 top-2 概率边际定义冲突点（margin < tau）
  2. 分成两类冲突：Base–Novel（top1/top2 跨 base/novel 边界）、Novel–Novel
  3. 在冲突点上做 3D 连通分量（体素 0.05m）
  4. Oracle：每个分量整体赋为「该分量的 GT 多数类」（多数占比 >= 0.5 才动），
     分量外严格不动 → 重算 Base/Novel/HM
  5. 对照：不用 GT，把分量整体赋为「预测多数类」（平滑），看是否也有增益

用法: python conflict_oracle.py [tau1,tau2,...]
"""
import glob
import os
import sys

import numpy as np

NO, NC = 12, 57
VAL = '/data01/jh/data/val_logits'
COORD = '/data01/jh/data/scannet200_gfs/val'
RADIUS = 0.05


def metrics(cm):
    d = np.diag(cm).astype(np.float64)
    r = cm.sum(1).astype(np.float64)
    c = cm.sum(0).astype(np.float64)
    iou = d / np.maximum(r + c - d, 1)
    base = float(iou[:NO].mean())
    nov = float(iou[NO:].mean())
    return base, nov, 2 * base * nov / max(base + nov, 1e-9)


def components(mask, coord, radius=RADIUS):
    from scipy import ndimage
    idx = np.where(mask)[0]
    if len(idx) < 20:
        return []
    pts = coord[idx]
    mn = pts.min(0)
    g = np.floor((pts - mn) / radius).astype(np.int64)
    shape = tuple(g.max(0) + 1)
    grid = np.zeros(shape, dtype=bool)
    grid[g[:, 0], g[:, 1], g[:, 2]] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=int))
    labels = lab[g[:, 0], g[:, 1], g[:, 2]]
    return [idx[labels == k] for k in range(1, n + 1)]


def main():
    taus = [float(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1
                               else ['0.1', '0.2', '0.3'])]
    files = sorted(glob.glob(VAL + '/*.npz'))
    print('scenes %d, taus %s' % (len(files), taus), flush=True)

    add = {t: np.zeros((NC, NC), dtype=np.int64) for t in taus}   # oracle 增量
    add_sm = {t: np.zeros((NC, NC), dtype=np.int64) for t in taus}  # 平滑增量
    stat = {t: dict(n_comp=0, n_pt=0, n_bn=0, n_nn=0) for t in taus}

    def cell(gt, pred):
        ok = gt >= 0
        return np.bincount(gt[ok] * NC + pred[ok], minlength=NC * NC).reshape(NC, NC)

    for fi, f in enumerate(files):
        sc = os.path.basename(f)[:-4]
        z = np.load(f)
        p = z['top5p'].astype(np.float32)
        i = z['top5i'].astype(np.int64)
        pred = z['pred'].astype(np.int64)
        gt = z['gt'].astype(np.int64)
        coord = np.load(os.path.join(COORD, sc, 'coord.npy')).astype(np.float32)
        if len(pred) != len(coord):
            print('!! len mismatch', sc, len(pred), len(coord))
            continue
        margin = p[:, 0] - p[:, 1]
        c1, c2 = i[:, 0], i[:, 1]
        is_bn = ((c1 < NO) & (c2 >= NO)) | ((c1 >= NO) & (c2 < NO))
        is_nn = (c1 >= NO) & (c2 >= NO) & (c1 != c2)

        base_cm = cell(gt, pred)
        for tau in taus:
            conf = (margin < tau) & (is_bn | is_nn)
            comps = components(conf, coord)
            orc = pred.copy()
            smo = pred.copy()
            for comp in comps:
                if len(comp) < 20:
                    continue
                g = gt[comp]
                g = g[g >= 0]
                if len(g) < 0.5 * len(comp):
                    continue
                maj = int(np.bincount(g, minlength=NC).argmax())
                if np.count_nonzero(g == maj) / len(g) < 0.5:
                    continue
                orc[comp] = maj
                pc = np.bincount(pred[comp], minlength=NC).argmax()
                smo[comp] = int(pc)
                stat[tau]['n_comp'] += 1
                stat[tau]['n_pt'] += len(comp)
                stat[tau]['n_bn'] += int(is_bn[comp].sum())
                stat[tau]['n_nn'] += int(is_nn[comp].sum())
            add[tau] += cell(gt, orc) - base_cm
            add_sm[tau] += cell(gt, smo) - base_cm
            stat[tau]['base_cm'] = stat[tau].get('base_cm', np.zeros((NC, NC), np.int64)) + base_cm
        if (fi + 1) % 50 == 0:
            print('  %d/%d' % (fi + 1, len(files)), flush=True)

    b0, n0, h0 = metrics(stat[taus[0]]['base_cm'])
    print('\n=== 基线（ONLINE_s42）===')
    print('Base %.4f  Novel %.4f  HM %.4f' % (b0, n0, h0))
    print('\n%-6s %8s %9s %8s %8s' % ('tau', '冲突点', '分量数', 'Oracle', '平滑'))
    for tau in taus:
        s = stat[tau]
        cm_o = s['base_cm'] + add[tau]
        cm_s = s['base_cm'] + add_sm[tau]
        bo, no, ho = metrics(cm_o)
        bs, ns, hs = metrics(cm_s)
        print('%-6.2f %8.2fM %9d  N %.3f→%.3f  H %.3f→%.3f | 平滑 N %.3f H %.3f'
              % (tau, s['n_pt'] / 1e6, s['n_comp'], n0, no, h0, ho, ns, hs))
        print('       其中 base-novel 点 %.2fM, novel-novel 点 %.2fM'
              % (s['n_bn'] / 1e6, s['n_nn'] / 1e6))


if __name__ == '__main__':
    main()
