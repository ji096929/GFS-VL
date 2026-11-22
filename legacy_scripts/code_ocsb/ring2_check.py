"""核查 ring2 几何统计的 4 处疑点。"""
import json
import os

import numpy as np

CACHE = '/data01/jh/data/ocsb_cache'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
NAMES_PATH = '/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt'


def main():
    names = [l.strip() for l in open(NAMES_PATH)]
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))
    eps = [m for m in metas if m['class'] == 13][:5]   # counter

    # ---------- 1) 特征值顺序 ----------
    print('=== 1) 特征值顺序核查 ===')
    d0 = np.load(os.path.join(CACHE, 'val_%03d.npz' % eps[0]['idx']))
    nm = eps[0]['u'][0]
    r = np.load(os.path.join(DATA, nm + '.npy'))
    seg = r[:, 7].astype(np.int64)
    uniq, inv = np.unique(seg, return_inverse=True)
    for r_i in range(3):
        pts = r[inv == r_i, 0:3].astype(np.float64)
        if len(pts) < 10:
            continue
        p = pts - pts.mean(0)
        cov = p.T @ p / len(pts)
        w, v = np.linalg.eigh(cov)
        l3, l2, l1 = w[0], w[1], w[2]
        print('  sp %d n=%d  eig(asc) %.3e %.3e %.3e' % (r_i, len(pts), l3, l2, l1))
        print('    planarity (l2-l3)/l1 = %.3f   linearity (l1-l2)/l1 = %.3f'
              % ((l2 - l3) / l1, (l1 - l2) / l1))
        print('    -> 若超点是平面薄片，应线性度高、平面度低（薄片=一维主导）')

    # ---------- 2) 边界坐标 ----------
    print('\n=== 2) 边界坐标核查 ===')
    xyz_all, segoff = [], []
    bounds = []
    for i, nm2 in enumerate(eps[0]['u']):
        rr = np.load(os.path.join(DATA, nm2 + '.npy'))
        x, y = rr[:, 0], rr[:, 1]
        bounds.append((float(x.min()), float(x.max()),
                       float(y.min()), float(y.max())))
        print('  block %d %s  x [%.2f, %.2f]  y [%.2f, %.2f]'
              % (i, nm2, x.min(), x.max(), y.min(), y.max()))

    # ---------- 3) 点数均值/中位数 ----------
    print('\n=== 3) 点数与体积分布 ===')
    n_pts, vols = [], []
    for m in eps:
        d = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
        xyz, segoff = [], []
        for i, nm2 in enumerate(m['u']):
            rr = np.load(os.path.join(DATA, nm2 + '.npy'))
            xyz.append(rr[:, 0:3].astype(np.float64))
            segoff.append(rr[:, 7].astype(np.int64) + i * 10 ** 7)
        xyz = np.concatenate(xyz)
        seg = np.concatenate(segoff)
        uniq, inv = np.unique(seg, return_inverse=True)
        n_u = len(uniq)
        cnt = np.bincount(inv, minlength=n_u)
        for r_i in range(n_u):
            sl = inv == r_i
            if sl.sum() < 4:
                continue
            pts = xyz[sl]
            mn, mx = pts.min(0), pts.max(0)
            d3 = mx - mn
            n_pts.append(int(sl.sum()))
            vols.append(float(d3[0] * d3[1] * d3[2]))
    n_pts = np.array(n_pts)
    vols = np.array(vols)
    print('  n_pts: mean %.1f  median %.1f  p25 %.1f  p75 %.1f'
          % (n_pts.mean(), np.median(n_pts),
             np.percentile(n_pts, 25), np.percentile(n_pts, 75)))
    print('  vol  : mean %.5f  median %.5f  p25 %.5f  p75 %.5f'
          % (vols.mean(), np.median(vols),
             np.percentile(vols, 25), np.percentile(vols, 75)))
    print('  说明: counter_007 总点数/超点数 = %.0f' % (25532 / 50))

    # ---------- 4) unannotated 与 B 精确率 ----------
    print('\n=== 4) unannotated 影响 ===')
    b_tot = b_pos = b_pos_anno = 0
    un_high = un_tot = 0
    for m in eps:
        d = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
        xyz, lab, segoff = [], [], []
        for i, nm2 in enumerate(m['u']):
            rr = np.load(os.path.join(DATA, nm2 + '.npy'))
            xyz.append(rr[:, 0:3].astype(np.float64))
            lab.append(rr[:, 6].astype(np.int64))
            segoff.append(rr[:, 7].astype(np.int64) + i * 10 ** 7)
        xyz = np.concatenate(xyz)
        lab = np.concatenate(lab)
        seg = np.concatenate(segoff)
        uniq, inv = np.unique(seg, return_inverse=True)
        n_u = len(uniq)
        u_seg = d['u_seg'].astype(np.int64)
        u_p0 = d['u_p0'].astype(np.float64)
        cnt = np.maximum(np.bincount(u_seg, minlength=n_u).astype(np.float64), 1)
        q = np.bincount(u_seg, weights=u_p0, minlength=n_u) / cnt
        B = q >= 0.6
        zmin = xyz[:, 2].min()
        for r_i in np.where(B)[0]:
            sl = inv == r_i
            la = lab[sl]
            vals, cnts = np.unique(la, return_counts=True)
            dom = int(vals[cnts.argmax()])
            b_tot += 1
            if dom == 13:
                b_pos += 1
            elif dom != 0:
                b_pos_anno += 1
            if dom == 0:
                un_tot += 1
                zr = float(xyz[sl, 2].mean() - zmin)
                if zr > 0.8:
                    un_high += 1
    print('  B total %d  B_pos %d  precision %.3f' % (b_tot, b_pos, b_pos / max(b_tot, 1)))
    print('  排除 unannotated 后: B total %d  pos %d  precision %.3f'
          % (b_tot - un_tot, b_pos_anno, b_pos_anno / max(b_tot - un_tot, 1)))
    print('  unannotated 占 B 的 %.1f%%，其中 z_rel>0.8 的占 %.1f%%'
          % (100 * un_tot / max(b_tot, 1), 100 * un_high / max(un_tot, 1)))


if __name__ == '__main__':
    main()
