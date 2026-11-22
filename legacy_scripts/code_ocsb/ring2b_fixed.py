"""10 类 unannotated 统计 + B_pos/B_neg 物理离地高度（按 block 分别算地面）。

z_rel 定义：超点内点的平均 z 减去【该超点所属 block 自己的】最低 z，单位米。
之前误用 3 个 U block 拼接后的全局最小 z，而 3 个 block 来自不同场景、
坐标系不同，导致高度被严重高估。
"""
import json
import os

import numpy as np

CACHE = '/data01/jh/data/ocsb_cache'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
NAMES_PATH = '/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
PER_CLASS = 15


def main():
    names = [l.strip() for l in open(NAMES_PATH)]
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))
    rows = []
    for c in NEW_IDS:
        eps = [m for m in metas if m['class'] == c][:PER_CLASS]
        b_tot = b_pos = b_un = 0
        z_pos, z_neg, z_un = [], [], []
        for m in eps:
            d = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
            xyz, lab, segoff = [], [], []
            block_zmin, block_off = [], []
            off = 0
            for i, nm in enumerate(m['u']):
                r = np.load(os.path.join(DATA, nm + '.npy'))
                n_b = len(r)
                xyz.append(r[:, 0:3].astype(np.float64))
                lab.append(r[:, 6].astype(np.int64))
                segoff.append(r[:, 7].astype(np.int64) + i * 10 ** 7)
                block_zmin.append(float(r[:, 2].min()))
                block_off.append((off, off + n_b))
                off += n_b
            xyz = np.concatenate(xyz)
            lab = np.concatenate(lab)
            seg = np.concatenate(segoff)
            block_id = np.zeros(len(xyz), dtype=np.int64)
            for i, (a, b) in enumerate(block_off):
                block_id[a:b] = i

            uniq, inv = np.unique(seg, return_inverse=True)
            n_u = len(uniq)
            u_seg = d['u_seg'].astype(np.int64)
            u_p0 = d['u_p0'].astype(np.float64)
            cnt = np.maximum(
                np.bincount(u_seg, minlength=n_u).astype(np.float64), 1)
            q = np.bincount(u_seg, weights=u_p0, minlength=n_u) / cnt
            B = q >= 0.6

            order = np.argsort(inv, kind='stable')
            bounds = np.searchsorted(inv[order], np.arange(n_u + 1))
            for r in np.where(B)[0]:
                sl = order[bounds[r]:bounds[r + 1]]
                if len(sl) < 4:
                    continue
                la = lab[sl]
                vals, cnts = np.unique(la, return_counts=True)
                dom = int(vals[cnts.argmax()])
                # 用该超点所属 block 自己的地面
                zmin = block_zmin[int(block_id[sl[0]])]
                zr = float(xyz[sl, 2].mean() - zmin)
                b_tot += 1
                z_neg.append(zr)
                if dom == c:
                    b_pos += 1
                    z_pos.append(zr)
                elif dom == 0:
                    b_un += 1
                    z_un.append(zr)
        rows.append({
            'class': names[c], 'n_eps': len(eps), 'B': b_tot, 'B_pos': b_pos,
            'prec': b_pos / max(b_tot, 1),
            'prec_anno': b_pos / max(b_tot - b_un, 1),
            'un_frac': b_un / max(b_tot, 1),
            'z_pos_mean': float(np.mean(z_pos)) if z_pos else 0,
            'z_pos_med': float(np.median(z_pos)) if z_pos else 0,
            'z_neg_mean': float(np.mean(z_neg)) if z_neg else 0,
            'z_neg_med': float(np.median(z_neg)) if z_neg else 0,
            'z_un_mean': float(np.mean(z_un)) if z_un else 0,
            'z_un_med': float(np.median(z_un)) if z_un else 0,
        })
        print('%-12s done (%d eps)' % (names[c], len(eps)), flush=True)

    print('\n=== B 精确率与 unannotated 占比 ===')
    print('%-12s %5s %6s %6s %9s %9s %9s'
          % ('class', 'eps', 'B', 'B_pos', 'prec', 'prec_anno', 'un_frac'))
    for r in rows:
        print('%-12s %5d %6d %6d %9.3f %9.3f %8.1f%%'
              % (r['class'], r['n_eps'], r['B'], r['B_pos'], r['prec'],
                 r['prec_anno'], 100 * r['un_frac']))

    print('\n=== 物理离地高度 z_rel（米，按 block 自己的地面）===')
    print('%-12s %18s %18s %18s'
          % ('class', 'B_pos mean/med', 'B_neg mean/med', 'unanno mean/med'))
    for r in rows:
        print('%-12s %8.2f / %-8.2f %8.2f / %-8.2f %8.2f / %-8.2f'
              % (r['class'], r['z_pos_mean'], r['z_pos_med'],
                 r['z_neg_mean'], r['z_neg_med'],
                 r['z_un_mean'], r['z_un_med']))

    json.dump(rows, open(os.path.join(CACHE, 'ring2b_fixed.json'), 'w'),
              indent=1)
    print('->', os.path.join(CACHE, 'ring2b_fixed.json'))


if __name__ == '__main__':
    main()
