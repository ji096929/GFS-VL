"""第二环补全：全 10 类 × 30 eps，按 block 自己的 z_min 算离地高度。

输出每类：B 精确率、unannotated 占比、排除 unannotated 后精确率、
B_neg Top3 混淆类别，以及 B_pos / B_neg 的 z_rel、|n_z|、RGB 亮度。
"""
import json
import os

import numpy as np

CACHE = '/data01/jh/data/ocsb_cache'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
NAMES_PATH = '/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
PER_CLASS = 30


def pca_nz(pts):
    if len(pts) < 4:
        return 0.0
    p = pts - pts.mean(0)
    w, v = np.linalg.eigh(p.T @ p / len(pts))
    return abs(float(v[2, 0]))


def main():
    names = [l.strip() for l in open(NAMES_PATH)]
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))
    rows = []
    for c in NEW_IDS:
        eps = [m for m in metas if m['class'] == c][:PER_CLASS]
        b_tot = b_pos = b_un = 0
        z_pos, z_neg = [], []
        nz_pos, nz_neg = [], []
        br_pos, br_neg = [], []
        top3 = {}
        for m in eps:
            d = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
            xyz, rgb, lab, seg = [], [], [], []
            zmin, boff = [], []
            off = 0
            for i, nm in enumerate(m['u']):
                r = np.load(os.path.join(DATA, nm + '.npy'))
                n_b = len(r)
                xyz.append(r[:, 0:3].astype(np.float64))
                rgb.append(r[:, 3:6].astype(np.float64))
                lab.append(r[:, 6].astype(np.int64))
                seg.append(r[:, 7].astype(np.int64) + i * 10 ** 7)
                zmin.append(float(r[:, 2].min()))
                boff.append((off, off + n_b))
                off += n_b
            xyz = np.concatenate(xyz)
            rgb = np.concatenate(rgb)
            lab = np.concatenate(lab)
            seg = np.concatenate(seg)
            bid = np.zeros(len(xyz), dtype=np.int64)
            for i, (a, b) in enumerate(boff):
                bid[a:b] = i

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
                vals, cts = np.unique(la, return_counts=True)
                dom = int(vals[cts.argmax()])
                zr = float(xyz[sl, 2].mean() - zmin[int(bid[sl[0]])])
                nz = pca_nz(xyz[sl])
                br = float(rgb[sl].mean())
                b_tot += 1
                z_neg.append(zr); nz_neg.append(nz); br_neg.append(br)
                if dom == c:
                    b_pos += 1
                    z_pos.append(zr); nz_pos.append(nz); br_pos.append(br)
                elif dom == 0:
                    b_un += 1
                else:
                    dn = names[dom] if dom < len(names) else str(dom)
                    top3[dn] = top3.get(dn, 0) + 1
        t3 = sorted(top3.items(), key=lambda x: -x[1])[:3]
        rows.append({
            'class': names[c], 'n_eps': len(eps), 'B': b_tot, 'B_pos': b_pos,
            'prec': b_pos / max(b_tot, 1),
            'prec_anno': b_pos / max(b_tot - b_un, 1),
            'un_frac': b_un / max(b_tot, 1),
            'top3': [[k, v / max(b_tot, 1)] for k, v in t3],
            'z_pos': [float(np.mean(z_pos)), float(np.median(z_pos))],
            'z_neg': [float(np.mean(z_neg)), float(np.median(z_neg))],
            'nz_pos': [float(np.mean(nz_pos)), float(np.median(nz_pos))],
            'nz_neg': [float(np.mean(nz_neg)), float(np.median(nz_neg))],
            'br_pos': [float(np.mean(br_pos)), float(np.median(br_pos))],
            'br_neg': [float(np.mean(br_neg)), float(np.median(br_neg))],
        })
        print('%-12s done (%d eps)' % (names[c], len(eps)), flush=True)

    print('\n=== 1) B 精确率与混淆 ===')
    print('%-12s %5s %6s %6s %8s %10s %8s  top3'
          % ('class', 'eps', 'B', 'B_pos', 'prec', 'prec_anno', 'un'))
    for r in rows:
        print('%-12s %5d %6d %6d %8.3f %10.3f %7.1f%%  %s'
              % (r['class'], r['n_eps'], r['B'], r['B_pos'], r['prec'],
                 r['prec_anno'], 100 * r['un_frac'],
                 ', '.join('%s %.0f%%' % (k, 100 * p) for k, p in r['top3'])))

    print('\n=== 2) 物理特征（均值 / 中位）===')
    print('%-12s %18s %18s' % ('class', 'z_rel B_pos|B_neg', '|nz| B_pos|B_neg'))
    print('%-12s %18s %18s %18s' % ('', '', '', 'bright B_pos|B_neg'))
    for r in rows:
        print('%-12s %7.2f|%-7.2f %7.2f|%-7.2f %7.1f|%-7.1f'
              % (r['class'], r['z_pos'][0], r['z_neg'][0],
                 r['nz_pos'][0], r['nz_neg'][0],
                 r['br_pos'][0], r['br_neg'][0]))

    json.dump(rows, open(os.path.join(CACHE, 'ring2c.json'), 'w'), indent=1)
    print('->', os.path.join(CACHE, 'ring2c.json'))


if __name__ == '__main__':
    main()
