"""第二环：导出 8 个典型任务的 U 点云（原始 RGB + 区域标注）。

颜色约定：B_pos 绿 (0,255,0)，B_neg 红 (255,0,0)，
C_pos 黄 (255,255,0)，其余背景灰 (128,128,128)。
"""
import json
import os

import numpy as np

CACHE = '/data01/jh/data/ocsb_cache'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
OUT = os.path.join(CACHE, 'ply')
NAMES_PATH = '/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt'
CLASSES = [13, 19, 10, 3]   # counter, cabinet, bathtub, chair
PER_CLASS = 2


def write_ply(path, xyz, rgb):
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write('element vertex %d\n' % len(xyz))
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\n'
                'property uchar blue\n')
        f.write('end_header\n')
        for i in range(len(xyz)):
            f.write('%.4f %.4f %.4f %d %d %d\n'
                    % (xyz[i, 0], xyz[i, 1], xyz[i, 2],
                       int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])))


def main():
    os.makedirs(OUT, exist_ok=True)
    names = [l.strip() for l in open(NAMES_PATH)]
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))
    index = []
    for c in CLASSES:
        eps = [m for m in metas if m['class'] == c][:PER_CLASS]
        for m in eps:
            cname = names[c]
            xyz, rgb, segoff = [], [], []
            for i, nm in enumerate(m['u']):
                r = np.load(os.path.join(DATA, nm + '.npy'))
                xyz.append(r[:, 0:3].astype(np.float64))
                rgb.append(r[:, 3:6].astype(np.float64))
                segoff.append(r[:, 7].astype(np.int64) + i * 10 ** 7)
            xyz = np.concatenate(xyz)
            rgb = np.concatenate(rgb)
            seg = np.concatenate(segoff)
            uniq, inv = np.unique(seg, return_inverse=True)
            n_u = len(uniq)

            d = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
            u_seg = d['u_seg'].astype(np.int64)
            u_p0 = d['u_p0'].astype(np.float64)
            u_y = d['u_y'].astype(np.float64)
            cnt = np.maximum(
                np.bincount(u_seg, minlength=n_u).astype(np.float64), 1)
            q = np.bincount(u_seg, weights=u_p0, minlength=n_u) / cnt
            fg = np.bincount(u_seg, weights=u_y, minlength=n_u) / cnt

            B = q >= 0.6
            B_pos = B & (fg > 0.5)
            B_neg = B & (fg <= 0.5)
            C_pos = (q < 0.5) & (fg > 0.5)

            tag = np.zeros(len(xyz), dtype=np.int8)
            tag[np.isin(inv, np.where(B_pos)[0])] = 1
            tag[np.isin(inv, np.where(B_neg)[0])] = 2
            tag[np.isin(inv, np.where(C_pos)[0])] = 3
            lab_rgb = np.full_like(rgb, 128)
            lab_rgb[tag == 1] = [0, 255, 0]
            lab_rgb[tag == 2] = [255, 0, 0]
            lab_rgb[tag == 3] = [255, 255, 0]

            base = '%s_%03d' % (cname, m['idx'])
            write_ply(os.path.join(OUT, base + '_rgb.ply'), xyz, rgb)
            write_ply(os.path.join(OUT, base + '_tag.ply'), xyz, lab_rgb)
            index.append({'file': base, 'class': cname, 'idx': m['idx'],
                          'u': m['u'], 'query': m['query'],
                          'support': m['support'], 'n_pts': int(len(xyz)),
                          'n_B_pos': int(B_pos.sum()),
                          'n_B_neg': int(B_neg.sum()),
                          'n_C_pos': int(C_pos.sum())})
            print('  %s  pts=%d B+%d B-%d C+%d'
                  % (base, len(xyz), B_pos.sum(), B_neg.sum(), C_pos.sum()),
                  flush=True)
    json.dump(index, open(os.path.join(OUT, 'index.json'), 'w'), indent=1)
    print('exported', len(index), '->', OUT)


if __name__ == '__main__':
    main()
