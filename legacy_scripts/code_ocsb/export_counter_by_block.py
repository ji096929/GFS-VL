"""按 U 块拆分导出 counter_007 的三个房间：rgb / gt / tag 三版。

tag 复现 ring2_export_ply 的超点级判定：
  B_pos 绿 (0,255,0)  B_neg 红 (255,0,0)  C_pos 黄 (255,255,0)  其余灰
gt: counter(13) 品红 (255,0,255)，其余原始 rgb 压暗
"""
import os

import numpy as np

DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
CACHE = '/data01/jh/data/ocsb_cache'
OUT = '/data01/jh/afford_diag/ocsb_logs'
NAMES = ['scene0338_02_block_13', 'scene0606_00_block_3',
         'scene0149_00_block_4']
IDX = 7
C = 13


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


# 与 ring2_export_ply 完全一致的全局超点映射
xyz, rgb, lab, segoff = [], [], [], []
for i, n in enumerate(NAMES):
    a = np.load(os.path.join(DATA, n + '.npy'))
    xyz.append(a[:, 0:3])
    rgb.append(a[:, 3:6])
    lab.append(a[:, 6].astype(int))
    segoff.append(a[:, 7].astype(np.int64) + i * 10 ** 7)
xyz = np.concatenate(xyz)
rgb = np.concatenate(rgb)
lab = np.concatenate(lab)
seg = np.concatenate(segoff)
uniq, inv = np.unique(seg, return_inverse=True)
n_u = len(uniq)

d = np.load(os.path.join(CACHE, 'val_%03d.npz' % IDX))
u_seg = d['u_seg'].astype(np.int64)
u_p0 = d['u_p0'].astype(np.float64)
u_y = d['u_y'].astype(np.float64)
cnt = np.maximum(np.bincount(u_seg, minlength=n_u).astype(np.float64), 1)
q = np.bincount(u_seg, weights=u_p0, minlength=n_u) / cnt
fg = np.bincount(u_seg, weights=u_y, minlength=n_u) / cnt
B_pos = (q >= 0.6) & (fg > 0.5)
B_neg = (q >= 0.6) & (fg <= 0.5)
C_pos = (q < 0.5) & (fg > 0.5)
tag = np.zeros(len(xyz), dtype=np.int8)
tag[np.isin(inv, np.where(B_pos)[0])] = 1
tag[np.isin(inv, np.where(B_neg)[0])] = 2
tag[np.isin(inv, np.where(C_pos)[0])] = 3

tag_rgb = np.full_like(rgb, 128)
tag_rgb[tag == 1] = [0, 255, 0]
tag_rgb[tag == 2] = [255, 0, 0]
tag_rgb[tag == 3] = [255, 255, 0]

gt_rgb = np.clip(rgb * 0.35, 0, 255)
gt_rgb[lab == C] = [255, 0, 255]

starts = np.cumsum([0] + [len(np.load(os.path.join(DATA, n + '.npy'),
                                     mmap_mode='r')) for n in NAMES])
for i, n in enumerate(NAMES):
    sl = slice(starts[i], starts[i + 1])
    base = 'counter_007_u%d_%s' % (i, n)
    write_ply(os.path.join(OUT, base + '_rgb.ply'), xyz[sl], rgb[sl])
    write_ply(os.path.join(OUT, base + '_gt.ply'), xyz[sl], gt_rgb[sl])
    write_ply(os.path.join(OUT, base + '_tag.ply'), xyz[sl], tag_rgb[sl])
    m = lab[sl] == C
    print('%-40s N=%6d counter=%5d (%.2f%%)  tag G/R/Y=%d/%d/%d'
          % (base, len(m), m.sum(), 100 * m.mean(),
             (tag[sl] == 1).sum(), (tag[sl] == 2).sum(),
             (tag[sl] == 3).sum()))
print('->', OUT)
