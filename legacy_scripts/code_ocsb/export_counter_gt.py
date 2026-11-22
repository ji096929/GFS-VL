"""导出 counter_007 的 U 池真值上色版：counter 品红，其余灰/暗原始色。"""
import os

import numpy as np

DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
OUT = '/data01/jh/afford_diag/ocsb_logs'
NAMES = ['scene0338_02_block_13', 'scene0606_00_block_3',
         'scene0149_00_block_4']
C = 13  # counter


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


xyz, rgb, lab = [], [], []
for n in NAMES:
    a = np.load(os.path.join(DATA, n + '.npy'))
    xyz.append(a[:, 0:3])
    rgb.append(a[:, 3:6])
    lab.append(a[:, 6].astype(int))
    m = lab[-1] == C
    print('%-22s N=%6d counter=%5d (%.2f%%) z=[%.2f, %.2f]'
          % (n, len(a), m.sum(), 100 * m.mean(),
             a[m, 2].min() if m.any() else -1,
             a[m, 2].max() if m.any() else -1))
xyz = np.concatenate(xyz)
rgb = np.concatenate(rgb)
lab = np.concatenate(lab)
m = lab == C
print('total %d  counter %d (%.2f%%)' % (len(xyz), m.sum(), 100 * m.mean()))

# 版本 A：counter 品红，其余灰
out = np.full((len(xyz), 3), 128, dtype=np.uint8)
out[m] = [255, 0, 255]
write_ply(os.path.join(OUT, 'counter_007_gt.ply'), xyz, out)

# 版本 B：counter 品红，其余原始 rgb 压暗（保留材质可读性）
out2 = np.clip(rgb * 0.35, 0, 255).astype(np.uint8)
out2[m] = [255, 0, 255]
write_ply(os.path.join(OUT, 'counter_007_gt_rgb.ply'), xyz, out2)
print('->', OUT)
