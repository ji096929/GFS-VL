"""导出 scene0606_00 整房间的 rgb / gt / tag 三版。

tag 来自该房间内 block_3 的 episode 区域判定，按坐标匹配映射回整房间：
  绿 B_pos / 红 B_neg / 黄 C_pos，块外点灰。
"""
import os

import numpy as np
from scipy.spatial import cKDTree

SCENE = '/data01/jh/data/ScanNet/scenes/data/scene0606_00.npy'
BLK = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
CACHE = '/data01/jh/data/ocsb_cache'
OUT = '/data01/jh/afford_diag/ocsb_logs'
U = ['scene0338_02_block_13', 'scene0606_00_block_3', 'scene0149_00_block_4']
C = 13
IDX = 7


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


s = np.load(SCENE)
print('scene', s.shape, 'rgb range', s[:, 3:6].min(), s[:, 3:6].max())
xyz = s[:, 0:3]
rgb = np.clip(s[:, 3:6], 0, 255)
lab = s[:, 6].astype(int)

# ---- 整房间 gt：counter 品红 + 暗底 ----
gt = np.clip(rgb * 0.35, 0, 255).astype(np.uint8)
gt[lab == C] = [255, 0, 255]
write_ply(os.path.join(OUT, 'room_scene0606_00_gt.ply'), xyz, gt)

# ---- 整房间 rgb ----
write_ply(os.path.join(OUT, 'room_scene0606_00_rgb.ply'),
          xyz, rgb.astype(np.uint8))

# ---- 块区域 tag 映射 ----
blocks = [np.load(os.path.join(BLK, n + '.npy')) for n in U]
seg = np.concatenate([b[:, 7].astype(np.int64) + i * 10 ** 7
                      for i, b in enumerate(blocks)])
uniq, inv = np.unique(seg, return_inverse=True)
n_u = len(uniq)
d = np.load(os.path.join(CACHE, 'val_%03d.npz' % IDX))
u_seg = d['u_seg'].astype(np.int64)
cnt = np.maximum(np.bincount(u_seg, minlength=n_u).astype(np.float64), 1)
q = np.bincount(u_seg, weights=d['u_p0'].astype(np.float64),
                minlength=n_u) / cnt
fg = np.bincount(u_seg, weights=d['u_y'].astype(np.float64),
                 minlength=n_u) / cnt
tag_u = np.zeros(len(seg), dtype=np.int8)
tag_u[np.isin(inv, np.where((q >= 0.6) & (fg > 0.5))[0])] = 1
tag_u[np.isin(inv, np.where((q >= 0.6) & (fg <= 0.5))[0])] = 2
tag_u[np.isin(inv, np.where((q < 0.5) & (fg > 0.5))[0])] = 3

starts = np.cumsum([0] + [len(b) for b in blocks])
b1 = blocks[1]                     # scene0606_00_block_3
t1 = tag_u[starts[1]:starts[2]]    # 该块的点级 tag
bs = b1[:, 7].astype(np.int64)
ss = s[:, 7].astype(np.int64)

# 块与整场景同超点体系（132 个块超点全在 scene 的 954 个里），
# 用 seg id 映射；块坐标 = scene 坐标 - scene_min，无需再做 KDTree。
tag_by_seg = {}
for g in np.unique(bs):
    tag_by_seg[int(g)] = int(t1[bs == g][0])
room_tag = np.array([tag_by_seg.get(int(g), 0) for g in ss], dtype=np.int8)
print('block segs %d  mapped %d' % (len(tag_by_seg),
                                    sum(1 for g in np.unique(ss)
                                        if int(g) in tag_by_seg)))
out = np.full((len(xyz), 3), 128, dtype=np.uint8)
out[room_tag == 1] = [0, 255, 0]
out[room_tag == 2] = [255, 0, 0]
out[room_tag == 3] = [255, 255, 0]
write_ply(os.path.join(OUT, 'room_scene0606_00_tag.ply'), xyz, out)
print('tag counts: G/R/Y = %d/%d/%d'
      % ((room_tag == 1).sum(), (room_tag == 2).sum(),
         (room_tag == 3).sum()))
print('->', OUT)
