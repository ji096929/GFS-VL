"""用原始 vh_clean_2.ply（164k 点）导出 scene0606_00 的 gt / tag。

原始 ply 是 binary_little_endian，顶点 x,y,z float32 + rgba uchar。
标签与超点从 scenes/data 的 voxel 0.02 npy 按最近邻映射（<0.01m）。
"""
import os

import numpy as np
from scipy.spatial import cKDTree

PLY = ('/data01/jh/data/scannet_raw/scans/scene0606_00/'
       'scene0606_00_vh_clean_2.ply')
NPY = '/data01/jh/data/ScanNet/scenes/data/scene0606_00.npy'
BLK = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
CACHE = '/data01/jh/data/ocsb_cache'
OUT = '/data01/jh/afford_diag/ocsb_logs'
U = ['scene0338_02_block_13', 'scene0606_00_block_3', 'scene0149_00_block_4']
C = 13
IDX = 7


def read_ply_bin(path):
    with open(path, 'rb') as f:
        n = 0
        props = []
        while True:
            line = f.readline().decode('ascii', 'ignore').strip()
            if line.startswith('element vertex'):
                n = int(line.split()[-1])
            elif line.startswith('property'):
                props.append(line.split()[-1])
            elif line.startswith('end_header'):
                break
        dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                       ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1')])
        data = np.frombuffer(f.read(), dtype=dt, count=n)
    return data, props


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


raw, props = read_ply_bin(PLY)
print('raw', raw.shape, 'props', props)
xyz = np.stack([raw['x'], raw['y'], raw['z']], 1).astype(np.float64)
rgb = np.stack([raw['r'], raw['g'], raw['b']], 1).astype(np.float64)

s = np.load(NPY)
tree = cKDTree(s[:, 0:3])
dist, nn = tree.query(xyz, k=1)
ok = dist < 0.01
print('map raw->npy: %d/%d (%.1f%%)' % (ok.sum(), len(ok), 100 * ok.mean()))
lab = np.full(len(xyz), -1, dtype=np.int64)
lab[ok] = s[nn[ok], 6].astype(np.int64)
seg_raw = np.full(len(xyz), -1, dtype=np.int64)
seg_raw[ok] = s[nn[ok], 7].astype(np.int64)

gt = np.clip(rgb * 0.35, 0, 255).astype(np.uint8)
gt[lab == C] = [255, 0, 255]
write_ply(os.path.join(OUT, 'room_scene0606_00_raw_gt.ply'), xyz, gt)
print('counter points', int((lab == C).sum()))

# 块 tag -> 场景 seg
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
b1 = blocks[1]
t1 = tag_u[starts[1]:starts[2]]
bs = b1[:, 7].astype(np.int64)
tag_by_seg = {}
for g in np.unique(bs):
    tag_by_seg[int(g)] = int(t1[bs == g][0])

room_tag = np.zeros(len(xyz), dtype=np.int8)
for i in range(len(xyz)):
    room_tag[i] = tag_by_seg.get(int(seg_raw[i]), 0)
out = np.full((len(xyz), 3), 128, dtype=np.uint8)
out[room_tag == 1] = [0, 255, 0]
out[room_tag == 2] = [255, 0, 0]
out[room_tag == 3] = [255, 255, 0]
write_ply(os.path.join(OUT, 'room_scene0606_00_raw_tag.ply'), xyz, out)
print('tag G/R/Y = %d/%d/%d'
      % ((room_tag == 1).sum(), (room_tag == 2).sum(), (room_tag == 3).sum()))
print('->', OUT)
