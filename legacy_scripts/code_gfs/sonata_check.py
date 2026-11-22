"""Sonata 环境验证（headless，不依赖 open3d 可视化）。

用本地 ScanNet block 构造点云（coord + color + 估计法向），
跑 Sonata forward，输出逐点特征统计并保存 PCA 着色 ply。

用法：
  CUDA_VISIBLE_DEVICES=0 python sonata_check.py [--block <npy>] [--out <ply>]
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

sys.path.insert(0, '/data01/jh/code/sonata')
import sonata  # noqa: E402

CKPT = '/data01/jh/code/sonata/ckpt/sonata.pth'
SMALL = '/data01/jh/code/sonata/ckpt/sonata_small.pth'
BLOCK = '/data01/jh/data/ScanNet/blocks_bs1_s1/data/scene0011_00_block_0.npy'


def estimate_normals(coord, k=30):
    """邻域 PCA，取最小特征值方向为法向，并统一朝上。"""
    n = len(coord)
    k = min(k, n)
    _, idx = cKDTree(coord).query(coord, k=k)
    nb = coord[idx]                                   # (N, k, 3)
    nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', nb, nb)           # (N, 3, 3)
    w, v = np.linalg.eigh(cov)
    nrm = v[:, :, 0]                                  # 最小特征值
    nrm[nrm[:, 2] < 0] *= -1.0
    return nrm.astype(np.float32)


def pca_color(feat, brightness=1.2):
    u, s, v = torch.pca_lowrank(feat, center=True, q=6, niter=5)
    proj = feat @ v
    proj = proj[:, :3] * 0.6 + proj[:, 3:6] * 0.4
    lo = proj.min(dim=-2, keepdim=True)[0]
    hi = proj.max(dim=-2, keepdim=True)[0]
    col = (proj - lo) / torch.clamp(hi - lo, min=1e-6) * brightness
    return col.clamp(0.0, 1.0)


def write_ply(path, xyz, rgb):
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write('element vertex %d\n' % len(xyz))
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for i in range(len(xyz)):
            f.write('%.4f %.4f %.4f %d %d %d\n'
                    % (xyz[i, 0], xyz[i, 1], xyz[i, 2],
                       int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', default=BLOCK)
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--out', default='/data01/jh/data/sonata_check.ply')
    ap.add_argument('--no-color', action='store_true')
    args = ap.parse_args()

    custom = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    t0 = time.time()
    model = sonata.load(args.ckpt, custom_config=custom).cuda().eval()
    print('[load] %.1fs  ckpt=%s' % (time.time() - t0, os.path.basename(args.ckpt)))
    n_par = sum(p.numel() for p in model.parameters())
    print('[load] params = %.2f M' % (n_par / 1e6))

    d = np.load(args.block)
    coord = d[:, 0:3].astype(np.float32)
    color = d[:, 3:6].astype(np.float32)
    print('[data] %s  N=%d  xyz[%g,%g]  rgb[%g,%g]'
          % (os.path.basename(args.block), len(coord),
             coord.min(), coord.max(), color.min(), color.max()))

    t0 = time.time()
    normal = estimate_normals(coord)
    print('[data] normals %.1fs' % (time.time() - t0))

    point = dict(coord=coord, color=color, normal=normal)
    transform = sonata.transform.default()
    point = transform(point)
    print('[tf] keys=%s' % sorted(point.keys()))
    print('[tf] coord=%s feat=%s inverse=%s'
          % (tuple(point['coord'].shape), tuple(point['feat'].shape),
             tuple(point['inverse'].shape)))

    with torch.inference_mode():
        for k in list(point.keys()):
            if isinstance(point[k], torch.Tensor):
                point[k] = point[k].cuda(non_blocking=True)
        t0 = time.time()
        point = model(point)
        torch.cuda.synchronize()
        print('[fwd] %.2fs' % (time.time() - t0))

        for _ in range(2):
            assert 'pooling_parent' in point.keys()
            parent = point.pop('pooling_parent')
            inv = point.pop('pooling_inverse')
            parent.feat = torch.cat([parent.feat, point.feat[inv]], dim=-1)
            point = parent
        while 'pooling_parent' in point.keys():
            parent = point.pop('pooling_parent')
            inv = point.pop('pooling_inverse')
            parent.feat = point.feat[inv]
            point = parent

        feat_ds = point.feat
        print('[out] downsampled feat %s' % (tuple(feat_ds.shape),))
        feat = feat_ds[point.inverse]
        print('[out] full-res feat %s  range[%.3f, %.3f]  mean=%.4f std=%.4f'
              % (tuple(feat.shape), float(feat.min()), float(feat.max()),
                 float(feat.mean()), float(feat.std())))

        col = pca_color(feat).cpu().numpy()

    xyz_out = point.coord.cpu().numpy()
    rgb_out = (col * 255).clip(0, 255).astype(np.uint8)
    write_ply(args.out, xyz_out, rgb_out)
    print('[save] %s  (%d points)' % (args.out, len(xyz_out)))


if __name__ == '__main__':
    main()
