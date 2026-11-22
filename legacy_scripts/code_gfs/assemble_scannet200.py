"""从服务器现有 ply + 本地传来的 segment200 组装 pointcept 格式的 ScanNet200。

点序已验证：本地 coord.npy 与服务器 ply 顶点逐点一致（81,369 点，首点相同）。
normal 置零（配置 in_channels=6 只用 coord+color）。
"""
import os
import sys
import zipfile
from multiprocessing import Pool

import numpy as np
from plyfile import PlyData

RAW = '/data01/jh/data/scannet_raw/scans'
OUT = '/data01/jh/data/scannet200_gfs'
ZIP = '/data01/jh/tmp/seg200.zip'
TMP = '/data01/jh/tmp/seg200'


def process(args):
    split, scene = args
    seg_path = os.path.join(TMP, split, scene, 'segment200.npy')
    if not os.path.exists(seg_path):
        return (scene, 'no_seg')
    ply = os.path.join(RAW, scene, scene + '_vh_clean_2.ply')
    if not os.path.exists(ply):
        return (scene, 'no_ply')
    v = PlyData.read(ply)['vertex']
    coord = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float32)
    color = np.stack([v['red'], v['green'], v['blue']], 1).astype(np.float32)
    seg = np.load(seg_path).astype(np.int32).reshape(-1)
    if len(seg) != len(coord):
        return (scene, 'len %d vs %d' % (len(seg), len(coord)))
    out = os.path.join(OUT, split, scene)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, 'coord.npy'), coord)
    np.save(os.path.join(out, 'color.npy'), color)
    np.save(os.path.join(out, 'normal.npy'), np.zeros_like(coord))
    np.save(os.path.join(out, 'segment200.npy'), seg)
    return (scene, 'ok')


def main():
    if not os.path.exists(TMP):
        os.makedirs(TMP)
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(TMP)
        print('extracted', flush=True)
    jobs = []
    for split in ('train', 'val'):
        d = os.path.join(TMP, split)
        if not os.path.isdir(d):
            continue
        for scene in sorted(os.listdir(d)):
            jobs.append((split, scene))
    print('jobs', len(jobs), flush=True)
    with Pool(16) as p:
        res = p.map(process, jobs)
    ok = sum(1 for r in res if r[1] == 'ok')
    bad = [r for r in res if r[1] != 'ok']
    print('ok %d  bad %d' % (ok, len(bad)))
    for r in bad[:10]:
        print('  bad:', r)


if __name__ == '__main__':
    main()
