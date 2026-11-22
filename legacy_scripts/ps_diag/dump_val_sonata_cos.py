"""为 312 val 场景预计算逐点 Sonata 对 45 个新类原型的余弦（仲裁用）。

输出 /data01/jh/data/val_sonata_cos/{scene}_cos.npy  float16[N,45]
点位顺序与 val/{scene}/coord.npy 一致。
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/ps_diag')
sys.path.insert(0, '/data01/jh/code/sonata')
sys.path.insert(0, '/data01/jh/code')
import sonata  # noqa: E402
from build_sonata_region import feat_sonata  # noqa: E402

NO, NN = 12, 45
DATA = '/data01/jh/data/scannet200_gfs/val'
PROTO = '/data01/jh/data/support_protos/sc200_regis_1_10_sonata.npy'
OUT = '/data01/jh/data/val_sonata_cos'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    pn = np.load(PROTO).astype(np.float32)
    pn = pn / (np.linalg.norm(pn, axis=1, keepdims=True) + 1e-8)
    cfg = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    model = sonata.load('/data01/jh/code/sonata/ckpt/sonata.pth',
                        custom_config=cfg).cuda().eval()

    scenes = sorted(os.listdir(DATA))[args.shard::args.shards]
    print('shard %d/%d: %d 场景' % (args.shard, args.shards, len(scenes)), flush=True)
    t0 = time.time()
    for i, sc in enumerate(scenes):
        p = os.path.join(DATA, sc)
        coord = np.load(p + '/coord.npy').astype(np.float32)
        color = np.load(p + '/color.npy').astype(np.float32)
        npath = p + '/normal.npy'
        normal = np.load(npath).astype(np.float32) if os.path.exists(npath) \
            else np.zeros_like(coord)
        f = feat_sonata(model, coord, color, normal)
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        cos = f @ pn.T
        np.save(os.path.join(OUT, sc + '_cos.npy'), cos.astype(np.float16))
        if (i + 1) % 50 == 0:
            print('  %d/%d %.0fs' % (i + 1, len(scenes), time.time() - t0), flush=True)
    print('-> shard %d done' % args.shard, flush=True)


if __name__ == '__main__':
    main()
