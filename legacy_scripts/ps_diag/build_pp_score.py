"""逐点 Sonata 打分：score_i = cos(f_sonata[i], proto[psai_i])，供软权重使用。

输出 /data01/jh/data/pp_score/{scene}_s.npy  float16[N]（无伪标签点 = nan）
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/ps_diag')
sys.path.insert(0, '/data01/jh/code/sonata')
sys.path.insert(0, '/data01/jh/code')
import sonata  # noqa: E402
import vlm_common as V  # noqa: E402
from build_sonata_region import feat_sonata  # noqa: E402

NO = 12
DATA = '/data01/jh/data/scannet200_gfs/train'
ONLINE = '/data01/jh/data/online_pl'
PROTO = '/data01/jh/data/support_protos/sc200_regis_1_10_sonata.npy'
OUT = '/data01/jh/data/pp_score'


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

    scenes = sorted(os.listdir(ONLINE))[args.shard::args.shards]
    print('shard %d/%d: %d 场景' % (args.shard, args.shards, len(scenes)), flush=True)
    for i, fn in enumerate(scenes):
        sc = fn[:-4]
        z = np.load(os.path.join(ONLINE, fn))
        pl = z['psai']
        base = os.path.join(DATA, sc)
        coord = np.load(base + '/coord.npy').astype(np.float32)
        color = np.load(base + '/color.npy').astype(np.float32)
        npth = base + '/normal.npy'
        normal = np.load(npth).astype(np.float32) if os.path.exists(npth) \
            else np.zeros_like(coord)
        f = feat_sonata(model, coord, color, normal)
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        s = np.full(len(pl), np.nan, dtype=np.float32)
        sel = pl >= NO
        if sel.any():
            idx = np.where(sel)[0]
            s[idx] = np.einsum('ij,ij->i', f[idx], pn[pl[idx] - NO])
        np.save(os.path.join(OUT, sc + '_s.npy'), s.astype(np.float16))
        if (i + 1) % 50 == 0:
            print('  %d/%d' % (i + 1, len(scenes)), flush=True)
    print('-> shard %d done' % args.shard, flush=True)


if __name__ == '__main__':
    main()
