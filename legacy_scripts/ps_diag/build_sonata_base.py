"""用 Sonata 池化 base 预测区的区域特征（AI 新增点的主要来源）。

与 build_sonata_region.py 完全相同的特征口径，只是输入换成 base_index
（VLM 判成 12 个基类之一的连通分量），输出 sonata_base_{split}_s{N}.pt。
"""
import argparse
import os
import sys

sys.path.insert(0, '/data01/jh/code/sonata')
sys.path.insert(0, '/data01/jh/code')
sys.path.insert(0, '/data01/jh/code/ps_diag')

import numpy as np  # noqa: E402
import torch  # noqa: E402

import sonata  # noqa: E402
from build_sonata_region import feat_sonata, estimate_normals  # noqa: E402

DATA = '/data01/jh/data/scannet200_gfs'
IDX = '/data01/jh/data/base_index'
OUT = '/data01/jh/data/sonata_base'
CKPT = '/data01/jh/code/sonata/ckpt/sonata.pth'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--split', default='train')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    p = os.path.join(IDX, 'base_index_%s_s%d.pt' % (args.split, args.shard))
    d = torch.load(p, map_location='cpu')
    print('shard %d scenes %d' % (args.shard, len(d)), flush=True)

    cfg = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    model = sonata.load(CKPT, custom_config=cfg).cuda().eval()

    out = {}
    for i, (sc, r) in enumerate(sorted(d.items())):
        base = os.path.join(DATA, args.split, sc)
        coord = np.load(os.path.join(base, 'coord.npy')).astype(np.float32)
        color = np.load(os.path.join(base, 'color.npy')).astype(np.float32)
        npath = os.path.join(base, 'normal.npy')
        normal = np.load(npath).astype(np.float32) if os.path.exists(npath) \
            else estimate_normals(coord)
        f = feat_sonata(model, coord, color, normal)
        if len(f) != len(r['pred']):
            print('  !! %s 点数不一致 %d vs %d' % (sc, len(f), len(r['pred'])))
            continue
        off, idx = r['off'], r['idx']
        rows = [f[idx[off[k]:off[k + 1]]].mean(0) for k in range(len(off) - 1)]
        feats = np.stack(rows).astype(np.float16) if rows else \
            np.zeros((0, f.shape[1]), np.float16)
        out[sc] = dict(feats=feats, off=np.array(off, dtype=np.int32),
                       pred=r['pred'][off[:-1]] if len(off) > 1
                       else np.empty(0, np.int16))
        if (i + 1) % 50 == 0:
            print('  %d/%d' % (i + 1, len(d)), flush=True)

    q = os.path.join(OUT, 'sonata_base_%s_s%d.pt' % (args.split, args.shard))
    torch.save(out, q)
    print('->', q, flush=True)


if __name__ == '__main__':
    main()
