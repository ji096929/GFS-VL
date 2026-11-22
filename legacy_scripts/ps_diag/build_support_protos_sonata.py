"""Sonata 视角的 support 原型（sonata 环境运行）。

与 VLM 版 (build_support_protos.py) 完全相同的 split 与 GT 选点逻辑，只是把
特征换成 Sonata 的 1232 维几何特征。输出 {split}_sonata.npy，形状 (45, 1232)。
"""
import argparse
import os
import runpy
import sys

sys.path.insert(0, '/data01/jh/code/sonata')
sys.path.insert(0, '/data01/jh/code')

import numpy as np  # noqa: E402
import torch  # noqa: E402

import sonata  # noqa: E402
from build_sonata_region import estimate_normals, feat_sonata  # noqa: E402

GFS = '/data01/jh/code/GFS-VL'
META = os.path.join(GFS, 'pointcept/datasets/preprocessing/scannet/meta_data')
DATA = '/data01/jh/data/scannet200_gfs'
OUT = '/data01/jh/data/support_protos'
CKPT = '/data01/jh/code/sonata/ckpt/sonata.pth'
SPLITS = ['sc200_regis_1_%d.txt' % s for s in (10, 20, 30, 40, 50)] + \
         ['sc200_regis_5_%d.txt' % s for s in (10, 20, 30, 40, 50)]


def build_lookup():
    c = runpy.run_path(os.path.join(META, 'scannet200_constants.py'))
    base = list(c['CLASS_LABELS_BASE'])
    novel = [n for n in c['CLASS_LABELS_BASE_NOVEL'] if n not in base]
    names = base + novel
    labels = list(c['CLASS_LABELS_200'])
    lk = np.full(len(labels), -1, dtype=np.int32)
    for i, n in enumerate(labels):
        if n in names:
            lk[i] = names.index(n)
    return names, len(base), lk


NAMES, NB, LOOKUP = build_lookup()


def parse_regis(name):
    out = []
    for line in open(os.path.join(DATA, name)).read().splitlines():
        if not line.strip():
            continue
        orig, rel = line.split('\t')
        gfs = int(LOOKUP[int(orig)])
        sp, sc = rel.replace('\\', '/').split('/')[-2:]
        out.append((gfs, sp, sc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default='')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    cfg = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    model = sonata.load(CKPT, custom_config=cfg).cuda().eval()
    print('[sonata] ready', flush=True)

    nin = len(NAMES) - NB
    splits = ([args.splits if args.splits.endswith('.txt')
               else args.splits + '.txt'] if args.splits else SPLITS)
    cache = {}
    for regis in splits:
        acc = np.zeros((nin, 1232), dtype=np.float64)
        cnt = np.zeros(nin, dtype=np.float64)
        for gfs_i, sp, sc in parse_regis(regis):
            if gfs_i < NB:
                continue
            base = os.path.join(DATA, sp, sc)
            if (sp, sc) not in cache:
                coord = np.load(os.path.join(base, 'coord.npy')).astype(np.float32)
                color = np.load(os.path.join(base, 'color.npy')).astype(np.float32)
                npth = os.path.join(base, 'normal.npy')
                normal = (np.load(npth).astype(np.float32)
                          if os.path.exists(npth) else estimate_normals(coord))
                seg = np.load(os.path.join(base, 'segment200.npy')).reshape(-1)
                g = np.full(seg.shape, -1, np.int32)
                v = seg != -1
                g[v] = LOOKUP[seg[v]]
                cache[(sp, sc)] = (feat_sonata(model, coord, color, normal), g)
            f, g = cache[(sp, sc)]
            mask = g == gfs_i
            if not mask.any():
                continue
            idx = gfs_i - NB
            acc[idx] += f[mask].astype(np.float64).sum(0)
            cnt[idx] += mask.sum()
        proto = acc / np.maximum(cnt[:, None], 1)
        p = os.path.join(OUT, regis.replace('.txt', '_sonata.npy'))
        np.save(p, proto.astype(np.float32))
        print('%s: %s  scenes %d' % (p, proto.shape, len(cache)), flush=True)


if __name__ == '__main__':
    main()
