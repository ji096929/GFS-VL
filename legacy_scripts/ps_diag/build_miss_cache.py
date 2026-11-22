"""缓存 VLM 漏检区（pred == -1）的几何区域特征。

region_cache 只收「预测为 novel 类」的区域，覆盖不到 PS 拒绝池之外的空白区。
这里补上 pred == -1 的部分：这些点 VLM 没有给出任何类，是真正的待填充区。

输出结构同 region_cache，但区域级 pred 恒为 -1（仅占位），靠目录区分来源。
"""
import argparse
import os
import sys

sys.path.insert(0, '/data01/jh/code/ps_diag')

import numpy as np  # noqa: E402
import torch  # noqa: E402

import vlm_common as V  # noqa: E402
from build_region_cache import (DATA, MIN_PTS, connected_components,
                                load_scene)  # noqa: E402

OUT = '/data01/jh/data/miss_cache'


def process_scene(model, cfg, split, scene):
    coord, color, gfs = load_scene(split, scene)
    feats, preds = V.vlm_forward(model, cfg, coord, color)
    feats_t = torch.as_tensor(feats)
    n_novel = len(V.CLASS_NAMES) - V.NUM_BASE

    mask = preds == -1
    if int(mask.sum()) < MIN_PTS:
        return None
    _, comps = connected_components(mask, coord)
    rows = []
    for comp in comps:
        if len(comp) < MIN_PTS:
            continue
        f_mean = feats_t[comp].mean(0).numpy()
        gt = gfs[comp]
        hist_n = np.bincount(gt[gt >= V.NUM_BASE] - V.NUM_BASE,
                             minlength=n_novel).astype(np.int32)
        hist_b = np.bincount(gt[(gt >= 0) & (gt < V.NUM_BASE)],
                             minlength=V.NUM_BASE).astype(np.int32)
        n_ign = int((gt < 0).sum())
        rows.append((len(comp), f_mean, hist_n, hist_b, n_ign))
    if not rows:
        return None
    pred = np.full(len(rows), -1, dtype=np.int16)
    sizes = np.array([r[0] for r in rows], dtype=np.int32)
    fmat = np.stack([r[1] for r in rows]).astype(np.float32)
    hist_n = np.stack([r[2] for r in rows]).astype(np.int32)
    hist_b = np.stack([r[3] for r in rows]).astype(np.int32)
    n_ign = np.array([r[4] for r in rows], dtype=np.int32)
    return dict(pred=pred, sizes=sizes, feats=fmat, hist_n=hist_n,
                hist_b=hist_b, n_ign=n_ign, scene=scene)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--split', default='train')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    scenes = sorted(os.listdir(os.path.join(DATA, args.split)))
    mine = scenes[args.shard::args.shards]
    print('scenes', len(mine), flush=True)

    cfg, model, text_embed, _, _ = V.build_vlm()
    print('[vlm] ready', text_embed.shape, flush=True)

    out = {}
    for i, sc in enumerate(mine):
        r = process_scene(model, cfg, args.split, sc)
        if r is not None:
            out[sc] = r
        if (i + 1) % 25 == 0:
            print('  %d / %d' % (i + 1, len(mine)), flush=True)
    p = os.path.join(OUT, 'miss_%s_s%d.pt' % (args.split, args.shard))
    torch.save(out, p)
    print('->', p)


if __name__ == '__main__':
    main()
