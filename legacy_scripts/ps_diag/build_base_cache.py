"""缓存「VLM 预测为基类」的几何区域（pred < 12）。

为什么不是 pred == -1：GFS-VL 的 VLM 输出是 57 类 argmax，**从不输出 -1**。
-1 是 PS 后处理把 base 类和被拒类置掉的。所以真正"待填充的空白区"是：
  - PS 拒绝的 novel 预测区  → 已在 region_cache 的 rejected 部分
  - VLM 判成 base 的区域    → 本文件覆盖

这批区域同时是「基类吞并」的直接证据：VLM 说它是 wall/floor/table，
但 GT 可能是 novel 类。

输出结构同 region_cache，pred 为对应的 base 类 id (0..11)。
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

OUT = '/data01/jh/data/base_cache'


def process_scene(model, cfg, split, scene):
    coord, color, gfs = load_scene(split, scene)
    feats, preds = V.vlm_forward(model, cfg, coord, color)
    feats_t = torch.as_tensor(feats)
    n_novel = len(V.CLASS_NAMES) - V.NUM_BASE

    rows = []
    for c in np.unique(preds):
        c = int(c)
        if c >= V.NUM_BASE:
            continue
        mask = preds == c
        if int(mask.sum()) < MIN_PTS:
            continue
        _, comps = connected_components(mask, coord)
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
            rows.append((c, len(comp), f_mean, hist_n, hist_b, n_ign))
    if not rows:
        return None
    pred = np.array([r[0] for r in rows], dtype=np.int16)
    sizes = np.array([r[1] for r in rows], dtype=np.int32)
    fmat = np.stack([r[2] for r in rows]).astype(np.float32)
    hist_n = np.stack([r[3] for r in rows]).astype(np.int32)
    hist_b = np.stack([r[4] for r in rows]).astype(np.int32)
    n_ign = np.array([r[5] for r in rows], dtype=np.int32)
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
    p = os.path.join(OUT, 'base_%s_s%d.pt' % (args.split, args.shard))
    torch.save(out, p)
    print('->', p)


if __name__ == '__main__':
    main()
