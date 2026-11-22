"""缓存训练场景的 VLM 预测区域特征（三组伪标签对照的底座）。

对每个训练场景：
  1. vlm_forward -> 每点 512 维特征 + 预测类（GFS 空间 0-56）
  2. 对每个 novel 预测类（>=12）做连通分量（radius=0.05）
  3. 每个区域存：均值特征、点数、GT 直方图（novel/base/ignore）

区域缓存只存特征，PS 分数逐 regis split 在实验时重算（support 原型依赖 split）。
"""
import argparse
import os
import sys

sys.path.insert(0, '/data01/jh/code/ps_diag')

import numpy as np  # noqa: E402
import torch  # noqa: E402

import vlm_common as V  # noqa: E402

DATA = '/data01/jh/data/scannet200_gfs'
OUT = '/data01/jh/data/region_cache'
RADIUS = 0.05
MIN_PTS = 20


def load_scene(split, scene):
    d = os.path.join(DATA, split, scene)
    coord = np.load(os.path.join(d, 'coord.npy')).astype(np.float32)
    color = np.load(os.path.join(d, 'color.npy'))
    seg200 = np.load(os.path.join(d, 'segment200.npy')).reshape(-1)
    return coord, color, V.convert_to_gfs(seg200)


def connected_components(mask, coord, radius=RADIUS):
    """3D 体素网格上的连通分量（scipy.ndimage.label，比 Python union-find 快两个量级）。"""
    from scipy import ndimage
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.empty(0, dtype=int), []
    pts = coord[idx]
    mn = pts.min(0)
    g = np.floor((pts - mn) / radius).astype(np.int64)
    shape = tuple(g.max(0) + 1)
    grid = np.zeros(shape, dtype=bool)
    grid[g[:, 0], g[:, 1], g[:, 2]] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=int))
    labels = lab[g[:, 0], g[:, 1], g[:, 2]]
    comps = [idx[labels == c] for c in range(1, n + 1)]
    return labels, comps


def process_scene(model, cfg, split, scene):
    coord, color, gfs = load_scene(split, scene)
    feats, preds = V.vlm_forward(model, cfg, coord, color)   # (N,512), (N,)
    feats_t = torch.as_tensor(feats)
    n_novel = len(V.CLASS_NAMES) - V.NUM_BASE

    rows = []
    for c in np.unique(preds):
        c = int(c)
        if c < V.NUM_BASE:      # 只看 novel 预测类
            continue
        mask = preds == c
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
    torch.save(out, os.path.join(OUT, 'regions_%s_s%d.pt'
                                 % (args.split, args.shard)))
    print('->', os.path.join(OUT, 'regions_%s_s%d.pt'
                            % (args.split, args.shard)))


if __name__ == '__main__':
    main()
