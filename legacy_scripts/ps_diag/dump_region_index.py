"""细节 4 第一步：导出每个训练场景的「点级 VLM 预测 + 区域点索引」。

跨视角对照要求 Sonata 和 VLM 用**完全相同的区域边界**。区域缓存只有区域级
pred，没有点级 pred，所以这里重跑一遍 VLM 前向并把索引落盘；随后在 sonata
环境里按同一批索引池化 Sonata 特征。

输出 regions_index_{split}_s{N}.pt：
  { scene: {pred: int16[N], off: int32[R+1], idx: int32[M]} }
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

OUT = '/data01/jh/data/region_index'


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
        coord, color, _ = load_scene(args.split, sc)
        _, preds = V.vlm_forward(model, cfg, coord, color)
        preds = preds.astype(np.int16)
        offs, idxs = [0], []
        for c in np.unique(preds):
            c = int(c)
            if c < V.NUM_BASE:
                continue
            mask = preds == c
            _, comps = connected_components(mask, coord)
            for comp in comps:
                if len(comp) < MIN_PTS:
                    continue
                idxs.append(comp.astype(np.int32))
                offs.append(offs[-1] + len(comp))
        out[sc] = dict(pred=preds,
                       off=np.array(offs, dtype=np.int32),
                       idx=(np.concatenate(idxs).astype(np.int32)
                            if idxs else np.empty(0, np.int32)))
        if (i + 1) % 25 == 0:
            print('  %d / %d' % (i + 1, len(mine)), flush=True)

    p = os.path.join(OUT, 'regions_index_%s_s%d.pt' % (args.split, args.shard))
    torch.save(out, p)
    print('->', p)


if __name__ == '__main__':
    main()
