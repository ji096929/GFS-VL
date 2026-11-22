"""导出 base 池（pred < 12）的区域点索引，供 AI 补标写入点级伪标签。"""
import argparse, os, sys
sys.path.insert(0, '/data01/jh/code/ps_diag')
import numpy as np
import torch
import vlm_common as V
from build_region_cache import DATA, MIN_PTS, connected_components, load_scene

OUT = '/data01/jh/data/base_index'

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
    cfg, model, te, _, _ = V.build_vlm()
    print('[vlm] ready', flush=True)
    out = {}
    for i, sc in enumerate(mine):
        coord, color, _ = load_scene(args.split, sc)
        _, preds = V.vlm_forward(model, cfg, coord, color)
        preds = preds.astype(np.int16)
        offs, idxs = [0], []
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
                idxs.append(comp.astype(np.int32))
                offs.append(offs[-1] + len(comp))
        out[sc] = dict(pred=preds, off=np.array(offs, dtype=np.int32),
                       idx=(np.concatenate(idxs).astype(np.int32) if idxs else np.empty(0, np.int32)))
        if (i + 1) % 25 == 0:
            print('  %d / %d' % (i + 1, len(mine)), flush=True)
    p = os.path.join(OUT, 'base_index_%s_s%d.pt' % (args.split, args.shard))
    torch.save(out, p)
    print('->', p)

if __name__ == '__main__':
    main()
