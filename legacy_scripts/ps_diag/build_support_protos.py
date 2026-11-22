"""计算 10 套 1-shot 划分的 support 原型（45 个 novel 类，512 维）。

带场景特征缓存，避免同一场景在多个 split 上重复前向。
"""
import os
import sys

sys.path.insert(0, '/data01/jh/code/ps_diag')

import numpy as np  # noqa: E402
import torch  # noqa: E402

import vlm_common as V  # noqa: E402
from ps_diag import DATA, load_scene  # noqa: E402

CACHE = '/data01/jh/data/scene_feats_cache'
OUT = '/data01/jh/data/support_protos'
SPLITS = ['sc200_regis_1_%d.txt' % s for s in (10, 20, 30, 40, 50)] + \
         ['sc200_regis_5_%d.txt' % s for s in (10, 20, 30, 40, 50)]


def parse_regis(name):
    out = []
    for line in (DATA / name).read_text().splitlines():
        if not line.strip():
            continue
        orig, rel = line.split('\t')
        gfs = V.CLASS_NAMES.index(V.LABELS200[int(orig)])
        sp, sc = rel.replace('\\', '/').split('/')[-2:]
        out.append((gfs, sp, sc))
    return out


def get_feats(model, cfg, split, scene):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, '%s_%s.npz' % (split, scene))
    if os.path.exists(p):
        d = np.load(p)
        return d['feats'], d['gfs']
    coord, color, gfs = load_scene(split, scene)
    feats, _ = V.vlm_forward(model, cfg, coord, color)
    np.savez_compressed(p, feats=feats.astype(np.float16), gfs=gfs)
    return feats, gfs


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg, model, text_embed, _, _ = V.build_vlm()
    print('[vlm] ready', text_embed.shape, flush=True)

    seen = set()
    for regis in SPLITS:
        entries = parse_regis(regis)
        acc = np.zeros((len(V.CLASS_NAMES) - V.NUM_BASE, 512), dtype=np.float64)
        cnt = np.zeros(len(V.CLASS_NAMES) - V.NUM_BASE, dtype=np.float64)
        for gfs_id, split, scene in entries:
            if gfs_id < V.NUM_BASE:
                continue
            feats, gfs = get_feats(model, cfg, split, scene)
            mask = gfs == gfs_id
            if mask.sum() == 0:
                continue
            idx = gfs_id - V.NUM_BASE
            acc[idx] += feats[mask].astype(np.float64).sum(0)
            cnt[idx] += mask.sum()
            seen.add((split, scene))
        proto = acc / np.maximum(cnt[:, None], 1)
        np.save(os.path.join(OUT, regis.replace('.txt', '.npy')),
                proto.astype(np.float32))
        print('%s: protos %s  scenes_seen %d' % (regis, proto.shape, len(seen)),
              flush=True)


if __name__ == '__main__':
    main()
