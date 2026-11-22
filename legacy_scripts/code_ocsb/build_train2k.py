"""从 train split 的 blocks 采 2000 个 base 类任务并提取缓存。

严格场景隔离：support / query / U 三者场景互不重叠。
输出 /data01/jh/data/ocsb_train2k/train_XXXX.npz + meta。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import diag_bc as D  # noqa: E402
from build_ocsb_cache import (BASE_FOLD0_NAMES, BASE_TO_SLOT,
                              build_train_specs, extract_episode)  # noqa: E402

DATA = '/data01/jh/data/scannet_train_v2/blocks_bs1.0_s1.0/data'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--out', default='/data01/jh/data/ocsb_train2k')
    ap.add_argument('--n-per-class', type=int, default=200)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    names = D.load_class_names()
    name2id = {n: i for i, n in enumerate(names)}
    base_ids = [name2id[n] for n in BASE_FOLD0_NAMES]
    assert set(BASE_TO_SLOT) == set(base_ids)
    print('base ids:', base_ids, flush=True)

    print('scanning train blocks...', flush=True)
    blocks = D.scan_blocks(args.data)
    info_by_name = {b['name']: b for b in blocks}
    class2blocks = {c: [b for b in blocks if c in b['classes']] for c in base_ids}
    print('blocks:', len(blocks), flush=True)
    for c in base_ids:
        print('  %2d %-16s %4d' % (c, names[c], len(class2blocks[c])))

    rng = np.random.RandomState(args.seed + 1)
    specs = build_train_specs(class2blocks, base_ids, args.n_per_class, rng)
    print('specs:', len(specs), flush=True)

    model, cfg = D.build_model()
    dev = torch.device('cuda:0')
    meta = []
    for i, sp in enumerate(specs):
        if i % args.nshard != args.shard:
            continue
        dst = os.path.join(args.out, 'train_%04d.npz' % i)
        s_info = info_by_name[sp['support']]
        q_info = info_by_name[sp['query']]
        u_infos = [info_by_name[n] for n in sp['u_cross']]
        if os.path.exists(dst):
            meta.append({'idx': i, 'class': int(sp['class']),
                         'support': s_info['name'], 'query': q_info['name'],
                         'u': [b['name'] for b in u_infos]})
            continue
        np.random.seed(30000 + i)
        torch.manual_seed(30000 + i)
        try:
            arr = extract_episode(model, cfg, s_info, q_info, u_infos,
                                  int(sp['class']), dev)
        except Exception as e:
            print('ep %d failed: %s' % (i, repr(e)[:80]), flush=True)
            continue
        np.savez_compressed(dst, **arr)
        meta.append({'idx': i, 'class': int(sp['class']),
                     'support': s_info['name'], 'query': q_info['name'],
                     'u': [b['name'] for b in u_infos]})
        if (i + 1) % 50 == 0:
            print('  %d / %d' % (i + 1, len(specs)), flush=True)
    json.dump(meta, open(os.path.join(
        args.out, 'meta_shard%d.json' % args.shard), 'w'), indent=1)
    print('done shard %d: %d' % (args.shard, len(meta)), flush=True)


if __name__ == '__main__':
    main()
