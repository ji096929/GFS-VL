"""并行收集 ScanNet train split 的场景 npy（第 8 列保留 seg id）。

用法：--shard 0 --nshard 4
输出：/data01/jh/data/scannet_train_v2/scenes/data/<scene>.npy
"""
import argparse
import os
import sys

sys.path.insert(0, '/data01/jh/code/COSeg')
import numpy as np

import preprocess.collect_scannet_data_sp as C
from preprocess.collect_scannet_data_sp import get_raw2scannet_label_map

DATA_PATH = '/data01/jh/data/scannet_raw/scans'
SAVE = '/data01/jh/data/scannet_train_v2/scenes/data'
META = '/data01/jh/code/COSeg/datasets/ScanNet/meta'
SPLIT = '/data01/jh/code/COSeg/datasets/ScanNet/meta/scannetv2_train.txt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(SAVE, exist_ok=True)
    names = [l.strip() for l in open(os.path.join(META, 'scannet_classnames.txt'))]
    C.CLASS_NAMES = names          # 必须先设置，get_raw2scannet_label_map 依赖它
    raw2scannet = get_raw2scannet_label_map(
        os.path.join(META, 'scannetv2-labels.combined.tsv'))
    C.RAW2SCANNET = raw2scannet

    scenes = [l.strip() for l in open(SPLIT) if l.strip()]
    if args.limit:
        scenes = scenes[:args.limit]
    mine = scenes[args.shard::args.nshard]
    print('shard %d/%d: %d scenes' % (args.shard, args.nshard, len(mine)),
          flush=True)

    done = 0
    for s in mine:
        out = os.path.join(SAVE, s + '.npy')
        if os.path.exists(out):
            continue
        try:
            C.collect_point_label(os.path.join(DATA_PATH, s), s, out)
            done += 1
        except Exception as e:
            print('fail', s, repr(e)[:80], flush=True)
        if done % 50 == 0 and done:
            print('  done', done, flush=True)
    print('finished', done, flush=True)


if __name__ == '__main__':
    main()
