"""预计算 500 个候选块的 COSeg 降采样特征（训练池随机采，不按标签筛）。

候选池固定 seed=0，所有任务共用；与测试的 312 个 val 场景零重叠。
"""
import argparse
import os
import random
import sys

sys.path.insert(0, '/data01/jh/code')
import fake_tp  # noqa: F401

sys.path.insert(0, '/data01/jh/code/COSeg')
sys.path.insert(0, '/data01/jh/code')

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ocsb_core import feats_one, load_model  # noqa: E402
from util.data_util import data_prepare_v101  # noqa: E402

TRAIN = '/data01/jh/data/scannet_train_v2/blocks_bs1.0_s1.0/data'
OUT = '/data01/jh/data/retrieval_cache'
POOL_N = 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    blocks = sorted(os.listdir(TRAIN))
    pool = random.Random(0).sample(blocks, POOL_N)
    mine = pool[args.shard::args.shards]
    print('pool %d  mine %d' % (len(pool), len(mine)), flush=True)

    model, cfg = load_model()
    dev = torch.device('cuda')
    cache = {}
    for i, name in enumerate(mine):
        r = np.load(os.path.join(TRAIN, name))
        coord, feat, label = data_prepare_v101(
            r[:, :3].astype(np.float32), r[:, 3:6].astype(np.float32),
            r[:, 6].astype(np.int64), 'train', 0.02, 20480, None, True, 0)
        x = torch.cat([coord, feat], 1).cuda()
        off = torch.tensor([len(x)], dtype=torch.int32)
        y = torch.zeros(len(x), dtype=torch.long).cuda()
        f = feats_one(model, x, y, dev)
        cache[name] = dict(
            fd=f['fd'].cpu(), cd=f['cd'].cpu(), ol=f['ol'].cpu(),
            yl=f['yl'].cpu(), xyz=f['xyz'].cpu(), off=f['off'].cpu(),
            ulab=label.numpy().astype(np.int64))
        if (i + 1) % 20 == 0:
            print('  %d / %d' % (i + 1, len(mine)), flush=True)
    torch.save(cache, os.path.join(OUT, 'ucache_%d.pt' % args.shard))
    print('->', os.path.join(OUT, 'ucache_%d.pt' % args.shard))


if __name__ == '__main__':
    main()
