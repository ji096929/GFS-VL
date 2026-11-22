"""实测 COSeg 训练前向+反向的显存与耗时。

对比：
  src=block   1m block（真实训练输入，voxel 后 2-5k 点）
  src=scene   整场景采样到 20480 点（输入上限）
  freeze=last 冻结骨干除最后一层外的所有层（classifier/bk_ffn 仍可训）
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, '/data01/jh/code')
import fake_tp  # noqa: F401

sys.path.insert(0, '/data01/jh/code/COSeg')

import numpy as np  # noqa: E402
import torch  # noqa: E402

from util import config  # noqa: E402
from util.data_util import data_prepare_v101  # noqa: E402
from model.coseg import COSeg  # noqa: E402

CFG = '/data01/jh/code/COSeg/config/scannetv2_COSeg_fs.yaml'
BLK = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
SCN = '/data01/jh/data/ScanNet/scenes/data'
TARGET = 4  # fold-0 的 base 类之一
# 官方训练集的 support_base_y 是映射到 base 空间后的标签（1..10），不是原始 21 类 id
FOLD0_BASE = {1: 1, 4: 2, 8: 3, 9: 4, 11: 5, 15: 6, 16: 7, 17: 8, 18: 9, 20: 10}


def map_base(label):
    by = np.zeros(label.shape, np.int64)
    for k, v in FOLD0_BASE.items():
        by[label == k] = v
    return torch.from_numpy(by).long()


def prep(path, seed):
    r = np.load(path)
    np.random.seed(seed)
    coord, feat, label = data_prepare_v101(
        r[:, :3].astype(np.float32), r[:, 3:6].astype(np.float32),
        r[:, 6].astype(np.int64), voxel_size=0.02, voxel_max=20480,
        shuffle_index=True)
    return coord, feat, label


def build(paths):
    xs, ys, bys = [], [], []
    for i, p in enumerate(paths):
        coord, feat, label = prep(p, i)
        xs.append(torch.cat([coord, feat], 1))
        ys.append((label == TARGET).long())
        bys.append(map_base(label))
    return xs, ys, bys


def bench(model, blocks, nq, amp, iters=5):
    xs, ys, bys = blocks
    sx = xs[0].cuda(); sy = ys[0].cuda(); sby = bys[0].cuda()
    qx = torch.cat(xs[1:1 + nq]).cuda()
    qy = torch.cat(ys[1:1 + nq]).cuda()
    qby = torch.cat(bys[1:1 + nq]).cuda()
    so = torch.tensor([len(xs[0])], dtype=torch.int32)
    qo = torch.tensor([sum(len(x) for x in xs[1:1 + i + 1])
                       for i in range(nq)], dtype=torch.int32)
    ts = []
    peak_alloc = peak_resv = 0.0
    for it in range(iters):
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.cuda.amp.autocast(enabled=amp):
            out, loss = model(so, sx, sy, qo, qx, qy, 6,
                              support_base_y=sby, query_base_y=qby,
                              sampled_classes=np.array([TARGET]))
        loss.backward()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
        peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated() / 1e9)
        peak_resv = max(peak_resv, torch.cuda.max_memory_reserved() / 1e9)
    return peak_alloc, peak_resv, float(np.median(ts[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='block', choices=['block', 'scene'])
    ap.add_argument('--freeze', default='none', choices=['none', 'last'])
    args = ap.parse_args()

    cfg = config.load_cfg_from_cfg_file(CFG)
    cfg.n_way = 1
    cfg.k_shot = 1
    cfg.n_queries = 2
    cfg.cvfold = 0
    cfg.save_path = '/data01/jh/afford_diag/tmp_coseg'
    cfg.ngpus_per_node = 1
    cfg.rank = 0
    cfg.multiprocessing_distributed = False
    cfg.distributed = False
    os.makedirs(cfg.save_path, exist_ok=True)
    model = COSeg(cfg).cuda().train()

    if args.freeze == 'last':
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.encoder.layers[-1].parameters():
            p.requires_grad = True
        for p in model.encoder.classifier.parameters():
            p.requires_grad = True
        for p in model.bk_ffn.parameters():
            p.requires_grad = True
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    nall = sum(p.numel() for p in model.parameters()) / 1e6
    print('params all %.1fM  trainable %.1fM  amp=%s  src=%s  freeze=%s'
          % (nall, ntr, cfg.use_amp, args.src, args.freeze))

    if args.src == 'block':
        c2s = pickle.load(open(os.path.join(os.path.dirname(BLK),
                                            'class2scans.pkl'), 'rb'))
        names = c2s[TARGET][:3]
        paths = [os.path.join(BLK, n + '.npy') for n in names]
    else:
        names = sorted(os.listdir(SCN))[:3]
        paths = [os.path.join(SCN, n) for n in names]
    print('inputs:', [os.path.basename(p) for p in paths])
    blocks = build(paths)
    print('pts:', [len(x) for x in blocks[0]])

    for nq in (1, 2):
        pa, pr, t = bench(model, blocks, nq, cfg.use_amp)
        print('1S+%dQ: peak_alloc %.2f GB  peak_reserved %.2f GB  median %.3f s/ep'
              % (nq, pa, pr, t), flush=True)


if __name__ == '__main__':
    main()
