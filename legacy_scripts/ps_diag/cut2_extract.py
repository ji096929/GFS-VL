"""Cut 2 阶段一：从指定 checkpoint 提取分层特征。

用法:
    python cut2_extract.py <S1|S2> train   [shard] [shards]
    python cut2_extract.py <S1|S2> support [shard] [shards]

输出 /data01/jh/data/cut2_feat/{tag}/
    train_s{i}.npz    f_enc4/f_dec3/f_dec2/f_dec1/f_dec0 (float16) + gt + n
    support_s{i}.npz  同上

采样规则：按 GT 分层，每类最多 CAP 点（分片内各自计）；不重复采样同一点；
与 val 推理完全一致的 fragment 管线（GridSample test mode）。
"""
import glob
import os
import sys
import time

import numpy as np
import torch

from cut2_common import (CFG, DIM, LAYERS, NC, NO, Probe, build_model_and_cfg,
                         make_ds, support_list)
from pointcept.datasets import collate_fn  # noqa: E402

OUT = '/data01/jh/data/cut2_feat'
CAP_TRAIN = 1500     # 每场景每类上限（控制单片产物约 3 GB）
CAP_SUP = 20000


def extract_scene(model, pb, ds, idx, cap):
    """跑一个场景，返回 (每层特征 dict, gt)。每场景内每类最多 cap 点。"""
    data_dict = ds[idx]
    frags = data_dict.pop('fragment_list')
    segment = data_dict.pop('segment')
    gt = np.asarray(segment).astype(np.int16)
    n = len(gt)
    seen = np.zeros(n, dtype=bool)
    cnt = np.zeros(NC, dtype=np.int64)
    buf = {L: [] for L in LAYERS}
    kept_gt = []
    for frag in frags:
        inp = collate_fn([frag])
        for k in inp:
            if isinstance(inp[k], torch.Tensor):
                inp[k] = inp[k].cuda(non_blocking=True)
        ip = inp['index'].detach().cpu().numpy()
        g = gt[ip]
        safe = np.clip(g, 0, NC - 1)
        cand = np.where((g >= 0) & (~seen[ip]) & (cnt[safe] < cap))[0]
        if len(cand) == 0:
            continue
        with torch.no_grad():
            model(inp)
        fs = pb.layers()
        for L in LAYERS:
            buf[L].append(fs[L][cand].half().cpu().numpy())
        kept_gt.append(g[cand])
        seen[ip[cand]] = True
        np.add.at(cnt, g[cand], 1)
    if not kept_gt:
        return {L: np.zeros((0, DIM[L]), np.float16) for L in LAYERS}, \
            np.zeros(0, np.int16)
    return {L: np.concatenate(buf[L], 0) for L in LAYERS}, \
        np.concatenate(kept_gt, 0)


PART_SCENES = 100   # 每多少场景落一次盘


def flush_part(acc, acc_gt, tag, mode, shard, part_idx, verbose=True):
    """把累积的特征写成一段并清空，控制内存峰值。"""
    if not acc_gt:
        return
    out = os.path.join(OUT, tag, '%s_s%d_p%02d.npz' % (mode, shard, part_idx))
    np.savez(
        out,
        **{'f_' + L: np.concatenate(acc[L], 0) for L in LAYERS},
        gt=np.concatenate(acc_gt, 0),
        n=np.array([len(acc_gt)]),
    )
    n_pts = sum(len(g) for g in acc_gt)
    for L in LAYERS:
        acc[L].clear()
    acc_gt.clear()
    if verbose:
        print('  -> 分段落盘 %s (%d 场景, %d 点)'
              % (os.path.basename(out), 0, n_pts), flush=True)
    return out


def main():
    tag = sys.argv[1]
    mode = sys.argv[2]
    shard = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    shards = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    cap = CAP_TRAIN if mode == 'train' else CAP_SUP
    os.makedirs(os.path.join(OUT, tag), exist_ok=True)
    dst = os.path.join(OUT, tag, '%s_s%d.npz' % (mode, shard))
    if os.path.exists(dst):
        print('已存在，跳过', dst, flush=True)
        return

    model, cfg = build_model_and_cfg(tag)
    pb = Probe()
    pb.attach(model.backbone)

    if mode == 'train':
        ds = make_ds(cfg, 'train')
        names = ds.data_list
        limit = int(os.environ.get('CUT2_LIMIT', '0'))
        if limit > 0:
            names = names[:limit]
    elif mode == 'support':
        names = support_list(cfg)
        ds = make_ds(cfg, 'train', data_list=names)
    else:
        raise ValueError(mode)
    ds.data_list = list(names)[shard::shards]
    print('%s %s: 本片场景 %d / 总 %d | cap=%d | ckpt=%s'
          % (tag, mode, len(ds.data_list), len(names), cap, tag), flush=True)

    acc = {L: [] for L in LAYERS}
    acc_gt = []
    t0 = time.time()
    for i in range(len(ds.data_list)):
        f, g = extract_scene(model, pb, ds, i, cap)
        if len(g) == 0:
            continue
        for L in LAYERS:
            acc[L].append(f[L])
        acc_gt.append(g)
        if (i + 1) % PART_SCENES == 0:
            flush_part(acc, acc_gt, tag, mode, shard, (i + 1) // PART_SCENES - 1)
        if (i + 1) % 10 == 0:
            print('  %d/%d  %.0fs  累计点 %d'
                  % (i + 1, len(ds.data_list), time.time() - t0,
                     sum(len(x) for x in acc_gt)), flush=True)

    if not acc_gt:
        print('-> 尾巴为空，跳过最终保存（场景已在分段文件里）', flush=True)
        return
    out = {}
    for L in LAYERS:
        out['f_' + L] = np.concatenate(acc[L], 0) if acc[L] else \
            np.zeros((0, DIM[L]), np.float16)
    out['gt'] = np.concatenate(acc_gt, 0) if acc_gt else np.zeros(0, np.int16)
    out['n'] = np.array([len(out['gt'])])
    np.savez(dst, **out)
    print('-> %s  %d 点  分布前10类 %s  %.0fs'
          % (dst, len(out['gt']),
             np.bincount(out['gt'], minlength=NC)[:12].tolist(),
             time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
