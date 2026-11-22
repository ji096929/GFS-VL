"""Cut 2 阶段三补：1-shot 支持原型 Probe（免训练）。

对每层：
  proto[c] = L2normalize(mean(feat[gt==c]))，c 取 45 个新类
在 312 val 的新类真值点上做 45-way 余弦最近原型，报 acc / mIoU。
输出 /data01/jh/data/cut2_feat/{tag}/oneshot_s{i}.npz

用法: python cut2_oneshot.py <S1|S2> [shard] [shards]
"""
import glob
import os
import sys
import time

import numpy as np
import torch

from cut2_common import (LAYERS, NC, NO, Probe, build_model_and_cfg, make_ds)
from pointcept.datasets import collate_fn  # noqa: E402

FEAT = '/data01/jh/data/cut2_feat'
MULTI = ['dec0', 'dec1', 'dec2', 'dec3']
NAMES = LAYERS + ['multi']
NN = NC - NO


def build_protos(tag):
    files = sorted(glob.glob(os.path.join(FEAT, tag, 'support_s*.npz')))
    assert files, '没有 support 特征'
    acc = {L: [] for L in LAYERS}
    gt = []
    for f in files:
        z = np.load(f)
        for L in LAYERS:
            acc[L].append(z['f_' + L])
        gt.append(z['gt'])
    gt = np.concatenate(gt, 0).astype(np.int64)
    feats = {L: np.concatenate(acc[L], 0).astype(np.float32) for L in LAYERS}
    protos = {}
    miss = []
    for nm in NAMES:
        if nm == 'multi':
            X = np.concatenate([feats[L] for L in MULTI], 1)
        else:
            X = feats[nm]
        P = np.zeros((NN, X.shape[1]), np.float32)
        for i, c in enumerate(range(NO, NC)):
            w = gt == c
            if w.sum() == 0:
                miss.append(c)
                continue
            v = X[w].mean(0)
            P[i] = v / (np.linalg.norm(v) + 1e-9)
        protos[nm] = P
    print('%s 支持集 %d 点 | 缺类 %d 个 %s'
          % (tag, len(gt), len(set(miss)), sorted(set(miss))), flush=True)
    return protos


def main():
    tag = sys.argv[1]
    shard = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    shards = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    out = os.path.join(FEAT, tag, 'oneshot_s%d.npz' % shard)
    if os.path.exists(out):
        print('已存在，跳过', out, flush=True)
        return

    protos = build_protos(tag)
    P = {nm: torch.tensor(v, device='cuda') for nm, v in protos.items()}
    model, cfg = build_model_and_cfg(tag)
    pb = Probe()
    pb.attach(model.backbone)
    ds = make_ds(cfg, 'val')
    ds.data_list = ds.data_list[shard::shards]
    print('%s val: 本片 %d 场景' % (tag, len(ds.data_list)), flush=True)

    cms = {nm: np.zeros((NN, NN), np.int64) for nm in NAMES}
    cover_cls = np.zeros(NN, np.int64)
    t0 = time.time()
    for i in range(len(ds.data_list)):
        data_dict = ds[i]
        frags = data_dict.pop('fragment_list')
        seg = data_dict.pop('segment')
        gt = np.asarray(seg).astype(np.int64)
        for frag in frags:
            inp = collate_fn([frag])
            for k in inp:
                if isinstance(inp[k], torch.Tensor):
                    inp[k] = inp[k].cuda(non_blocking=True)
            ip = inp['index']
            g = gt[ip.detach().cpu().numpy()]
            m = g >= NO
            if not m.any():
                continue
            with torch.no_grad():
                model(inp)
                fs = pb.layers()
                for nm in NAMES:
                    if nm == 'multi':
                        X = torch.cat([fs[L] for L in MULTI], 1)
                    else:
                        X = fs[nm]
                    X = X[m]
                    X = X / (X.norm(dim=1, keepdim=True) + 1e-9)
                    pred = (X @ P[nm].T).argmax(1).cpu().numpy()
                    gg = g[m] - NO
                    cms[nm] += np.bincount(gg * NN + pred,
                                           minlength=NN * NN).reshape(NN, NN)
        if (i + 1) % 20 == 0:
            print('  %d/%d  %.0fs' % (i + 1, len(ds.data_list),
                                      time.time() - t0), flush=True)

    np.savez(out, **cms)
    for nm in NAMES:
        cm = cms[nm]
        d = np.diag(cm).astype(np.float64)
        r = cm.sum(1).astype(np.float64)
        c_ = cm.sum(0).astype(np.float64)
        iou = d / np.maximum(r + c_ - d, 1)
        print('%-5s 45way acc %.4f  mIoU %.4f'
              % (nm, d.sum() / max(cm.sum(), 1), iou.mean()), flush=True)
    print('-> %s' % out, flush=True)


if __name__ == '__main__':
    main()
