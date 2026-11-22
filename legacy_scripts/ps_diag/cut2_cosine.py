"""盲区 A 补：Cosine Classifier / 权重归一化对 Base→Novel 假阳性的影响。

用法: python cut2_cosine.py <mode> [shard] [shards]
  mode = cos    两个头权重都 L2 归一化 + bias 清零
  mode = segcos 只归一化 12 类基类头（新类头保持原样）
输出 /data01/jh/data/cut2_feat/cos/{mode}_cm_s{i}.npz
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data01/jh/code/GFS-VL')
from pointcept.datasets import build_dataset, collate_fn  # noqa: E402
from pointcept.engines.defaults import (  # noqa: E402
    default_config_parser, default_setup)
from pointcept.models import build_model  # noqa: E402

CFG = '/data01/jh/code/GFS-VL/configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1.py'
W = '/data01/jh/code/GFS-VL/exp/scannet200/sc200_k1_ONLINE_s42/model/regis1_model_last.pth'
NO, NC = 12, 57
OUT = '/data01/jh/data/cut2_feat/cos'


def metrics(cm):
    d = np.diag(cm).astype(np.float64)
    r = cm.sum(1).astype(np.float64)
    c = cm.sum(0).astype(np.float64)
    iou = d / np.maximum(r + c - d, 1)
    b = float(iou[:NO].mean())
    n = float(iou[NO:].mean())
    return b, n, 2 * b * n / max(b + n, 1e-9)


def main():
    mode = sys.argv[1]
    shard = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    shards = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, '%s_cm_s%d.npz' % (mode, shard))
    if os.path.exists(dst):
        print('已存在，跳过', dst, flush=True)
        return

    cfg = default_config_parser(CFG, {})
    cfg.save_path = '/tmp/cut2_cos_%s_%d' % (mode, shard)
    cfg.weight = W
    cfg = default_setup(cfg)
    model = build_model(cfg.model).cuda().eval()
    ck = torch.load(W, map_location='cpu')
    sd = {k[7:] if k.startswith('module.') else k: v
          for k, v in ck['state_dict'].items()}
    model.load_state_dict(sd, strict=False)
    model.backbone.shuffle_orders = False
    with torch.no_grad():
        if mode in ('cos', 'segcos'):
            model.seg_head.weight.data = F.normalize(
                model.seg_head.weight.data, dim=1)
            model.seg_head.bias.data.zero_()
        if mode == 'cos':
            model.novel_seg_head.weight.data = F.normalize(
                model.novel_seg_head.weight.data, dim=1)
    print('%s: seg_head |W| %.3f novel |W| %.3f'
          % (mode, model.seg_head.weight.norm().item(),
             model.novel_seg_head.weight.norm().item()), flush=True)

    ds = build_dataset(cfg.data.test)
    ds.data_list = ds.data_list[shard::shards]
    print('val 本片 %d 场景' % len(ds.data_list), flush=True)
    cm = np.zeros((NC, NC), np.int64)
    t0 = time.time()
    for i in range(len(ds.data_list)):
        d = ds[i]
        frags = d.pop('fragment_list')
        seg = d.pop('segment')
        gt = np.asarray(seg).astype(np.int64)
        n = len(gt)
        prob = torch.zeros((n, NC), device='cuda')
        cover = torch.zeros(n, device='cuda')
        for frag in frags:
            inp = collate_fn([frag])
            for k in inp:
                if isinstance(inp[k], torch.Tensor):
                    inp[k] = inp[k].cuda(non_blocking=True)
            ip = inp['index']
            with torch.no_grad():
                lg = model(inp)['seg_logits']
                prob[ip] += torch.softmax(lg, -1)
            cover[ip] += 1
        pred = (prob / cover.clamp(min=1)[:, None]).argmax(1).cpu().numpy()
        ok = gt >= 0
        cm += np.bincount(gt[ok] * NC + pred[ok],
                          minlength=NC * NC).reshape(NC, NC)
        if (i + 1) % 20 == 0:
            print('  %d/%d %.0fs' % (i + 1, len(ds.data_list),
                                     time.time() - t0), flush=True)
    np.savez(dst, cm=cm)
    b, nn, h = metrics(cm)
    bn = int(cm[:NO, NO:].sum())
    print('-> %s Base %.4f Novel %.4f HM %.4f | Base→Novel 假阳性 %d'
          % (dst, b, nn, h, bn), flush=True)


if __name__ == '__main__':
    main()
