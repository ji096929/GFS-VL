"""疑点 2：ptv3_bk 原生 12 类 seg_head 在 312 val 上的 mIoU-B。

用来交叉验证「S1 dec0 Linear Probe 只有 0.5136」是不是均匀采样造成的。
S1 的 novel_seg_head 是随机初始化，所以只走 model.seg_head(dec0)。
"""
import sys

import numpy as np
import torch

from cut2_common import NO, Probe, build_model_and_cfg, make_ds
from pointcept.datasets import collate_fn  # noqa: E402


def main():
    model, cfg = build_model_and_cfg('S1')
    pb = Probe()
    pb.attach(model.backbone)
    ds = make_ds(cfg, 'val')
    shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    shards = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ds.data_list = ds.data_list[shard::shards]
    print('本片 %d 场景' % len(ds.data_list), flush=True)
    cm = np.zeros((NO, NO), np.int64)
    for i in range(len(ds.data_list)):
        d = ds[i]
        frags = d.pop('fragment_list')
        seg = d.pop('segment')
        gt = np.asarray(seg).astype(np.int64)
        n = len(gt)
        prob = torch.zeros((n, NO), device='cuda')
        cover = torch.zeros(n, device='cuda')
        for frag in frags:
            inp = collate_fn([frag])
            for k in inp:
                if isinstance(inp[k], torch.Tensor):
                    inp[k] = inp[k].cuda(non_blocking=True)
            ip = inp['index']
            with torch.no_grad():
                model(inp)
                fs = pb.layers()
                lg = model.seg_head(fs['dec0'])
                prob[ip] += torch.softmax(lg.float(), -1)
            cover[ip] += 1
        pred = (prob / cover.clamp(min=1)[:, None]).argmax(1).cpu().numpy()
        ok = (gt >= 0) & (gt < NO)
        cm += np.bincount(gt[ok] * NO + pred[ok],
                          minlength=NO * NO).reshape(NO, NO)
        if (i + 1) % 20 == 0:
            print('  %d/%d' % (i + 1, len(ds.data_list)), flush=True)
    np.savez('/data01/jh/data/cut2_feat/audit_S1_base12_cm_s%d.npz' % shard,
             cm=cm)
    d = np.diag(cm).astype(float)
    r = cm.sum(1).astype(float)
    c = cm.sum(0).astype(float)
    iou = d / np.maximum(r + c - d, 1)
    print('ptv3_bk 原生 12 类 seg_head mIoU-B = %.4f' % iou.mean())
    print('  每类: %s' % np.round(iou, 3).tolist())


if __name__ == '__main__':
    main()
