"""修正版对拍：官方 forward（fp16/fp32）vs predict_agg（fp32）。

上次 probe 的错误：把 dump 的 s_lab（0/1 二值）当成 21 类 id 传给
`(label == c)`，导致 support 前景原型为空。
"""
import glob
import os
import sys

sys.path.insert(0, '/data01/jh/code')
import fake_tp  # noqa: F401

sys.path.insert(0, '/data01/jh/code/COSeg')
sys.path.insert(0, '/data01/jh/code')

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ocsb_core import load_model, feats_one, build_proto, predict_agg  # noqa: E402

files = sorted(glob.glob('/data01/jh/data/sonata_dump/ep*.npz'))[:20]
m, cfg = load_model()
dev = torch.device('cuda')
print('model.training =', m.training)


def prep(d):
    sx = torch.cat([
        torch.from_numpy(d['s_coord'].astype(np.float32)),
        torch.from_numpy(d['s_rgb'].astype(np.float32)) / 255.], 1).cuda()
    qx = torch.cat([
        torch.from_numpy(d['q_coord'].astype(np.float32)),
        torch.from_numpy(d['q_rgb'].astype(np.float32)) / 255.], 1).cuda()
    sy = torch.from_numpy(d['s_lab'].astype(np.int64)).cuda()   # 0/1 二值
    qy = torch.from_numpy(d['q_lab'].astype(np.int64)).cuda()
    so = torch.tensor([len(sx)], dtype=torch.int32)
    qo = torch.tensor([len(qx)], dtype=torch.int32)
    return sx, sy, qx, qy, so, qo


def official(d, amp):
    c = int(d['cls'][0])
    sx, sy, qx, qy, so, qo = prep(d)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
        out, _ = m(so, sx, sy, qo, qx, qy, 5, support_base_y=None,
                   query_base_y=None, sampled_classes=np.array([c]))
    return F.softmax(out[0].t().float(), 1)[:, 1].cpu().numpy()


def repro(d, amp=False):
    sx, sy, qx, qy, so, qo = prep(d)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
        sf = feats_one(m, sx, sy, dev)
        qf = feats_one(m, qx, qy, dev)
        fg = sf['yl'].bool()
        sp = torch.cat([build_proto(m, sf, ~fg), build_proto(m, sf, fg)], 0)
        pred, _ = predict_agg(m, qf, [sp], 0.0, dev)
    return F.softmax(pred.float(), 0)[1].cpu().numpy()


def iou(p, d):
    gt = d['q_lab'].astype(bool)
    pr = p > 0.5
    return float((pr & gt).sum() / max(int((pr | gt).sum()), 1))


acc = {'off_fp16': [], 'off_fp32': [], 'agg_fp32': [], 'agg_fp16': []}
diffs = {'off16_vs_off32': [], 'agg32_vs_off32': [], 'agg16_vs_off16': []}
for f in files:
    d = np.load(f)
    p16 = official(d, True)
    p32 = official(d, False)
    a32 = repro(d, False)
    a16 = repro(d, True)
    acc['off_fp16'].append(iou(p16, d))
    acc['off_fp32'].append(iou(p32, d))
    acc['agg_fp32'].append(iou(a32, d))
    acc['agg_fp16'].append(iou(a16, d))
    diffs['off16_vs_off32'].append(np.abs(p16 - p32).max())
    diffs['agg32_vs_off32'].append(np.abs(a32 - p32).max())
    diffs['agg16_vs_off16'].append(np.abs(a16 - p16).max())

print('官方 fp16  mean IoU %.4f' % np.mean(acc['off_fp16']))
print('官方 fp32  mean IoU %.4f' % np.mean(acc['off_fp32']))
print('agg  fp32  mean IoU %.4f' % np.mean(acc['agg_fp32']))
print('agg  fp16  mean IoU %.4f' % np.mean(acc['agg_fp16']))
for k, v in diffs.items():
    print('%s  p0 maxdiff  mean %.2e  max %.2e' % (k, np.mean(v), np.max(v)))
