#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XL-1 实验 1b v2：冻结主干 + 45 类线性探针。

按要求：
  * 104 场景用固定 seed 打乱后切 70/30（消除顺序偏差）
  * 每层按训练集 (X-mean)/std 标准化
  * GPU 上带 bias 的线性头 + 类别逆频次加权 CrossEntropy，Adam 150 步
  * 报 dec0/dec1/dec2 单层的 45 novel 类 mIoU 与 macro Acc
"""
import json, os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, '/data01/jh/code/GFS-VL')
from pointcept.datasets import build_dataset, collate_fn
from pointcept.engines.defaults import default_config_parser
from pointcept.models import build_model

R = '/data01/jh/code/GFS-VL/'
STRUCT_CFG = R + 'configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_sym_lovasz.py'
NB, NC = 12, 57
NOV = NC - NB
PER_SCENE = 2048
CASES = [
    ('base_lovasz_s7', R + 'exp/scannet200/base_lovasz_s7/model/regis1_model_last.pth'),
    ('sym_lovasz_s7',  R + 'exp/scannet200/sym_lovasz_s7/model/regis1_model_last.pth'),
    ('detach_sym_lovasz_s7', R + 'exp/scannet200/detach_sym_lovasz_s7/model/regis1_model_last.pth'),
]

def load(cfg, ck_p):
    m = build_model(cfg.model)
    ck = torch.load(ck_p, map_location='cpu', weights_only=False)
    sd = {k.replace('module.', '', 1): v for k, v in ck.get('state_dict', ck).items()}
    msd = m.state_dict()
    kept = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    m.load_state_dict(kept, strict=False)
    m.eval().cuda()
    return m

def miou_macro(pred, gt):
    inter = np.zeros(NOV); union = np.zeros(NOV); per = []
    for j in range(NOV):
        c = j + NB
        p = pred == c; g = gt == c
        inter[j] = (p & g).sum(); union[j] = (p | g).sum()
        if g.sum() > 0:
            per.append(float((p & g).sum()) / float(g.sum()))
    iou = inter / np.maximum(union, 1)
    return float(np.mean(iou)), float(np.mean(per))

def main():
    assert os.environ.get('GFS_CAPTURE_LAYERS')
    cfg = default_config_parser(STRUCT_CFG, options=None)
    ds = build_dataset(cfg.data.val)
    idxs = list(range(0, len(ds), 3))
    loader = DataLoader(Subset(ds, idxs), batch_size=1, shuffle=False,
                        num_workers=8, collate_fn=collate_fn)
    def _to_cpu(v, k):
        if not torch.is_tensor(v): return v
        v = v.cpu()
        if k == 'offset': return v.reshape(-1)
        if v.dim() > 1 and v.shape[0] == 1: return v.squeeze(0)
        return v
    scenes = [{k: _to_cpu(v, k) for k, v in d.items()} for d in loader]
    print('loaded %d scenes' % len(scenes), flush=True)

    rng = np.random.default_rng(20261124)
    order = rng.permutation(len(scenes))
    n_tr = int(len(order) * 0.7)
    tr_set, te_set = set(order[:n_tr].tolist()), set(order[n_tr:].tolist())

    samp = []
    for d in scenes:
        n = d['coord'].shape[0]
        samp.append(np.sort(rng.choice(n, min(PER_SCENE, n), replace=False)))

    out = {}
    for tag, ck_p in CASES:
        print('=== %s ===' % tag, flush=True)
        np.random.seed(42)
        torch.manual_seed(42)
        m = load(cfg, ck_p)
        feats = {L: [] for L in ('dec0', 'dec1', 'dec2')}
        labs = []; sid = []
        for si, d in enumerate(scenes):
            dd = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in d.items()}
            with torch.no_grad():
                m(dd)
            cr = getattr(m, '_cr_last', None) or {}
            keep = torch.from_numpy(samp[si]).cuda()
            for L in ('dec0', 'dec1', 'dec2'):
                f = cr.get(L)
                if f is None: continue
                x = f.float().view(-1, f.shape[-1])[keep]
                feats[L].append(x.cpu().numpy().astype(np.float32))
            seg = d['segment'].view(-1).numpy()
            labs.append(seg[samp[si]]); sid.append(np.full(samp[si].size, si))
            if (si + 1) % 20 == 0:
                print('  [%d/%d]' % (si + 1, len(scenes)), flush=True)
        Y = np.concatenate(labs); S = np.concatenate(sid)
        tr = np.isin(S, list(tr_set)); te = ~tr
        mt = (Y >= NB) & (Y < NC)
        tr = tr & mt; te = te & mt
        row = {}
        for L in ('dec0', 'dec1', 'dec2'):
            # 每层独立固定 seed：线性头的 nn.Linear 初始化是随机的，
            # 不固定会让两次重跑的数字有 0.3~0.5 的漂移。
            np.random.seed(42)
            torch.manual_seed(42)
            if not feats[L]: continue
            X = np.concatenate(feats[L])
            mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
            Xn = (X - mu) / sd
            Xtr = torch.from_numpy(Xn[tr]).float().cuda()
            ytr = torch.from_numpy(Y[tr] - NB).long().cuda()
            # 类别逆频次权重
            cnt = torch.bincount(ytr, minlength=NOV).float()
            w = (cnt.sum() / (cnt.clamp(min=1) * NOV)).cuda()
            head = nn.Linear(Xtr.shape[1], NOV, bias=True).cuda()
            opt = torch.optim.Adam(head.parameters(), lr=1e-3)
            crit = nn.CrossEntropyLoss(weight=w)
            head.train()
            for step in range(150):
                opt.zero_grad()
                loss = crit(head(Xtr), ytr)
                loss.backward(); opt.step()
            head.eval()
            with torch.no_grad():
                pred = head(torch.from_numpy(Xn[te]).float().cuda()).argmax(1).cpu().numpy() + NB
            gt = Y[te]
            mi, ma = miou_macro(pred, gt)
            row[L] = dict(miou=mi, macro_acc=ma, n_tr=int(tr.sum()), n_te=int(te.sum()))
            print('  %-5s mIoU=%.4f  macroAcc=%.4f  (train=%d test=%d)' % (
                L, mi, ma, tr.sum(), te.sum()), flush=True)
        out[tag] = row
        del m
        torch.cuda.empty_cache()
    with open('/data01/jh/data/linear_probe_v2.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('\n%-18s %12s %12s %12s' % ('case','dec0 mIoU','dec1 mIoU','dec2 mIoU'))
    for tag, row in out.items():
        print('%-18s %12.4f %12.4f %12.4f' % (
            tag, row.get('dec0',{}).get('miou',float('nan')),
            row.get('dec1',{}).get('miou',float('nan')),
            row.get('dec2',{}).get('miou',float('nan'))))

if __name__ == '__main__':
    main()
