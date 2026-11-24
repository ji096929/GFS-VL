#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XL-1 实验 1b：冻结主干，用 dec0 / dec1 / dec2 单层做 45 类线性探针。

设计：
  * 只取 novel 45 类（label 12..56）的点
  * 每个场景随机采样 2048 点
  * 按场景切分：前 70% 场景训练，后 30% 验证（无点级泄漏）
  * 闭式 ridge regression（one-hot 目标），报验证集 overall acc 与逐类 acc 均值

用法：
  python tools/probe_linear.py
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/GFS-VL')
from pointcept.datasets import build_dataset
from pointcept.engines.defaults import default_config_parser
from pointcept.models import build_model

R = '/data01/jh/code/GFS-VL/'
STRUCT_CFG = R + 'configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_sym_lovasz.py'
NB, NC = 12, 57
PER_SCENE = 2048

CASES = [
    ('base_lovasz_s7', R + 'exp/scannet200/base_lovasz_s7/model/regis1_model_last.pth'),
    ('sym_lovasz_s7',  R + 'exp/scannet200/sym_lovasz_s7/model/regis1_model_last.pth'),
]

def load_model(cfg, ck_p):
    m = build_model(cfg.model)
    ck = torch.load(ck_p, map_location='cpu', weights_only=False)
    sd = {k.replace('module.', '', 1): v for k, v in ck.get('state_dict', ck).items()}
    msd = m.state_dict()
    kept = {}
    for k, v in sd.items():
        if k in msd and msd[k].shape == v.shape:
            kept[k] = v
    m.load_state_dict(kept, strict=False)
    m.eval().cuda()
    return m

def main():
    cfg = default_config_parser(STRUCT_CFG, options=None)
    ds = build_dataset(cfg.data.val)
    rng = np.random.default_rng(20261124)
    idxs = list(range(0, len(ds), 3))     # 104 场景
    n_tr = int(len(idxs) * 0.7)
    feats = {L: [] for L in ('dec0', 'dec1', 'dec2')}
    labels = []
    scene_id = []
    out = {}
    for tag, ck_p in CASES:
        print('=== %s ===' % tag, flush=True)
        model = load_model(cfg, ck_p)
        feats = {L: [] for L in ('dec0', 'dec1', 'dec2')}
        labels = []; scene_id = []
        for n, i in enumerate(idxs):
            d = ds[i]
            seg = d['segment'].view(-1).numpy()
            m = (seg >= NB) & (seg < NC)
            if m.sum() < 10:
                continue
            keep = np.where(m)[0]
            if keep.size > PER_SCENE:
                keep = rng.choice(keep, PER_SCENE, replace=False)
            for k, v in list(d.items()):
                if isinstance(v, torch.Tensor):
                    d[k] = v.to('cuda')
            with torch.no_grad():
                model(d)
            cr = getattr(model, '_cr_last', None) or {}
            for L in ('dec0', 'dec1', 'dec2'):
                f = cr.get(L)
                if f is None:
                    continue
                x = f.float().view(-1, f.shape[-1])[torch.from_numpy(keep).cuda()]
                feats[L].append(x.cpu().numpy().astype(np.float32))
            labels.append(seg[keep].astype(np.int64))
            scene_id.append(np.full(keep.size, n, dtype=np.int64))
            if (n + 1) % 20 == 0:
                print('  [%d/%d]' % (n + 1, len(idxs)), flush=True)
        if not labels:
            print('  no data'); continue
        Y = np.concatenate(labels); S = np.concatenate(scene_id)
        tr = S < n_tr; te = ~tr
        row = {}
        for L in ('dec0', 'dec1', 'dec2'):
            if not feats[L]:
                continue
            X = np.concatenate(feats[L])
            mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
            Xn = (X - mu) / sd
            # 闭式 ridge：W = (X^T X + λI)^-1 X^T Y
            Xtr = torch.from_numpy(Xn[tr]).double()
            Ytr = torch.zeros(Xtr.shape[0], NC - NB, dtype=torch.float64)
            yl = torch.from_numpy(Y[tr] - NB)
            Ytr[torch.arange(Xtr.shape[0]), yl] = 1.0
            D = Xtr.shape[1]
            A = Xtr.t() @ Xtr + 1.0 * torch.eye(D, dtype=torch.float64)
            B = Xtr.t() @ Ytr
            W = torch.linalg.solve(A, B)
            Xte = torch.from_numpy(Xn[te]).double()
            pred = (Xte @ W).argmax(1).numpy() + NB
            gt = Y[te]
            acc = float((pred == gt).mean())
            per = []
            for c in range(NB, NC):
                mc = gt == c
                if mc.sum() > 0:
                    per.append(float((pred[mc] == c).mean()))
            row[L] = dict(acc=acc, macro=float(np.mean(per)), n_tr=int(tr.sum()), n_te=int(te.sum()))
            print('  %-5s acc=%.4f  macro=%.4f  (train=%d test=%d)' % (
                L, acc, row[L]['macro'], tr.sum(), te.sum()), flush=True)
        out[tag] = row
        del model
        torch.cuda.empty_cache()
    with open('/data01/jh/data/linear_probe.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('saved /data01/jh/data/linear_probe.json')
    print('\n%-18s %10s %10s %10s' % ('case', 'dec0 acc', 'dec1 acc', 'dec2 acc'))
    for tag, row in out.items():
        print('%-18s %10.4f %10.4f %10.4f' % (
            tag, row.get('dec0', {}).get('acc', float('nan')),
            row.get('dec1', {}).get('acc', float('nan')),
            row.get('dec2', {}).get('acc', float('nan'))))

if __name__ == '__main__':
    main()
