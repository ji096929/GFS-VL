#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XL-2 诊断：冻结主干 + novel 类线性探针（参数化版）。

关键修正：
  * 每场景先在 novel 点 (NB<=seg<NC) 内采样，避免 85% 预算浪费在墙/地上
  * mIoU / macroAcc 只对测试集出现过的类 (union>0) 求均值，并打印 n_present
  * 线性头 Adam(lr=2e-2, wd=1e-4) 250 步，打印首末 loss 确认收敛
  * 每层固定 torch seed，保证可复现

需要 GFS_CAPTURE_LAYERS=1。
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, '/data01/jh/code/GFS-VL')
from pointcept.datasets import build_dataset, collate_fn
from pointcept.engines.defaults import default_config_parser
from pointcept.models import build_model

PER_SCENE = 2048
STEPS = 250
LR = 2e-2
WD = 1e-4

def load_model(cfg, ck_p, tag, NB):
    m = build_model(cfg.model)
    ck = torch.load(ck_p, map_location='cpu', weights_only=False)
    sd = {k.replace('module.', '', 1): v for k, v in ck.get('state_dict', ck).items()}
    msd = m.state_dict()
    kept = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    bk = [k for k in sd if k.startswith('backbone.')]
    bk_ok = sum(1 for k in bk if k in kept)
    assert bk_ok == len(bk), '[%s] backbone 只加载 %d/%d' % (tag, bk_ok, len(bk))
    m.load_state_dict(kept, strict=False)
    print('  [%s] backbone %d/%d kept=%d/%d' % (tag, bk_ok, len(bk), len(kept), len(sd)), flush=True)
    m.eval().cuda()
    return m

def _to_cpu(v, k):
    if not torch.is_tensor(v):
        return v
    v = v.cpu()
    if k == 'offset':
        return v.reshape(-1)
    if v.dim() > 1 and v.shape[0] == 1:
        return v.squeeze(0)
    return v

def miou_macro(pred, gt, NB, NC):
    ious, accs = [], []
    for c in range(NB, NC):
        p = pred == c; g = gt == c
        u = (p | g).sum()
        if u > 0:
            ious.append(float((p & g).sum()) / float(u))
        if g.sum() > 0:
            accs.append(float((p & g).sum()) / float(g.sum()))
    return (float(np.mean(ious)) if ious else float('nan'),
            float(np.mean(accs)) if accs else float('nan'),
            len(ious))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpts', nargs='+', required=True, help='name=path')
    ap.add_argument('--num-base', type=int, default=12)
    ap.add_argument('--num-total', type=int, default=57)
    ap.add_argument('--step', type=int, default=3)
    ap.add_argument('--out', default='/data01/jh/data/linear_probe_v2.json')
    a = ap.parse_args()
    assert os.environ.get('GFS_CAPTURE_LAYERS')
    NB, NC = a.num_base, a.num_total
    NOV = NC - NB

    print("DBG config=", repr(a.config), flush=True)
    cfg = default_config_parser(a.config, options=None)
    ds = build_dataset(cfg.data.val)
    idxs = list(range(0, len(ds), a.step))
    print('scenes: %d / %d (step=%d)  NB=%d NC=%d' % (len(idxs), len(ds), a.step, NB, NC), flush=True)
    loader = DataLoader(Subset(ds, idxs), batch_size=1, shuffle=False,
                        num_workers=8, collate_fn=collate_fn)
    scenes = [{k: _to_cpu(v, k) for k, v in d.items()} for d in loader]

    rng = np.random.default_rng(20261124)
    samp = []
    for d in scenes:
        seg = d['segment'].view(-1).numpy()
        nov = np.where((seg >= NB) & (seg < NC))[0]
        if nov.size == 0:
            samp.append(np.zeros(0, dtype=np.int64)); continue
        k = min(PER_SCENE, nov.size)
        samp.append(np.sort(rng.choice(nov, k, replace=False)))
    n_valid = sum(1 for s in samp if s.size)
    print('scenes with novel points: %d' % n_valid, flush=True)

    order = rng.permutation(len(scenes))
    n_tr = int(len(order) * 0.7)
    tr_set = set(order[:n_tr].tolist())

    out = {}
    for spec in a.ckpts:
        tag, ck_p = spec.split('=', 1)
        if not os.path.exists(ck_p):
            print('[skip] ' + tag, flush=True); continue
        print('=== %s ===' % tag, flush=True)
        np.random.seed(42); torch.manual_seed(42)
        m = load_model(cfg, ck_p, tag, NB)
        feats = {L: [] for L in ('dec0','dec1','dec2')}
        labs, sid = [], []
        for si, d in enumerate(scenes):
            if samp[si].size == 0:
                continue
            dd = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in d.items()}
            with torch.no_grad():
                m(dd)
            cr = getattr(m, '_cr_last', None) or {}
            keep = torch.from_numpy(samp[si]).cuda()
            for L in ('dec0','dec1','dec2'):
                f = cr.get(L)
                if f is None: continue
                feats[L].append(f.float().view(-1, f.shape[-1])[keep].cpu().numpy().astype(np.float32))
            seg = d['segment'].view(-1).numpy()
            labs.append(seg[samp[si]]); sid.append(np.full(samp[si].size, si))
            if (si + 1) % 20 == 0:
                print('  [%d/%d]' % (si + 1, len(scenes)), flush=True)
        Y = np.concatenate(labs); S = np.concatenate(sid)
        mt = (Y >= NB) & (Y < NC)
        tr = mt & np.isin(S, list(tr_set)); te = mt & ~tr
        row = {}
        for L in ('dec0','dec1','dec2'):
            if not feats[L]: continue
            torch.manual_seed(42)
            X = np.concatenate(feats[L])
            mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
            Xn = (X - mu) / sd
            Xtr = torch.from_numpy(Xn[tr]).float().cuda()
            ytr = torch.from_numpy(Y[tr] - NB).long().cuda()
            cnt = torch.bincount(ytr, minlength=NOV).float()
            w = (cnt.sum() / (cnt.clamp(min=1) * NOV)).cuda()
            head = nn.Linear(Xtr.shape[1], NOV, bias=True).cuda()
            opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WD)
            crit = nn.CrossEntropyLoss(weight=w)
            head.train()
            l0 = lN = None
            for step in range(STEPS):
                opt.zero_grad()
                loss = crit(head(Xtr), ytr)
                loss.backward(); opt.step()
                if step == 0: l0 = float(loss)
                lN = float(loss)
            head.eval()
            with torch.no_grad():
                pred = head(torch.from_numpy(Xn[te]).float().cuda()).argmax(1).cpu().numpy() + NB
            mi, ma, npres = miou_macro(pred, Y[te], NB, NC)
            row[L] = dict(miou=mi, macro_acc=ma, n_present=npres,
                          loss_first=l0, loss_last=lN, n_tr=int(tr.sum()), n_te=int(te.sum()))
            print('  %-5s mIoU=%.4f macroAcc=%.4f present=%d/%d loss %.4f->%.4f (tr=%d te=%d)' % (
                L, mi, ma, npres, NOV, l0, lN, tr.sum(), te.sum()), flush=True)
        out[tag] = row
        del m
        torch.cuda.empty_cache()
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    print('\n%-24s %12s %12s %12s' % ('case','dec0 mIoU','dec1 mIoU','dec2 mIoU'))
    for tag, row in out.items():
        print('%-24s %12.4f %12.4f %12.4f' % (
            tag, row.get('dec0',{}).get('miou',float('nan')),
            row.get('dec1',{}).get('miou',float('nan')),
            row.get('dec2',{}).get('miou',float('nan'))))
    print('saved ' + a.out)

if __name__ == '__main__':
    main()
