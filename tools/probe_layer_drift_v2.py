#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XL-1 实验 1a v2：三层漂移（eRank + 均值余弦 + 点级余弦）。

改进：
  * DataLoader(num_workers=8) 一次性把 52 个场景读进 CPU RAM 复用
  * 需要 GFS_CAPTURE_LAYERS=1
  * Stage-1 锚点固定为 exp/scannet200/ptv3_bk/model/model_best.pth
  * 每场景固定 512 个采样点，额外报「点级逐点余弦」（同一场景同一点）
"""
import json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, '/data01/jh/code/GFS-VL')
from pointcept.datasets import build_dataset, collate_fn
from pointcept.engines.defaults import default_config_parser
from pointcept.models import build_model

R = '/data01/jh/code/GFS-VL/'
STRUCT_CFG = R + 'configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_sym_lovasz.py'
STAGE1 = R + 'exp/scannet200/ptv3_bk/model/model_best.pth'
CASES = [
    ('Stage1_K12_ep81', STAGE1),
    ('base_lovasz_s7',  R + 'exp/scannet200/base_lovasz_s7/model/regis1_model_last.pth'),
    ('sym_lovasz_s7',   R + 'exp/scannet200/sym_lovasz_s7/model/regis1_model_last.pth'),
    ('rep3_lovasz_s7',  R + 'exp/scannet200/rep3_lovasz_s7/model/regis1_model_last.pth'),
    ('lnonly_s42',      R + 'exp/scannet200/lnonly_s42/model/regis1_model_last.pth'),
]
NSAMP = 512

def erank(cov):
    s = torch.linalg.eigvalsh(cov).clamp(min=0)
    s = s[s > 1e-12]
    if s.numel() == 0:
        return float('nan')
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))

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
    assert os.environ.get('GFS_CAPTURE_LAYERS'), '需要 GFS_CAPTURE_LAYERS=1'
    cfg = default_config_parser(STRUCT_CFG, options=None)
    ds = build_dataset(cfg.data.val)
    idxs = list(range(0, len(ds), 6))
    print('scenes: %d' % len(idxs), flush=True)
    # 一次性读进 CPU RAM
    loader = DataLoader(Subset(ds, idxs), batch_size=1, shuffle=False,
                        num_workers=8, collate_fn=collate_fn)
    def _to_cpu(v, k):
        if not torch.is_tensor(v):
            return v
        v = v.cpu()
        if k == 'offset':
            return v.reshape(-1)
        if v.dim() > 1 and v.shape[0] == 1:
            return v.squeeze(0)
        return v

    scenes = []
    for d in loader:
        scenes.append({k: _to_cpu(v, k) for k, v in d.items()})
    print('loaded %d scenes to RAM' % len(scenes), flush=True)

    rng = np.random.default_rng(20261124)
    samp_idx = []
    for d in scenes:
        n = d['coord'].shape[0]
        k = min(NSAMP, n)
        samp_idx.append(np.sort(rng.choice(n, k, replace=False)))

    out = {}
    point_feats = {}
    for tag, ck_p in CASES:
        if not os.path.exists(ck_p):
            print('[skip] ' + tag, flush=True); continue
        print('=== %s ===' % tag, flush=True)
        m = load_model(cfg, ck_p)
        acc = {}
        pf = {L: [] for L in ('dec0', 'dec1', 'dec2')}
        for si, d in enumerate(scenes):
            dd = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in d.items()}
            with torch.no_grad():
                m(dd)
            cr = getattr(m, '_cr_last', None) or {}
            for L in ('dec0', 'dec1', 'dec2'):
                f = cr.get(L)
                if f is None:
                    continue
                x = f.float().view(-1, f.shape[-1])
                xi = x[torch.from_numpy(samp_idx[si]).cuda()]
                pf[L].append(xi.cpu().numpy().astype(np.float32))
                xd = x.double()
                if L not in acc:
                    D = xd.shape[1]
                    acc[L] = dict(sum=torch.zeros(D, dtype=torch.float64, device='cuda'),
                                  sxx=torch.zeros(D, D, dtype=torch.float64, device='cuda'),
                                  n=0)
                a = acc[L]
                a['sum'] += xd.sum(0); a['sxx'] += xd.t() @ xd; a['n'] += xd.shape[0]
            if (si + 1) % 20 == 0:
                print('  [%d/%d]' % (si + 1, len(scenes)), flush=True)
        row = {}
        for L, a in acc.items():
            mu = a['sum'] / a['n']
            cov = a['sxx'] / a['n'] - mu.unsqueeze(1) @ mu.unsqueeze(0)
            row[L] = dict(erank=erank(cov), dim=int(mu.numel()),
                          mean_norm=float(mu.norm()), mu=mu.cpu().tolist())
        out[tag] = row
        point_feats[tag] = {L: np.stack(v) for L, v in pf.items() if v}
        print('  ' + str({k: (round(v['erank'], 2), round(v['mean_norm'], 2)) for k, v in row.items()}), flush=True)
        del m
        torch.cuda.empty_cache()

    # 相对 Stage-1 的余弦
    S = point_feats.get('Stage1_K12_ep81', {})
    Smu = {L: np.array(out['Stage1_K12_ep81'][L]['mu']) for L in S}
    print('\n=== 相对 Stage-1 的漂移 ===')
    print('%-18s %8s | %10s %10s | %10s %10s | %10s %10s' % (
        'case','',  'dec0 mu','dec0 pt','dec1 mu','dec1 pt','dec2 mu','dec2 pt'))
    for tag, row in out.items():
        if tag == 'Stage1_K12_ep81':
            continue
        cells = []
        for L in ('dec0','dec1','dec2'):
            m_ = np.array(row[L]['mu'])
            cmu = float(Smu[L] @ m_ / (np.linalg.norm(Smu[L]) * np.linalg.norm(m_) + 1e-12))
            a = point_feats[tag].get(L); b = S.get(L)
            if a is not None and b is not None and a.shape == b.shape:
                an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
                bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
                cpt = float((an * bn).sum(-1).mean())
            else:
                cpt = float('nan')
            cells.append((cmu, cpt))
        print('%-18s %8s | %10.4f %10.4f | %10.4f %10.4f | %10.4f %10.4f' % (
            tag, '', cells[0][0], cells[0][1], cells[1][0], cells[1][1], cells[2][0], cells[2][1]))
    with open('/data01/jh/data/layer_drift_v2.json', 'w') as f:
        json.dump(out, f)
    print('\nsaved /data01/jh/data/layer_drift_v2.json')

if __name__ == '__main__':
    main()
