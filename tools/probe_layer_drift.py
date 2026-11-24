#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XL-1 实验 1a：三层（dec0/dec1/dec2）相对 Stage-1 的漂移。

统一用 sym 的结构（cross_layers=(1,2)）加载每个权重，只换 state_dict，
这样 Stage-1 / base / sym / rep3 / lnonly 都能抓到同样三层特征。

用法：
  python tools/probe_layer_drift.py
"""
import json, os, sys
import torch

sys.path.insert(0, '/data01/jh/code/GFS-VL')
from pointcept.datasets import build_dataset
from pointcept.engines.defaults import default_config_parser
from pointcept.models import build_model

R = '/data01/jh/code/GFS-VL/'
STRUCT_CFG = R + 'configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_sym_lovasz.py'

CASES = [
    ('Stage1_K12_ep81', R + 'exp/scannet200/ptv3_bk/model/model_best.pth'),
    ('base_lovasz_s7',  R + 'exp/scannet200/base_lovasz_s7/model/regis1_model_last.pth'),
    ('sym_lovasz_s7',   R + 'exp/scannet200/sym_lovasz_s7/model/regis1_model_last.pth'),
    ('rep3_lovasz_s7',  R + 'exp/scannet200/rep3_lovasz_s7/model/regis1_model_last.pth'),
    ('lnonly_s42',      R + 'exp/scannet200/lnonly_s42/model/regis1_model_last.pth'),
]

def erank(cov):
    s = torch.linalg.eigvalsh(cov).clamp(min=0)
    s = s[s > 1e-12]
    if s.numel() == 0:
        return float('nan')
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))

def main():
    cfg = default_config_parser(STRUCT_CFG, options=None)
    ds = build_dataset(cfg.data.val)
    idxs = list(range(0, len(ds), 6))   # 52 / 312
    out = {}
    for tag, ck_p in CASES:
        if not os.path.exists(ck_p):
            print('[skip] %s' % tag, flush=True); continue
        print('=== %s ===' % tag, flush=True)
        model = build_model(cfg.model)
        ck = torch.load(ck_p, map_location='cpu', weights_only=False)
        sd = {k.replace('module.', '', 1): v for k, v in ck.get('state_dict', ck).items()}
        msd = model.state_dict()
        kept = {}
        skipped = []
        for k, v in sd.items():
            if k not in msd:
                continue
            if msd[k].shape == v.shape:
                kept[k] = v
            else:
                skipped.append(k)
        model.load_state_dict(kept, strict=False)
        print('  load: kept=%d/%d shape_mismatch=%d (novel 头按结构跳过)'
              % (len(kept), len(sd), len(skipped)), flush=True)
        model.eval().cuda()
        acc = {}
        for n, i in enumerate(idxs):
            d = ds[i]
            for k, v in list(d.items()):
                if isinstance(v, torch.Tensor):
                    d[k] = v.to('cuda')
            with torch.no_grad():
                model(d)
            cr = getattr(model, '_cr_last', None) or {}
            for key in ('dec0', 'dec1', 'dec2'):
                f = cr.get(key)
                if f is None:
                    continue
                x = f.float().view(-1, f.shape[-1]).double()
                if key not in acc:
                    D = x.shape[1]
                    acc[key] = dict(sum=torch.zeros(D, dtype=torch.float64, device='cuda'),
                                    sxx=torch.zeros(D, D, dtype=torch.float64, device='cuda'), n=0)
                a = acc[key]
                a['sum'] += x.sum(0); a['sxx'] += x.t() @ x; a['n'] += x.shape[0]
            if (n + 1) % 20 == 0:
                print('  [%d/%d]' % (n + 1, len(idxs)), flush=True)
        row = {}
        for key, a in acc.items():
            mu = a['sum'] / a['n']
            cov = a['sxx'] / a['n'] - mu.unsqueeze(1) @ mu.unsqueeze(0)
            row[key] = dict(erank=erank(cov), dim=int(mu.numel()),
                            mean_norm=float(mu.norm()), n=int(a['n']), mu=mu.cpu().tolist())
        out[tag] = row
        print('  ' + str({k: (round(v['erank'], 2), round(v['mean_norm'], 2)) for k, v in row.items()}), flush=True)
        del model
        torch.cuda.empty_cache()
    with open('/data01/jh/data/layer_drift.json', 'w') as f:
        json.dump(out, f)
    print('saved /data01/jh/data/layer_drift.json')

if __name__ == '__main__':
    main()
