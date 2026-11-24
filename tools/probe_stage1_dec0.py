#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""补 1/补 2：Stage-1 与 U1 的 dec0 特征 eRank + 均值。

用途：
  1) base/sym/rep3/lnonly 的 dec0 均值相对 Stage-1 的余弦（是否锁在原位）
  2) 同数据集 K=12（Stage-1）vs K=30（U1 fullsup）的 dec0 eRank

用法：
  python tools/probe_stage1_dec0.py
"""
import json, os, sys
import torch

sys.path.insert(0, '/data01/jh/code/GFS-VL')
from pointcept.datasets import build_dataset
from pointcept.engines.defaults import default_config_parser
from pointcept.models import build_model

def erank_from_cov(cov):
    s = torch.linalg.eigvalsh(cov).clamp(min=0)
    s = s[s > 1e-12]
    if s.numel() == 0:
        return float('nan')
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))

CASES = [
    ('Sc200_Stage1_K12',   'exp/scannet200/base_lovasz_s42/config.py',
                           'exp/scannet200/ptv3_bk_ontr.pth', 6),
    ('Scpp_Stage1_K12',    'exp/scannetpp/scpp_sym_lovasz_k5/config.py',
                           'exp/scannetpp/ptv3_bk/model/model_best.pth', 1),
    ('Scpp_U1_K30fullsup', 'exp/scannetpp/scpp_oracle_fullsup20/config.py',
                           'exp/scannetpp/scpp_oracle_fullsup20/model/regis1_model_last.pth', 1),
]

def main():
    out = {}
    for tag, cfg_p, ck_p, step in CASES:
        if not (os.path.exists(cfg_p) and os.path.exists(ck_p)):
            print('[skip] %s (missing file)' % tag); continue
        print('=== %s ===' % tag, flush=True)
        cfg = default_config_parser(cfg_p, options=None)
        model = build_model(cfg.model)
        ck = torch.load(ck_p, map_location='cpu', weights_only=False)
        sd = {k.replace('module.', '', 1): v for k, v in ck.get('state_dict', ck).items()}
        miss, unexp = model.load_state_dict(sd, strict=False)
        print('  load: missing=%d unexpected=%d' % (len(miss), len(unexp)), flush=True)
        model.eval().cuda()
        ds = build_dataset(cfg.data.val)
        idxs = list(range(0, len(ds), step))
        acc = None
        for n, i in enumerate(idxs):
            d = ds[i]
            for k, v in list(d.items()):
                if isinstance(v, torch.Tensor):
                    d[k] = v.to('cuda')
            with torch.no_grad():
                model(d)
            cr = getattr(model, '_cr_last', None) or {}
            if 'dec0' not in cr:
                continue
            x = cr['dec0'].float().view(-1, cr['dec0'].shape[-1]).double()
            if acc is None:
                D = x.shape[1]
                acc = dict(sum=torch.zeros(D, dtype=torch.float64, device='cuda'),
                           sxx=torch.zeros(D, D, dtype=torch.float64, device='cuda'),
                           n=0)
            acc['sum'] += x.sum(0)
            acc['sxx'] += x.t() @ x
            acc['n'] += x.shape[0]
            if (n + 1) % 20 == 0:
                print('  [%d/%d]' % (n + 1, len(idxs)), flush=True)
        if acc is None:
            print('  no dec0 captured'); continue
        mu = acc['sum'] / acc['n']
        cov = acc['sxx'] / acc['n'] - mu.unsqueeze(1) @ mu.unsqueeze(0)
        r = dict(erank=erank_from_cov(cov), dim=int(mu.numel()),
                 mean_norm=float(mu.norm()), n=int(acc['n']), mu=mu.cpu().tolist())
        out[tag] = r
        print('  dec0 eRank=%.2f  |mu|=%.3f  n=%d' % (r['erank'], r['mean_norm'], r['n']), flush=True)
        del model
        torch.cuda.empty_cache()
    with open('/data01/jh/data/stage1_dec0.json', 'w') as f:
        json.dump(out, f)
    print('saved /data01/jh/data/stage1_dec0.json')

if __name__ == '__main__':
    main()
