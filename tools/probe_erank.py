#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""X1/X8：逐层有效秩（eRank）+ 特征均值。

对每个 Stage-2 checkpoint：
  * 用它保存的 config.py 构建结构（base 无 cross_layers，sym/rep3 有）
  * 加载权重 -> val 集抽样前向 -> 累积每层 sum(x) 与 sum(xx^T)
  * eRank = exp(-Σ p_i log p_i)，p_i 为协方差特征值占比
  * 同时存均值向量，供跨模型特征漂移比较

用法：
  python tools/probe_erank.py --names base_lovasz_s42 sym_lovasz_s42 rep3_lovasz_s42
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, '/data01/jh/code/GFS-VL')
from pointcept.datasets import build_dataset
from pointcept.engines.defaults import default_config_parser
from pointcept.models import build_model

ROOT = '/data01/jh/code/GFS-VL/exp/scannet200'
CKPT = 'model/regis1_model_last.pth'

def erank_from_cov(cov):
    s = torch.linalg.eigvalsh(cov).clamp(min=0)
    s = s[s > 1e-12]
    if s.numel() == 0:
        return float('nan')
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--step', type=int, default=6)
    ap.add_argument('--out', default='/data01/jh/data/erank_stage2.json')
    a = ap.parse_args()
    results = {}
    for name in a.names:
        cfg_path = os.path.join(ROOT, name, 'config.py')
        ck_path = os.path.join(ROOT, name, CKPT)
        if not (os.path.exists(cfg_path) and os.path.exists(ck_path)):
            print('[skip] %s missing config/ckpt' % name); continue
        print('=== %s ===' % name, flush=True)
        cfg = default_config_parser(cfg_path, options=None)
        model = build_model(cfg.model)
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        sd = {k.replace('module.', '', 1): v for k, v in ck['state_dict'].items()}
        miss, unexp = model.load_state_dict(sd, strict=False)
        print('  load: missing=%d unexpected=%d' % (len(miss), len(unexp)), flush=True)
        model.eval().cuda()
        ds = build_dataset(cfg.data.val)
        idxs = list(range(0, len(ds), a.step))
        acc = {}
        for n, i in enumerate(idxs):
            d = ds[i]
            for k, v in list(d.items()):
                if isinstance(v, torch.Tensor):
                    d[k] = v.to('cuda')
            with torch.no_grad():
                model(d)
            cr = getattr(model, '_cr_last', None) or {}
            for key, f in cr.items():
                x = f.float().view(-1, f.shape[-1]).double()
                if key not in acc:
                    D = x.shape[1]
                    acc[key] = dict(
                        sum=torch.zeros(D, dtype=torch.float64, device='cuda'),
                        sxx=torch.zeros(D, D, dtype=torch.float64, device='cuda'),
                        n=0)
                a_ = acc[key]
                a_['sum'] += x.sum(0)
                a_['sxx'] += x.t() @ x
                a_['n'] += x.shape[0]
            if (n + 1) % 10 == 0:
                print('  [%d/%d]' % (n + 1, len(idxs)), flush=True)
        row = {}
        for key, a_ in acc.items():
            mu = a_['sum'] / a_['n']
            cov = a_['sxx'] / a_['n'] - mu.unsqueeze(1) @ mu.unsqueeze(0)
            row[key] = dict(erank=erank_from_cov(cov), dim=int(mu.numel()),
                            mean_norm=float(mu.norm()), n=int(a_['n']),
                            mu=mu.cpu().tolist())
        results[name] = row
        print('  ' + str({k: round(v['erank'], 2) for k, v in row.items()}), flush=True)
        del model
        torch.cuda.empty_cache()
    with open(a.out, 'w') as f:
        json.dump(results, f)
    print('saved ' + a.out)
    print('\n%-24s %10s %10s %10s' % ('name', 'dec0', 'dec1', 'dec2'))
    for name, row in results.items():
        print('%-24s %10.2f %10.2f %10.2f' % (
            name,
            row.get('dec0', {}).get('erank', float('nan')),
            row.get('dec1', {}).get('erank', float('nan')),
            row.get('dec2', {}).get('erank', float('nan'))))

if __name__ == '__main__':
    main()
