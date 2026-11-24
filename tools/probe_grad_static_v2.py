#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XL-1 辅助 v2：8 场景平均的梯度对比（ep0 随机头 vs ep20 收敛态）。

* ep0：Stage-1 backbone+seg_head，novel/cross 头随机初始化（固定 seed）
* ep20：sym_lovasz_s7 完整权重
* 两种状态都对比 detach_cross=False vs True 的 dec0/dec1/dec2 L2 与 RMS
* eval() 模式保证同一状态两次前向逐比特一致
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
CFG = R + 'configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_sym_lovasz.py'
STAGE1 = R + 'exp/scannet200/ptv3_bk/model/model_best.pth'
SYM20 = R + 'exp/scannet200/sym_lovasz_s7/model/regis1_model_last.pth'
NSCENE = 8

def build(ck_p, seed):
    torch.manual_seed(seed)
    cfg = default_config_parser(CFG, options=None)
    m = build_model(cfg.model)
    ck = torch.load(ck_p, map_location='cpu', weights_only=False)
    sd = {k.replace('module.', '', 1): v for k, v in ck.get('state_dict', ck).items()}
    msd = m.state_dict()
    kept = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    m.load_state_dict(kept, strict=False)
    return cfg, m

def gn(m):
    acc = {g: [0.0, 0.0] for g in ('dec0', 'dec1', 'dec2', 'other')}
    for n, p in m.named_parameters():
        if p.grad is None: continue
        g = p.grad.detach().float()
        grp = 'other'
        if 'dec0.' in n: grp = 'dec0'
        elif 'dec1.' in n: grp = 'dec1'
        elif 'dec2.' in n: grp = 'dec2'
        acc[grp][0] += float((g * g).sum().item())
        acc[grp][1] += float(g.numel())
    return {k: dict(L2=v[0] ** 0.5, RMS=(v[0] ** 0.5) / max(v[1] ** 0.5, 1.0)) for k, v in acc.items()}

def main():
    cfg = default_config_parser(CFG, options=None)
    ds = build_dataset(cfg.data.train)
    idxs = list(range(0, len(ds), 150))[:NSCENE]
    loader = DataLoader(Subset(ds, idxs), batch_size=1, shuffle=False,
                        num_workers=8, collate_fn=collate_fn)
    def _to(v, k):
        if not torch.is_tensor(v): return v
        if k == 'offset': return v.reshape(-1)
        if v.dim() > 1 and v.shape[0] == 1: return v.squeeze(0)
        return v
    scenes = [{k: _to(v, k) for k, v in d.items()} for d in loader]
    print('scenes: %d' % len(scenes), flush=True)

    out = {}
    for state, ck_p in (('ep0_random_head', STAGE1), ('ep20_sym_s7', SYM20)):
        print('=== %s ===' % state, flush=True)
        cfg2, m = build(ck_p, seed=7)
        m.eval().cuda()
        out[state] = {}
        for detach in (False, True):
            m.detach_cross = detach
            agg = {g: [0.0, 0.0] for g in ('dec0', 'dec1', 'dec2', 'other')}
            for d in scenes:
                dd = {k: (v.clone().cuda() if torch.is_tensor(v) else v) for k, v in d.items()}
                m.zero_grad(set_to_none=True)
                o = m(dd)
                loss = torch.nn.functional.cross_entropy(
                    o['seg_logits'], dd['segment'].view(-1), ignore_index=-1)
                loss.backward()
                r = gn(m)
                for g in agg:
                    agg[g][0] += r[g]['L2'] ** 2
                    agg[g][1] += 1
            res = {}
            for g in agg:
                mean_l2 = (agg[g][0] / agg[g][1]) ** 0.5
                res[g] = dict(L2=mean_l2)
            out[state]['detach' if detach else 'nodetach'] = res
            print('  detach=%-5s | ' % detach + ' | '.join(
                '%s L2=%.3e' % (g, res[g]['L2']) for g in ('dec0','dec1','dec2','other')), flush=True)
        m.zero_grad(set_to_none=True)
        del m
        torch.cuda.empty_cache()
    with open('/data01/jh/data/grad_static_v2.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('\nsaved /data01/jh/data/grad_static_v2.json')

if __name__ == '__main__':
    main()
