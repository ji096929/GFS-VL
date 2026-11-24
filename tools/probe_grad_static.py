#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XL-1 辅助：同一权重下，detach_cross=False/True 的 1-batch 梯度对比。

用途：训练日志里的 [GradNorm] 受参数名匹配影响，且需要长训练才稳定；
这里直接在一个 batch 上对比两种模式的 dec0/dec1/dec2 梯度范数，
回答「截断跨层支路后，dec1/dec2 的梯度到底少了多少」。
"""
import sys
import torch

sys.path.insert(0, '/data01/jh/code/GFS-VL')
from pointcept.datasets import build_dataset
from pointcept.engines.defaults import default_config_parser
from pointcept.models import build_model

R = '/data01/jh/code/GFS-VL/'
CFG = R + 'configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_sym_lovasz.py'
CKPT = R + 'exp/scannet200/sym_lovasz_s7/model/regis1_model_last.pth'

def load():
    cfg = default_config_parser(CFG, options=None)
    m = build_model(cfg.model)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    sd = {k.replace('module.', '', 1): v for k, v in ck.get('state_dict', ck).items()}
    msd = m.state_dict()
    kept = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    m.load_state_dict(kept, strict=False)
    return cfg, m

def grad_norms(m):
    d1 = d2 = dc0 = oth = 0.0
    for n, p in m.named_parameters():
        if p.grad is None:
            continue
        g = float(p.grad.detach().float().norm()) ** 2
        if 'dec1.' in n:
            d1 += g
        elif 'dec2.' in n:
            d2 += g
        elif 'dec0.' in n:
            dc0 += g
        else:
            oth += g
    return d1 ** 0.5, d2 ** 0.5, dc0 ** 0.5, oth ** 0.5

def main():
    cfg, m = load()
    m.cuda().train()
    ds = build_dataset(cfg.data.train)
    print('train scenes:', len(ds), flush=True)
    # 固定同一份数据 + eval（关掉 drop_path），否则两次前向不可比
    m.eval()
    d0 = ds[0]
    d = {}
    for k, v in d0.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.clone().to('cuda')
        else:
            d[k] = v
    for detach in (False, True):
        m.detach_cross = detach
        m.zero_grad(set_to_none=True)
        out = m(d)
        seg = out['seg_logits']
        loss = torch.nn.functional.cross_entropy(
            seg, d['segment'].view(-1), ignore_index=-1)
        loss.backward()
        d1, d2, dc0, oth = grad_norms(m)
        print('detach_cross=%-5s  loss=%.4f | dec1=%.3e dec2=%.3e dec0=%.3e other=%.3e' % (
            detach, float(loss), d1, d2, dc0, oth), flush=True)
        m.zero_grad(set_to_none=True)

if __name__ == '__main__':
    main()
