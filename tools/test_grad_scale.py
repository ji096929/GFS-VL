#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cross_grad_scale 单测：前向不变 + 跨层梯度线性按 g 缩放。

验证方式：梯度是线性的，所以
    grad(g=0.2) == grad(g=0) + 0.2 * (grad(g=1) - grad(g=0))
测的是 backbone.dec.dec2 模块输出的梯度（含 backbone 反向 + 跨层支路两部分）。
"""
import sys
import torch

sys.path.insert(0, '/data01/jh/code/GFS-VL')
from pointcept.datasets import build_dataset
from pointcept.engines.defaults import default_config_parser
from pointcept.models import build_model

cfg = default_config_parser('exp/scannetpp/scpp_sym_lovasz_k5/config.py', options=None)
m = build_model(cfg.model).eval().cuda()

ds = build_dataset(cfg.data.val)
d = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in ds[0].items()}

feats = {}
def mk(s):
    def hook(mod, inp, out):
        f = out.feat
        f.retain_grad()
        feats[s] = f
    return hook
m.backbone.dec.dec2.register_forward_hook(mk(2))
m.backbone.dec.dec1.register_forward_hook(mk(1))

def run(g, detach=False):
    m.cross_grad_scale = g
    m.detach_cross = detach
    m.zero_grad(set_to_none=True)
    out = m(d)
    logits = out['seg_logits'].detach().clone()
    loss = out['loss']
    loss.backward()
    return logits, {s: feats[s].grad.detach().clone() for s in feats}

l1, g1 = run(1.0)
l0, g0 = run(0.0, detach=True)
l2, g2 = run(0.2)

print('forward max|g1-g02| = %.3e' % float((l1 - l2).abs().max()))
print('forward max|g1-g0|  = %.3e' % float((l1 - l0).abs().max()))
for s in (1, 2):
    pred = g0[s] + 0.2 * (g1[s] - g0[s])
    err = float((pred - g2[s]).abs().max())
    rel = err / max(float(g1[s].abs().max()), 1e-12)
    print('dec%d: |grad| g1=%.4e g0=%.4e g02=%.4e | 线性预测误差 %.3e (rel %.3e)' % (
        s, float(g1[s].norm()), float(g0[s].norm()), float(g2[s].norm()), err, rel))
    assert rel < 1e-4, 'dec%d 的梯度未按 g 线性缩放' % s
print('PASS')
