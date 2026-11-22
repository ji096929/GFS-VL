"""四种候选信号在 TP/FN/FP/TN 上的四分组诊断。

S1  旧公式：Σ ŵ_R relu(z_R·z_q - c0)          （复现 TN 倒置）
S2  纯加权相似度：Σ ŵ_R (z_R·z_q)
S3  减查询点背景得分：Σ ŵ_R relu(z_R·z_q - c_bg)
S4  模块一/二基准也改成背景裕度后，再算 S3
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ocsb_cmc import OCSBCMC  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
CKPT = '/data01/jh/weights/sc0_1w1s.pth'
GROUPS = ('TP', 'FN', 'FP', 'TN')


def to_t(a):
    if a.dtype == np.float16:
        return torch.from_numpy(a.astype(np.float32)).cuda()
    if a.dtype.kind in 'iu':
        return torch.from_numpy(a.astype(np.int64)).cuda()
    return torch.from_numpy(a.astype(np.float32)).cuda()


def group_masks(ep):
    y = ep['q_y'] > 0
    pos = ep['q_margin'] > 0
    return {'TP': y & pos, 'FN': y & (~pos), 'FP': (~y) & pos,
            'TN': (~y) & (~pos)}


def signals(ep, out, cmc):
    z_q = out['z_q']
    M = out['M_feat']
    w = out['M_w']
    if M.shape[0] == 0:
        z = torch.zeros(z_q.shape[0], device=z_q.device)
        return z, z, z
    wh = w / (w.sum() + 1e-6)
    sim = z_q @ M.t()
    c0 = cmc.topm_cosine(z_q, out['a_fg'])
    c_bg = (z_q @ out['a_bg'].t()).max(1).values
    if ep.get('base_protos') is not None and ep['base_protos'].numel():
        bp = F.normalize(ep['base_protos'].to(z_q.dtype), dim=-1)
        c_bg = torch.maximum(c_bg, (z_q @ bp.t()).max(1).values)
    S1 = (wh.unsqueeze(0) * F.relu(sim - c0.unsqueeze(1))).sum(1)
    S2 = (wh.unsqueeze(0) * sim).sum(1)
    S3 = (wh.unsqueeze(0) * F.relu(sim - c_bg.unsqueeze(1))).sum(1)
    return S1, S2, S3


def main():
    table = torch.load(CKPT, map_location='cpu')['state_dict'][
        'module.base_prototypes'].float().cuda()
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))[:100]

    res = {}
    for tag, bg_base in (('B∪C', False), ('B-only', False),
                         ('B∪C-bgb', True)):
        cmc = None
        agg = {g: {s: [] for s in ('S1', 'S2', 'S3')} for g in GROUPS}
        for m in metas:
            d = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
            ep = {k: to_t(d[k]) for k in d.files}
            ep['base_protos'] = table
            if cmc is None:
                cmc = OCSBCMC(int(ep['s_h'].shape[1]), tau=0.1, tau_b=0.6,
                              m_anchor=3).cuda()
            with torch.no_grad():
                out = cmc(ep, use_csr=True, bg_baseline=bg_base)
            if tag == 'B-only' and out['z_B'].shape[0]:
                keep = out['w_B'] > 0
                M2 = out['z_B'][keep] if keep.any() else out['z_B']
                w2 = out['w_B'][keep] if keep.any() else out['w_B']
                out = dict(out)
                out['M_feat'] = M2
                out['M_w'] = w2
            S1, S2, S3 = signals(ep, out, cmc)
            masks = group_masks(ep)
            for g, mk in masks.items():
                if mk.any():
                    agg[g]['S1'].append(float(S1[mk].mean()))
                    agg[g]['S2'].append(float(S2[mk].mean()))
                    agg[g]['S3'].append(float(S3[mk].mean()))
        res[tag] = agg

    for tag, agg in res.items():
        print('\n=== %s ===' % tag)
        print('%-4s %9s %9s %9s' % ('grp', 'S1(old)', 'S2(pure)', 'S3(bg)'))
        for g in GROUPS:
            row = agg[g]
            print('%-4s %9.4f %9.4f %9.4f'
                  % (g, np.mean(row['S1']), np.mean(row['S2']),
                     np.mean(row['S3'])))
        for s in ('S1', 'S2', 'S3'):
            fn = np.mean(agg['FN'][s])
            tn = np.mean(agg['TN'][s])
            fp = np.mean(agg['FP'][s])
            tp = np.mean(agg['TP'][s])
            print('  %s: FN-TN %+.4f  FN-FP %+.4f  TP-TN %+.4f'
                  % (s, fn - tn, fn - fp, tp - tn))


if __name__ == '__main__':
    main()
