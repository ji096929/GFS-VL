#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XL-2 诊断：三层（dec0/dec1/dec2）eRank + 相对 Stage-1 的均值/点级余弦。

参数化版本：
  --config   结构配置（决定 cross_layers；无 cross 的权重会跳过 novel 头）
  --stage1   Stage-1 锚点权重
  --ckpts    name=path 列表
  --step     每 step 个 val 场景取 1 个（ScanNet200 用 6，ScanNet++ 用 1）
  --out      输出 json

需要 GFS_CAPTURE_LAYERS=1。
"""
import argparse, json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, '/data01/jh/code/GFS-VL')
from pointcept.datasets import build_dataset, collate_fn
from pointcept.engines.defaults import default_config_parser
from pointcept.models import build_model

NSAMP = 512

def erank(cov):
    s = torch.linalg.eigvalsh(cov).clamp(min=0)
    s = s[s > 1e-12]
    if s.numel() == 0:
        return float('nan')
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))

def load_model(cfg, ck_p, tag):
    m = build_model(cfg.model)
    ck = torch.load(ck_p, map_location='cpu', weights_only=False)
    sd = {k.replace('module.', '', 1): v for k, v in ck.get('state_dict', ck).items()}
    msd = m.state_dict()
    kept, skipped = {}, []
    for k, v in sd.items():
        if k not in msd:
            continue
        if msd[k].shape == v.shape:
            kept[k] = v
        else:
            skipped.append(k)
    bk = [k for k in sd if k.startswith('backbone.')]
    bk_ok = sum(1 for k in bk if k in kept)
    assert bk_ok == len(bk), '[%s] backbone 只加载 %d/%d' % (tag, bk_ok, len(bk))
    m.load_state_dict(kept, strict=False)
    print('  [%s] backbone %d/%d  shape_mismatch=%d' % (tag, bk_ok, len(bk), len(skipped)), flush=True)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--stage1', required=True)
    ap.add_argument('--ckpts', nargs='+', required=True, help='name=path')
    ap.add_argument('--step', type=int, default=6)
    ap.add_argument('--out', default='/data01/jh/data/layer_drift_v2.json')
    a = ap.parse_args()
    assert os.environ.get('GFS_CAPTURE_LAYERS'), '需要 GFS_CAPTURE_LAYERS=1'

    cfg = default_config_parser(a.config, options=None)
    ds = build_dataset(cfg.data.val)
    idxs = list(range(0, len(ds), a.step))
    print('scenes: %d / %d (step=%d)' % (len(idxs), len(ds), a.step), flush=True)
    loader = DataLoader(Subset(ds, idxs), batch_size=1, shuffle=False,
                        num_workers=8, collate_fn=collate_fn)
    scenes = [{k: _to_cpu(v, k) for k, v in d.items()} for d in loader]
    print('loaded %d scenes to RAM' % len(scenes), flush=True)

    rng = np.random.default_rng(20261124)
    samp = []
    for d in scenes:
        n = d['coord'].shape[0]
        samp.append(np.sort(rng.choice(n, min(NSAMP, n), replace=False)))

    cases = [('Stage1', a.stage1)]
    for spec in a.ckpts:
        name, path = spec.split('=', 1)
        cases.append((name, path))

    out, point_feats = {}, {}
    for tag, ck_p in cases:
        if not os.path.exists(ck_p):
            print('[skip] ' + tag, flush=True); continue
        print('=== %s ===' % tag, flush=True)
        m = load_model(cfg, ck_p, tag)
        acc, pf = {}, {L: [] for L in ('dec0', 'dec1', 'dec2')}
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
                xi = x[torch.from_numpy(samp[si]).cuda()]
                pf[L].append(xi.cpu().numpy().astype(np.float32))
                xd = x.double()
                if L not in acc:
                    D = xd.shape[1]
                    acc[L] = dict(sum=torch.zeros(D, dtype=torch.float64, device='cuda'),
                                  sxx=torch.zeros(D, D, dtype=torch.float64, device='cuda'),
                                  n=0)
                aa = acc[L]
                aa['sum'] += xd.sum(0); aa['sxx'] += xd.t() @ xd; aa['n'] += xd.shape[0]
            if (si + 1) % 20 == 0:
                print('  [%d/%d]' % (si + 1, len(scenes)), flush=True)
        row = {}
        for L, aa in acc.items():
            mu = aa['sum'] / aa['n']
            cov = aa['sxx'] / aa['n'] - mu.unsqueeze(1) @ mu.unsqueeze(0)
            row[L] = dict(erank=erank(cov), dim=int(mu.numel()),
                          mean_norm=float(mu.norm()), mu=mu.cpu().tolist())
        out[tag] = row
        point_feats[tag] = {L: np.concatenate(v, axis=0) for L, v in pf.items() if v}
        print('  ' + str({k: (round(v['erank'], 2), round(v['mean_norm'], 2)) for k, v in row.items()}), flush=True)
        del m
        torch.cuda.empty_cache()

    S = point_feats.get('Stage1', {})
    Smu = {L: np.array(out['Stage1'][L]['mu']) for L in S}
    print('\n=== 相对 Stage1 的漂移（均值余弦 / 点级余弦）===')
    print('%-24s | %10s %10s | %10s %10s | %10s %10s' % ('case','dec0 mu','dec0 pt','dec1 mu','dec1 pt','dec2 mu','dec2 pt'))
    for tag, row in out.items():
        if tag == 'Stage1':
            continue
        cells = []
        for L in ('dec0','dec1','dec2'):
            mm = np.array(row[L]['mu'])
            cmu = float(Smu[L] @ mm / (np.linalg.norm(Smu[L]) * np.linalg.norm(mm) + 1e-12))
            aa = point_feats.get(tag, {}).get(L); bb = S.get(L)
            if aa is not None and bb is not None and aa.shape == bb.shape:
                an = aa / (np.linalg.norm(aa, axis=-1, keepdims=True) + 1e-12)
                bn = bb / (np.linalg.norm(bb, axis=-1, keepdims=True) + 1e-12)
                cpt = float((an * bn).sum(-1).mean())
            else:
                cpt = float('nan')
            cells.append((cmu, cpt))
        print('%-24s | %10.4f %10.4f | %10.4f %10.4f | %10.4f %10.4f' % (
            tag, cells[0][0], cells[0][1], cells[1][0], cells[1][1], cells[2][0], cells[2][1]))
    with open(a.out, 'w') as f:
        json.dump(out, f)
    print('\nsaved ' + a.out)

if __name__ == '__main__':
    main()
