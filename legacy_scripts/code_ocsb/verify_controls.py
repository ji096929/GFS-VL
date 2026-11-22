"""三个对照，检验 Step 4 的增益是否来自真实空间匹配。

检验 1  常数阈值平移：ell = q_margin - theta（完全不用 U）
检验 2  空间打乱：把 delta_mem 替换为任务均值 / 随机打乱
检验 3  官方 Class-Cumulative mIoU：按类累加 sum(TP)/sum(Union)
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ocsb_faithful import OCSB  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
THETAS = [-0.4, -0.2, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def to_t(a):
    if a.dtype == np.float16:
        return torch.from_numpy(a.astype(np.float32)).cuda()
    if a.dtype.kind in 'iu':
        return torch.from_numpy(a.astype(np.int64)).cuda()
    return torch.from_numpy(a.astype(np.float32)).cuda()


def interp(cf, cd, val, k=3):
    d, idx = cKDTree(cd).query(cf, k=k)
    w = 1.0 / (d + 1e-8)
    w /= w.sum(1, keepdims=True)
    return (val[idx] * w).sum(1)


def tp_union(ell_down, ep):
    e = interp(ep['q_coord_full'].cpu().numpy().astype(np.float64),
               ep['q_coord'].cpu().numpy().astype(np.float64),
               ell_down.detach().cpu().numpy().astype(np.float64))
    y = ep['q_y_full'].cpu().numpy()
    pred = e > 0
    tp = int((pred & (y > 0)).sum())
    un = int((pred | (y > 0)).sum())
    return tp, un


def main():
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))[:100]
    eps = [None] * len(metas)
    for i, m in enumerate(metas):
        d = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
        ep = {k: to_t(d[k]) for k in d.files}
        ep['cls'] = m['class']
        ep['y_full'] = d['q_y_full'].astype(np.int64)
        eps[i] = ep
    dim = int(eps[0]['s_h'].shape[1])

    # ---------- 检验 1：常数阈值平移 ----------
    print('=' * 70)
    print('检验 1  常数阈值平移  ell = q_margin - theta')
    print('%-7s %8s %8s' % ('theta', 'mIoU', 'delta'))
    base = None
    base_cum = None
    for th in THETAS:
        ious, preds = [], []
        for ep in eps:
            e = interp(ep['q_coord_full'].cpu().numpy().astype(np.float64),
                       ep['q_coord'].cpu().numpy().astype(np.float64),
                       (ep['q_margin'] - th).detach().cpu().numpy()
                       .astype(np.float64))
            pred = e > 0
            y = ep['q_y_full'].cpu().numpy()
            tp = int((pred & (y > 0)).sum())
            un = int((pred | (y > 0)).sum())
            ious.append(tp / max(un, 1))
            preds.append(pred)
        v = float(np.mean(ious))
        if th == 0.0:
            base = v
        # 累计 mIoU
        tp_c, un_c = {}, {}
        for pred, ep in zip(preds, eps):
            c = ep['cls']
            y = ep['q_y_full'].cpu().numpy()
            tp_c[c] = tp_c.get(c, 0) + int((pred & (y > 0)).sum())
            un_c[c] = un_c.get(c, 0) + int((pred | (y > 0)).sum())
        cum = float(np.mean([tp_c[c] / max(un_c[c], 1) for c in tp_c]))
        if th == 0.0:
            base_cum = cum
        print('%-7.2f %8.4f %8s  cum %.4f (%+.4f)'
              % (th, v, '' if base is None else '%+.4f' % (v - base), cum,
                 cum - (base_cum if base_cum is not None else cum)))
    print('baseline (theta=0) = %.4f' % base)

    # ---------- 检验 2：空间打乱 ----------
    print('\n' + '=' * 70)
    print('检验 2  空间打乱对照（Step 4，形式 A）')
    m = OCSB(dim, tau=0.1, tau_b=0.6, fusion_mode='A', gamma_fuse=1.0).cuda()
    deltas, miss, fas = [], [], []
    with torch.no_grad():
        for ep in eps:
            out = m(ep, enable_csr=True)
            deltas.append(out['delta_mem'].detach())
    rng = np.random.RandomState(0)
    print('%-24s %8s %8s %8s %8s'
          % ('mode', 'gamma', 'mIoU', 'delta', 'cumul'))
    for g in (1.75, 2.0):
        real, meanr, shuf = [], [], []
        ells = {'real': [], 'mean': [], 'shuf': []}
        for ep, d in zip(eps, deltas):
            dn = d.cpu().numpy()
            perm = rng.permutation(len(dn))
            e_real = ep['q_margin'] + g * d
            e_mean = ep['q_margin'] + g * float(dn.mean())
            e_shuf = ep['q_margin'] + g * torch.from_numpy(
                dn[perm]).cuda().to(d.dtype)
            real.append(tp_union(e_real, ep))
            meanr.append(tp_union(e_mean, ep))
            shuf.append(tp_union(e_shuf, ep))
            ells['real'].append(e_real)
            ells['mean'].append(e_mean)
            ells['shuf'].append(e_shuf)
        for tag, tt, key in (('真实空间对齐', real, 'real'),
                             ('替换为任务均值', meanr, 'mean'),
                             ('随机打乱顺序', shuf, 'shuf')):
            ious = [tp / max(un, 1) for tp, un in tt]
            tp_c, un_c = {}, {}
            for ell, ep in zip(ells[key], eps):
                v = interp(ep['q_coord_full'].cpu().numpy().astype(np.float64),
                           ep['q_coord'].cpu().numpy().astype(np.float64),
                           ell.detach().cpu().numpy().astype(np.float64))
                pred = v > 0
                y = ep['q_y_full'].cpu().numpy()
                c = ep['cls']
                tp_c[c] = tp_c.get(c, 0) + int((pred & (y > 0)).sum())
                un_c[c] = un_c.get(c, 0) + int((pred | (y > 0)).sum())
            cum = float(np.mean([tp_c[c] / max(un_c[c], 1) for c in tp_c]))
            print('%-24s %8.2f %8.4f %+8.4f %8.4f'
                  % (tag, g, float(np.mean(ious)),
                     float(np.mean(ious)) - base, cum))

    # ---------- 检验 3：官方累计 mIoU ----------
    print('\n' + '=' * 70)
    print('检验 3  Class-Cumulative mIoU（按类累加 TP / Union）')
    cls_ids = sorted({ep['cls'] for ep in eps})
    print('%-18s %8s %8s %8s' % ('version', 'mIoU', 'cum-mIoU', 'delta'))
    for step, g in (('Step1', 0.0), ('Step2', 1.75), ('Step3', 1.75),
                    ('Step4', 1.75), ('Step4', 2.0)):
        tp_c = {c: 0 for c in cls_ids}
        un_c = {c: 0 for c in cls_ids}
        per = []
        for ep, d in zip(eps, deltas):
            if step == 'Step1':
                ell = ep['q_margin']
            elif step == 'Step2':
                o2 = m(ep)
                B = o2['B_idx']
                zq = m.project(ep['q_h'])
                s0_q = (zq @ o2['p_fg']) / m.tau
                dd = m.memory_delta(s0_q, o2['z_R'][B], o2['q_reg'][B], zq)
                ell = ep['q_margin'] + g * dd
            else:
                csr = step == 'Step4'
                out = m(ep, enable_csr=csr)
                ell = ep['q_margin'] + g * out['delta_mem']
            tp, un = tp_union(ell, ep)
            tp_c[ep['cls']] += tp
            un_c[ep['cls']] += un
            per.append(tp / max(un, 1))
        cums = [tp_c[c] / max(un_c[c], 1) for c in cls_ids]
        print('%-18s %8.4f %8.4f %8.4f'
              % ('%s g=%.2f' % (step, g), float(np.mean(per)),
                 float(np.mean(cums)), float(np.mean(cums)) - base))
        print('    per-class cum: ' + ' '.join(
            '%d:%.3f' % (c, tp_c[c] / max(un_c[c], 1)) for c in cls_ids))

    json.dump({'theta_base': base}, open(
        os.path.join(CACHE, 'controls.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
