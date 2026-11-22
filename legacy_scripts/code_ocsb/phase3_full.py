"""Phase 3 完整版：β_orth 正交投影 + 边界腐蚀 + 自适应阈值。

1) β_orth ∈ {0,0.2,0.5,0.8,1.0}：把支持/查询特征沿背景原型方向正交化，
   重算支持前景原型，用余弦得分做超点平滑后的 IoU。
2) 边界腐蚀 0/20%/30%：按"与背景原型相似度"剔除最像背景的支持前景点后重提原型。
3) 自适应阈值：已在 phase3_calib.py 验证（lam=0 最优），此处复算确认。

全部在 560 eps 上与 COSeg 基线（margin > 0）同口径对比。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

CACHE = '/data01/jh/data/ocsb_cache'
INTRA = os.path.join(CACHE, 'intra')

BETAS = (0.0, 0.2, 0.5, 0.8, 1.0)
ERODE = (0.0, 0.2, 0.3)
TAU = 0.1


DEV = torch.device('cuda:0')


def to_t(a):
    if a.dtype == np.float16:
        return torch.from_numpy(a.astype(np.float32)).to(DEV)
    if a.dtype.kind in 'iu':
        return torch.from_numpy(a.astype(np.int64)).to(DEV)
    return torch.from_numpy(a.astype(np.float32)).to(DEV)


def iou_of(pred, y):
    tp = int((pred & (y > 0)).sum())
    un = int((pred | (y > 0)).sum())
    return tp / max(un, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=560)
    ap.add_argument('--out', default=os.path.join(CACHE, 'phase3_full.json'))
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    if args.cpu:
        global DEV
        DEV = torch.device('cpu')

    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))[:args.n]
    data = []
    for i, m in enumerate(metas):
        v = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
        a = np.load(os.path.join(INTRA, 'intra_%03d.npz' % i))
        s_h = to_t(v['s_h'])
        q_h = to_t(v['q_h'])
        s_y = to_t(v['s_y'].astype(np.float32))
        seg = a['q_seg'].astype(np.int64)
        margin = to_t(v['q_margin']).cpu().numpy()
        y = v['q_y'].astype(np.int64)
        uniq, inv = np.unique(seg, return_inverse=True)
        n_reg = len(uniq)
        cnt = np.bincount(inv, minlength=n_reg).astype(np.float64)
        m_sp = np.bincount(inv, weights=margin, minlength=n_reg) / np.maximum(cnt, 1)
        data.append({'s_h': s_h, 'q_h': q_h, 's_y': s_y, 'seg': inv,
                     'n_reg': n_reg, 'y': y, 'cls': int(m['class']),
                     'margin': v['q_margin'].astype(np.float64),
                     'y_full': v['q_y_full'].astype(np.int64),
                     'seg_full': a['q_seg_full'].astype(np.int64),
                     'm_sp': m_sp})
        if (i + 1) % 100 == 0:
            print('  loaded %d / %d' % (i + 1, len(metas)), flush=True)
    print('episodes:', len(data), flush=True)

    # baseline：margin > 0（downsampled 点，与 cosine 得分同空间）
    base_ious = [iou_of(d['margin'] > 0, d['y']) for d in data]
    print('baseline (margin>0) arith %.4f' % np.mean(base_ious), flush=True)

    out = {'baseline': float(np.mean(base_ious)), 'beta': [], 'erode': []}

    # ---------- 1) beta_orth ----------
    print('\n=== beta_orth (cosine prototype, superpoint smoothed) ===',
          flush=True)
    for beta in BETAS:
        ious = []
        for d in data:
            z_s = F.normalize(d['s_h'], dim=-1)
            z_q = F.normalize(d['q_h'], dim=-1)
            fg = d['s_y'] > 0.5
            if fg.sum() == 0 or (~fg).sum() == 0:
                ious.append(iou_of(d['margin'] > 0, d['y']))
                continue
            p_bg = F.normalize(z_s[~fg].mean(0, keepdim=True), dim=-1)
            if beta > 0:
                z_s_c = z_s - beta * (z_s @ p_bg.t()) * p_bg
                z_q_c = z_q - beta * (z_q @ p_bg.t()) * p_bg
                z_s_c = F.normalize(z_s_c, dim=-1)
                z_q_c = F.normalize(z_q_c, dim=-1)
            else:
                z_s_c, z_q_c = z_s, z_q
            p_fg = F.normalize(z_s_c[fg].mean(0, keepdim=True), dim=-1)
            s = (z_q_c @ p_fg.t()).squeeze(1) / TAU
            s_np = s.cpu().numpy().astype(np.float64)
            m_sp = np.bincount(d['seg'], weights=s_np,
                               minlength=d['n_reg']) / np.maximum(
                np.bincount(d['seg'], minlength=d['n_reg']), 1)
            ious.append(iou_of(m_sp[d['seg']] > 0.5, d['y']))
        out['beta'].append({'beta': beta, 'arith': float(np.mean(ious))})
        print(' beta %.1f  arith %.4f  (d %+.4f)'
              % (beta, np.mean(ious), np.mean(ious) - np.mean(base_ious)),
              flush=True)

    # ---------- 2) 边界腐蚀 ----------
    print('\n=== boundary erosion (by bg-similarity) ===', flush=True)
    for frac in ERODE:
        ious = []
        for d in data:
            z_s = F.normalize(d['s_h'], dim=-1)
            z_q = F.normalize(d['q_h'], dim=-1)
            fg = d['s_y'] > 0.5
            if fg.sum() < 5 or (~fg).sum() == 0:
                ious.append(iou_of(d['margin'] > 0, d['y']))
                continue
            p_bg = F.normalize(z_s[~fg].mean(0, keepdim=True), dim=-1)
            sim_bg = (z_s[fg] @ p_bg.t()).squeeze(1)
            if frac > 0:
                th = torch.quantile(sim_bg, 1.0 - frac)
                keep = sim_bg < th
            else:
                keep = torch.ones_like(sim_bg, dtype=torch.bool)
            if int(keep.sum()) < 3:
                ious.append(iou_of(d['margin'] > 0, d['y']))
                continue
            p_fg = F.normalize(z_s[fg][keep].mean(0, keepdim=True), dim=-1)
            s = (z_q @ p_fg.t()).squeeze(1) / TAU
            s_np = s.cpu().numpy().astype(np.float64)
            m_sp = np.bincount(d['seg'], weights=s_np,
                               minlength=d['n_reg']) / np.maximum(
                np.bincount(d['seg'], minlength=d['n_reg']), 1)
            ious.append(iou_of(m_sp[d['seg']] > 0.5, d['y']))
        out['erode'].append({'frac': frac, 'arith': float(np.mean(ious))})
        print(' erode %.1f  arith %.4f  (d %+.4f)'
              % (frac, np.mean(ious), np.mean(ious) - np.mean(base_ious)),
              flush=True)

    # ---------- 3) 自适应阈值复算 ----------
    print('\n=== adaptive theta (confirm) ===', flush=True)
    for lam in (0.0, 0.5, 1.0):
        ious = []
        for d in data:
            fg0 = float((d['m_sp'] > 0.5).mean())
            th = 0.5 + lam * max(fg0 - 0.15, 0.0)
            ious.append(iou_of(d['m_sp'][d['seg']] > th, d['y']))
        print(' lam %.1f  arith %.4f  (d %+.4f)'
              % (lam, np.mean(ious), np.mean(ious) - np.mean(base_ious)),
              flush=True)
        out.setdefault('adapt', []).append({'lam': lam,
                                            'arith': float(np.mean(ious))})

    json.dump(out, open(args.out, 'w'), indent=1)
    print('->', args.out)


if __name__ == '__main__':
    main()
