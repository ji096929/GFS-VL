"""Phase 3 补充：基于支持集三维距离 d_to_bg 的边界腐蚀。

对每个支持前景点，计算它到最近支持背景点的欧氏距离 d_to_bg，
剔除 d_to_bg 最小的 20% / 30%，重提支持前景原型，再算查询集 IoU。

d_to_bg 在 voxel 后的点上计算（物理坐标），再按超点 id 映射回 downsampled 点。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import diag_bc as D  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
INTRA = os.path.join(CACHE, 'intra')
FRACS = (0.0, 0.2, 0.3, 0.4)
TAU = 0.1


def to_t(a):
    if a.dtype == np.float16:
        return torch.from_numpy(a.astype(np.float32)).cuda()
    if a.dtype.kind in 'iu':
        return torch.from_numpy(a.astype(np.int64)).cuda()
    return torch.from_numpy(a.astype(np.float32)).cuda()


def iou_of(pred, y):
    tp = int((pred & (y > 0)).sum())
    un = int((pred | (y > 0)).sum())
    return tp / max(un, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=560)
    ap.add_argument('--out', default=os.path.join(CACHE, 'phase3_erode3d.json'))
    args = ap.parse_args()

    model, cfg = D.build_model()
    dev = torch.device('cuda:0')
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))[:args.n]
    print('episodes:', len(metas), flush=True)

    ious = {f: [] for f in FRACS}
    tp_c = {f: {} for f in FRACS}
    un_c = {f: {} for f in FRACS}
    base_ious = []
    d_stats = []

    for i, m in enumerate(metas):
        v = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
        a = np.load(os.path.join(INTRA, 'intra_%03d.npz' % i))
        cls = int(m['class'])

        # support block 重新预处理，拿 voxel 后的 coord/label/seg
        b = np.load(os.path.join(DATA, m['support'] + '.npy'))
        s_coord_t, s_feat, s_label_t, s_seg_t = D.data_prepare_seg(
            b[:, 0:3], b[:, 3:6], b[:, 6], b[:, 7],
            cfg.voxel_size, cfg.voxel_max)
        s_coord = s_coord_t.numpy()
        s_label = s_label_t.numpy()
        s_seg_full = s_seg_t.numpy()

        fg = s_label == cls
        bg = ~fg
        if fg.sum() < 5 or bg.sum() < 1:
            for f in FRACS:
                ious[f].append(iou_of(v['q_margin'] > 0, v['q_y']))
            base_ious.append(iou_of(v['q_margin'] > 0, v['q_y']))
            continue

        # 每个前景点的 d_to_bg
        tree_bg = cKDTree(s_coord[bg])
        d_fg = tree_bg.query(s_coord[fg])[0]
        # 按超点 id 聚合（voxel 后的 seg）
        seg_fg = s_seg_full[fg]
        seg_d = {}
        for sid in np.unique(seg_fg):
            seg_d[int(sid)] = float(d_fg[seg_fg == sid].mean())

        # 映射到 downsampled 点：重跑 getFeatures 拿 downsampled 坐标，
        # 再按最近邻取 voxel 后的 seg（缓存里的 s_seg 已被 remap，不能用）
        support_x = torch.cat([s_coord_t, s_feat], 1).to(dev)
        s_off = torch.tensor([s_coord.shape[0]], dtype=torch.int32).to(dev)
        s_bin = torch.from_numpy((s_label == cls).astype(np.int64)).to(dev)
        with torch.no_grad():
            _, s_cd, _, _, _, _ = model.getFeatures(
                support_x, s_off, s_bin, None)
        idx_nn = cKDTree(s_coord).query(s_cd.cpu().numpy())[1]
        s_seg_down = s_seg_full[idx_nn].astype(np.int64)
        d_down = np.array([seg_d.get(int(s), np.nan) for s in s_seg_down])
        d_stats.append(np.nanmean(d_down))
        s_y = to_t(v['s_y'].astype(np.float32))
        s_h = to_t(v['s_h'])
        z_s = F.normalize(s_h, dim=-1)
        z_q = F.normalize(to_t(v['q_h']), dim=-1)
        q_seg = a['q_seg'].astype(np.int64)
        uniq, inv = np.unique(q_seg, return_inverse=True)
        n_reg = len(uniq)
        margin = v['q_margin'].astype(np.float64)
        q_y = v['q_y'].astype(np.int64)

        base_ious.append(iou_of(margin > 0, q_y))
        fg_mask = (s_y > 0.5).cpu().numpy()
        valid = fg_mask & ~np.isnan(d_down)
        for f in FRACS:
            if f == 0 or valid.sum() < 5:
                keep = fg_mask
            else:
                th = np.quantile(d_down[valid], f)
                keep = fg_mask & ~((d_down <= th) & valid)
            if keep.sum() < 3:
                keep = fg_mask
            keep_t = torch.from_numpy(keep).to(z_s.device)
            p_fg = F.normalize(z_s[keep_t].mean(0, keepdim=True), dim=-1)
            s = (z_q @ p_fg.t()).squeeze(1) / TAU
            m_sp = np.bincount(inv, weights=s.cpu().numpy().astype(np.float64),
                               minlength=n_reg) / np.maximum(
                np.bincount(inv, minlength=n_reg), 1)
            pred = m_sp[inv] > 0.5
            ious[f].append(iou_of(pred, q_y))
            c = cls
            tp_c[f][c] = tp_c[f].get(c, 0) + int((pred & (q_y > 0)).sum())
            un_c[f][c] = un_c[f].get(c, 0) + int((pred | (q_y > 0)).sum())
        if (i + 1) % 100 == 0:
            print('  %d / %d' % (i + 1, len(metas)), flush=True)

    base = float(np.mean(base_ious))
    print('\nbaseline arith %.4f  mean d_to_bg %.4f' % (base, np.nanmean(d_stats)))
    out = {'baseline': base, 'mean_d_to_bg': float(np.nanmean(d_stats)),
           'rows': []}
    for f in FRACS:
        v = np.array(ious[f])
        cum = float(np.mean([tp_c[f][c] / max(un_c[f][c], 1)
                             for c in tp_c[f]]))
        out['rows'].append({'frac': f, 'arith': float(v.mean()), 'cum': cum,
                            'delta': float(v.mean() - base)})
        print(' erode3d %.1f  arith %.4f (d %+.4f)  cum %.4f'
              % (f, v.mean(), v.mean() - base, cum), flush=True)
    json.dump(out, open(args.out, 'w'), indent=1)
    print('->', args.out)


if __name__ == '__main__':
    main()
