"""第四环对照诊断：B 超点是否与查询集点云空间重合。

data_prepare_seg 会对每个 block 独立中心化，U 与 Q 的坐标不可比，
所以这里复现其 voxelize/crop 并保留到原始 .npy 的索引，
用世界坐标计算 U 的 B 点与 Q 点云的最近距离。

输出每类：
  - B 原始点数、落在查询点云 2/5/10 cm 内的比例
  - B_pos（真值目标类）vs B_neg 的重合比例
  - 查询点云被 B 覆盖的比例
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, '/data01/jh/code/COSeg')
sys.path.insert(0, '/data01/jh/code')
from util.data_util import voxelize  # noqa: E402
import diag_bc as D  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
NAMES_PATH = '/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
PER_CLASS = 30
RADII = (0.02, 0.05, 0.10)


def prepare_with_idx(coord, label, seg, voxel_size, voxel_max):
    """复现 data_prepare_seg，返回保留点在原始数组中的索引。"""
    idx_all = np.arange(len(coord))
    coord = coord.astype(np.float32).copy()
    if voxel_size:
        coord -= np.min(coord, 0)
        uq = voxelize(coord, voxel_size)
        idx_all = idx_all[uq]
        coord, label, seg = coord[uq], label[uq], seg[uq]
    if voxel_max and label.shape[0] > voxel_max:
        init = label.shape[0] // 2
        crop = np.argsort(np.sum(np.square(coord - coord[init]), 1))[:voxel_max]
        idx_all = idx_all[crop]
        label, seg = label[crop], seg[crop]
    return idx_all, label, seg


def main():
    cfg = D.load_cfg_from_cfg_file(D.CFG_PATH)
    cfg = D.merge_cfg_from_list(cfg, [
        'data_root', DATA, 'n_way', '1', 'k_shot', '1'])
    names = [l.strip() for l in open(NAMES_PATH)]
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))
    rows = []
    for c in NEW_IDS:
        eps = [m for m in metas if m['class'] == c][:PER_CLASS]
        acc = {r: [] for r in RADII}
        acc_pos = {r: [] for r in RADII}
        acc_neg = {r: [] for r in RADII}
        acc_qcov = []
        n_bpts, n_qpts = [], []
        for m in eps:
            d = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
            xyz_w, lab, seg, bids = [], [], [], []
            for i, nm in enumerate(m['u']):
                r = np.load(os.path.join(DATA, nm + '.npy'))
                keep, l, s = prepare_with_idx(
                    r[:, 0:3], r[:, 6], r[:, 7],
                    cfg.voxel_size, cfg.voxel_max)
                xyz_w.append(r[keep, 0:3].astype(np.float64))
                lab.append(l); seg.append(s + i * 10 ** 7)
                bids.append(np.full(len(keep), i, np.int64))
            xyz_w = np.concatenate(xyz_w)
            lab = np.concatenate(lab)
            seg = np.concatenate(seg)
            bid = np.concatenate(bids)
            uniq, inv = np.unique(seg, return_inverse=True)
            n_u = len(uniq)

            u_seg = d['u_seg'].astype(np.int64)
            if int(u_seg.max()) + 1 > n_u:
                print('  !! %s #%d 编号超界 u_seg.max=%d n_u=%d'
                      % (names[c], m['idx'], u_seg.max(), n_u), flush=True)
            u_p0 = d['u_p0'].astype(np.float64)
            u_y = d['u_y'].astype(np.float64)
            cnt = np.maximum(
                np.bincount(u_seg, minlength=n_u).astype(np.float64), 1)
            q_sp = np.bincount(u_seg, weights=u_p0, minlength=n_u) / cnt
            fg_sp = np.bincount(u_seg, weights=u_y, minlength=n_u) / cnt
            B = q_sp >= 0.6
            if not B.any():
                continue

            is_b = np.isin(inv, np.where(B)[0])
            sp_fg = np.zeros(n_u, dtype=bool)
            b_idx = np.where(B)[0]
            sp_fg[b_idx] = fg_sp[b_idx] > 0.5
            is_pos = is_b & sp_fg[inv]
            is_neg = is_b & ~sp_fg[inv]

            rq = np.load(os.path.join(DATA, m['query'] + '.npy'))
            qx = rq[:, 0:3].astype(np.float64)
            dist, _ = cKDTree(qx).query(xyz_w)
            for r in RADII:
                acc[r].append(float((dist[is_b] <= r).mean()))
                if is_pos.any():
                    acc_pos[r].append(float((dist[is_pos] <= r).mean()))
                if is_neg.any():
                    acc_neg[r].append(float((dist[is_neg] <= r).mean()))
            dq, _ = cKDTree(xyz_w[is_b]).query(qx)
            acc_qcov.append(float((dq <= 0.05).mean()))
            n_bpts.append(int(is_b.sum()))
            n_qpts.append(int(len(qx)))
        rows.append({
            'class': names[c], 'n': len(eps),
            'n_B': float(np.mean(n_bpts)) if n_bpts else 0,
            'n_Q': float(np.mean(n_qpts)) if n_qpts else 0,
            **{'b_%g' % r: float(np.mean(acc[r])) for r in RADII if acc[r]},
            **{'pos_%g' % r: float(np.mean(acc_pos[r])) for r in RADII if acc_pos[r]},
            **{'neg_%g' % r: float(np.mean(acc_neg[r])) for r in RADII if acc_neg[r]},
            'q_cov': float(np.mean(acc_qcov)) if acc_qcov else 0,
        })
        print('%-12s done' % names[c], flush=True)

    print('\n%-12s %7s %7s %9s %9s %9s %9s'
          % ('class', 'B_pts', 'Q_pts', 'B@2cm', 'B@5cm', 'B@10cm', 'Qcov5cm'))
    for r in rows:
        print('%-12s %7.0f %7.0f %8.3f %8.3f %8.3f %8.3f'
              % (r['class'], r['n_B'], r['n_Q'], r['b_0.02'], r['b_0.05'],
                 r['b_0.1'], r['q_cov']))
    print('\n%-12s %9s %9s %9s' % ('class', 'B_pos@5', 'B_neg@5', 'gap'))
    for r in rows:
        p = r.get('pos_0.05', 0)
        n = r.get('neg_0.05', 0)
        print('%-12s %8.3f %8.3f %+8.3f' % (r['class'], p, n, p - n))
    print('\n%-12s %9s %9s' % ('MEAN', 'B@5cm', 'Qcov'))
    print('%-12s %8.3f %8.3f'
          % ('', np.mean([r['b_0.05'] for r in rows]),
             np.mean([r['q_cov'] for r in rows])))
    json.dump(rows, open(os.path.join(CACHE, 'ring4_overlap.json'), 'w'),
              indent=1)
    print('->', os.path.join(CACHE, 'ring4_overlap.json'))


if __name__ == '__main__':
    main()
