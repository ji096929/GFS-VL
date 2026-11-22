"""第三环：B 提纯策略 A/B/C（全 10 类 × 30 eps）。

指标：过滤后 B 精确率、B_pos 保留率、仍有 B 的任务比例。
禁止按类别写死阈值，所有规则都是类无关的。
"""
import json
import os

import numpy as np

CACHE = '/data01/jh/data/ocsb_cache'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
NAMES_PATH = '/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
PER_CLASS = 30


def pca_nz(pts):
    if len(pts) < 4:
        return 0.0
    p = pts - pts.mean(0)
    w, v = np.linalg.eigh(p.T @ p / len(pts))
    return abs(float(v[2, 0]))


def load_ep(m):
    d = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
    xyz, rgb, lab, seg = [], [], [], []
    zmin, boff = [], []
    off = 0
    for i, nm in enumerate(m['u']):
        r = np.load(os.path.join(DATA, nm + '.npy'))
        n_b = len(r)
        xyz.append(r[:, 0:3].astype(np.float64))
        rgb.append(r[:, 3:6].astype(np.float64))
        lab.append(r[:, 6].astype(np.int64))
        seg.append(r[:, 7].astype(np.int64) + i * 10 ** 7)
        zmin.append(float(r[:, 2].min()))
        boff.append((off, off + n_b))
        off += n_b
    xyz = np.concatenate(xyz)
    rgb = np.concatenate(rgb)
    lab = np.concatenate(lab)
    seg = np.concatenate(seg)
    bid = np.zeros(len(xyz), dtype=np.int64)
    for i, (a, b) in enumerate(boff):
        bid[a:b] = i
    uniq, inv = np.unique(seg, return_inverse=True)
    n_u = len(uniq)
    u_seg = d['u_seg'].astype(np.int64)
    u_p0 = d['u_p0'].astype(np.float64)
    cnt = np.maximum(
        np.bincount(u_seg, minlength=n_u).astype(np.float64), 1)
    q = np.bincount(u_seg, weights=u_p0, minlength=n_u) / cnt
    order = np.argsort(inv, kind='stable')
    bounds = np.searchsorted(inv[order], np.arange(n_u + 1))

    # 每个超点的几何 / 标签
    zr = np.zeros(n_u)
    nz = np.zeros(n_u)
    ctr = np.zeros((n_u, 3))
    dom = np.full(n_u, -1, dtype=np.int64)
    sbid = np.zeros(n_u, dtype=np.int64)
    for r in range(n_u):
        sl = order[bounds[r]:bounds[r + 1]]
        if len(sl) < 4:
            continue
        pts = xyz[sl]
        sbid[r] = int(bid[sl[0]])
        zr[r] = pts[:, 2].mean() - zmin[int(bid[sl[0]])]
        nz[r] = pca_nz(pts)
        ctr[r] = pts.mean(0)
        v, c = np.unique(lab[sl], return_counts=True)
        dom[r] = int(v[c.argmax()])

    # 支持集前景的物理姿态（S 的超点）
    s_h = d['s_h'].astype(np.float64)
    s_y = d['s_y'].astype(np.float64)
    s_seg = d['s_seg'].astype(np.int64)
    # 支持集坐标不可用（缓存没存 s_coord_down），用特征空间代替：
    # 这里改用支持集前景概率的分布作为姿态代理不可行，故策略 B 用
    # "B 集合自身的中位姿态"做类无关一致性（不依赖支持集坐标）
    return {'q': q, 'zr': zr, 'nz': nz, 'ctr': ctr, 'dom': dom, 'sbid': sbid,
            'bid': bid, 'inv': inv, 'order': order, 'bounds': bounds,
            'n_u': n_u, 'cls': m['class']}


def run_strategy(eps_data, mode, param):
    """返回 (prec, pos_keep, task_with_B)。"""
    tot = pos = kept_pos = kept = n_task_with = 0
    for e in eps_data:
        q, zr, nz = e['q'], e['zr'], e['nz']
        c = e['cls']
        B = q >= 0.6
        if not B.any():
            continue
        pos_mask = B & (e['dom'] == c)
        if mode == 'A':
            keep = B & (q >= param)
        elif mode == 'B':
            # 类无关姿态一致性：用 B 自身的中位 |nz| 作为参照
            if B.sum() >= 3:
                ref = float(np.median(nz[B]))
                keep = B & (np.abs(nz - ref) <= param)
            else:
                keep = B
        elif mode == 'C':
            # 类无关空间连通：每个 block 内只保留与最高置信超点距离 < param
            keep = np.zeros_like(B)
            for b in np.unique(e['sbid'][B]):
                idx = np.where(B & (e['sbid'] == b))[0]
                if len(idx) == 0:
                    continue
                core = idx[np.argmax(q[idx])]
                d = np.linalg.norm(e['ctr'][idx] - e['ctr'][core], axis=1)
                keep[idx[d <= param]] = True
        else:
            keep = B
        kept += int(keep.sum())
        tot += int(B.sum())
        pos += int(pos_mask.sum())
        kept_pos += int((keep & pos_mask).sum())
        if keep.any():
            n_task_with += 1
    return (kept_pos / max(kept, 1), kept_pos / max(pos, 1),
            n_task_with / max(len(eps_data), 1))


def main():
    names = [l.strip() for l in open(NAMES_PATH)]
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))
    data = {}
    for c in NEW_IDS:
        eps = [m for m in metas if m['class'] == c][:PER_CLASS]
        data[c] = [load_ep(m) for m in eps]
        print('%-12s loaded %d' % (names[c], len(eps)), flush=True)

    configs = [('baseline', None, 0),
               ('A tau0.7', 'A', 0.7), ('A tau0.8', 'A', 0.8),
               ('A tau0.85', 'A', 0.85),
               ('B nz0.20', 'B', 0.20), ('B nz0.30', 'B', 0.30),
               ('B nz0.45', 'B', 0.45),
               ('C r0.30', 'C', 0.30), ('C r0.50', 'C', 0.50),
               ('C r0.80', 'C', 0.80)]
    out = []
    for tag, mode, p in configs:
        print('\n### %s' % tag)
        print('%-12s %8s %8s %8s' % ('class', 'prec', 'pos_keep', 'task%'))
        agg_p, agg_k, agg_t = [], [], []
        for c in NEW_IDS:
            pr, pk, tt = run_strategy(data[c], mode, p)
            print('%-12s %8.3f %8.3f %8.1f%%'
                  % (names[c], pr, pk, 100 * tt))
            agg_p.append(pr); agg_k.append(pk); agg_t.append(tt)
        print('%-12s %8.3f %8.3f %8.1f%%'
              % ('MEAN', np.mean(agg_p), np.mean(agg_k), 100 * np.mean(agg_t)))
        out.append({'tag': tag, 'prec': float(np.mean(agg_p)),
                    'pos_keep': float(np.mean(agg_k)),
                    'task': float(np.mean(agg_t))})

    print('\n=== 汇总 ===')
    print('%-12s %8s %10s %8s' % ('config', 'prec', 'pos_keep', 'task%'))
    for r in out:
        print('%-12s %8.3f %10.3f %7.1f%%'
              % (r['tag'], r['prec'], r['pos_keep'], 100 * r['task']))

    json.dump(out, open(os.path.join(CACHE, 'ring3.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
