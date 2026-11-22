"""第二环：B_pos / B_neg / C_pos 的语义成分与稠密几何特征。

在原始 .npy 稠密点上按 .segs.json 超点分组，计算几何量。
"""
import argparse
import json
import os
import sys

import numpy as np

CACHE = '/data01/jh/data/ocsb_cache'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
NAMES_PATH = '/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt'
CLASSES = [13, 19, 10, 3]   # counter, cabinet, bathtub, chair
PER_CLASS = 10


def pca_feats(pts):
    n = len(pts)
    if n < 4:
        return None
    mn, mx = pts.min(0), pts.max(0)
    bbox = mx - mn
    vol = float(bbox[0] * bbox[1] * bbox[2])
    p = pts - pts.mean(0)
    cov = p.T @ p / n
    w, v = np.linalg.eigh(cov)
    w = np.maximum(w, 1e-12)
    l3, l2, l1 = w[0], w[1], w[2]
    return (n, bbox, vol, (l2 - l3) / l1, (l1 - l2) / l1, l3 / l1,
            abs(float(v[2, 0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=PER_CLASS)
    ap.add_argument('--out', default=os.path.join(CACHE, 'ring2.json'))
    args = ap.parse_args()

    names = [l.strip() for l in open(NAMES_PATH)]
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))

    feats = {k: [] for k in ('B_pos', 'B_neg', 'C_pos')}
    sem = {}
    rgb_stat = {k: [] for k in ('B_pos', 'B_neg', 'C_pos')}
    trunc = {k: [] for k in ('B_pos', 'B_neg', 'C_pos')}
    zrel = {k: [] for k in ('B_pos', 'B_neg', 'C_pos')}
    n_ep = 0

    for c in CLASSES:
        eps = [m for m in metas if m['class'] == c][:args.per_class]
        cname = names[c]
        sem.setdefault(cname, {'mixed': 0, 'pure': 0, 'total': 0, 'top3': {}})
        for m in eps:
            d = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
            xyz, rgb, lab, segoff = [], [], [], []
            for i, nm in enumerate(m['u']):
                r = np.load(os.path.join(DATA, nm + '.npy'))
                xyz.append(r[:, 0:3].astype(np.float64))
                rgb.append(r[:, 3:6].astype(np.float64))
                lab.append(r[:, 6].astype(np.int64))
                segoff.append(r[:, 7].astype(np.int64) + i * 10 ** 7)
            xyz = np.concatenate(xyz)
            rgb = np.concatenate(rgb)
            lab = np.concatenate(lab)
            seg = np.concatenate(segoff)
            uniq, inv = np.unique(seg, return_inverse=True)
            n_u = len(uniq)

            u_seg = d['u_seg'].astype(np.int64)
            u_p0 = d['u_p0'].astype(np.float64)
            u_y = d['u_y'].astype(np.float64)
            cnt = np.bincount(u_seg, minlength=n_u).astype(np.float64)
            q_reg = np.bincount(u_seg, weights=u_p0,
                                minlength=n_u) / np.maximum(cnt, 1)
            fg = np.bincount(u_seg, weights=u_y,
                             minlength=n_u) / np.maximum(cnt, 1)

            B = q_reg >= 0.6
            B_pos = B & (fg > 0.5)
            B_neg = B & (fg <= 0.5)
            C_pos = (q_reg < 0.5) & (fg > 0.5)

            order = np.argsort(inv, kind='stable')
            sorted_inv = inv[order]
            bounds = np.searchsorted(sorted_inv, np.arange(n_u + 1))
            z_min = xyz[order, 2].min()
            # 每个 block 的 1m 边界（场景绝对坐标）
            pts_block = np.zeros(len(xyz), dtype=np.int64)
            off = 0
            block_bounds = []
            for i, nm2 in enumerate(m['u']):
                rr = np.load(os.path.join(DATA, nm2 + '.npy'))
                n_b = len(rr)
                pts_block[off:off + n_b] = i
                x, y = rr[:, 0], rr[:, 1]
                block_bounds.append((float(x.min()), float(x.max()),
                                     float(y.min()), float(y.max())))
                off += n_b
            for r in range(n_u):
                sl = order[bounds[r]:bounds[r + 1]]
                if len(sl) < 4:
                    continue
                pts = xyz[sl]
                pf = pca_feats(pts)
                if pf is None:
                    continue
                n_pt, bbox, vol, plan, lin, scat, nz = pf
                rgbs = rgb[sl]
                bright = float(rgbs.mean())
                std = float(rgbs.std())
                blk = int(pts_block[sl[0]])
                bx0, bx1, by0, by1 = block_bounds[blk]
                edge = ((np.abs(pts[:, 0] - bx1) < 0.05) |
                        (np.abs(pts[:, 0] - bx0) < 0.05) |
                        (np.abs(pts[:, 1] - by1) < 0.05) |
                        (np.abs(pts[:, 1] - by0) < 0.05))
                edge_frac = float(edge.mean())
                z_rel = float(pts[:, 2].mean() - z_min)
                for key, mask in (('B_pos', B_pos), ('B_neg', B_neg),
                                  ('C_pos', C_pos)):
                    if not mask[r]:
                        continue
                    feats[key].append((n_pt, bbox[0], bbox[1], bbox[2], vol,
                                       plan, lin, scat, nz))
                    rgb_stat[key].append((bright, std))
                    trunc[key].append(edge_frac)
                    zrel[key].append(z_rel)
                if B_neg[r]:
                    sem[cname]['total'] += 1
                    if fg[r] == 0.0:
                        sem[cname]['pure'] += 1
                    else:
                        sem[cname]['mixed'] += 1
                    la = lab[sl]
                    vals, cnts = np.unique(la, return_counts=True)
                    dom = int(vals[cnts.argmax()])
                    dn = names[dom] if dom < len(names) else str(dom)
                    if dn != cname:
                        sem[cname]['top3'][dn] = sem[cname]['top3'].get(dn, 0) + 1
            n_ep += 1
        print('class %-12s done (%d eps)' % (cname, len(eps)), flush=True)

    out = {'n_ep': n_ep, 'per_class': {}, 'feats': {}, 'rgb': {},
           'trunc': {}, 'zrel': {}}
    for cname in sem:
        s = sem[cname]
        tot = max(s['total'], 1)
        top3 = sorted(s['top3'].items(), key=lambda x: -x[1])[:3]
        out['per_class'][cname] = {
            'n_B_neg': s['total'],
            'mixed_pct': s['mixed'] / tot,
            'pure_pct': s['pure'] / tot,
            'top3': [[k, v / tot] for k, v in top3]}
    for key in ('B_pos', 'B_neg', 'C_pos'):
        f = np.array(feats[key]) if feats[key] else np.zeros((0, 9))
        out['feats'][key] = {
            'n': int(len(f)),
            'n_pt': float(np.median(f[:, 0])) if len(f) else 0,
            'bbox_x': float(np.median(f[:, 1])) if len(f) else 0,
            'bbox_y': float(np.median(f[:, 2])) if len(f) else 0,
            'bbox_z': float(np.median(f[:, 3])) if len(f) else 0,
            'vol': float(np.median(f[:, 4])) if len(f) else 0,
            'planarity': float(np.median(f[:, 5])) if len(f) else 0,
            'linearity': float(np.median(f[:, 6])) if len(f) else 0,
            'scattering': float(np.median(f[:, 7])) if len(f) else 0,
            'nz': float(np.median(f[:, 8])) if len(f) else 0,
        }
        r = np.array(rgb_stat[key]) if rgb_stat[key] else np.zeros((0, 2))
        out['rgb'][key] = {'bright': float(np.median(r[:, 0])) if len(r) else 0,
                           'std': float(np.median(r[:, 1])) if len(r) else 0}
        t = np.array(trunc[key]) if trunc[key] else np.zeros(0)
        out['trunc'][key] = float(np.median(t)) if len(t) else 0
        z = np.array(zrel[key]) if zrel[key] else np.zeros(0)
        out['zrel'][key] = float(np.median(z)) if len(z) else 0

    json.dump(out, open(args.out, 'w'), indent=1)
    print('\n=== B_neg 语义成分 ===')
    for cname, v in out['per_class'].items():
        print('%-12s n=%4d  mixed %.1f%%  pure %.1f%%  top3: %s'
              % (cname, v['n_B_neg'], 100 * v['mixed_pct'],
                 100 * v['pure_pct'],
                 ', '.join('%s %.0f%%' % (k, 100 * p) for k, p in v['top3'])))
    print('\n=== 稠密几何 ===')
    print('%-6s %6s %7s %7s %9s %9s %9s %9s %6s %6s %6s %6s'
          % ('grp', 'n', 'pts', 'vol', 'planar', 'linear', 'scat', '|nz|',
             'bright', 'std', 'edge', 'zrel'))
    for key in ('B_pos', 'B_neg', 'C_pos'):
        f = out['feats'][key]
        r = out['rgb'][key]
        print('%-6s %6d %7.0f %7.3f %9.3f %9.3f %9.3f %9.3f %6.0f %6.1f %6.2f %6.2f'
              % (key, f['n'], f['n_pt'], f['vol'], f['planarity'],
                 f['linearity'], f['scattering'], f['nz'], r['bright'],
                 r['std'], out['trunc'][key], out['zrel'][key]))
    print('->', args.out)


if __name__ == '__main__':
    main()
