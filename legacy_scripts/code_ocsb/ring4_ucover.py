"""第四环补充：U 中真值同类物体对 Q 前景的形态覆盖。

不依赖模型，纯几何。对每个 episode：
  - U 中 label==cls 的原始点（真值前景）点数 / bbox 体积
  - Q 中 label==cls 的原始点（真值前景）点数 / bbox 体积
  - 两者点数比、体积比、最近距离
用来判断"桥梁可提供的形态信息上限"。
"""
import json
import os

import numpy as np
from scipy.spatial import cKDTree

CACHE = '/data01/jh/data/ocsb_cache'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
NAMES_PATH = '/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
PER_CLASS = 30
HULL = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])


def bbox_vol(p):
    if len(p) < 4:
        return 0.0
    e = p.max(0) - p.min(0)
    return float(np.prod(np.maximum(e, 1e-3)))


def main():
    names = [l.strip() for l in open(NAMES_PATH)]
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))
    rows = []
    for c in NEW_IDS:
        eps = [m for m in metas if m['class'] == c][:PER_CLASS]
        ratio_p, ratio_v, mind = [], [], []
        n_u_pts, n_q_pts = [], []
        n_eps_with_u = 0
        for m in eps:
            up, uext = [], []
            for nm in m['u']:
                r = np.load(os.path.join(DATA, nm + '.npy'))
                p = r[r[:, 6] == c, 0:3].astype(np.float64)
                if len(p):
                    up.append(p)
            r = np.load(os.path.join(DATA, m['query'] + '.npy'))
            qp = r[r[:, 6] == c, 0:3].astype(np.float64)
            if not len(qp):
                continue
            qu = len(up)
            if qu == 0:
                ratio_p.append(0.0); ratio_v.append(0.0)
                n_u_pts.append(0); n_q_pts.append(len(qp))
                continue
            n_eps_with_u += 1
            up = np.concatenate(up)
            ratio_p.append(len(up) / len(qp))
            vu, vq = bbox_vol(up), bbox_vol(qp)
            ratio_v.append(vu / max(vq, 1e-6))
            d, _ = cKDTree(up).query(qp)
            mind.append(float(np.percentile(d, 5)))
            n_u_pts.append(len(up)); n_q_pts.append(len(qp))
        rows.append({
            'class': names[c], 'n': len(eps),
            'has_u': n_eps_with_u / max(len(eps), 1),
            'u_pts': float(np.mean(n_u_pts)) if n_u_pts else 0,
            'q_pts': float(np.mean(n_q_pts)) if n_q_pts else 0,
            'ratio_pts': float(np.median(ratio_p)) if ratio_p else 0,
            'ratio_vol': float(np.median(ratio_v)) if ratio_v else 0,
            'd5': float(np.median(mind)) if mind else -1,
        })
        print('%-12s done' % names[c], flush=True)

    print('\n%-12s %7s %8s %8s %10s %10s %8s'
          % ('class', 'hasU%', 'U_pts', 'Q_pts', 'medPtRatio', 'medVolRatio', 'd5(m)'))
    for r in rows:
        print('%-12s %6.0f%% %8.0f %8.0f %10.2f %10.2f %8.2f'
              % (r['class'], 100 * r['has_u'], r['u_pts'], r['q_pts'],
                 r['ratio_pts'], r['ratio_vol'], r['d5']))
    print('%-12s %6.0f%% %8.0f %8.0f %10.2f %10.2f'
          % ('MEAN', 100 * np.mean([r['has_u'] for r in rows]),
             np.mean([r['u_pts'] for r in rows]),
             np.mean([r['q_pts'] for r in rows]),
             np.median([r['ratio_pts'] for r in rows]),
             np.median([r['ratio_vol'] for r in rows])))
    json.dump(rows, open(os.path.join(CACHE, 'ring4_ucover.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
