"""第一环：数据与任务划分均衡性核查（10 个 novel 类）。"""
import json
import os
import sys

import numpy as np

CACHE = '/data01/jh/data/ocsb_cache'
BLOCK_CLS = os.path.join(CACHE, 'block_classes.json')
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
NAMES_PATH = '/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt'


def main():
    names = [l.strip() for l in open(NAMES_PATH)]
    bcls = json.load(open(BLOCK_CLS))
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))

    # 312 场景的类别
    scene_cls = {}
    for name, cls in bcls.items():
        sc = name.split('_block_')[0]
        scene_cls.setdefault(sc, set()).update(cls)

    # 每类的场景/块数
    cls_scenes = {c: set() for c in NEW_IDS}
    cls_blocks = {c: 0 for c in NEW_IDS}
    for name, cls in bcls.items():
        sc = name.split('_block_')[0]
        for c in cls:
            if c in cls_scenes:
                cls_scenes[c].add(sc)
                cls_blocks[c] += 1

    # per-episode 缓存统计
    ep_stats = {}
    for m in metas:
        p = os.path.join(CACHE, 'val_%03d.npz' % m['idx'])
        d = np.load(p)
        y = d['q_y_full'].astype(np.float64)
        a = np.load(os.path.join(CACHE, 'intra', 'intra_%03d.npz' % m['idx']))
        seg = a['q_seg_full'].astype(np.int64)
        fg = float(y.mean())
        n_sp = int(len(np.unique(seg)))
        # B precision（用 u_p0 / u_y）
        u_seg = d['u_seg'].astype(np.int64)
        u_p0 = d['u_p0'].astype(np.float64)
        u_y = d['u_y'].astype(np.float64)
        n_u = int(u_seg.max()) + 1
        cnt_u = np.bincount(u_seg, minlength=n_u).astype(np.float64)
        cnt_u = np.maximum(cnt_u, 1)
        q_reg = np.bincount(u_seg, weights=u_p0, minlength=n_u) / cnt_u
        fg_frac = np.bincount(u_seg, weights=u_y, minlength=n_u) / cnt_u
        B = q_reg >= 0.6
        B_pos = B & (fg_frac > 0.5)
        prec = float(B_pos.sum() / max(B.sum(), 1))
        ep_stats[m['idx']] = {'fg': fg, 'n_sp': n_sp, 'B_prec': prec,
                              'n_B': int(B.sum())}

    # 逐类汇总
    print('%-16s %6s %7s %6s %8s %8s %16s %14s %9s %9s'
          % ('class', 'scenes', 'blocks', 'tasks', 'q_sc', 's_sc', 'fg_med (IQR)',
             'sp_med (IQR)', 'base', 'Bprec'))
    rows = []
    for c in NEW_IDS:
        eps = [m for m in metas if m['class'] == c]
        q_sc = set(m['query'].split('_block_')[0] for m in eps)
        s_sc = set(m['support'].split('_block_')[0] for m in eps)
        fgs = np.array([ep_stats[m['idx']]['fg'] for m in eps])
        sps = np.array([ep_stats[m['idx']]['n_sp'] for m in eps])
        bps = np.array([ep_stats[m['idx']]['B_prec'] for m in eps])
        # 基线 mIoU 从 ablation_cum175.json 读
        rows.append({'class': c, 'name': names[c], 'scenes': len(cls_scenes[c]),
                     'blocks': cls_blocks[c], 'tasks': len(eps),
                     'q_scenes': len(q_sc), 's_scenes': len(s_sc),
                     'fg_med': float(np.median(fgs)),
                     'fg_iqr': [float(np.percentile(fgs, 25)),
                                float(np.percentile(fgs, 75))],
                     'sp_med': float(np.median(sps)),
                     'sp_iqr': [float(np.percentile(sps, 25)),
                                float(np.percentile(sps, 75))],
                     'b_prec': float(np.mean(bps))})

    # 基线 mIoU
    abl = json.load(open(os.path.join(CACHE, 'ablation_cum175.json')))
    per = abl['Step4 OASB+CSR_per_ep']
    base_by_cls = {}
    for r in per:
        base_by_cls.setdefault(r['class'], []).append(r['base_iou'])
    for r in rows:
        r['base'] = float(np.mean(base_by_cls.get(r['class'], [np.nan])))

    for r in rows:
        print('%-16s %6d %7d %6d %8d %8d %7.3f [%.3f,%.3f] %6.0f [%3.0f,%3.0f]'
              ' %9.4f %9.4f'
              % (r['name'], r['scenes'], r['blocks'], r['tasks'],
                 r['q_scenes'], r['s_scenes'], r['fg_med'], r['fg_iqr'][0],
                 r['fg_iqr'][1], r['sp_med'], r['sp_iqr'][0], r['sp_iqr'][1],
                 r['base'], r['b_prec']))

    json.dump(rows, open(os.path.join(CACHE, 'ring1.json'), 'w'), indent=1)
    print('->', os.path.join(CACHE, 'ring1.json'))


if __name__ == '__main__':
    main()
