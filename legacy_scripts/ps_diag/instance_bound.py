"""细节 2：几何实例边界约束 AI 补标范围的上限对照。

两种容器边界：
  A. 真实实例边界  instance.npy 的每个 instance id
  B. 无监督边界    region_cache 里「预测 novel 类内的 3D 连通分量」(radius=0.05)

对每种容器统计：
  - 容器数 / 总点数 / 平均大小
  - 容器 novel 纯度 = GT novel 点 / (GT novel + GT base)，ignore 不计入分母
  - 按点数加权的整体纯度
  - 在不同纯度门槛下可「安全补标」的点数

另算实例 -> 区域的碎片化：一个真实实例被切成几个连通块、
一个连通块跨几个真实实例（决定 AI 补标会不会越界）。
"""
import argparse
import glob
import os

import numpy as np
import torch

DATA = '/data01/jh/data/scannet200_gfs'
REGION = '/data01/jh/data/region_cache'
BASE = 12
NOVEL = 45
IGN = -1
# ScanNet200 -> GFS 57 类的映射（base 12 在 0..11，novel 45 在 12..56）
# 与 vlm_common.convert_to_gfs 一致
import sys
sys.path.insert(0, '/data01/jh/code/ps_diag')
import vlm_common as V  # noqa: E402


def load_scene(split, scene):
    d = os.path.join(DATA, split, scene)
    coord = np.load(os.path.join(d, 'coord.npy')).astype(np.float32)
    seg200 = np.load(os.path.join(d, 'segment200.npy')).reshape(-1)
    inst = np.load(os.path.join(d, 'instance.npy')).reshape(-1)
    return coord, V.convert_to_gfs(seg200), inst


def stats_from_hist(hn, hb, n_ign):
    """容器纯度：novel / (novel + base)，ignore 不计入分母。"""
    n_nov = int(hn.sum())
    n_bas = int(hb.sum())
    denom = n_nov + n_bas
    return (n_nov + n_bas + int(n_ign), n_nov, n_bas,
            float(n_nov / denom) if denom else -1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 个场景，0=全部')
    args = ap.parse_args()

    scenes = sorted(os.listdir(os.path.join(DATA, args.split)))
    if args.limit:
        scenes = scenes[:args.limit]

    # ---------- A. 真实实例边界 ----------
    rows_inst = []   # (size, novelty_purity, n_nov, n_bas, n_ign)
    for si, sc in enumerate(scenes):
        coord, gfs, inst = load_scene(args.split, sc)
        ids = np.unique(inst)
        ids = ids[ids > 0]
        for i in ids:
            m = inst == i
            g = gfs[m]
            hn = np.bincount(g[(g >= BASE)], minlength=NOVEL)
            hb = np.bincount(g[(g >= 0) & (g < BASE)], minlength=BASE)
            n_ign = int((g < 0).sum())
            size, nn, nb, pur = stats_from_hist(hn, hb, n_ign)
            rows_inst.append((size, pur, nn, nb, n_ign))
        if (si + 1) % 100 == 0:
            print('  inst %d / %d' % (si + 1, len(scenes)), flush=True)

    # ---------- B. 无监督连通分量（来自 region_cache） ----------
    rows_reg = []
    for f in sorted(glob.glob(os.path.join(REGION, 'regions_%s_s*.pt' % args.split))):
        d = torch.load(f, map_location='cpu')
        for sc, r in sorted(d.items()):
            if args.limit and sc not in scenes:
                continue
            for i in range(len(r['pred'])):
                hn = r['hist_n'][i].astype(np.int64)
                hb = r['hist_b'][i].astype(np.int64)
                n_ign = int(r['n_ign'][i])
                size, nn, nb, pur = stats_from_hist(hn, hb, n_ign)
                rows_reg.append((size, pur, nn, nb, n_ign))

    def report(name, rows):
        if not rows:
            print('\n=== %s ===\n（空）' % name)
            return
        a = np.array(rows, dtype=np.float64)
        size, pur = a[:, 0], a[:, 1]
        valid = pur >= 0
        tot = size.sum()
        wpur = (size[valid] * pur[valid]).sum() / max(size[valid].sum(), 1)
        print('\n=== %s ===' % name)
        print('容器数 %d  总点数 %.0f  平均大小 %.1f  中位大小 %.0f'
              % (len(rows), tot, size.mean(), np.median(size)))
        print('纯度(按点数加权) %.4f  纯度(容器平均) %.4f  有效纯度容器 %.1f%%'
              % (wpur, pur[valid].mean(), 100.0 * valid.mean()))
        print('%-8s %10s %10s' % ('纯度>=', '容器占比', '可补点数'))
        for th in (0.5, 0.7, 0.8, 0.9, 0.95, 1.0):
            m = pur >= th
            print('%-8.2f %9.1f%% %10.0f' % (th, 100.0 * m.mean(), size[m].sum()))
        print('novel 点合计 %.0f  base 点合计 %.0f  ignore 点合计 %.0f'
              % (a[:, 2].sum(), a[:, 3].sum(), a[:, 4].sum()))

    report('A. 真实实例边界 (instance.npy)', rows_inst)
    report('B. 无监督边界 (预测类内连通分量)', rows_reg)

    # ---------- 碎片化：实例 vs 连通分量 ----------
    # 用 region_cache 里存不了点索引，这里重算一个场景级别的粗对齐：
    # 统计「同一预测类内连通分量」与「真实实例」的重叠拓扑
    print('\n=== 碎片化（前 50 场景抽样）===')
    from scipy import ndimage
    npairs, n_split, n_merge = 0, [], []
    for sc in scenes[:50]:
        coord, gfs, inst = load_scene(args.split, sc)
        # 只对和实例重叠的 novel 连通块做拓扑统计，复用圆心算法
        # 全局一次连通分量太粗（房间连通），所以按 instance 空间近似：
        # 每个真实实例内部看是否被 level-set 切成多块（用体素 0.05 的连通性）
        ids = np.unique(inst)
        ids = ids[ids > 0]
        for i in ids[:64]:
            m = inst == i
            pts = coord[m]
            if len(pts) < 20:
                continue
            mn = pts.min(0)
            g = np.floor((pts - mn) / 0.05).astype(np.int64)
            shape = tuple(g.max(0) + 1)
            grid = np.zeros(shape, dtype=bool)
            grid[g[:, 0], g[:, 1], g[:, 2]] = True
            _, n = ndimage.label(grid, structure=np.ones((3, 3, 3), int))
            n_split.append(n)
            npairs += 1
    if n_split:
        print('实例内体素连通块数：平均 %.2f  中位 %.0f  >1 的实例占 %.1f%%'
              % (np.mean(n_split), np.median(n_split),
                 100.0 * np.mean(np.array(n_split) > 1)))
        print('（体素网格只做可达性近似，用于估计实例的几何碎片化程度）')


if __name__ == '__main__':
    main()
