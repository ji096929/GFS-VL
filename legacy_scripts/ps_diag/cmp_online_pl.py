"""Task 1 汇总：官方在线 PS / PS+逐点 AI / 我们 A 的逐类点数与伪标签质量。

全部用同一口径：只统计写入 Y_b == -1 的点（训练时 dataset 唯一会填的位置）。

输入
  /data01/jh/data/online_pl_shard0.npz   dump_online_pl.py 的逐类统计
  /data01/jh/data/offline_pl/sc200_regis_1_10_A/*.npy   A 组点级伪标签
  /data01/jh/data/scannet200_gfs/train/*/segment200.npy   GT（用于 A 的命中统计）

输出
  /data01/jh/afford_diag/gfs_build/online_pl_pts.log
"""
import os
import sys

import numpy as np

sys.path.insert(0, '/data01/jh/code/ps_diag')
import vlm_common as V  # noqa: E402

DATA = '/data01/jh/data/scannet200_gfs/train'
PL_A = '/data01/jh/data/offline_pl/sc200_regis_1_10_A'
SHARD = '/data01/jh/data/online_pl_shard0.npz'
NO, NN = 12, 45
NAMES = V.CLASS_NAMES[NO:]


def a_stats():
    cnt = np.zeros(NN, dtype=np.int64)
    hit = np.zeros(NN, dtype=np.int64)
    paths = sorted(os.listdir(PL_A))
    for i, fn in enumerate(paths):
        sc = fn[:-4]
        pl = np.load(os.path.join(PL_A, fn))
        seg = np.load(os.path.join(DATA, sc, 'segment200.npy')).reshape(-1)
        gfs = V.convert_to_gfs(seg)
        lab = np.where((gfs >= 0) & (gfs < NO), gfs, -1)
        w = lab == -1
        if len(pl) != len(w):
            continue
        v = pl[w].astype(np.int64)
        ok = v >= NO
        vv = v[ok]
        cnt += np.bincount(vv - NO, minlength=NN)[:NN]
        g = gfs[w][ok]
        m = (g >= NO) & (g == vv)
        hit += np.bincount(vv[m] - NO, minlength=NN)[:NN]
        if (i + 1) % 300 == 0:
            print('  A stats %d/%d' % (i + 1, len(paths)), flush=True)
    return cnt, hit


def miou(P, I, G):
    ious = []
    for c in range(NN):
        p, it, g = P[c], I[c], G[c]
        if g == 0 and p == 0:
            continue
        u = p + g - it
        ious.append(it / u if u > 0 else 0.0)
    return float(np.mean(ious)), len(ious)


def main():
    d = np.load(SHARD)
    gt = d['gt']
    print('场景集合 GT novel 点合计 %.2f M' % (gt.sum() / 1e6), flush=True)
    print('PS 拒绝点（VLM 判为 novel 但被 PS 整类删掉）%d' % int(d['reject_pts'][0]),
          flush=True)

    print('统计 A 的逐类点数与命中 ...', flush=True)
    a_cnt, a_hit = a_stats()

    # 参考：全库 1201 场景 GT（gc_full）
    gf = np.load('/data01/jh/data/gc_full.npy').astype(np.int64)

    print('\n%-24s %10s %10s %10s %10s %8s' %
          ('class', 'GT(1195)', 'onlinePS', 'PS+AI', 'A', 'A/PS+AI'))
    print('-' * 80)
    for c in range(NN):
        print('%-24s %10d %10d %10d %10d %7.2fx'
              % (NAMES[c], gt[c], d['ps_cnt'][c], d['psai_cnt'][c], a_cnt[c],
                 a_cnt[c] / max(d['psai_cnt'][c], 1)))
    print('-' * 80)
    print('%-24s %10d %10d %10d %10d' %
          ('合计', gt.sum(), d['ps_cnt'].sum(), d['psai_cnt'].sum(), a_cnt.sum()))

    print('\n=== 伪标签质量（P=写入点数，I=其中 GT 命中，G=GT 点数）===')
    for tag, P, I in (('online PS  ', d['ps_cnt'], d['ps_hit']),
                      ('online PS+AI', d['psai_cnt'], d['psai_hit']),
                      ('A (离线)    ', a_cnt, a_hit)):
        m, n = miou(P, I, gt)
        mf, _ = miou(P, I, gf)
        print('%-14s mIoU-N(G=1195场景) %.4f | (G=gc_full) %.4f | '
              '微精度 %.3f | 召回 %.3f | 写入 %.2f M'
              % (tag, m, mf, I.sum() / max(P.sum(), 1),
                 I.sum() / max(gt.sum(), 1), P.sum() / 1e6))
    print('AI 新增部分: 写入 %d 点，其中命中 %d，微精度 %.3f'
          % (d['ai_cnt'].sum(), d['ai_hit'].sum(),
             d['ai_hit'].sum() / max(d['ai_cnt'].sum(), 1)))
    print('\n=== 逐类精度（online PS+AI vs A）===')
    print('%-24s %8s %8s %8s %8s' % ('class', 'PS+AI P', 'A P', 'PS+AI R', 'A R'))
    for c in range(NN):
        print('%-24s %8.3f %8.3f %8.3f %8.3f'
              % (NAMES[c],
                 d['psai_hit'][c] / max(d['psai_cnt'][c], 1),
                 a_hit[c] / max(a_cnt[c], 1),
                 d['psai_hit'][c] / max(gt[c], 1),
                 a_hit[c] / max(gt[c], 1)))


if __name__ == '__main__':
    main()
