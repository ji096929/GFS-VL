"""用【逐点】Sonata 分数生成软权重，并分 PS / AI 两组做五等分精度校验。

权重区间（按建议）
  PS keep : w = 0.25 + 0.75 * sigmoid(2z)      ∈ [0.25, 1.00]，均值 ≈ 0.63
  AI 新增 : w = 0.05 + 0.55 * sigmoid(2z)      ∈ [0.05, 0.60]，均值 ≈ 0.32
  GT / NB-Mix : 1.0
z 为类内标准化（PS+AI+补标 合并同一份分布）。
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, '/data01/jh/code/ps_diag')
import vlm_common as V  # noqa: E402

NO, NN = 12, 45
DATA = '/data01/jh/data/scannet200_gfs/train'
ONLINE = '/data01/jh/data/online_pl'
PLW = '/data01/jh/data/online_pl_w'
PP = '/data01/jh/data/pp_score'


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-8 else np.zeros_like(x)


def main():
    ws, cs, grp = [], [], []
    n_scene = 0
    for f in sorted(glob.glob(PP + '/*_s.npy')):
        sc = os.path.basename(f)[:-6]
        pl_path = os.path.join(PLW, sc + '_pl.npy')
        if not os.path.exists(pl_path):
            continue
        s = np.load(f).astype(np.float32)
        lab = np.load(pl_path)
        z = np.load(os.path.join(ONLINE, sc + '.npz'))
        ps = z['ps']
        if len(s) != len(lab):
            print('!! len mismatch', sc)
            continue
        valid = lab >= NO
        if not valid.any():
            continue
        # 缺失分数（补标点 / 未覆盖）用同类均值填充
        miss = ~np.isfinite(s)
        for c in range(NO, NO + NN):
            m = (lab == c) & ~miss
            if m.any():
                s[(lab == c) & miss] = s[m].mean()
        s[~np.isfinite(s)] = 0.0

        zz = np.zeros(len(lab), dtype=np.float32)
        for c in range(NO, NO + NN):
            m = lab == c
            if m.sum() > 1:
                zz[m] = zscore(s[m])

        w = np.ones(len(lab), dtype=np.float32)
        is_ps = (ps >= NO) & valid
        is_ai = (ps < NO) & valid
        sig = 1.0 / (1.0 + np.exp(-2.0 * zz))
        w[is_ps] = 0.25 + 0.75 * sig[is_ps]
        w[is_ai] = 0.05 + 0.55 * sig[is_ai]
        np.save(os.path.join(PLW, sc + '_w.npy'), w)
        n_scene += 1

        gfs = V.convert_to_gfs(np.load(os.path.join(DATA, sc, 'segment200.npy')).reshape(-1))
        sel = valid
        ws.append(w[sel])
        cs.append(((gfs[sel] >= NO) & (gfs[sel] == lab[sel])).astype(np.int8))
        grp.append(np.where(ps[sel] >= NO, 0, 1).astype(np.int8))
        if n_scene % 300 == 0:
            print('  %d 场景' % n_scene, flush=True)

    w = np.concatenate(ws)
    c = np.concatenate(cs)
    g = np.concatenate(grp)
    print('\n处理 %d 场景，PL 点 %.2f M，精度 %.3f，权重 mean %.3f'
          % (n_scene, len(w) / 1e6, c.mean(), w.mean()))
    for k, name in ((0, 'PS keep'), (1, 'AI 新增')):
        m = g == k
        if m.any():
            print('  %-8s 点 %.2f M 精度 %.3f 权重 mean %.3f [%.2f, %.2f]'
                  % (name, m.sum() / 1e6, c[m].mean(), w[m].mean(),
                     w[m].min(), w[m].max()))
    print('\n分来源五等分：')
    for k, name in ((0, 'PS'), (1, 'AI')):
        m0 = g == k
        if not m0.any():
            continue
        q = np.quantile(w[m0], [0, .2, .4, .6, .8, 1.0])
        row = []
        for j in range(5):
            mm = m0 & (w >= q[j]) & (w <= q[j + 1])
            row.append('%.3f' % c[mm].mean() if mm.any() else '-')
        print('  %-3s w分桶 -> 精度: %s' % (name, ' | '.join(row)))


if __name__ == '__main__':
    main()
