"""校验软权重：权重大小是否与伪标签真实精度单调相关。"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, '/data01/jh/code/ps_diag')
import vlm_common as V  # noqa: E402

NO = 12
DATA = '/data01/jh/data/scannet200_gfs/train'
PLW = '/data01/jh/data/online_pl_w'
ONLINE = '/data01/jh/data/online_pl'


def main():
    ws, cs, grp = [], [], []
    for f in sorted(glob.glob(PLW + '/*_w.npy')):
        sc = os.path.basename(f)[:-6]
        w = np.load(f)
        pl = np.load(os.path.join(PLW, sc + '_pl.npy'))
        seg = np.load(os.path.join(DATA, sc, 'segment200.npy')).reshape(-1)
        gfs = V.convert_to_gfs(seg)
        sel = pl >= 0
        if not sel.any():
            continue
        ws.append(w[sel])
        cs.append(((gfs[sel] >= NO) & (gfs[sel] == pl[sel])).astype(np.int8))
        z = np.load(os.path.join(ONLINE, sc + '.npz'))
        ps = z['ps'][sel]
        g = np.zeros(len(ps), dtype=np.int8)
        g[ps >= NO] = 0          # PS keep
        g[(ps < NO) & (pl[sel] >= NO)] = 1   # AI 新增
        grp.append(g)
    w = np.concatenate(ws)
    c = np.concatenate(cs)
    g = np.concatenate(grp)
    print('PL 点 %d，总精度 %.3f，权重 mean %.3f min %.3f max %.3f'
          % (len(w), c.mean(), w.mean(), w.min(), w.max()))
    for k, name in ((0, 'PS keep'), (1, 'AI 新增')):
        m = g == k
        if m.any():
            print('  %-10s 点 %8d  精度 %.3f  权重 mean %.3f  min %.3f  max %.3f'
                  % (name, int(m.sum()), c[m].mean(), w[m].mean(),
                     w[m].min(), w[m].max()))
    print('\n按权重五等分：')
    q = np.quantile(w, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for i in range(5):
        m = (w >= q[i]) & (w <= q[i + 1])
        if m.any():
            print('  桶 %d: w≈%.2f  点 %8d  精度 %.3f' % (i + 1, w[m].mean(),
                                                          int(m.sum()), c[m].mean()))


if __name__ == '__main__':
    main()
