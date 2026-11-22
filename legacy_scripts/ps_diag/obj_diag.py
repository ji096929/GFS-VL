"""把伪标签误差拆成「objectness（是不是物体）」与「class identity（是不是该类）」。

对每个 PL 点，用逐点 Sonata 分数（= cos 到其被赋类原型）做两个二分类：
  objectness: 真值是新类(>=12)  vs  真值是 ignore(-1)
  identity  : 在真值为新类的点里，类是否对
分 PS / AI 两组报告 AUC。
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, '/data01/jh/code/ps_diag')
import vlm_common as V  # noqa: E402

NO = 12
DATA = '/data01/jh/data/scannet200_gfs/train'
ONLINE = '/data01/jh/data/online_pl'
PP = '/data01/jh/data/pp_score'


def main():
    S, L, G, SRC = [], [], [], []
    for f in sorted(glob.glob(PP + '/*_s.npy')):
        sc = os.path.basename(f)[:-6]
        s = np.load(f).astype(np.float32)
        z = np.load(os.path.join(ONLINE, sc + '.npz'))
        ps, psai = z['ps'], z['psai']
        if len(s) != len(ps):
            continue
        gfs = V.convert_to_gfs(
            np.load(os.path.join(DATA, sc, 'segment200.npy')).reshape(-1))
        sel = psai >= NO
        if not sel.any():
            continue
        S.append(s[sel]); L.append(psai[sel].astype(np.int64))
        G.append(gfs[sel]); SRC.append((ps[sel] >= NO).astype(np.int8))

    s = np.concatenate(S); l = np.concatenate(L)
    g = np.concatenate(G); src = np.concatenate(SRC)
    print('PL 点 %.2f M' % (len(s) / 1e6))

    from sklearn.metrics import roc_auc_score
    # ---- objectness：真值是新类 vs ignore ----
    y_obj = (g >= NO).astype(int)
    print('\n[objectness] 真值新类 vs 背景/ignore  正例率 %.3f' % y_obj.mean())
    for k, name in ((0, 'PS'), (1, 'AI')):
        m = src == k
        if m.any() and y_obj[m].min() != y_obj[m].max():
            print('  %-3s n=%.2fM  AUC %.4f' %
                  (name, m.sum() / 1e6, roc_auc_score(y_obj[m], s[m])))
    print('  全体 AUC %.4f' % roc_auc_score(y_obj, s))

    q = np.quantile(s, [0, .2, .4, .6, .8, 1.0])
    print('  Sonata 五分位 -> "真值确实是新类"的比例：', ' | '.join(
        '%.3f' % y_obj[(s >= q[j]) & (s <= q[j + 1])].mean() for j in range(5)))

    # ---- class identity：在真新类里，类对不对 ----
    m = g >= NO
    y_cls = (g[m] == l[m]).astype(int)
    print('\n[class identity] 真新类点里类是否正确  正例率 %.3f' % y_cls.mean())
    for k, name in ((0, 'PS'), (1, 'AI')):
        mm = m & (src == k)
        if mm.any() and y_obj[mm].min() != y_obj[mm].max():
            yy = (g[mm] == l[mm]).astype(int)
            if yy.min() != yy.max():
                print('  %-3s n=%.2fM  AUC %.4f' % (name, mm.sum() / 1e6,
                                                   roc_auc_score(yy, s[mm])))
    print('  全体 AUC %.4f' % roc_auc_score(y_cls, s[m]))


if __name__ == '__main__':
    main()
