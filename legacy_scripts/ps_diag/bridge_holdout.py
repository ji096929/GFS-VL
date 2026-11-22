"""留出集检验：提纯多实例原型 P~ 是否比 1-shot 支持原型 p 更能排序区域。

做法
  用奇数场景的区域（按支持原型挑 top10%）构造 P~_c，
  在偶数场景的区域上比较 AUC（正例 = GT 该类占比 >= 0.5）。
  这样 P~ 的构造与评估不重叠，避免选择偏差。
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/ps_diag')
import vlm_common as V  # noqa: E402

NO, NN = 12, 45
REGION = '/data01/jh/data/region_cache'
SONATA = '/data01/jh/data/sonata_region'
PROTO = '/data01/jh/data/support_protos/sc200_regis_1_10_sonata.npy'
TOP = 0.10


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    pn = unit(np.load(PROTO).astype(np.float32))
    son = {}
    for f in sorted(glob.glob(SONATA + '/sonata_regions_train_s*.pt')):
        for sc, r in torch.load(f, map_location='cpu').items():
            son[sc] = r['feats'].astype(np.float32)

    Fs, Cs, Ps, Sc = [], [], [], []
    for f in sorted(glob.glob(REGION + '/regions_train_s*.pt')):
        for sc, r in torch.load(f, map_location='cpu').items():
            if sc not in son:
                continue
            n = len(r['pred'])
            S = son[sc][:n]
            if len(S) != n:
                continue
            hb = r['hist_b'].sum(1)
            hn = r['hist_n']
            sz = r['sizes']
            for i in range(n):
                c = int(r['pred'][i])
                if c < NO:
                    continue
                denom = max(int(sz[i] - hb[i]), 1)
                Fs.append(unit(S[i:i + 1])[0])
                Cs.append(c)
                Ps.append(float(hn[i][c - NO]) / denom)
                Sc.append(int(sc.split('_')[-1]) % 2)   # 用场景名尾号奇偶做划分
    F = np.stack(Fs); C = np.array(Cs); P = np.array(Ps); H = np.array(Sc)
    print('区域 %d，训练半 %d，评估半 %d' % (len(C), (H == 0).sum(), (H == 1).sum()))

    from sklearn.metrics import roc_auc_score
    auc_p, auc_pe, auc_mix = [], [], []
    rows = []
    for c in range(NO, NO + NN):
        tr = (C == c) & (H == 0)
        te = (C == c) & (H == 1)
        if tr.sum() < 20 or te.sum() < 20:
            continue
        y = (P[te] >= 0.5).astype(int)
        if y.min() == y.max():
            continue
        s = F[tr] @ pn[c - NO]
        k = max(5, int(TOP * tr.sum()))
        idx = np.argsort(-s)[:k]
        Pt = unit(F[tr][idx].mean(0))
        a0 = roc_auc_score(y, F[te] @ pn[c - NO])
        a1 = roc_auc_score(y, F[te] @ Pt)
        a2 = roc_auc_score(y, 0.5 * (F[te] @ pn[c - NO]) + 0.5 * (F[te] @ Pt))
        auc_p.append(a0); auc_pe.append(a1); auc_mix.append(a2)
        rows.append((V.CLASS_NAMES[c], a0, a1, a2, int(te.sum())))
    print('\n%-22s %8s %8s %8s' % ('class', 'p', 'P~', 'avg'))
    for n, a0, a1, a2, t in sorted(rows, key=lambda x: -(x[2] - x[1]))[:8]:
        print('%-22s %8.4f %8.4f %8.4f' % (n, a0, a1, a2))
    print('  ...')
    for n, a0, a1, a2, t in sorted(rows, key=lambda x: (x[2] - x[1]))[:5]:
        print('%-22s %8.4f %8.4f %8.4f' % (n, a0, a1, a2))
    print('\n宏平均（%d 类）：支持原型 %.4f | 提纯多实例 %.4f | 两者平均 %.4f'
          % (len(auc_p), np.mean(auc_p), np.mean(auc_pe), np.mean(auc_mix)))
    print('P~ 更好的类 %d/%d' % (sum(1 for r in rows if r[2] > r[1]), len(rows)))


if __name__ == '__main__':
    main()
