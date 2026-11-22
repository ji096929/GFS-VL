"""逐点 Sonata 打分 + 假阳性成分分析（抽样 30 场景）。

回答两个问题：
  1. PS / AI 的假阳性点里，真值分别是 base(0-11) / novel(12-56) / ignore(-1) 各占多少
  2. 用【逐点】Sonata 特征算 cos 与支持原型，类内四分位精度是否单调（对比区域级 AUC 0.659）
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/ps_diag')
sys.path.insert(0, '/data01/jh/code/sonata')
sys.path.insert(0, '/data01/jh/code')
import sonata  # noqa: E402
import vlm_common as V  # noqa: E402
from build_sonata_region import feat_sonata  # noqa: E402

NO, NN = 12, 45
DATA = '/data01/jh/data/scannet200_gfs/train'
ONLINE = '/data01/jh/data/online_pl'
PROTO = '/data01/jh/data/support_protos/sc200_regis_1_10_sonata.npy'


def main():
    pn = np.load(PROTO).astype(np.float32)
    pn = pn / (np.linalg.norm(pn, axis=1, keepdims=True) + 1e-8)
    cfg = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    model = sonata.load('/data01/jh/code/sonata/ckpt/sonata.pth',
                        custom_config=cfg).cuda().eval()

    scenes = sorted(os.listdir(ONLINE))[::40][:30]   # 等距抽样
    S, L, C, G, SRC = [], [], [], [], []
    for i, fn in enumerate(scenes):
        sc = fn[:-4]
        z = np.load(os.path.join(ONLINE, fn))
        ps, psai = z['ps'], z['psai']
        base = os.path.join(DATA, sc)
        coord = np.load(base + '/coord.npy').astype(np.float32)
        color = np.load(base + '/color.npy').astype(np.float32)
        npth = base + '/normal.npy'
        normal = np.load(npth).astype(np.float32) if os.path.exists(npth) \
            else np.zeros_like(coord)
        gfs = V.convert_to_gfs(np.load(base + '/segment200.npy').reshape(-1))
        f = feat_sonata(model, coord, color, normal)
        fn_ = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        sel = psai >= NO
        idx = np.where(sel)[0]
        sc_ = np.einsum('ij,ij->i', fn_[idx], pn[psai[idx] - NO])
        S.append(sc_)
        L.append(psai[idx].astype(np.int64))
        C.append(((gfs[idx] >= NO) & (gfs[idx] == psai[idx])).astype(np.int8))
        G.append(gfs[idx])
        SRC.append((ps[idx] >= NO).astype(np.int8))
        if (i + 1) % 10 == 0:
            print('  %d/%d' % (i + 1, len(scenes)), flush=True)

    s = np.concatenate(S); l = np.concatenate(L); c = np.concatenate(C)
    g = np.concatenate(G); src = np.concatenate(SRC)
    print('\n抽样 PL 点 %d，精度 %.3f' % (len(s), c.mean()))

    # ---- (a) 假阳性成分 ----
    print('\n(a) 假阳性成分（只看判错的点）')
    for k, name in ((1, 'PS'), (0, 'AI')):
        m = src == k
        wrong = m & (c == 0)
        if not wrong.any():
            continue
        gg = g[wrong]
        nb = int(((gg >= 0) & (gg < NO)).sum())
        nn = int((gg >= NO).sum())
        ni = int((gg == -1).sum())
        tot = wrong.sum()
        print('  %-3s 错 %8d 点: 真值 base %5.1f%% | 其他 novel %5.1f%% | 背景ignore %5.1f%%'
              % (name, tot, 100 * nb / tot, 100 * nn / tot, 100 * ni / tot))

    # ---- (b) 逐点 Sonata 的类内单调性 ----
    print('\n(b) 逐点 Sonata：类内四分位精度（低→高）')
    from sklearn.metrics import roc_auc_score
    aucs = []
    mono = 0
    for k in range(NO, NO + NN):
        m = l == k
        if m.sum() < 2000:
            continue
        y = c[m]
        if y.min() == y.max():
            continue
        aucs.append(roc_auc_score(y, s[m]))
        q = np.quantile(s[m], [0, .25, .5, .75, 1.0])
        ps_ = []
        for j in range(4):
            mm = m & (s >= q[j]) & (s <= q[j + 1])
            ps_.append(c[mm].mean() if mm.any() else -1)
        if ps_[-1] > ps_[0]:
            mono += 1
        if V.CLASS_NAMES[k] in ('book', 'pillow', 'tv', 'office chair', 'microwave'):
            print('  %-14s n=%7d  四分位 %s' % (V.CLASS_NAMES[k], int(m.sum()),
                                               ' '.join('%.2f' % x for x in ps_)))
    print('  宏平均 AUC %.4f（%d 类），高>低 的类 %d/%d'
          % (np.mean(aucs), len(aucs), mono, len(aucs)))

    q = np.quantile(s, [0, .2, .4, .6, .8, 1.0])
    print('\n  全体五分位精度：', ' '.join(
        '%.3f' % c[(s >= q[j]) & (s <= q[j + 1])].mean() for j in range(5)))


if __name__ == '__main__':
    main()
