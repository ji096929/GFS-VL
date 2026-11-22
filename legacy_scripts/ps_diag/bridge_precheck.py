"""零 GPU 预检：用缓存区域特征构造「提纯多实例原型」，看与 1-shot 支持原型差多少。"""
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

    Fs, Cs = [], []
    for f in sorted(glob.glob(REGION + '/regions_train_s*.pt')):
        for sc, r in torch.load(f, map_location='cpu').items():
            if sc not in son:
                continue
            n = len(r['pred'])
            S = son[sc][:n]
            if len(S) != n:
                continue
            Fs.append(unit(S))
            Cs.append(r['pred'].astype(np.int64))
    F = np.concatenate(Fs, axis=0)
    C = np.concatenate(Cs)
    print('区域 %d' % len(C))

    print('\n%-22s %6s %10s %8s' % ('class', 'core', 'cos(P~,p)', 'cos 中位'))
    ratios = []
    for c in range(NO, NO + NN):
        m = C == c
        if m.sum() < 20:
            continue
        s = F[m] @ pn[c - NO]
        k = max(5, int(TOP * m.sum()))
        idx = np.argsort(-s)[:k]
        P = unit(F[m][idx].mean(0))
        cosv = float(P @ pn[c - NO])
        ratios.append((V.CLASS_NAMES[c], cosv, k, int(m.sum())))
        if c < NO + 12:
            print('%-22s %6d %10.3f %8.3f'
                  % (V.CLASS_NAMES[c], k, cosv, float(np.median(s))))

    r = np.array([x[1] for x in ratios])
    print('\n%d 类：cos(P~, p) 中位数 %.3f | >0.95 的类 %d | <0.90 的类 %d'
          % (len(r), float(np.median(r)), int((r > 0.95).sum()), int((r < 0.90).sum())))
    print('\n最不相似的 8 个类（最可能带来新信息的）：')
    for n, v, k, tot in sorted(ratios, key=lambda x: x[1])[:8]:
        print('  %-22s cos %.3f  核心 %d/%d 区域' % (n, v, k, tot))


if __name__ == '__main__':
    main()
