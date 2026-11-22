"""测 Sonata 原型分在【AI 新增点】上的逐类宏平均 AUC（对应你问的 0.83 是否真实）。"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/ps_diag')
import vlm_common as V  # noqa: E402

NO, NN = 12, 45
DATA = '/data01/jh/data/scannet200_gfs/train'
ONLINE = '/data01/jh/data/online_pl'
REGION_IDX = '/data01/jh/data/region_index'
BASE_IDX = '/data01/jh/data/base_index'
SON_REG = '/data01/jh/data/sonata_region'
SON_BASE = '/data01/jh/data/sonata_base'
PROTO = '/data01/jh/data/support_protos/sc200_regis_1_10_sonata.npy'


def main():
    pn = np.load(PROTO).astype(np.float32)
    pn = pn / (np.linalg.norm(pn, axis=1, keepdims=True) + 1e-8)
    ridx, bidx, sreg, sbase = {}, {}, {}, {}
    for f in glob.glob(REGION_IDX + '/regions_index_train_s*.pt'):
        for sc, r in torch.load(f, map_location='cpu').items():
            ridx[sc] = r
    for f in glob.glob(BASE_IDX + '/base_index_train_s*.pt'):
        for sc, r in torch.load(f, map_location='cpu').items():
            bidx[sc] = r
    for f in glob.glob(SON_REG + '/sonata_regions_train_s*.pt'):
        for sc, r in torch.load(f, map_location='cpu').items():
            sreg[sc] = r['feats']
    for f in glob.glob(SON_BASE + '/sonata_base_train_s*.pt'):
        for sc, r in torch.load(f, map_location='cpu').items():
            sbase[sc] = r['feats']
    print('sonata_base %d 场景' % len(sbase), flush=True)

    scores, labels, ok = [], [], []
    for i, fn in enumerate(sorted(os.listdir(ONLINE))):
        sc = fn[:-4]
        z = np.load(os.path.join(ONLINE, fn))
        ps, psai = z['ps'], z['psai']
        seg = np.load(os.path.join(DATA, sc, 'segment200.npy')).reshape(-1)
        gfs = V.convert_to_gfs(seg)
        score = np.full(len(ps), np.nan, dtype=np.float32)
        if sc in ridx and sc in sreg:
            r = ridx[sc]
            S = sreg[sc].astype(np.float32)
            S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
            off, idxs = r['off'], r['idx']
            for k in range(len(off) - 1):
                s_ = idxs[off[k]:off[k + 1]]
                c = int(r['pred'][off[k]])
                if c >= NO and len(s_):
                    score[s_] = S[k] @ pn[c - NO]
        if sc in bidx and sc in sbase:
            r = bidx[sc]
            S = sbase[sc].astype(np.float32)
            S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
            off, idxs = r['off'], r['idx']
            for k in range(len(off) - 1):
                s_ = idxs[off[k]:off[k + 1]]
                v = psai[s_]
                m = v >= NO
                if not m.any():
                    continue
                for c in np.unique(v[m]):
                    sel = s_[m][v[m] == c]
                    score[sel] = S[k] @ pn[int(c) - NO]
        ai = (ps < NO) & (psai >= NO) & np.isfinite(score)
        if not ai.any():
            continue
        scores.append(score[ai])
        labels.append(psai[ai].astype(np.int64))
        ok.append((gfs[ai] >= NO) & (gfs[ai] == psai[ai]))
        if (i + 1) % 300 == 0:
            print('  %d' % (i + 1), flush=True)

    s = np.concatenate(scores); lab = np.concatenate(labels)
    corr = np.concatenate(ok).astype(int)
    print('AI 点 %d，精度 %.3f' % (len(s), corr.mean()), flush=True)

    from sklearn.metrics import roc_auc_score
    aucs = []
    print('\n%-24s %8s %8s %8s' % ('class', 'n', 'pos', 'AUC'))
    for c in range(NO, NO + NN):
        m = lab == c
        if m.sum() < 50:
            continue
        y = corr[m]
        if y.min() == y.max():
            continue
        a = roc_auc_score(y, s[m])
        aucs.append(a)
        print('%-24s %8d %8d %8.4f' % (V.CLASS_NAMES[c], int(m.sum()), int(y.sum()), a))
    print('宏平均 AUC %.4f（%d 类）' % (np.mean(aucs), len(aucs)))

    q = np.quantile(s, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    print('\nz-score 五分桶的真实精度：')
    for i in range(5):
        m = (s >= q[i]) & (s <= q[i + 1])
        if m.any():
            print('  桶 %d/5: %8d 点  精度 %.3f' % (i + 1, int(m.sum()), corr[m].mean()))


if __name__ == '__main__':
    main()
