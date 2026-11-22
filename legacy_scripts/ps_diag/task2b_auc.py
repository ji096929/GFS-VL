"""补充：Sonata+D_base 的【逐类】区分度（宏平均 AUC）与 G3z 逐类精度。"""
import glob
import json
import sys

import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/ps_diag')
import vlm_common as V  # noqa: E402

REGION = '/data01/jh/data/region_cache'
SONATA = '/data01/jh/data/sonata_region'
PROTO = '/data01/jh/data/support_protos'
REJ = '/data01/jh/data/online_pl_rej_s0.json'
NO, NN = 12, 45
NAMES = V.CLASS_NAMES[NO:]


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-8 else np.zeros_like(x)


def main():
    rej = json.load(open(REJ))
    scenes = {}
    for f in sorted(glob.glob(REGION + '/regions_train_s*.pt')):
        for sc, r in torch.load(f, map_location='cpu').items():
            scenes[sc] = r
    son = {}
    for f in sorted(glob.glob(SONATA + '/sonata_regions_train_s*.pt')):
        for sc, r in torch.load(f, map_location='cpu').items():
            son[sc] = r['feats'].astype(np.float32)

    sp = np.load(PROTO + '/sc200_regis_1_10_sonata.npy').astype(np.float32)
    spn = sp / (np.linalg.norm(sp, axis=1, keepdims=True) + 1e-8)

    sc_l, cls_l, sz_l, hn_l, hb_l, sn_l = [], [], [], [], [], []
    for sc, r in scenes.items():
        if sc not in son:
            continue
        S = son[sc]
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
        for i in range(len(r['pred'])):
            sc_l.append(sc); cls_l.append(int(r['pred'][i]))
            sz_l.append(int(r['sizes'][i])); hn_l.append(r['hist_n'][i])
            hb_l.append(r['hist_b'][i]); sn_l.append(S[i])
    cls = np.array(cls_l); sz = np.array(sz_l); hb_sum = np.stack(hb_l).sum(1)
    sn = np.stack(sn_l)
    is_rej = np.array([rej.get(s) is not None and c in rej[s]
                       for s, c in zip(sc_l, cls_l)])

    Dn = sn[np.stack(hb_l).max(1) >= 0.5 * np.maximum(sz, 1)]
    Dn = Dn / (np.linalg.norm(Dn, axis=1, keepdims=True) + 1e-8)
    s = np.empty(len(cls), dtype=np.float32)
    for c in range(NO, NO + NN):
        m = cls == c
        if not m.any():
            continue
        kf = sn[m & ~is_rej]
        dmax = (sn[m] @ Dn.T).max(1) if len(Dn) else 0
        s[m] = ((sn[m] @ kf.T).max(1) if len(kf) else -1) - dmax
    z = np.zeros(len(cls), dtype=np.float32)
    for c in range(NO, NO + NN):
        m = cls == c
        if m.any():
            z[m] = zscore(s[m])

    pur = np.array([hn_l[i][cls[i] - NO] / max(sz[i] - hb_sum[i], 1)
                    for i in range(len(cls))])
    from sklearn.metrics import roc_auc_score
    print('\n逐类 AUC（keep 区域，正例 = 纯度 >= 0.5）')
    print('%-24s %6s %6s %8s %8s' % ('class', 'n', 'pos', 'AUC', 'Top20%精度'))
    aucs = []
    for c in range(NO, NO + NN):
        m = (~is_rej) & (cls == c) & (sz - hb_sum > 0)
        if m.sum() < 30:
            continue
        y = (pur[m] >= 0.5).astype(int)
        if y.min() == y.max():
            continue
        a = roc_auc_score(y, z[m])
        k = max(1, int(0.2 * m.sum()))
        top = np.argsort(-z[m])[:k]
        aucs.append(a)
        print('%-24s %6d %6d %8.4f %8.3f'
              % (NAMES[c - NO], int(m.sum()), int(y.sum()), a, pur[m][top].mean()))
    print('宏平均 AUC %.4f（%d 类）' % (np.mean(aucs), len(aucs)))

    # G3z 选中区域的逐类精度
    print('\nG3z 在 reject 池 top-K 的逐类精度（按 z 排序）')
    idx = np.where(is_rej)[0]
    idx = idx[np.argsort(-z[idx], kind='stable')]
    tot = 0
    hit = np.zeros(NN); cnt = np.zeros(NN)
    for j in idx:
        if tot >= 1000000:
            break
        add = int(sz[j] - hb_sum[j])
        if add <= 0:
            continue
        c = cls[j] - NO
        cnt[c] += add
        hit[c] += int(hn_l[j][c])
        tot += add
    print('总 %.2f M 点，微精度 %.3f' % (cnt.sum() / 1e6, hit.sum() / max(cnt.sum(), 1)))
    for c in np.where(cnt > 0)[0]:
        print('  %-24s %8d 点  精度 %.3f' % (NAMES[c], int(cnt[c]), hit[c] / cnt[c]))


if __name__ == '__main__':
    main()
