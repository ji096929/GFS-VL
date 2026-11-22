"""对比多种候选分数在 keep 区域上的逐类 AUC（判断软权重是否可行）。"""
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


def macro_auc(cls, z, pur, mask):
    from sklearn.metrics import roc_auc_score
    out = []
    for c in range(NO, NO + NN):
        m = mask & (cls == c)
        if m.sum() < 30:
            continue
        y = (pur[m] >= 0.5).astype(int)
        if y.min() == y.max():
            continue
        out.append(roc_auc_score(y, z[m]))
    return float(np.mean(out)), len(out)


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

    pv = np.load(PROTO + '/sc200_regis_1_10.npy').astype(np.float32)
    pvn = pv / (np.linalg.norm(pv, axis=1, keepdims=True) + 1e-8)
    sp = np.load(PROTO + '/sc200_regis_1_10_sonata.npy').astype(np.float32)
    spn = sp / (np.linalg.norm(sp, axis=1, keepdims=True) + 1e-8)

    sc_l, cls_l, sz_l, hn_l, hb_l = [], [], [], [], []
    fv_l, sv_l = [], []
    for sc, r in scenes.items():
        if sc not in son:
            continue
        S = son[sc]
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
        F = r['feats'].astype(np.float32)
        F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
        for i in range(len(r['pred'])):
            sc_l.append(sc); cls_l.append(int(r['pred'][i]))
            sz_l.append(int(r['sizes'][i])); hn_l.append(r['hist_n'][i])
            hb_l.append(r['hist_b'][i]); fv_l.append(F[i]); sv_l.append(S[i])
    cls = np.array(cls_l); sz = np.array(sz_l)
    hb_sum = np.stack(hb_l).sum(1)
    fv = np.stack(fv_l); sv = np.stack(sv_l)
    is_rej = np.array([rej.get(s) is not None and c in rej[s]
                       for s, c in zip(sc_l, cls_l)])
    pur = np.array([hn_l[i][cls[i] - NO] / max(sz[i] - hb_sum[i], 1)
                    for i in range(len(cls))])
    mask = (~is_rej) & (sz - hb_sum > 0)

    # D_base 两套
    hbmax = np.stack(hb_l).max(1)
    Dv = fv[hbmax >= 0.5 * np.maximum(sz, 1)]
    Dv = Dv / (np.linalg.norm(Dv, axis=1, keepdims=True) + 1e-8)
    Ds = sv[hbmax >= 0.5 * np.maximum(sz, 1)]
    Ds = Ds / (np.linalg.norm(Ds, axis=1, keepdims=True) + 1e-8)

    def per_cls_raw(X, prot, D):
        s1 = np.empty(len(cls), dtype=np.float32)
        for c in range(NO, NO + NN):
            m = cls == c
            if m.any():
                s1[m] = X[m] @ prot[c - NO]
        return s1, (X @ D.T).max(1) if len(D) else np.zeros(len(cls))

    def per_cls_bridge(X, keep):
        s = np.empty(len(cls), dtype=np.float32)
        for c in range(NO, NO + NN):
            m = cls == c
            if not m.any():
                continue
            kf = X[m & keep]
            s[m] = (X[m] @ kf.T).max(1) if len(kf) else -1
        return s

    cands = {}
    for tag, X, prot, D in (('sonata', sv, spn, Ds), ('vlm', fv, pvn, Dv)):
        s1, dmax = per_cls_raw(X, prot, D)
        cands[tag + ' 原型'] = s1
        cands[tag + ' 原型-maxD'] = s1 - dmax
        br = per_cls_bridge(X, ~is_rej)
        cands[tag + ' 桥梁'] = br
        cands[tag + ' 桥梁-maxD'] = br - dmax

    print('%-24s %8s %6s' % ('score', 'macroAUC', 'classes'))
    for k, v in cands.items():
        z = np.zeros(len(cls), dtype=np.float32)
        for c in range(NO, NO + NN):
            m = cls == c
            if m.any():
                z[m] = zscore(v[m])
        a, nc = macro_auc(cls, z, pur, mask)
        print('%-24s %8.4f %6d' % (k, a, nc))


if __name__ == '__main__':
    main()
