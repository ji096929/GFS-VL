"""用 Sonata 原型相似度（而非桥梁池）做 G3z 选点，对比 mIoU-N。"""
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
SHARD = '/data01/jh/data/online_pl_shard0.npz'
NO, NN = 12, 45
NAMES = V.CLASS_NAMES[NO:]


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-8 else np.zeros_like(x)


def miou(P, I, G):
    ious = []
    for c in range(NN):
        p, it, g = P[c], I[c], G[c]
        if g == 0 and p == 0:
            continue
        u = p + g - it
        ious.append(it / u if u > 0 else 0.0)
    return float(np.mean(ious))


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

    cls_l, sz_l, hn_l, hb_l, sv_l, sc_l = [], [], [], [], [], []
    for sc, r in scenes.items():
        if sc not in son:
            continue
        S = son[sc]
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
        for i in range(len(r['pred'])):
            sc_l.append(sc); cls_l.append(int(r['pred'][i]))
            sz_l.append(int(r['sizes'][i])); hn_l.append(r['hist_n'][i])
            hb_l.append(r['hist_b'][i]); sv_l.append(S[i])
    cls = np.array(cls_l); sz = np.array(sz_l); sv = np.stack(sv_l)
    hb_sum = np.stack(hb_l).sum(1)
    is_rej = np.array([rej.get(s) is not None and c in rej[s]
                       for s, c in zip(sc_l, cls_l)])

    s1 = np.empty(len(cls), dtype=np.float32)
    for c in range(NO, NO + NN):
        m = cls == c
        if m.any():
            s1[m] = sv[m] @ spn[c - NO]
    z = np.zeros(len(cls), dtype=np.float32)
    for c in range(NO, NO + NN):
        m = cls == c
        if m.any():
            z[m] = zscore(s1[m])

    d = np.load(SHARD)
    G = d['gt']
    print('reject 区域 %d / %.2f M 点' % (int(is_rej.sum()), sz[is_rej].sum() / 1e6))
    for tag, P0, I0 in (('online PS', d['ps_cnt'], d['ps_hit']),
                        ('online PS+AI', d['psai_cnt'], d['psai_hit'])):
        m0 = miou(P0, I0, G)
        print('\n底盘 %-14s mIoU-N %.4f' % (tag, m0))
        idx = np.where(is_rej)[0]
        idx = idx[np.argsort(-z[idx], kind='stable')]
        for budget in (0.5e6, 1.0e6, 2.0e6, 3.5e6):
            P, I = P0.astype(np.int64).copy(), I0.astype(np.int64).copy()
            tot = 0
            hitC = np.zeros(NN); cntC = np.zeros(NN)
            for j in idx:
                if tot >= budget:
                    break
                add = int(sz[j] - hb_sum[j])
                if add <= 0:
                    continue
                c = cls[j] - NO
                P[c] += add; I[c] += int(hn_l[j][c])
                cntC[c] += add; hitC[c] += int(hn_l[j][c])
                tot += add
            m1 = miou(P, I, G)
            print('  + G3z(原型分) %5.1f 万点 实际 %.2f M 微精度 %.3f: mIoU-N %.4f (%+0.4f)'
                  % (budget / 1e4, tot / 1e6, hitC.sum() / max(cntC.sum(), 1), m1, m1 - m0))
        if tag == 'online PS+AI':
            print('  逐类构成（budget=1M）:')
            for c in np.where(cntC > 0)[0]:
                print('    %-22s %7d 点 精度 %.3f' % (NAMES[c], int(cntC[c]), hitC[c] / cntC[c]))


if __name__ == '__main__':
    main()
