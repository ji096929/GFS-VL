"""伪标签层的 mIoU-N：原版 PS keep vs B 池提纯 vs + 补标。

支持两套打分视角
----------------
vlm    原型/桥梁池相似度 - 基类负例（512 维 CLIP 空间）
sonata 同一公式，但换成 Sonata 的 1232 维几何特征（区域索引与 region_cache 对齐）

口径限制（必须一起报告）
------------------------
G_c（每类 GT 点数）只能从 region_cache 的直方图得到，也就是「VLM 至少预测出
一个 novel 类的场景」里的 GT。105 个没有任何 novel 预测的场景不在覆盖内，
所以这里是**缓存覆盖范围内**的相对指标，不等于 val 集上的模型 mIoU。
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/ps_diag')
import vlm_common as V  # noqa: E402

REGION = '/data01/jh/data/region_cache'
SONATA = '/data01/jh/data/sonata_region'
PROTO = '/data01/jh/data/support_protos'
NO, NN = 12, 45
PS_THRESH = 0.6
CLASS_NAMES = V.CLASS_NAMES


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-8 else np.zeros_like(x)


def load_regions():
    scenes = []
    for f in sorted(glob.glob(os.path.join(REGION, 'regions_train_s*.pt'))):
        d = torch.load(f, map_location='cpu')
        for sc, r in sorted(d.items()):
            scenes.append(dict(name=sc,
                               pred=r['pred'].astype(np.int64),
                               sizes=r['sizes'].astype(np.int64),
                               feats=r['feats'].astype(np.float32),
                               hist_n=r['hist_n'].astype(np.int64),
                               hist_b=r['hist_b'].astype(np.int64),
                               n_ign=r['n_ign'].astype(np.int64)))
    return scenes


def load_sonata():
    out = {}
    for f in sorted(glob.glob(os.path.join(SONATA, 'sonata_regions_train_s*.pt'))):
        d = torch.load(f, map_location='cpu')
        for sc, r in d.items():
            out[sc] = r['feats'].astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='sc200_regis_1_10')
    ap.add_argument('--ratios', default='0.2,0.3,0.5,1.0')
    ap.add_argument('--budgets', default='100000,400000,1000000')
    args = ap.parse_args()

    scenes = load_regions()
    proto = np.load(os.path.join(PROTO, args.split + '.npy')).astype(np.float32)
    pn = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)

    sp_path = os.path.join(PROTO, args.split + '_sonata.npy')
    has_s = os.path.exists(sp_path) and os.path.isdir(SONATA)
    son = load_sonata() if has_s else {}
    if has_s:
        sp = np.load(sp_path).astype(np.float32)
        spn = sp / (np.linalg.norm(sp, axis=1, keepdims=True) + 1e-8)
        print('sonata 视角可用：%d 场景, proto %s' % (len(son), sp.shape))
    else:
        spn = None
        print('没有 sonata 原型，只跑 VLM 视角')

    kept_idx, rej_idx = [], []
    for si, sc in enumerate(scenes):
        for c in np.unique(sc['pred']):
            c = int(c)
            if c < NO:
                continue
            m = sc['pred'] == c
            w = sc['sizes'][m].astype(np.float64)
            pooled = (sc['feats'][m] * w[:, None]).sum(0) / w.sum()
            pooled /= np.linalg.norm(pooled) + 1e-8
            tgt = kept_idx if float(pooled @ pn[c - NO]) >= PS_THRESH else rej_idx
            tgt.extend((si, int(i), c) for i in np.where(m)[0])

    def pack(items):
        si = np.array([a[0] for a in items])
        ri = np.array([a[1] for a in items])
        cc = np.array([a[2] for a in items])
        F = np.stack([scenes[s]['feats'][i] for s, i, c in items])
        SZ = np.array([scenes[s]['sizes'][i] for s, i, c in items], dtype=np.int64)
        HN = np.stack([scenes[s]['hist_n'][i] for s, i, c in items])
        return si, ri, cc, F, SZ, HN

    si_k, ri_k, cc_k, Fk, SZk, HNk = pack(kept_idx)
    si_r, ri_r, cc_r, Fr, SZr, HNr = pack(rej_idx)
    print('keep %d 区域 %d 点   reject %d 区域 %d 点'
          % (len(kept_idx), SZk.sum(), len(rej_idx), SZr.sum()))

    # D：仅用 base 真值
    Ditem = []
    for si, sc in enumerate(scenes):
        gb = sc['hist_b']
        for i in range(len(sc['pred'])):
            vp = int(sc['sizes'][i]) - int(sc['n_ign'][i])
            if vp > 0 and gb[i].sum() > 0 and int(gb[i].max()) >= 0.5 * vp:
                Ditem.append((si, i))
    D = np.stack([scenes[s]['feats'][i] for s, i in Ditem])
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)

    G = np.zeros(NN, dtype=np.int64)
    for s, i, c in kept_idx + rej_idx:
        G += scenes[s]['hist_n'][i]

    def views(items, si_a, ri_a):
        """返回 {view: (feats_n, proto_n, D_n)}。"""
        out = {}
        F = np.stack([scenes[s]['feats'][i] for s, i, c in items])
        Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
        out['vlm'] = (Fn, pn, Dn)
        if has_s:
            rows = []
            ok = True
            for s, i, c in items:
                arr = son.get(scenes[s]['name'])
                if arr is None or i >= len(arr):
                    ok = False
                    break
                rows.append(arr[i])
            if ok:
                S = np.stack(rows)
                Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
                SD = np.stack([son[scenes[s]['name']][i] for s, i in Ditem])
                SDn = SD / (np.linalg.norm(SD, axis=1, keepdims=True) + 1e-8)
                out['sonata'] = (Sn, spn, SDn)
        return out

    Vk = views(kept_idx, si_k, ri_k)
    Vr = views(rej_idx, si_r, ri_r)
    if 'sonata' not in Vk or 'sonata' not in Vr:
        print('!! sonata 区域缺失，只跑 vlm')
        Vk.pop('sonata', None)
        Vr.pop('sonata', None)

    def score_pool(Fn, prot, Dref, cls):
        """返回 G1 原型分、G2 桥梁池分、G3 扣负例分、G3z。"""
        s1 = np.zeros(len(Fn))
        for c in range(NO, NO + NN):
            m = cls == c
            if m.any():
                s1[m] = Fn[m] @ prot[c - NO]
        dmax = np.empty(len(Fn), dtype=np.float32)
        for st in range(0, len(Fn), 4096):
            en = min(st + 4096, len(Fn))
            dmax[st:en] = (Fn[st:en] @ Dref.T).max(1)
        return s1, dmax

    # B 池（keep）在每种视角下的 margin 分
    mg_k, ps_k = {}, {}
    for view, (Fn, prot, Dref) in Vk.items():
        s1, dmax = score_pool(Fn, prot, Dref, cc_k)
        ps_k[view] = s1
        mg_k[view] = s1 - dmax

    # C 池（reject）：G2 用同类 B 池最近邻，再扣负例，类内标准化
    sc2, s3z = {}, {}
    for view, (Fn, prot, Dref) in Vr.items():
        Fk_view = Vk[view][0]
        s2 = np.zeros(len(Fn))
        for c in range(NO, NO + NN):
            m = cc_r == c
            if not m.any():
                continue
            Bm = Fk_view[cc_k == c]
            if len(Bm):
                s2[m] = (Fn[m] @ Bm.T).max(1)
            else:
                s2[m] = -1
        _, dmax = score_pool(Fn, prot, Dref, cc_r)
        z = np.zeros(len(Fn))
        for c in range(NO, NO + NN):
            m = cc_r == c
            if m.any():
                z[m] = zscore(s2[m] - dmax[m])
        sc2[view] = s2 - dmax
        s3z[view] = z

    def miou(pp, ii):
        ious = []
        for c in range(NN):
            p, it, g = pp[c], ii[c], G[c]
            if p == 0 and g == 0:
                continue
            u = p + g - it
            ious.append(it / u if u > 0 else 0.0)
        return float(np.mean(ious)), len(ious)

    def eval_sel(k_sel, r_sel):
        pp = np.zeros(NN)
        ii = np.zeros(NN)
        for c, sz, hn in zip(cc_k[k_sel], SZk[k_sel], HNk[k_sel]):
            pp[c - NO] += sz
            ii[c - NO] += hn[c - NO]
        for c, sz, hn in zip(cc_r[r_sel], SZr[r_sel], HNr[r_sel]):
            pp[c - NO] += sz
            ii[c - NO] += hn[c - NO]
        return miou(pp, ii)

    all_k = np.arange(len(Fk))
    m0, n0 = eval_sel(all_k, np.array([], dtype=int))
    print('\n=== 缓存覆盖内 mIoU-N（%d 类参与）===' % n0)
    print('  PS keep 全部                     %.4f' % m0)

    ratios = [float(x) for x in args.ratios.split(',')]
    budgets = [int(x) for x in args.budgets.split(',')]
    FOCUS = ['ceiling', 'office chair', 'armchair', 'nightstand', 'radiator']

    def keep_rates(sel, view):
        """每类保留区域比例（诊断自适应淘汰有没有误伤小类）。"""
        out = {}
        for c in range(NO, NO + NN):
            idx = np.where(cc_k == c)[0]
            if len(idx) == 0:
                continue
            out[CLASS_NAMES[c]] = len(np.intersect1d(sel, idx, assume_unique=False)) / len(idx)
        return out

    # B 池的类内标准化分数：自适应淘汰必须作用在相对同类的位置上，
    # 因为 raw(s1-max_D) 的绝对值会被「max over 1.3 万个 D」的极值偏差压到全负。
    z_k = {}
    for view in Vk:
        z = np.zeros(len(Fk))
        for c in range(NO, NO + NN):
            m = cc_k == c
            if m.any():
                z[m] = zscore(mg_k[view][m])
        z_k[view] = z

    for view in Vk:
        print('\n  ---- %s 视角：按比例砍（对照，已知伤小类）----' % view)
        for r in ratios:
            sel = []
            for c in range(NO, NO + NN):
                idx = np.where(cc_k == c)[0]
                if len(idx) == 0:
                    continue
                k = max(1, int(round(r * len(idx))))
                sel.append(idx[np.argsort(-mg_k[view][idx], kind='stable')[:k]])
            sel = np.concatenate(sel)
            m, _ = eval_sel(sel, np.array([], dtype=int))
            print('  margin r=%.1f 保留                %.4f  (%+0.4f)'
                  % (r, m, m - m0))
            order = np.argsort(-s3z[view], kind='stable')
            cum = np.cumsum(SZr[order])
            for b in budgets:
                k = int(np.searchsorted(cum, b))
                k = min(k + 1, len(order))
                m2, _ = eval_sel(sel, order[:k])
                print('      + G3z %7d 补标          %.4f  (%+0.4f)'
                      % (b, m2, m2 - m0))

        print('\n  ---- %s 视角：自适应淘汰（类内 z >= 阈值）----' % view)
        for th in (-1.5, -1.0, -0.5, 0.0):
            sel = np.where(z_k[view] >= th)[0]
            m, _ = eval_sel(sel, np.array([], dtype=int))
            kr = keep_rates(sel, view)
            foc = '  '.join('%s %.0f%%' % (n, 100 * kr[n])
                            for n in FOCUS if n in kr)
            print('  z>=%+.1f  保留 %.1f%% 区域  mIoU %.4f (%+0.4f)  | %s'
                  % (th, 100 * len(sel) / len(Fk), m, m - m0, foc))
            order = np.argsort(-s3z[view], kind='stable')
            cum = np.cumsum(SZr[order])
            for b in budgets:
                k = int(np.searchsorted(cum, b))
                k = min(k + 1, len(order))
                m2, _ = eval_sel(sel, order[:k])
                print('      + G3z %7d 补标          %.4f  (%+0.4f)'
                      % (b, m2, m2 - m0))


if __name__ == '__main__':
    main()
