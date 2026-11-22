"""完整伪标签管线：统一 G_c_full 口径，一次跑完所有消融。

方案
----
ps_full     PS keep 全部区域（原版 PS 输出）
ps_ai       原版 PS + AI：对 pred==-1 且未标注的点，cos(区域, 原型) >= 0.9 才赋类
            （复现 train.py:786-800 的 AI 分支）
sonata_z0   Sonata 打分自适应淘汰 z>=0
  +no_neg   消融：Sonata 不减 max_D（验证基类负对照的净贡献）
  +G3z_C    G3z 在 PS 拒绝池上补标 100 万
  +G2_base  G2 在 Y_b==-1 的 base 池上补标 100 万 / 320 万
  +G1_C     G1 在 PS 拒绝池上补标 100 万（补标策略消融）
  +G2_C     G2 在 PS 拒绝池上补标 100 万（补标策略消融）

G_c 用完整 1201 场景的 GT 点数（gc_full.npy），不再用缓存覆盖口径。
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
BASE = '/data01/jh/data/base_cache'
SONATA = '/data01/jh/data/sonata_region'
PROTO = '/data01/jh/data/support_protos'
GC = '/data01/jh/data/gc_full.npy'
NO, NN = 12, 45
PS_THRESH = 0.6
AI_THRESH = 0.9
NAMES = V.CLASS_NAMES


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


def load_base():
    F, SZ, HN, IGN = [], [], [], []
    for f in sorted(glob.glob(os.path.join(BASE, 'base_train_s*.pt'))):
        d = torch.load(f, map_location='cpu')
        for sc, r in d.items():
            F.append(r['feats'].astype(np.float32))
            SZ.append(r['sizes'].astype(np.int64))
            HN.append(r['hist_n'].astype(np.int32))
            IGN.append(r['n_ign'].astype(np.int64))
    F = np.concatenate(F, axis=0)
    SZ = np.concatenate(SZ)
    HN = np.concatenate(HN, axis=0)
    IGN = np.concatenate(IGN)
    unlab = HN.sum(1) + IGN
    sel = unlab >= 0.5 * SZ
    print('base 池 %d -> Y_b==-1 过滤后 %d 区域 %d 点'
          % (len(F), int(sel.sum()), int(SZ[sel].sum())))
    return F[sel], SZ[sel], HN[sel], IGN[sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='sc200_regis_1_10')
    args = ap.parse_args()

    G = np.load(GC).astype(np.float64)
    proto = np.load(os.path.join(PROTO, args.split + '.npy')).astype(np.float32)
    pn = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
    sp_path = os.path.join(PROTO, args.split + '_sonata.npy')
    has_s = os.path.exists(sp_path)
    if has_s:
        sp = np.load(sp_path).astype(np.float32)
        spn = sp / (np.linalg.norm(sp, axis=1, keepdims=True) + 1e-8)
        son = load_sonata()

    scenes = load_regions()
    kept, rej = [], []
    for si, sc in enumerate(scenes):
        for c in np.unique(sc['pred']):
            c = int(c)
            if c < NO:
                continue
            m = sc['pred'] == c
            w = sc['sizes'][m].astype(np.float64)
            pooled = (sc['feats'][m] * w[:, None]).sum(0) / w.sum()
            pooled /= np.linalg.norm(pooled) + 1e-8
            (kept if float(pooled @ pn[c - NO]) >= PS_THRESH else rej).extend(
                (si, int(i), c) for i in np.where(m)[0])

    def pack(items):
        si = np.array([a[0] for a in items])
        ri = np.array([a[1] for a in items])
        cc = np.array([a[2] for a in items])
        F = np.stack([scenes[s]['feats'][i] for s, i, c in items])
        SZ = np.array([scenes[s]['sizes'][i] for s, i, c in items], dtype=np.int64)
        HN = np.stack([scenes[s]['hist_n'][i] for s, i, c in items])
        HB = np.stack([scenes[s]['hist_b'][i] for s, i, c in items])
        return si, ri, cc, F, SZ, HN, HB

    _, _, cc_k, Fk, SZk, HNk, HBk = pack(kept)
    _, _, cc_r, Fr, SZr, HNr, HBr = pack(rej)

    # D：仅用 base 真值
    Ditem = []
    for si, sc in enumerate(scenes):
        gb = sc['hist_b']
        for i in range(len(sc['pred'])):
            vp = int(sc['sizes'][i]) - int(sc['n_ign'][i])
            if vp > 0 and gb[i].sum() > 0 and int(gb[i].max()) >= 0.5 * vp:
                Ditem.append((si, i))
    Dn = np.stack([scenes[s]['feats'][i] for s, i in Ditem])
    Dn = Dn / (np.linalg.norm(Dn, axis=1, keepdims=True) + 1e-8)

    # 三种视角下的 B 池打分
    def view_feats(items, si, ri):
        F = np.stack([scenes[s]['feats'][i] for s, i, c in items])
        out = {'vlm': (F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8),
                       pn, Dn)}
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
                SD = np.stack([son[scenes[s]['name']][i] for s, i in Ditem])
                out['sonata'] = (S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8),
                                 spn,
                                 SD / (np.linalg.norm(SD, axis=1, keepdims=True) + 1e-8))
        return out

    Vk = view_feats(kept, None, None)
    Vr = view_feats(rej, None, None)

    def scores(Fn, prot, Dref, cls):
        s1 = np.zeros(len(Fn))
        for c in range(NO, NO + NN):
            m = cls == c
            if m.any():
                s1[m] = Fn[m] @ prot[c - NO]
        dm = np.empty(len(Fn), dtype=np.float32)
        for st in range(0, len(Fn), 4096):
            en = min(st + 4096, len(Fn))
            dm[st:en] = (Fn[st:en] @ Dref.T).max(1)
        return s1, dm

    # 每视角：B 池的 (s1, s1-dmax)
    k_s1, k_mg = {}, {}
    for view, (Fn, prot, Dref) in Vk.items():
        s1, dm = scores(Fn, prot, Dref, cc_k)
        k_s1[view] = s1
        k_mg[view] = s1 - dm
    # C 池的 G1/G2/G3z
    c_s1, c_s2, c_g3z = {}, {}, {}
    for view, (Fn, prot, Dref) in Vr.items():
        s1, dm = scores(Fn, prot, Dref, cc_r)
        Fkv = Vk[view][0]
        s2 = np.zeros(len(Fn))
        for c in range(NO, NO + NN):
            m = cc_r == c
            if not m.any():
                continue
            Bm = Fkv[cc_k == c]
            if len(Bm):
                s2[m] = (Fn[m] @ Bm.T).max(1)
            else:
                s2[m] = -1
        c_s1[view] = s1
        c_s2[view] = s2
        z = np.zeros(len(Fn))
        for c in range(NO, NO + NN):
            m = cc_r == c
            if m.any():
                z[m] = zscore(s2[m] - dm[m])
        c_g3z[view] = z

    # base 池：VLM 视角的 G2（B 池最近邻）
    Fb, SZb, HNb, IGNb = load_base()
    Fbn = Fb / (np.linalg.norm(Fb, axis=1, keepdims=True) + 1e-8)
    b_s2 = np.full(len(Fb), -1.0, dtype=np.float32)
    b_c = np.full(len(Fb), -1, dtype=np.int32)
    for c in range(NO, NO + NN):
        Bm = Vk['vlm'][0][cc_k == c]
        if not len(Bm):
            continue
        d = np.empty(len(Fb), dtype=np.float32)
        for st in range(0, len(Fb), 4096):
            en = min(st + 4096, len(Fb))
            d[st:en] = (Fbn[st:en] @ Bm.T).max(1)
        upd = d > b_s2
        b_s2[upd] = d[upd]
        b_c[upd] = c

    def miou(P, I):
        ious = []
        for c in range(NN):
            p, it, g = P[c], I[c], G[c]
            if p == 0 and g == 0:
                continue
            u = p + g - it
            ious.append(it / u if u > 0 else 0.0)
        return float(np.mean(ious))

    def add(P, I, cls, sz, hn, sel=None):
        idx = np.arange(len(cls)) if sel is None else sel
        for c, s, h in zip(cls[idx], sz[idx], hn[idx]):
            P[c - NO] += s
            I[c - NO] += h[c - NO]

    allk = np.arange(len(Fk))
    P0 = np.zeros(NN)
    I0 = np.zeros(NN)
    add(P0, I0, cc_k, SZk, HNk)
    base_line = miou(P0, I0)
    print('\n=== 全库口径 mIoU-N（G_c_full = %.1f M 点）==='
          % (G.sum() / 1e6))
    print('  ps_full（PS keep 全部）            %.4f' % base_line)

    # --- 原版 PS + AI ---
    # AI 语义：对 pred==-1 且未标注的点，用区域特征与 45 个原型的余弦相似度，
    # >= 0.9 才赋类。PS 把 base 类也置成了 -1，所以 base 池同样参与。
    # 原版用 batch 内背景区自算原型，这里用 support 原型（口径差异，已在报告注明）。
    P, I = P0.copy(), I0.copy()
    ai_ok = 0
    for c, s, h, s1v in zip(cc_r, SZr, HNr, c_s1['vlm']):
        if s1v >= AI_THRESH:
            P[c - NO] += s
            I[c - NO] += h[c - NO]
            ai_ok += 1
    b_s1 = np.empty((len(Fb), NN), dtype=np.float32)
    for st in range(0, len(Fb), 8192):
        en = min(st + 8192, len(Fb))
        b_s1[st:en] = Fbn[st:en] @ pn.T
    b_best = b_s1.max(1)
    b_ai_cls = b_s1.argmax(1) + NO
    ai_b = np.where(b_best >= AI_THRESH)[0]
    add(P, I, b_ai_cls, SZb, HNb, ai_b)
    print('  ps_ai（PS + AI δ=0.9）            %.4f  (C 池 %d 区 + base 池 %d 区)'
          % (miou(P, I), ai_ok, len(ai_b)))

    # --- Sonata z>=0 ---
    def zk(view):
        z = np.zeros(len(Fk))
        for c in range(NO, NO + NN):
            m = cc_k == c
            if m.any():
                z[m] = zscore(k_mg[view][m])
        return z

    for view in Vk:
        z = zk(view)
        sel = np.where(z >= 0)[0]
        P, I = np.zeros(NN), np.zeros(NN)
        add(P, I, cc_k, SZk, HNk, sel)
        m_z = miou(P, I)
        print('  %s z>=0 提纯%s      %.4f'
              % (view, ' ' * (18 - len(view)), m_z))
        # 无负对照消融：只用 s1
        z2 = np.zeros(len(Fk))
        for c in range(NO, NO + NN):
            m = cc_k == c
            if m.any():
                z2[m] = zscore(k_s1[view][m])
        s2 = np.where(z2 >= 0)[0]
        Pa, Ia = np.zeros(NN), np.zeros(NN)
        add(Pa, Ia, cc_k, SZk, HNk, s2)
        print('    消融-不减 max_D               %.4f  (%+0.4f)'
              % (miou(Pa, Ia), miou(Pa, Ia) - m_z))

        # 补标消融：C 池 G1 / G2 / G3z，各 100 万
        for tag, sc_ in (('G1', c_s1[view]), ('G2', c_s2[view]),
                         ('G3z', c_g3z[view])):
            order = np.argsort(-sc_, kind='stable')
            cum = np.cumsum(SZr[order])
            k = int(np.searchsorted(cum, 1000000))
            k = min(k + 1, len(order))
            Pc, Ic = np.zeros(NN), np.zeros(NN)
            add(Pc, Ic, cc_k, SZk, HNk, sel)
            add(Pc, Ic, cc_r, SZr, HNr, order[:k])
            print('    + %s 补 C 池 100 万            %.4f  (%+0.4f)'
                  % (tag, miou(Pc, Ic), miou(Pc, Ic) - m_z))

        # base 池补标（G2，仅 VLM 视角有意义）
        if view == 'vlm':
            order = np.argsort(-b_s2, kind='stable')
            cum = np.cumsum(SZb[order])
            for budget in (1000000, 3200000):
                k = int(np.searchsorted(cum, budget))
                k = min(k + 1, len(order))
                Pc, Ic = np.zeros(NN), np.zeros(NN)
                add(Pc, Ic, cc_k, SZk, HNk, sel)
                add(Pc, Ic, b_c, SZb, HNb, order[:k])
                print('    + G2 补 base 池 %d 万        %.4f  (%+0.4f)'
                      % (budget // 10000, miou(Pc, Ic), miou(Pc, Ic) - m_z))


if __name__ == '__main__':
    main()
