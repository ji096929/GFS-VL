"""第 2 点：在「VLM 判成基类」的漏检区上重跑三组补标对照。

候选池来自 base_cache：VLM 把这些点判成了 12 个基类之一（pred < 12），
所以它们既不在 PS 拒绝池里，也没有 novel 标签。这批区域才是真正等着
自适应填充的空白区。

类别必须由打分方法自己分配：
  G1  45 个 support 原型里取 argmax
  G2  同类 B 池最近邻取 argmax
  G3  同 G2 的类别，分数换成 cos(B) - max_D cos(D)（基类负例增量）
  G3z G2 的类别 + 类内 z-score

指标
  precision = 选中区域里 GT 逐点等于「分配类」的点数 / 有效 GT 点数
  recall    = 捞回的 GT novel 点 / 该 split 下全部 novel 点（缓存覆盖内）
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
PROTO = '/data01/jh/data/support_protos'
PS_THRESH = 0.6
NO, NN = 12, 45
NAMES = V.CLASS_NAMES
TARGETS = [100000, 200000, 400000, 800000, 1600000, 3200000]


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-8 else np.zeros_like(x)


def load_b_pool(pn):
    """PS keep 的区域，按类分组；同时返回 D（只用 base 真值）。"""
    Bf = {c: [] for c in range(NO, NO + NN)}
    D = []
    for f in sorted(glob.glob(os.path.join(REGION, 'regions_train_s*.pt'))):
        d = torch.load(f, map_location='cpu')
        for sc, r in d.items():
            pred = r['pred'].astype(np.int64)
            feats = r['feats'].astype(np.float32)
            for c in np.unique(pred):
                c = int(c)
                if c < NO:
                    continue
                m = pred == c
                w = r['sizes'][m].astype(np.float64)
                pooled = (feats[m] * w[:, None]).sum(0) / w.sum()
                pooled /= np.linalg.norm(pooled) + 1e-8
                if float(pooled @ pn[c - NO]) >= PS_THRESH:
                    Bf[c].extend(feats[m])
            for i in range(len(pred)):
                vp = int(r['sizes'][i]) - int(r['n_ign'][i])
                gb = r['hist_b'][i]
                if vp > 0 and gb.sum() > 0 and int(gb.max()) >= 0.5 * vp:
                    D.append(feats[i])
    return Bf, np.stack(D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='sc200_regis_1_10')
    args = ap.parse_args()

    proto = np.load(os.path.join(PROTO, args.split + '.npy')).astype(np.float32)
    pn = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
    Bf, D = load_b_pool(pn)
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    # 按类拼成一个大矩阵，用于分块最近邻
    sizes = [len(v) for v in Bf.values() if v]
    Ball = np.concatenate([np.stack(v) for v in Bf.values() if v])
    cls_of_B = np.concatenate([np.full(len(v), c) for c, v in Bf.items() if v])
    Balln = Ball / (np.linalg.norm(Ball, axis=1, keepdims=True) + 1e-8)
    o = np.argsort(cls_of_B)
    Balln, cls_sorted = Balln[o], cls_of_B[o]
    starts = np.searchsorted(cls_sorted, np.arange(NO, NO + NN))
    print('B 池 %d 区域  D %d 区域' % (len(Ball), len(D)), flush=True)

    # ---- base 池：先按 Y_b == -1 过滤 ----
    # GFS-VL Eq.6 里 Y_b >= 0 的点保留真值，AI 只允许在未标注区补标。
    # 所以排除「base 点占多数」的区域，只留 novel + ignore 点构成的空白区。
    feats, sizes_b, hns, vals = [], [], [], []
    igns = []
    for f in sorted(glob.glob(os.path.join(BASE, 'base_train_s*.pt'))):
        d = torch.load(f, map_location='cpu')
        for sc, r in d.items():
            feats.append(r['feats'].astype(np.float32))
            sizes_b.append(r['sizes'].astype(np.int64))
            hns.append(r['hist_n'].astype(np.int32))
            igns.append(r['n_ign'].astype(np.int64))
    F = np.concatenate(feats, axis=0)
    SZ = np.concatenate(sizes_b)
    HN = np.concatenate(hns, axis=0)
    IGN = np.concatenate(igns)
    unlab = HN.sum(1) + IGN           # novel + ignore = Y_b == -1 意义上的点数
    sel = unlab >= 0.5 * SZ           # base 点不占多数
    print('base 池全量 %d 区域 %d 点' % (len(F), SZ.sum()))
    print('Y_b==-1 过滤后 %d 区域 %d 点（%.1f%%）'
          % (int(sel.sum()), SZ[sel].sum(), 100 * SZ[sel].sum() / SZ.sum()))
    F, SZ, HN, IGN = F[sel], SZ[sel], HN[sel], IGN[sel]
    VAL = unlab[sel]                   # 分母排除 base 真值点
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    print('  其中 GT novel 点 %d（正例率 %.1f%%）'
          % (int(HN.sum()), 100 * HN.sum() / max(VAL.sum(), 1)), flush=True)

    # ---- G1：原型 argmax ----
    s1 = np.empty((len(F), NN), dtype=np.float32)
    for st in range(0, len(F), 8192):
        en = min(st + 8192, len(F))
        s1[st:en] = Fn[st:en] @ pn.T
    c1 = s1.argmax(1) + NO
    sc1 = s1.max(1)

    # ---- G2：分块最近邻 + 按类 reduceat ----
    best = np.full(len(F), -1.0, dtype=np.float32)
    bestc = np.full(len(F), -1, dtype=np.int32)
    dD = np.empty(len(F), dtype=np.float32)
    for st in range(0, len(F), 4096):
        en = min(st + 4096, len(F))
        sim = Fn[st:en] @ Balln.T                    # (chunk, |B|)
        cmax = np.maximum.reduceat(sim, starts, axis=1)   # (chunk,45)
        b = cmax.argmax(1)
        best[st:en] = cmax[np.arange(en - st), b]
        bestc[st:en] = b + NO
        dD[st:en] = (Fn[st:en] @ Dn.T).max(1)
        if (st // 4096) % 4 == 0:
            print('  nn %d / %d' % (en, len(F)), flush=True)
    sc2 = best
    sc3 = best - dD
    z = np.zeros(len(F))
    for c in range(NO, NO + NN):
        m = bestc == c
        if m.any():
            z[m] = zscore(sc3[m])

    # ---- 真值 ----
    gt_novel_all = float(HN.sum())
    print('\nbase 池里 GT novel 点合计 %.0f' % gt_novel_all)

    def curve(score, pred_c):
        order = np.argsort(-score, kind='stable')
        cum = np.cumsum(SZ[order])
        rows = []
        for t in TARGETS:
            k = int(np.searchsorted(cum, t))
            k = min(k + 1, len(order))
            sel = order[:k]
            v = int(VAL[sel].sum())
            cc = pred_c[sel] - NO
            ok = (cc >= 0) & (cc < NN)
            hit = float(HN[sel][ok, cc[ok]].sum()) if ok.any() else 0.0
            rows.append((int(SZ[sel].sum()), hit / max(v, 1),
                         hit / max(gt_novel_all, 1)))
        return rows

    print('\n=== base 池补标：精度 / 召回 ===')
    print('%-10s %16s %16s %16s %16s'
          % ('+点数', 'G1 原型', 'G2 桥梁池', 'G3 扣负例', 'G3z'))
    rs = [curve(sc1, c1), curve(sc2, bestc), curve(sc3, bestc),
          curve(z, bestc)]
    for i, t in enumerate(TARGETS):
        row = ''
        for r in rs:
            row += '  %.3f/%.3f' % (r[i][1], r[i][2])
        print('%-10d %s' % (t, row))

    # ---- 重点类召回 ----
    print('\n=== 重点类：能被分配到多少点数（G3z）===')
    order = np.argsort(-z, kind='stable')
    cum = np.cumsum(SZ[order])
    for t in (400000, 1600000):
        k = int(np.searchsorted(cum, t))
        k = min(k + 1, len(order))
        sel = order[:k]
        cc = bestc[sel] - NO
        ok = (cc >= 0) & (cc < NN)
        hit = np.zeros(NN)
        if ok.any():
            np.add.at(hit, cc[ok], HN[sel][ok, cc[ok]])
        print('  +%d 点时:' % t)
        for name in ('office chair', 'armchair', 'nightstand', 'ceiling',
                     'shelf', 'radiator'):
            c = NAMES.index(name)
            allc = float(HN[:, c - NO].sum())
            print('    %-14s 捞回 %8.0f / %8.0f  (%.1f%%)'
                  % (name, hit[c - NO], allc,
                     100 * hit[c - NO] / max(allc, 1)))


if __name__ == '__main__':
    main()
