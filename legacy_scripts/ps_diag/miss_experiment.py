"""第 2 点：在真实待填充区（pred == -1 的漏检区）上重跑三组补标对照。

与 group_experiment 的区别
--------------------------
- 候选池来自 miss_cache：VLM 对整片区域没有给出任何预测（pred == -1），
  不是 PS 拒绝池。这批区域没有"预测类"，所以类别必须由打分方法自己分配：
    G1  45 个 support 原型里取 argmax，分数 = max cos
    G2  B 池最近邻取 argmax，分数 = max cos
    G3  同 G2 的类别，分数 = cos(B) - max_D cos(D)（基类负例增量）
- 精度 = 分配类 == 区域 GT 多数类 的点数 / 有效 GT 点数。
- 真值只在排序截断之后参与计分。
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
MISS = '/data01/jh/data/miss_cache'
PROTO = '/data01/jh/data/support_protos'
PS_THRESH = 0.6
NO = 12
NN = 45
NAMES = V.CLASS_NAMES
TARGETS = [100000, 200000, 400000, 600000, 800000, 1000000]


def gt_major(hn, hb, ni):
    bn = int(hn.max()) if len(hn) else 0
    bb = int(hb.max()) if len(hb) else 0
    if ni >= max(bn, bb):
        return -1
    return NO + int(hn.argmax()) if bn >= bb else int(hb.argmax())


def load_keep_pool():
    """已废弃：B 池与 D 在 main 里一次性读出。"""
    raise NotImplementedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='sc200_regis_1_10')
    args = ap.parse_args()

    proto = np.load(os.path.join(PROTO, args.split + '.npy')).astype(np.float32)
    pn = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)

    # ---- 重新读 region_cache，同时构建 B 池和 D ----
    B = {c: [] for c in range(NO, NO + NN)}
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
                    B[c].extend(feats[m])
            for i in range(len(pred)):
                vp = int(r['sizes'][i]) - int(r['n_ign'][i])
                gb = r['hist_b'][i]
                if vp > 0 and gb.sum() > 0 and int(gb.max()) >= 0.5 * vp:
                    D.append(feats[i])
    Bn = {}
    for c, v in B.items():
        if v:
            a = np.stack(v)
            Bn[c] = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    D = np.stack(D)
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    print('B 池 %d 类有区域（合计 %d）  D_base %d'
          % (len(Bn), sum(len(v) for v in Bn.values()), len(D)), flush=True)

    # ---- miss 区域 ----
    feats, sizes, hn_list, valid = [], [], [], []
    for f in sorted(glob.glob(os.path.join(MISS, 'miss_train_s*.pt'))):
        d = torch.load(f, map_location='cpu')
        for sc, r in d.items():
            feats.append(r['feats'].astype(np.float32))
            sizes.append(r['sizes'].astype(np.int64))
            hn_list.append(r['hist_n'].astype(np.int32))
            for i in range(len(r['sizes'])):
                valid.append(int(r['hist_n'][i].sum() + r['hist_b'][i].sum()))
    F = np.stack(feats)
    SZ = np.concatenate(sizes)
    HN = np.concatenate(hn_list)          # (R,45) 每个区域的 novel GT 直方图
    VAL = np.array(valid)
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    in_cls = HN.sum(1) > 0
    print('miss 区域 %d  %d 点  有效 GT %d 点  含 novel GT 的区域 %d'
          % (len(F), SZ.sum(), VAL.sum(), int(in_cls.sum())), flush=True)

    # ---- G1：原型 argmax ----
    s1_all = Fn @ pn.T                       # (R,45)
    c1 = s1_all.argmax(1) + NO
    sc1 = s1_all.max(1)

    # ---- G2/G3：B 池最近邻 ----
    sc2 = np.full(len(F), -1.0, dtype=np.float32)
    c2 = np.full(len(F), -1, dtype=np.int32)
    for c, Bc in Bn.items():
        d = np.empty(len(F), dtype=np.float32)
        for st in range(0, len(F), 4096):
            en = min(st + 4096, len(F))
            d[st:en] = (Fn[st:en] @ Bc.T).max(1)
        upd = d > sc2
        sc2[upd] = d[upd]
        c2[upd] = c
    dD = np.empty(len(F), dtype=np.float32)
    for st in range(0, len(F), 4096):
        en = min(st + 4096, len(F))
        dD[st:en] = (Fn[st:en] @ Dn.T).max(1)
    sc3 = sc2 - dD

    # ---- 精度：分配类在区域 GT 直方图里的占比 ----
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
            rows.append((int(SZ[sel].sum()), hit / max(v, 1)))
        return rows

    print('\n=== 漏检区补标精度（分配类的 GT 点 / 有效 GT 点）===')
    print('%-10s %14s %14s %14s' % ('+点数', 'G1 原型', 'G2 桥梁池', 'G3 扣负例'))
    r1 = curve(sc1, c1)
    r2 = curve(sc2, c2)
    r3 = curve(sc3, c2)
    for i in range(len(TARGETS)):
        print('%-10d  %.3f(%d)   %.3f(%d)   %.3f(%d)'
              % (TARGETS[i], r1[i][1], r1[i][0] // 1000,
                 r2[i][1], r2[i][0] // 1000,
                 r3[i][1], r3[i][0] // 1000))

    # ---- 覆盖率：有多少类真的被分配到 ----
    for tag, cc in (('G1', c1), ('G2', c2)):
        u = np.unique(cc[cc >= NO])
        print('%s 覆盖 %d 个类' % (tag, len(u)))


if __name__ == '__main__':
    main()
