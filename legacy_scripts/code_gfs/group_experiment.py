"""三组伪标签对照：保留 PS 已接受，只考察 PS 拒绝的区域。

Group 1: support 原型直接匹配 C（放宽阈值）
Group 2: 加入跨场景 B（PS 共识池），按 B 对 C 的增量排序
Group 3: Group2 再扣除 B 对真实基类负例 D_base 的增量（有符号差，不做 relu）

输出：
  - 每类桥梁池可用度（B 数量、B 的真实精度）
  - 同等新增点数（10万..100万）下三组的新增伪标签精度
  - PS 误删正确点的找回率（分开 VLM 判对 / VLM 判错）
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

REGION = '/data01/jh/data/region_cache'
PROTO = '/data01/jh/data/support_protos'
OUT = '/data01/jh/data/group_exp'
PS_THRESH = 0.6
SPLITS = ['sc200_regis_1_%d' % s for s in (10, 20, 30, 40, 50)] + \
         ['sc200_regis_5_%d' % s for s in (10, 20, 30, 40, 50)]
NOVEL_OFF = 12
N_NOVEL = 45
N_BASE = 12


def load_regions():
    scenes = []
    for f in sorted(glob.glob(os.path.join(REGION, 'regions_train_s*.pt'))):
        d = torch.load(f, map_location='cpu')
        for sc, r in sorted(d.items()):
            scenes.append(dict(
                name=sc,
                pred=r['pred'].astype(np.int64),
                sizes=r['sizes'].astype(np.int64),
                feats=r['feats'].astype(np.float32),
                hist_n=r['hist_n'].astype(np.int64),
                hist_b=r['hist_b'].astype(np.int64),
                n_ign=r['n_ign'].astype(np.int64),
            ))
    print('scenes %d  regions %d' % (len(scenes),
                                     sum(len(s['pred']) for s in scenes)))
    return scenes


def gt_major(scene, i):
    """区域 i 的 GT 多数类（GFS 空间），ignore 胜出返回 -1。"""
    hn = scene['hist_n'][i]
    hb = scene['hist_b'][i]
    ni = scene['n_ign'][i]
    best_n = int(hn.max()) if len(hn) else 0
    best_b = int(hb.max()) if len(hb) else 0
    if ni >= max(best_n, best_b):
        return -1
    if best_n >= best_b:
        return NOVEL_OFF + int(hn.argmax())
    return int(hb.argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='sc200_regis_1_10')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    scenes = load_regions()
    proto = np.load(os.path.join(PROTO, args.split + '.npy'))  # (45,512)
    proto_n = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)

    # ---- 每场景每预测类的 PS 决策 ----
    kept = []      # (scene_idx, region_idx) -> B
    rejected = []  # -> C
    scene_cls_pooled = {}
    for si, sc in enumerate(scenes):
        uniq = np.unique(sc['pred'])
        for c in uniq:
            c = int(c)
            if c < NOVEL_OFF:
                continue
            m = sc['pred'] == c
            w = sc['sizes'][m].astype(np.float64)
            pooled = (sc['feats'][m].astype(np.float64) * w[:, None]).sum(0) / w.sum()
            pooled /= np.linalg.norm(pooled) + 1e-8
            score = float(pooled @ proto_n[c - NOVEL_OFF])
            idxs = np.where(m)[0]
            if score >= PS_THRESH:
                kept.extend((si, i, c) for i in idxs)
            else:
                rejected.extend((si, i, c) for i in idxs)
    print('kept %d  rejected %d' % (len(kept), len(rejected)), flush=True)

    # ---- 桥梁池可用度 ----
    print('\n=== 桥梁池可用度（PS keep 的区域）===')
    print('%-22s %8s %8s %8s' % ('class', 'n_B', 'mean_size', 'B_precision'))
    bridge_stats = {}
    for c in range(NOVEL_OFF, NOVEL_OFF + N_NOVEL):
        bs = [(si, i) for (si, i, cc) in kept if cc == c]
        if not bs:
            bridge_stats[c] = (0, 0.0, -1.0)
            continue
        sizes = np.array([scenes[si]['sizes'][i] for si, i in bs])
        corr = np.array([gt_major(scenes[si], i) == c for si, i in bs])
        bridge_stats[c] = (len(bs), sizes.mean(), corr.mean())
    names = None
    for c in range(NOVEL_OFF, NOVEL_OFF + N_NOVEL):
        n, msz, prec = bridge_stats[c]
        print('%-22s %8d %8.0f %8.3f' % (c, n, msz, prec))

    # ---- 三组打分 ----
    C = np.array([[si, i, c] for (si, i, c) in rejected], dtype=np.int64)
    cf = np.stack([scenes[si]['feats'][i] for si, i, c in C])
    csz = np.array([scenes[si]['sizes'][i] for si, i, c in C])
    cpred = C[:, 2]
    cf_n = cf / (np.linalg.norm(cf, axis=1, keepdims=True) + 1e-8)

    # B 池：按类存特征矩阵
    B_by_cls = {c: [] for c in range(NOVEL_OFF, NOVEL_OFF + N_NOVEL)}
    for (si, i, c) in kept:
        B_by_cls[c].append(scenes[si]['feats'][i])
    B_by_cls = {c: (np.stack(v) if v else None)
                for c, v in B_by_cls.items()}

    # D_base：GT 多数为 base 的区域
    D_base = []
    for si, sc in enumerate(scenes):
        gb = sc['hist_b']
        for i in range(len(sc['pred'])):
            if gb[i].sum() > 0 and gb[i].max() >= sc['hist_n'][i].max():
                D_base.append(sc['feats'][i])
    D = np.stack(D_base) if D_base else None
    D_n = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8) if D is not None else None
    print('\nD_base regions', 0 if D is None else len(D))

    s1 = np.zeros(len(C))
    s2 = np.zeros(len(C))
    s3 = np.zeros(len(C))
    for c in range(NOVEL_OFF, NOVEL_OFF + N_NOVEL):
        m = cpred == c
        if not m.any():
            continue
        p = proto_n[c - NOVEL_OFF]
        s1[m] = cf_n[m] @ p
        B = B_by_cls[c]
        if B is not None:
            B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
            s2[m] = (cf_n[m] @ B_n.T).max(1)
        else:
            s2[m] = -1.0
        if D_n is not None:
            dmax = (cf_n[m] @ D_n.T).max(1)
            s3[m] = s2[m] - dmax
        else:
            s3[m] = s2[m]

    # ---- 真值标签 ----
    gt_novel_cnt = np.zeros(len(C), dtype=np.int64)   # 区域内 GT novel 点数
    gt_any = np.zeros(len(C), dtype=np.int64)
    correct = np.zeros(len(C), dtype=bool)
    for k, (si, i, c) in enumerate(C):
        sc = scenes[si]
        gt_novel_cnt[k] = sc['hist_n'][i].sum()
        gt_any[k] = sc['hist_n'][i].sum() + sc['hist_b'][i].sum()
        gm = gt_major(sc, i)
        correct[k] = (gm == c)

    # ---- 曲线 ----
    print('\n=== 同等新增点数下三组的新增伪标签精度 ===')
    print('%-10s %8s %8s %8s' % ('+点数', 'Group1', 'Group2', 'Group3'))
    targets = [100000, 200000, 400000, 600000, 800000, 1000000]
    for t in targets:
        row = []
        for s in (s1, s2, s3):
            order = np.argsort(-s)
            cum = np.cumsum(csz[order])
            k = int(np.searchsorted(cum, t))
            k = min(k + 1, len(order))
            sel = order[:k]
            prec = gt_novel_cnt[sel].sum() / max(gt_any[sel].sum(), 1)
            row.append(prec)
        print('%-10d %8.3f %8.3f %8.3f' % (t, row[0], row[1], row[2]))

    # ---- PS 误删找回率 ----
    n_miss_correct = int((csz * correct).sum())
    n_miss_wrong = int((csz * (~correct)).sum())
    print('\nPS 误删（rejected）点数: 判对 %d / 判错 %d' %
          (n_miss_correct, n_miss_wrong))
    print('%-10s %14s %14s' % ('+点数', '判对找回率', '判错混入率'))
    for t in targets:
        row = []
        for s in (s1, s2, s3):
            order = np.argsort(-s)
            cum = np.cumsum(csz[order])
            k = int(np.searchsorted(cum, t))
            k = min(k + 1, len(order))
            sel = order[:k]
            row.append((
                float((csz[sel] * correct[sel]).sum() / max(n_miss_correct, 1)),
                float((csz[sel] * (~correct[sel])).sum() / max(n_miss_correct, 1)),
            ))
        print('%-10d %6.3f/%6.3f %6.3f/%6.3f %6.3f/%6.3f'
              % (t, row[0][0], row[0][1], row[1][0], row[1][1],
                 row[2][0], row[2][1]))

    if args.out:
        json.dump(dict(split=args.split, bridge=bridge_stats,
                       n_rejected=len(C)), open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
