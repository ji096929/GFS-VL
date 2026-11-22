"""三组伪标签对照：保留 PS 已接受的结果，只考察 PS 拒绝的区域。

固定条件
--------
- 候选池 C：同一批 PS 拒绝的区域（VLM 预测为 novel 类 c，但场景级 pooled
  cos(u_c, p_c) < 0.6），区域边界 = 预测类内 3D 连通分量（radius 0.05）。
- 支持样本：同一套 1-shot split 的 support 原型（45x512）。
- 三组只改变「候选区域的打分」，不改变候选集、不改变阈值位置。

三组打分
--------
G1  支持原型直接匹配        s1 = cos(C, p_c)                  （放宽阈值＝按分数取前 k）
G2  可信桥梁池最近邻        s2 = max_{B~c} cos(C, B)          （B = PS 已接受的同类区域）
G3  扣除基类负例的增量      s3 = s2 - max_{D in base} cos(C, D)   （有符号，不 relu）
G3z 无监督标准化增量        s3z = (s2 - mean_c) / std_c        （只用同类候选分布，无 GT）

真值只在决策后计分：
  - 曲线按分数排序、按累计点数截断，然后才查 GT。
  - 类正确率 = Σ GT==c 的点 / Σ 有效 GT 点；novel 纯度 = Σ GT novel / Σ 有效 GT。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/ps_diag')
import vlm_common as V  # noqa: E402

REGION = '/data01/jh/data/region_cache'
PROTO = '/data01/jh/data/support_protos'
OUT = '/data01/jh/data/group_exp'
PS_THRESH = 0.6
NOVEL_OFF = 12
N_NOVEL = 45
N_BASE = 12
CLASS_NAMES = V.CLASS_NAMES
FOCUS = ['armchair', 'office chair', 'nightstand', 'radiator']


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
    print('scenes %d  regions %d'
          % (len(scenes), sum(len(s['pred']) for s in scenes)))
    # int32 修复的健全性检查
    bad = sum(int((s['hist_n'] < 0).sum() + (s['hist_b'] < 0).sum())
              for s in scenes)
    if bad:
        print('!! 直方图出现负值 %d 处，缓存仍是 int16 版，结果不可信' % bad)
    return scenes


def gt_major(scene, i):
    hn, hb, ni = scene['hist_n'][i], scene['hist_b'][i], scene['n_ign'][i]
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
    pn = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)

    # ---------- 每场景每预测类的 PS 决策 ----------
    kept, rejected = [], []
    for si, sc in enumerate(scenes):
        for c in np.unique(sc['pred']):
            c = int(c)
            if c < NOVEL_OFF:
                continue
            m = sc['pred'] == c
            w = sc['sizes'][m].astype(np.float64)
            pooled = (sc['feats'][m] * w[:, None]).sum(0) / w.sum()
            pooled /= np.linalg.norm(pooled) + 1e-8
            score = float(pooled @ pn[c - NOVEL_OFF])
            idxs = np.where(m)[0]
            (kept if score >= PS_THRESH else rejected).extend(
                (si, int(i), c) for i in idxs)
    n_k = len(kept)
    n_r = len(rejected)
    print('PS keep %d 个区域 / rejected %d 个区域' % (n_k, n_r), flush=True)

    # ---------- 细节 1：45 类桥梁池可用度 ----------
    print('\n=== 桥梁池可用度（PS keep 区域，B 池）===')
    print('%-24s %6s %10s %10s' % ('class', 'n_B', 'mean_size', 'B_acc'))
    bridge = {}
    for c in range(NOVEL_OFF, NOVEL_OFF + N_NOVEL):
        bs = [(si, i) for (si, i, cc) in kept if cc == c]
        if not bs:
            bridge[c] = (0, 0.0, -1.0)
            print('%-24s %6d %10s %10s'
                  % (CLASS_NAMES[c], 0, '-', '-'))
            continue
        sz = np.array([scenes[si]['sizes'][i] for si, i in bs])
        ok = np.array([gt_major(scenes[si], i) == c for si, i in bs])
        bridge[c] = (len(bs), float(sz.mean()), float(ok.mean()))
        print('%-24s %6d %10.0f %10.3f%s'
              % (CLASS_NAMES[c], len(bs), sz.mean(), ok.mean(),
                 '  <<' if CLASS_NAMES[c] in FOCUS else ''))

    # ---------- 候选池 ----------
    C = np.array(rejected, dtype=np.int64)
    cf = np.stack([scenes[si]['feats'][i] for si, i, c in C])
    csz = np.array([scenes[si]['sizes'][i] for si, i, c in C])
    cpred = C[:, 2]
    cfn = cf / (np.linalg.norm(cf, axis=1, keepdims=True) + 1e-8)
    print('\n候选池 C：%d 区域  %d 点' % (len(C), csz.sum()))

    # B 池按类
    B_by_cls = {}
    for c in range(NOVEL_OFF, NOVEL_OFF + N_NOVEL):
        v = [scenes[si]['feats'][i] for (si, i, cc) in kept if cc == c]
        B_by_cls[c] = np.stack(v) if v else None

    # D：真实基类负例。只用 12 个 base 类的真值（GFS-VL 设定下 base 有标注，
    # 等价于原文 Eq.6 直接用 Y_b），Novel 真值不参与构造。
    # 判定：区域内 base 多数类点数 >= 有效点数的一半。
    D_feats = []
    for si, sc in enumerate(scenes):
        gb = sc['hist_b']
        for i in range(len(sc['pred'])):
            valid_pts = int(sc['sizes'][i]) - int(sc['n_ign'][i])
            if valid_pts <= 0 or gb[i].sum() <= 0:
                continue
            if int(gb[i].max()) >= 0.5 * valid_pts:
                D_feats.append(sc['feats'][i])
    Dn = None
    if D_feats:
        D = np.stack(D_feats)
        Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    print('D_base（GT 多数为 base）%d 区域' % (0 if Dn is None else len(Dn)))

    # ---------- 打分 ----------
    s1 = np.zeros(len(C))
    s2 = np.zeros(len(C))
    s3 = np.zeros(len(C))
    s3z = np.zeros(len(C))
    for c in range(NOVEL_OFF, NOVEL_OFF + N_NOVEL):
        m = cpred == c
        if not m.any():
            continue
        s1[m] = cfn[m] @ pn[c - NOVEL_OFF]
        B = B_by_cls[c]
        if B is not None:
            Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
            s2[m] = (cfn[m] @ Bn.T).max(1)
        else:
            s2[m] = -1.0
        v = s2[m]
        sd = float(v.std())
        s3z[m] = (v - v.mean()) / sd if sd > 1e-8 else np.zeros_like(v)
        s3[m] = s2[m]
    if Dn is not None:
        # 分块算 C x D 相似度，避免 30万 x 3万 的稠密矩阵爆内存
        dmax = np.empty(len(C), dtype=np.float32)
        for st in range(0, len(C), 4096):
            en = min(st + 4096, len(C))
            dmax[st:en] = (cfn[st:en] @ Dn.T).max(1)
        s3 = s2 - dmax

    # ---------- 真值 ----------
    nn = np.zeros(len(C), dtype=np.int64)     # 区域内 GT novel 点数（任意 novel 类）
    nb = np.zeros(len(C), dtype=np.int64)
    ncls = np.zeros(len(C), dtype=np.int64)   # 区域内 GT == 预测类 c 的点数
    correct = np.zeros(len(C), dtype=bool)
    for k, (si, i, c) in enumerate(C):
        sc = scenes[si]
        nn[k] = sc['hist_n'][i].sum()
        nb[k] = sc['hist_b'][i].sum()
        ncls[k] = sc['hist_n'][i][c - NOVEL_OFF]
        correct[k] = gt_major(sc, i) == c
    valid = nn + nb

    def curve(score, name):
        order = np.argsort(-score, kind='stable')
        cum = np.cumsum(csz[order])
        out = []
        for t in TARGETS:
            k = int(np.searchsorted(cum, t))
            k = min(k + 1, len(order))
            sel = order[:k]
            tot = max(int(valid[sel].sum()), 1)
            out.append(dict(
                target=int(t), pts=int(csz[sel].sum()), n=int(k),
                cls_acc=float(ncls[sel].sum() / tot),
                novel_pur=float(nn[sel].sum() / tot),
                recall=float(ncls[sel].sum() / max(int(ncls.sum()), 1)),
            ))
        return out

    def auc_point(score, idx=None):
        """点级加权 AUC：每个点是一个样本，标签 = GT==预测类，分数取所属区域的分。

        每个区域贡献 ncls 个正例 + (valid-ncls) 个负例，同区域的点分数相同，
        sklearn 会把这种并列对按 0.5 处理。
        """
        from sklearn.metrics import roc_auc_score
        if idx is None:
            idx = np.arange(len(C))
        wp = ncls[idx].astype(np.float64)
        wn = (valid[idx] - ncls[idx]).astype(np.float64)
        keep = (wp + wn) > 0
        y = np.concatenate([np.ones(keep.sum()), np.zeros(keep.sum())])
        s = np.concatenate([score[idx][keep], score[idx][keep]])
        w = np.concatenate([wp[keep], wn[keep]])
        nz = w > 0
        if y[nz].sum() == 0 or (1 - y[nz]).sum() == 0:
            return float('nan')
        return float(roc_auc_score(y[nz], s[nz], sample_weight=w[nz]))

    def auc_macro(score):
        """45 类各自算点级 AUC 再取宏平均。"""
        vals = []
        for c in range(NOVEL_OFF, NOVEL_OFF + N_NOVEL):
            m = cpred == c
            if m.sum() < 2:
                continue
            v = auc_point(score, np.where(m)[0])
            if not np.isnan(v):
                vals.append(v)
        return (float(np.mean(vals)) if vals else float('nan'), len(vals))

    TARGETS = [100000, 200000, 400000, 600000, 800000, 1000000]
    groups = [('G1 原型', s1), ('G2 桥梁池', s2), ('G3 扣负例', s3),
              ('G3z 标准化', s3z)]

    print('\n=== 同等新增点数下的伪标签精度（类正确率 / novel 纯度）===')
    print('%-10s' % '+点数' + ''.join('%16s' % g[0] for g in groups))
    curves = {g[0]: curve(g[1], g[0]) for g in groups}
    for i, t in enumerate(TARGETS):
        row = ''
        for g in groups:
            c = curves[g[0]][i]
            row += '  %.3f/%.3f(%d)' % (c['cls_acc'], c['novel_pur'],
                                        c['pts'] // 1000)
        print('%-10d' % t + row)

    print('\n=== AUC（点级加权：标签=GT==预测类；macro=45类各自算再平均）===')
    print('  %-12s %10s %12s' % ('group', 'pooled', 'macro(n)'))
    for g in groups:
        a, n = auc_macro(g[1])
        print('  %-12s %10.4f %12.4f(%d)'
              % (g[0], auc_point(g[1]), a, n))

    # ---------- 逐类宏平均：防止全局曲线被大类主导 ----------
    RATIOS = (0.1, 0.2, 0.3, 0.5, 1.0)
    print('\n=== 逐类 top-r 宏平均精度（每类各取前 r 比例区域）===')
    print('  %-10s' % 'group' + ''.join('%12s' % ('r=%.1f' % r) for r in RATIOS))
    macro_prec = {}
    for g in groups:
        row, store = '', {}
        for r in RATIOS:
            ps = []
            for c in range(NOVEL_OFF, NOVEL_OFF + N_NOVEL):
                idx = np.where(cpred == c)[0]
                if len(idx) == 0:
                    continue
                k = max(1, int(round(r * len(idx))))
                sub = idx[np.argsort(-g[1][idx], kind='stable')[:k]]
                v = int(valid[sub].sum())
                if v > 0:
                    ps.append(float(ncls[sub].sum()) / v)
            store[r] = float(np.mean(ps)) if ps else float('nan')
            row += '%12s' % ('%.3f' % store[r] if ps else '-')
        macro_prec[g[0]] = store
        print('  %-10s' % g[0] + row)

    print('\n=== 逐类 Pseudo-IoU 宏平均（选中区域 ∪ 该类 GT 点）===')
    print('  %-10s' % 'group' + ''.join('%12s' % ('r=%.1f' % r) for r in RATIOS))
    macro_iou = {}
    for g in groups:
        row, store = '', {}
        for r in RATIOS:
            vs = []
            for c in range(NOVEL_OFF, NOVEL_OFF + N_NOVEL):
                idx = np.where(cpred == c)[0]
                if len(idx) == 0:
                    continue
                k = max(1, int(round(r * len(idx))))
                sub = idx[np.argsort(-g[1][idx], kind='stable')[:k]]
                inter = float(ncls[sub].sum())
                union = float(csz[sub].sum()) + float(ncls[idx].sum()) - inter
                if union > 0:
                    vs.append(inter / union)
            store[r] = float(np.mean(vs)) if vs else float('nan')
            row += '%12s' % ('%.3f' % store[r] if vs else '-')
        macro_iou[g[0]] = store
        print('  %-10s' % g[0] + row)

    # ---------- 误删正确点召回 ----------
    n_ok = int((csz * correct).sum())
    n_bad = int((csz * ~correct).sum())
    print('\nPS 误删点数：判对 %d / 判错 %d' % (n_ok, n_bad))
    print('%-10s' % '+点数' + ''.join('%18s' % g[0] for g in groups))
    for i, t in enumerate(TARGETS):
        row = ''
        for g in groups:
            c = curves[g[0]][i]
            # 找回率（相对全部误删正确点）；误入率（相对全部误删错误点）
            k = int(np.searchsorted(np.cumsum(csz[np.argsort(-g[1], kind='stable')]), t))
            k = min(k + 1, len(C))
            sel = np.argsort(-g[1], kind='stable')[:k]
            ok = float((csz[sel] * correct[sel]).sum() / max(n_ok, 1))
            bad = float((csz[sel] * ~correct[sel]).sum() / max(n_bad, 1))
            row += '  %5.3f/%5.3f' % (ok, bad)
        print('%-10d' % t + row)

    # ---------- 重点类 ----------
    print('\n=== 重点类：B 池与 G2 增益 ===')
    print('%-14s %6s %6s %8s %8s'
          % ('class', 'n_B', 'n_C', 's1_mean', 's2_mean'))
    for c in range(NOVEL_OFF, NOVEL_OFF + N_NOVEL):
        if CLASS_NAMES[c] not in FOCUS:
            continue
        m = cpred == c
        print('%-14s %6d %6d %8.3f %8.3f'
              % (CLASS_NAMES[c], bridge[c][0], int(m.sum()),
                 s1[m].mean() if m.any() else float('nan'),
                 s2[m].mean() if m.any() else float('nan')))

    if args.out:
        json.dump(dict(split=args.split, n_kept=n_k, n_rejected=n_r,
                       pool_points=int(csz.sum()),
                       bridge={str(k): v for k, v in bridge.items()},
                       curves=curves,
                       auc_pooled={g[0]: auc_point(g[1]) for g in groups},
                       auc_macro={g[0]: auc_macro(g[1])[0] for g in groups},
                       macro_prec={k: {str(r): v for r, v in s.items()}
                                   for k, s in macro_prec.items()},
                       macro_iou={k: {str(r): v for r, v in s.items()}
                                  for k, s in macro_iou.items()},
                       n_miss_correct=n_ok, n_miss_wrong=n_bad),
                  open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
