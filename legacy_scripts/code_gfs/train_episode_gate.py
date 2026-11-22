"""任务级 EpisodeGate + 点级保 TP 融合。

为什么这样拆
------------
上一版把 8 维任务特征和 3 维点级特征混进一个 200 参数的 MLP，用逐点 CE 盲训。
一折只有 270 个任务，逐点损失会沿着"点"这个维度过拟合，导致两折的 Q2/Q3 表现分叉。
而且 `0.8*sigmoid(z)` 处处大于 0，在根本不该开桥的任务上仍输出非零权重，Q5 倒扣。

这一版拆成两层：

1. EpisodeGate（任务级，约 11 个参数）
   标签是**任务真实的盈亏**：y = 1[IoU(pred_U) − IoU(pred_S) > δ]。
   用逻辑回归直接学这个二分类，再按 p_open 排序，只对 Top-k% 的任务开门；
   其余任务 λ = 0，严格退化为 pred_S（零损失）。

2. 点级融合（两种，对比）
   bilinear  : (1−λ)·pred_S + λ·pred_U
   one_sided : pred_S + λ·relu(pred_U − pred_S)，U 只能往上抬，结构上保 TP
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

CACHE = '/data01/jh/data/ocsb_cache'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]


def load_recs(prefix='v3h_s[0-7].pt'):
    recs = []
    for f in sorted(glob.glob(os.path.join(CACHE, prefix))):
        recs += torch.load(f)
    recs.sort(key=lambda r: r['g_ep'])
    return recs


def iou_of(pred, gt):
    t = int(((pred == 1) & (gt == 1)).sum())
    f = int(((pred == 1) & (gt == 0)).sum())
    n = int(((pred == 0) & (gt == 1)).sum())
    return t / max(t + f + n, 1)


def ep_feats(rec, key='pred_u_solo'):
    """任务级特征：8 维统计量 + 两条预测的一致性。"""
    ps = rec['pred_s'].argmax(0)
    pu = rec[key].argmax(0)
    inter = int(((ps == 1) & (pu == 1)).sum())
    union = int(((ps == 1) | (pu == 1)).sum())
    cons = inter / max(union, 1)
    agree = float((ps == pu).mean())
    return np.concatenate([rec['x_task'].astype(np.float64),
                           [cons, agree]])


def fuse(rec, lam, mode, key='pred_u_solo'):
    ps = rec['pred_s']
    pu = rec[key]
    if lam <= 0:
        return ps
    if mode == 'bilinear':
        return (1 - lam) * ps + lam * pu
    # one_sided：逐 logit 取 relu，U 只能抬升不能压低
    return ps + lam * np.maximum(pu - ps, 0.0)


def evaluate(recs, lam_map, mode, key='pred_u_solo', base_ious=None):
    tp = {c: 0 for c in NEW_IDS}
    fp = {c: 0 for c in NEW_IDS}
    fn = {c: 0 for c in NEW_IDS}
    per_ep = {}
    for rec in recs:
        lam = lam_map.get(rec['g_ep'], 0.0)
        pred = fuse(rec, lam, mode, key).argmax(0)
        gt = rec['gt']
        c = rec['cls']
        t_ = int(((pred == 1) & (gt == 1)).sum())
        f_ = int(((pred == 1) & (gt == 0)).sum())
        n_ = int(((pred == 0) & (gt == 1)).sum())
        tp[c] += t_
        fp[c] += f_
        fn[c] += n_
        per_ep[rec['g_ep']] = t_ / max(t_ + f_ + n_, 1)
    per = [tp[c] / max(tp[c] + fp[c] + fn[c], 1) for c in NEW_IDS]
    return float(np.mean(per)), per, per_ep


def bootstrap(recs, lam_map, mode, base_ious, n_boot=1000, seed=0,
              key='pred_u_solo'):
    """按 episode 有放回重采样，返回 dCumul 的 95% CI。"""
    eps = [r['g_ep'] for r in recs]
    by = {r['g_ep']: r for r in recs}
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_boot):
        samp = rng.choice(eps, size=len(eps), replace=True)
        tp = {c: 0.0 for c in NEW_IDS}
        fp = {c: 0.0 for c in NEW_IDS}
        fn = {c: 0.0 for c in NEW_IDS}
        btp = {c: 0.0 for c in NEW_IDS}
        bfp = {c: 0.0 for c in NEW_IDS}
        bfn = {c: 0.0 for c in NEW_IDS}
        for e in samp:
            rec = by[e]
            lam = lam_map.get(e, 0.0)
            pred = fuse(rec, lam, mode, key).argmax(0)
            base = rec['pred_s'].argmax(0)
            gt = rec['gt']
            c = rec['cls']
            tp[c] += ((pred == 1) & (gt == 1)).sum()
            fp[c] += ((pred == 1) & (gt == 0)).sum()
            fn[c] += ((pred == 0) & (gt == 1)).sum()
            btp[c] += ((base == 1) & (gt == 1)).sum()
            bfp[c] += ((base == 1) & (gt == 0)).sum()
            bfn[c] += ((base == 0) & (gt == 1)).sum()
        cu = np.mean([tp[c] / max(tp[c] + fp[c] + fn[c], 1e-9) for c in NEW_IDS])
        cb = np.mean([btp[c] / max(btp[c] + bfp[c] + bfn[c], 1e-9) for c in NEW_IDS])
        ds.append(cu - cb)
    ds = np.array(ds)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), \
        float((ds > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--delta', type=float, default=0.0,
                    help='任务标签阈值：ΔIoU > delta 记为正')
    ap.add_argument('--lam', type=float, default=0.5)
    ap.add_argument('--off-ratios', default='0.15,0.20,0.30',
                    help='关掉预测亏损风险最高的这一比例任务，其余全开')
    ap.add_argument('--n-boot', type=int, default=200)
    ap.add_argument('--u-key', default='pred_u_solo',
                    help='用哪种 pred_U：pred_u_solo / pred_u_l0.3 / pred_u_l0.5 / pred_u_l0.7')
    ap.add_argument('--out', default=os.path.join(CACHE, 'episode_gate.json'))
    args = ap.parse_args()

    recs = load_recs()
    print('episodes %d' % len(recs))
    key = args.u_key
    base_ious = {r['g_ep']: iou_of(r['pred_s'].argmax(0), r['gt']) for r in recs}
    u_ious = {r['g_ep']: iou_of(r[key].argmax(0), r['gt']) for r in recs}
    delta_iou = {e: u_ious[e] - base_ious[e] for e in base_ious}
    print('pred_S 平均 IoU %.4f   pred_U %.4f   差值 %+.4f' % (
        np.mean(list(base_ious.values())), np.mean(list(u_ious.values())),
        np.mean(list(delta_iou.values()))))
    print('pred_U 更好的比例 %.3f  （Δ>5pp 的比例 %.3f）' % (
        np.mean([delta_iou[e] > 0 for e in delta_iou]),
        np.mean([delta_iou[e] > 0.05 for e in delta_iou])))

    X = np.stack([ep_feats(r, key) for r in recs])
    y = np.array([1 if delta_iou[r['g_ep']] > args.delta else 0 for r in recs])
    print('任务级正样本 %d / %d' % (int(y.sum()), len(y)))

    eps_all = [r['g_ep'] for r in recs]
    half = len(eps_all) // 2
    folds = [(set(eps_all[:half]), set(eps_all[half:]), 'A->B'),
             (set(eps_all[half:]), set(eps_all[:half]), 'B->A')]
    offs = [float(x) for x in args.off_ratios.split(',')]

    all_res = {}
    for tr, te, tag in folds:
        mtr = np.isin(eps_all, list(tr))
        mte = np.isin(eps_all, list(te))
        sc = StandardScaler().fit(X[mtr])
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')
        clf.fit(sc.transform(X[mtr]), y[mtr])
        p_tr = clf.predict_proba(sc.transform(X[mtr]))[:, 1]
        p_te = clf.predict_proba(sc.transform(X[mte]))[:, 1]
        try:
            auc = roc_auc_score(y[mte], p_te)
        except ValueError:
            auc = float('nan')
        print('\n[%s]  EpisodeGate 测试折 AUC %.4f  参数 %d' % (
            tag, auc, clf.coef_.size + clf.intercept_.size))
        te_recs = [r for r in recs if r['g_ep'] in te]
        print('%-8s %6s %9s %9s %9s %9s' % (
            'topk', 'n_on', 'mode', 'Cumul', 'dCumul', 'Q1-Q3'))
        for r in offs:
            # 关掉训练折里 p_open 最低的 r（这些任务 pred_U 更可能更差）
            thr = np.quantile(p_tr, r)
            on = {e for e, pp in zip([r['g_ep'] for r in te_recs], p_te)
                  if pp >= thr}
            lam_map = {e: (args.lam if e in on else 0.0) for e in on}
            for mode in ('bilinear', 'one_sided'):
                cu, _, per_ep = evaluate(te_recs, lam_map, mode, key)
                base_cu = float(np.mean([
                    tp / max(tp + fp + fn, 1) for tp, fp, fn in [
                        (sum(((rec['pred_s'].argmax(0) == 1) & (rec['gt'] == 1)).sum()
                             for rec in te_recs if rec['cls'] == c),
                         sum(((rec['pred_s'].argmax(0) == 1) & (rec['gt'] == 0)).sum()
                             for rec in te_recs if rec['cls'] == c),
                         sum(((rec['pred_s'].argmax(0) == 0) & (rec['gt'] == 1)).sum()
                             for rec in te_recs if rec['cls'] == c)) for c in NEW_IDS]]))
                order = sorted([e for e in per_ep], key=lambda e: base_ious[e])
                q = len(order) // 5
                bins = []
                for i in range(5):
                    seg = order[i * q:(i + 1) * q] if i < 4 else order[4 * q:]
                    bins.append(np.mean([per_ep[e] - base_ious[e] for e in seg]) * 100)
                lo, hi, prob = bootstrap(te_recs, lam_map, mode, base_ious,
                                         n_boot=args.n_boot, key=key)
                print('%-8.2f %6d %9s %9.4f %+9.4f %+9.2f   [%+.2f,%+.2f] P=%.2f  Q4/Q5 %+.2f/%+.2f' % (
                    1 - r, len(on), mode, cu, cu - base_cu,
                    float(np.mean(bins[:3])), lo * 100, hi * 100, prob,
                    bins[3], bins[4]))
                all_res.setdefault(tag, []).append(dict(
                    keep=1 - r, mode=mode, cumul=cu, d=cu - base_cu,
                    bins=bins, ci=[lo, hi], prob=prob, n_on=len(on)))

    json.dump(all_res, open(args.out, 'w'), indent=1)
    print('\n->', args.out)


if __name__ == '__main__':
    main()
