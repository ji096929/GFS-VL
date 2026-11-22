"""检查1(提纯) + 检查2(零训练分割水位) 的汇总。

检查1: 在 B={p0>0.7} 上对比 7 种打分的 AUC 与 Top-X% 精度
检查2: 6 种 Sonata 打分当零训练 1-shot 分割器，全 query 点算 Cumul/Arith
        阈值两折交叉验证（折内 Cumul 最优）
"""
import glob
import os

import numpy as np
import torch
from scipy.stats import mannwhitneyu

DUMP = '/data01/jh/data/sonata_dump'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
NAMES = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter',
         'curtain', 'desk', 'door', 'floor']
TOP_FRACS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]


def load_all():
    gids, sc, p0, lab, cls = [], {}, {}, {}, {}
    fs = sorted(glob.glob(os.path.join(DUMP, 'p2_s*.pt')))
    for f in fs:
        d = torch.load(f, map_location='cpu', weights_only=False)
        gids += list(d['gid'])
        for k, v in d['scores'].items():
            sc.setdefault(k, {})
            for g, t in v.items():
                sc[k][g] = t.float().numpy()
        for g, v in d['p0'].items():
            p0[g] = np.asarray(v, np.float32)
        for g, v in d['lab'].items():
            lab[g] = np.asarray(v, np.uint8)
        cls.update(d['cls'])
    gids = sorted(set(gids))
    print('files %d  tasks %d  scores %s'
          % (len(fs), len(gids), sorted(sc)), flush=True)
    return gids, sc, p0, lab, cls


def auc_stats(gids, sc, p0, lab, thr_b):
    out = {}
    for k in ['p0'] + sorted(sc):
        pos, neg, s_all, l_all, ep_auc = [], [], [], [], []
        for g in gids:
            m = p0[g] > thr_b
            s = (p0[g] if k == 'p0' else sc[k][g])[m]
            l = (lab[g] > 0)[m]
            if not l.any() or l.all():
                continue
            pos.append(s[l]); neg.append(s[~l])
            s_all.append(s); l_all.append(l)
            u = mannwhitneyu(s[l], s[~l], alternative='greater').statistic
            ep_auc.append(u / (l.sum() * (~l).sum()))
        pos = np.concatenate(pos); neg = np.concatenate(neg)
        u = mannwhitneyu(pos, neg, alternative='greater').statistic
        auc = u / (len(pos) * len(neg))
        s_all = np.concatenate(s_all); l_all = np.concatenate(l_all)
        o = np.argsort(-s_all)
        n = len(o)
        tops = {fr: float(l_all[o[:max(int(n * fr), 1)]].mean())
                for fr in TOP_FRACS}
        out[k] = dict(auc=auc, ep_auc=float(np.mean(ep_auc)),
                      n_pos=len(pos), n_neg=len(neg), top=tops,
                      n_B=int(len(s_all)))
    return out


def zs(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def ensemble_scores(gids, sc, p0, key, w):
    return {g: w * zs(sc[key][g]) + zs(p0[g]) for g in gids}


def evaluate(gids, score, lab, cls, thrs):
    """逐集 IoU -> Arith；按类累计 TP/FP/FN -> Cumul。"""
    tp = {c: 0 for c in NEW_IDS}
    fp = {c: 0 for c in NEW_IDS}
    fn = {c: 0 for c in NEW_IDS}
    ious = []
    for g in gids:
        pred = score[g] > thrs[g]
        l = lab[g] > 0
        c = int(cls[g])
        t = int((pred & l).sum()); f = int((pred & ~l).sum())
        n = int((~pred & l).sum())
        tp[c] += t; fp[c] += f; fn[c] += n
        ious.append(t / max(t + f + n, 1))
    per = [tp[c] / max(tp[c] + fp[c] + fn[c], 1) for c in NEW_IDS]
    return float(np.mean(per)), float(np.mean(ious)), per


class Eval:
    """按类预排序，O(log n) 查任意阈值的 Cumul mIoU。"""

    def __init__(self, gids, score, lab, cls):
        self.per_cls = {}
        for c in NEW_IDS:
            gs = [g for g in gids if int(cls[g]) == c]
            if not gs:      # 某个折里可能整类缺席
                self.per_cls[c] = (np.zeros(0, np.float32),
                                   np.zeros(0, np.int64), 0)
                continue
            ss = np.concatenate([score[g] for g in gs])
            ll = np.concatenate([lab[g] > 0 for g in gs])
            o = np.argsort(ss)
            ss, ll = ss[o], ll[o]
            self.per_cls[c] = (ss, np.cumsum(ll).astype(np.int64), int(ll.sum()))
        alls = np.concatenate([score[g] for g in gids])
        self.cand = np.quantile(alls, np.linspace(0.005, 0.995, 199))

    def cumul_at(self, th):
        per = []
        for c in NEW_IDS:
            ss, cum, npos = self.per_cls[c]
            k = int(np.searchsorted(ss, th, side='right'))
            tp = npos - int(cum[k]) if k < len(cum) else 0
            npred = len(ss) - k
            fp = npred - tp
            fn = npos - tp
            per.append(tp / max(tp + fp + fn, 1))
        return float(np.mean(per))

    def best_thr(self):
        vals = [(self.cumul_at(th), float(th)) for th in self.cand]
        return max(vals)[1]


def main(thr_b=0.7):
    gids, sc, p0, lab, cls = load_all()

    # ---------- 检查 1 ----------
    res = auc_stats(gids, sc, p0, lab, thr_b)
    n_pos = res['p0']['n_pos']; n_neg = res['p0']['n_neg']
    print('\n=== 检查1: B={p0>%.1f} 共 %d 点 (pos %d / neg %d, 正例率 %.1f%%) ==='
          % (thr_b, res['p0']['n_B'], n_pos, n_neg,
             100.0 * n_pos / max(n_pos + n_neg, 1)))
    print('%-12s %8s %10s %s'
          % ('score', 'AUC', '逐任务AUC',
             ''.join('%8s' % ('T%g%%' % (f * 100)) for f in TOP_FRACS)))
    for k in ['p0'] + sorted(sc):
        r = res[k]
        print('%-12s %.4f %10.4f %s' % (
            k, r['auc'], r['ep_auc'],
            ''.join('%8.4f' % r['top'][f] for f in TOP_FRACS)))

    best_key = max(sorted(sc), key=lambda k: res[k]['auc'])
    print('best sonata cfg (by AUC) ->', best_key, flush=True)

    print('\n=== 协同: w*z(%s) + z(p0), 再算 AUC/TopX ===' % best_key)
    print('%-8s %8s %s'
          % ('w', 'AUC', ''.join('%8s' % ('T%g%%' % (f * 100)) for f in TOP_FRACS)))
    for w in (0.25, 0.5, 0.75, 1.0):
        ens = ensemble_scores(gids, sc, p0, best_key, w)
        pos, neg, s_all, l_all = [], [], [], []
        for g in gids:
            m = p0[g] > thr_b
            s = ens[g][m]; l = lab[g][m] > 0
            pos.append(s[l]); neg.append(s[~l])
            s_all.append(s); l_all.append(l)
        pos = np.concatenate(pos); neg = np.concatenate(neg)
        u = mannwhitneyu(pos, neg, alternative='greater').statistic
        auc = u / (len(pos) * len(neg))
        s_all = np.concatenate(s_all); l_all = np.concatenate(l_all)
        o = np.argsort(-s_all); n = len(o)
        tops = {fr: float(l_all[o[:max(int(n * fr), 1)]].mean())
                for fr in TOP_FRACS}
        print('%-8.2f %.4f %s' % (w, auc,
                                  ''.join('%8.4f' % tops[f] for f in TOP_FRACS)))

    # ---------- 检查 2 ----------
    print('\n=== 检查2: 全 query 点零训练 1-shot 分割 (%d 集) ===' % len(gids))
    base_cum, base_arith, base_per = evaluate(
        gids, p0, lab, cls, {g: np.float32(0.5) for g in gids})
    print('%-30s %8s %8s' % ('config', 'Cumul', 'Arith'))
    print('%-30s %8.4f %8.4f' % ('COSeg p0 > 0.5', base_cum, base_arith))

    A = [g for g in gids if g % 2 == 0]
    B = [g for g in gids if g % 2 == 1]
    tab = {}
    for key in sorted(sc):
        s = sc[key]
        cum0, arith0, per0 = evaluate(
            gids, s, lab, cls, {g: np.float32(0.0) for g in gids})
        thA = Eval(A, s, lab, cls).best_thr()
        thB = Eval(B, s, lab, cls).best_thr()
        thr_cv = {g: (thA if g % 2 == 0 else thB) for g in gids}
        cumCV, arithCV, perCV = evaluate(gids, s, lab, cls, thr_cv)
        tab[key] = dict(cum0=cum0, arith0=arith0, cumCV=cumCV,
                        arithCV=arithCV, thA=thA, thB=thB,
                        per0=per0, perCV=perCV)
        print('%-30s %8.4f %8.4f' % (key + ' > 0', cum0, arith0))
        print('%-30s %8.4f %8.4f  (thetaA=%.4f thetaB=%.4f)'
              % (key + ' CV-theta', cumCV, arithCV, thA, thB))

    best2 = max(sorted(sc), key=lambda k: tab[k]['cumCV'])
    print('\nbest zero-shot cfg (by CV Cumul) ->', best2)
    print('\n逐类 IoU (p0 vs %s)' % best2)
    print('%-10s %10s %10s %10s' % ('class', 'p0', best2 + '>0', 'CVth'))
    for i, nm in enumerate(NAMES):
        print('%-10s %10.4f %10.4f %10.4f'
              % (nm, base_per[i], tab[best2]['per0'][i], tab[best2]['perCV'][i]))

    s_all = np.concatenate([sc[best2][g] for g in gids])
    print('\nscore 分位 (全 query 点合并):',
          np.round(np.percentile(s_all, [1, 5, 25, 50, 75, 95, 99]), 4))
    print('p0 argmax 前景占比: %.4f' % float(np.mean(np.concatenate(
        [p0[g] > 0.5 for g in gids]))))
    print('score>0 前景占比: %.4f' % float(np.mean(np.concatenate(
        [sc[best2][g] > 0 for g in gids]))))


if __name__ == '__main__':
    main()
