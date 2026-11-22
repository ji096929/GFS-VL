"""验A + 验B(CPU 部分)：全图融合与 Sonata 自选头部纯度。

验A: 融合得分 w*z(cat_top3) + z(p0)，两折 CV 定阈值，357 集 Cumul/Arith。
      纯后处理，不碰 GPU。
验B-1: Sonata 自选高置信前景（全 query 点按分数取 top X%）的逐任务纯度，
       对比 COSeg 的 B 池（p0>0.7）基线纯度 50.4%。
验B-2: p0 与 Sonata 的漏检互补：各自 FN 里有多少能被对方救回。
"""
import glob
import os

import numpy as np
import torch

DUMP = '/data01/jh/data/sonata_dump'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
NAMES = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter',
         'curtain', 'desk', 'door', 'floor']
KEY = 'cat_top3'
WS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
TOPS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]


def load_all():
    gids, sc, p0, lab, cls = [], {}, {}, {}, {}
    for f in sorted(glob.glob(os.path.join(DUMP, 'p2_s*.pt'))):
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
    print('tasks %d  scores %s' % (len(gids), sorted(sc)), flush=True)
    return gids, sc, p0, lab, cls


def zs(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def evaluate(gids, score, lab, cls, thrs):
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
    def __init__(self, gids, score, lab, cls):
        self.per_cls = {}
        for c in NEW_IDS:
            gs = [g for g in gids if int(cls[g]) == c]
            if not gs:
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
        return max((self.cumul_at(th), float(th)) for th in self.cand)[1]


def cv_eval(gids, score, lab, cls):
    A = [g for g in gids if g % 2 == 0]
    B = [g for g in gids if g % 2 == 1]
    thA = Eval(A, score, lab, cls).best_thr()
    thB = Eval(B, score, lab, cls).best_thr()
    thrs = {g: (thA if g % 2 == 0 else thB) for g in gids}
    cum, arith, per = evaluate(gids, score, lab, cls, thrs)
    fg_ratio = float(np.mean(np.concatenate(
        [score[g] > thrs[g] for g in gids])))
    return cum, arith, per, thA, thB, fg_ratio


def top_purity(gids, score, lab, fracs):
    """逐任务 top-X% 的平均精度 + 全局合并 top-X%。"""
    per_ep = {f: [] for f in fracs}
    all_s, all_l = [], []
    for g in gids:
        s = score[g]; l = lab[g] > 0
        o = np.argsort(-s)
        n = len(o)
        for f in fracs:
            k = max(int(n * f), 1)
            per_ep[f].append(float(l[o[:k]].mean()))
        all_s.append(s); all_l.append(l)
    all_s = np.concatenate(all_s); all_l = np.concatenate(all_l)
    o = np.argsort(-all_s); n = len(o)
    glob = {f: float(all_l[o[:max(int(n * f), 1)]].mean()) for f in fracs}
    return {f: float(np.mean(per_ep[f])) for f in fracs}, glob


def main():
    gids, sc, p0, lab, cls = load_all()

    # ---------- 验 A ----------
    print('\n=== 验A: 融合 w*z(%s) + z(p0)，两折 CV 阈值 ===' % KEY)
    print('%-6s %8s %8s %10s %10s %8s'
          % ('w', 'Cumul', 'Arith', 'thetaA', 'thetaB', 'predFG'))
    best = None
    for w in WS:
        fused = {g: w * zs(sc[KEY][g]) + zs(p0[g]) for g in gids}
        cum, arith, per, thA, thB, fg = cv_eval(gids, fused, lab, cls)
        print('%-6.2f %8.4f %8.4f %10.4f %10.4f %8.3f'
              % (w, cum, arith, thA, thB, fg))
        if best is None or cum > best[1]:
            best = (w, cum, arith, per)

    w, cum, arith, per = best
    print('\nbest w = %.2f  Cumul %.4f  Arith %.4f' % (w, cum, arith))
    print('\n逐类 IoU (融合 w=%.2f vs p0)' % w)
    print('%-10s %10s %10s' % ('class', 'p0', 'fused'))
    # p0 在 CV 阈值下的逐类
    fused = {g: w * zs(sc[KEY][g]) + zs(p0[g]) for g in gids}
    _, _, per_f, _, _, _ = cv_eval(gids, fused, lab, cls)
    _, _, per_p0, _, _, _ = cv_eval(gids, p0, lab, cls)
    for i, nm in enumerate(NAMES):
        print('%-10s %10.4f %10.4f' % (nm, per_p0[i], per_f[i]))

    # ---------- 验 B-1 ----------
    print('\n=== 验B-1: 自选高置信前景的头部纯度（逐任务平均 / 全局合并）===')
    print('%-10s %-18s %s' % ('score', '口径',
                              ' '.join('Top%d%%' % (f * 100) for f in TOPS)))
    for tag, s in (('p0', p0), (KEY, sc[KEY])):
        ep, gl = top_purity(gids, s, lab, TOPS)
        print('%-10s %-18s %s' % (tag, '逐任务平均',
                                  ' '.join('%7.4f' % ep[f] for f in TOPS)))
        print('%-10s %-18s %s' % ('', '全局合并',
                                  ' '.join('%7.4f' % gl[f] for f in TOPS)))
    # 对照：COSeg B 池纯度
    bp, bn = [], []
    for g in gids:
        m = p0[g] > 0.7
        l = lab[g][m] > 0
        bp.append(l.sum()); bn.append(len(l) - l.sum())
    print('COSeg B池(p0>0.7): 全局纯度 %.4f  (%d pos / %d neg)'
          % (sum(bp) / max(sum(bp) + sum(bn), 1), sum(bp), sum(bn)))

    # ---------- 验 B-2 ----------
    print('\n=== 验B-2: 漏检互补（CV阈值分割）===')
    # p0 用 CV 阈值；Sonata 用 CV 阈值
    _, _, _, thA, thB, _ = cv_eval(gids, p0, lab, cls)
    p0_thr = {g: (thA if g % 2 == 0 else thB) for g in gids}
    _, _, _, thA, thB, _ = cv_eval(gids, sc[KEY], lab, cls)
    s_thr = {g: (thA if g % 2 == 0 else thB) for g in gids}

    stat = dict(gt=0, pred_s=0, pred_p0=0,
                fn_s=0, fn_s_p0hit=0, fn_p0=0, fn_p0_shit=0,
                tp_s=0, tp_p0=0, fp_s=0, fp_p0=0)
    for g in gids:
        l = lab[g] > 0
        ps = sc[KEY][g] > s_thr[g]
        pp = p0[g] > p0_thr[g]
        stat['gt'] += int(l.sum())
        stat['pred_s'] += int(ps.sum()); stat['pred_p0'] += int(pp.sum())
        stat['fn_s'] += int((~ps & l).sum())
        stat['fn_s_p0hit'] += int((~ps & l & pp).sum())
        stat['fn_p0'] += int((~pp & l).sum())
        stat['fn_p0_shit'] += int((~pp & l & ps).sum())
        stat['tp_s'] += int((ps & l).sum()); stat['tp_p0'] += int((pp & l).sum())
        stat['fp_s'] += int((ps & ~l).sum()); stat['fp_p0'] += int((pp & ~l).sum())

    gt = stat['gt']
    print('总前景点 %d' % gt)
    print('Sonata:  predFG %d (%.1f%%)  TP %d (召回 %.1f%%)  FP %d'
          % (stat['pred_s'], 100.0 * stat['pred_s'] / gt,
             stat['tp_s'], 100.0 * stat['tp_s'] / gt, stat['fp_s']))
    print('p0    :  predFG %d (%.1f%%)  TP %d (召回 %.1f%%)  FP %d'
          % (stat['pred_p0'], 100.0 * stat['pred_p0'] / gt,
             stat['tp_p0'], 100.0 * stat['tp_p0'] / gt, stat['fp_p0']))
    print('Sonata 漏检 %d 点，其中 COSeg 判前景 %d (%.1f%%)'
          % (stat['fn_s'], stat['fn_s_p0hit'],
             100.0 * stat['fn_s_p0hit'] / max(stat['fn_s'], 1)))
    print('COSeg  漏检 %d 点，其中 Sonata 判前景 %d (%.1f%%)'
          % (stat['fn_p0'], stat['fn_p0_shit'],
             100.0 * stat['fn_p0_shit'] / max(stat['fn_p0'], 1)))
    both_fn = sum(int((~(sc[KEY][g] > s_thr[g]) & ~(p0[g] > p0_thr[g])
                       & (lab[g] > 0)).sum()) for g in gids)
    print('两者都漏的公共 FN: %d 点 (%.1f%% of GT)'
          % (both_fn, 100.0 * both_fn / max(gt, 1)))


if __name__ == '__main__':
    main()
