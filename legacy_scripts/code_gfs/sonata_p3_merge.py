"""检验B 汇总：原型采样空间（feat/nfeat/coord）× k 的 CV Cumul/Arith + 头部纯度。"""
import glob
import os

import numpy as np
import torch

DUMP = '/data01/jh/data/sonata_dump'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
NAMES = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter',
         'curtain', 'desk', 'door', 'floor']
MODES = ('feat', 'nfeat', 'coord')
KS = (3, 10, 30, 100)
RULES = ('multi', 'top3')
TOPS = [0.01, 0.02, 0.05, 0.10]


def load_all():
    gids, sc, p0, lab, cls = [], {}, {}, {}, {}
    for f in sorted(glob.glob(os.path.join(DUMP, 'p3_s*.pt'))):
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
    # 清洗：7 个任务的 support 全前景或全背景（gid 139/159/161/319/482/491/503），
    # mean 原型对空类算出 NaN，protos_space 则直接缺席。
    # 统一填 0，与 p2 的 score_one 空原型返回 zeros 口径一致。
    for k in list(sc):
        for g in gids:
            a = sc[k].get(g)
            if a is None or not np.isfinite(a).all():
                sc[k][g] = np.zeros_like(p0[g], np.float32)
    print('files tasks %d  variants %d' % (len(gids), len(sc)), flush=True)
    return gids, sc, p0, lab, cls


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
    return cum, arith, per, thA, thB


def purity(gids, score, lab, fracs):
    out = {f: [] for f in fracs}
    for g in gids:
        s = score[g]; l = lab[g] > 0
        o = np.argsort(-s); n = len(o)
        for f in fracs:
            k = max(int(n * f), 1)
            out[f].append(float(l[o[:k]].mean()))
    return {f: float(np.mean(out[f])) for f in fracs}


def main():
    gids, sc, p0, lab, cls = load_all()
    keys = sorted(sc)
    res = {}
    print('\n=== 变体 CV Cumul/Arith + 头部纯度（逐任务平均）===')
    print('%-16s %8s %8s %8s %8s %8s'
          % ('cfg', 'Cumul', 'Arith', 'Top1%', 'Top2%', 'Top5%'))
    for key in ['mean'] + [k for k in keys if k != 'mean']:
        if key not in sc:
            continue
        cum, arith, per, thA, thB = cv_eval(gids, sc[key], lab, cls)
        pu = purity(gids, sc[key], lab, TOPS)
        res[key] = dict(cum=cum, arith=arith, per=per, pu=pu,
                        thA=thA, thB=thB)
        print('%-16s %8.4f %8.4f %8.4f %8.4f %8.4f'
              % (key, cum, arith, pu[0.01], pu[0.02], pu[0.05]))

    # k 扫描表
    for rule in RULES:
        print('\n=== k 扫描 (%s): Cumul / Arith ===' % rule)
        print('%-6s %18s %18s %18s'
              % ('k', 'feat', 'nfeat', 'coord'))
        for k in KS:
            row = []
            for m in MODES:
                key = '%s_k%d_%s' % (m, k, rule)
                if key in res:
                    row.append('%7.4f/%6.3f' % (res[key]['cum'],
                                                 res[key]['arith']))
                else:
                    row.append('%13s' % '-')
            print('%-6d %18s %18s %18s' % (k, row[0], row[1], row[2]))

    best = max(res, key=lambda k: res[k]['cum'])
    print('\nbest cfg -> %s  Cumul %.4f  Arith %.4f'
          % (best, res[best]['cum'], res[best]['arith']))
    print('\n逐类 IoU (%s vs 均值原型)' % best)
    print('%-10s %10s %10s' % ('class', 'mean', best))
    for i, nm in enumerate(NAMES):
        print('%-10s %10.4f %10.4f'
              % (nm, res['mean']['per'][i], res[best]['per'][i]))


if __name__ == '__main__':
    main()
