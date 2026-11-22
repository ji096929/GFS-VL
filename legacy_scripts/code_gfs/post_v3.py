"""v3 后处理：合并分片、2 折 CV 选 (λ, gate_thr)、分层表、TP/FP/FN。"""
import argparse
import glob
import json
import os
import re

import numpy as np

CACHE = '/data01/jh/data/ocsb_cache'
NAMES_PATH = '/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
NSHARDS = 4
BASE_STEPS = ['Baseline', 'Naive-B', 'P1_cleanB', 'P2_onesided']


def load_rows():
    files = sorted(glob.glob(os.path.join(CACHE, 'v3_s*.json')))
    files = [f for f in files if '.partial' not in f]
    rows = []
    for f in files:
        mt = re.search(r'v3_s(\d+)\.json', os.path.basename(f))
        if not mt:
            continue
        i = int(mt.group(1))
        d = json.load(open(f))
        for r in d['rows']:
            r['g_ep'] = i + NSHARDS * r['ep']
            rows.append(r)
    keys = {(r['step'], r['g_ep']) for r in rows}
    if len(keys) != len(rows):
        raise SystemExit('!! 重复 (step, g_ep)：%d 行 / %d 唯一' % (
            len(rows), len(keys)))
    return files, rows


def cumul(rows, step, eps_set=None):
    tp = {c: 0 for c in NEW_IDS}
    fp = {c: 0 for c in NEW_IDS}
    fn = {c: 0 for c in NEW_IDS}
    for r in rows:
        if r['step'] != step:
            continue
        if eps_set is not None and r['g_ep'] not in eps_set:
            continue
        tp[r['cls']] += r['tp']
        fp[r['cls']] += r['fp']
        fn[r['cls']] += r['fn']
    per = [tp[c] / max(tp[c] + fp[c] + fn[c], 1) for c in NEW_IDS]
    return float(np.mean(per)), per, (tp, fp, fn)


def ep_map(rows, step):
    return {r['g_ep']: r['iou'] for r in rows if r['step'] == step}


def paired(dmap, bmap):
    ks = sorted(set(dmap) & set(bmap))
    v = np.array([dmap[k] - bmap[k] for k in ks])
    pse = float(v.std(ddof=1)) / np.sqrt(len(v))
    return float(v.mean()), pse, float(v.mean() / pse) if pse > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(CACHE, 'v3_summary.json'))
    args = ap.parse_args()

    files, rows = load_rows()
    print('shards: %s' % [os.path.basename(f) for f in files])
    steps = []
    for r in rows:
        if r['step'] not in steps:
            steps.append(r['step'])
    p3 = [s for s in steps if s.startswith('P3_')]
    eps_all = sorted({r['g_ep'] for r in rows})
    half = len(eps_all) // 2
    fold_a, fold_b = set(eps_all[:half]), set(eps_all[half:])
    print('episodes %d  steps %d（P3 变体 %d）' % (
        len(eps_all), len(steps), len(p3)))

    bmap = ep_map(rows, 'Baseline')
    print('\n=== 表1 全量 %d 集 ===' % len(eps_all))
    print('%-18s %9s %9s %9s %8s %7s' % (
        'step', 'Cumul', 'Arith', 'dCumul', 'dArith', 't'))
    full = {}
    base_c = cumul(rows, 'Baseline')[0]
    for s in steps:
        cu, per, _ = cumul(rows, s)
        dm = ep_map(rows, s)
        av = float(np.mean(list(dm.values())))
        full[s] = dict(cumul=cu, arith=av, per_class=per)
        if s == 'Baseline':
            print('%-18s %9.4f %9.4f %9s %9s %7s' % (s, cu, av, '-', '-', '-'))
        else:
            m, pse, t = paired(dm, bmap)
            print('%-18s %9.4f %9.4f %+9.4f %+9.4f %7.2f' % (
                s, cu, av, cu - base_c, m, t))

    print('\n=== 表2 2 折 CV（A 折选最优 P3 变体，B 折盲测；再互换） ===')
    picks = []
    for tr, te, tag in ((fold_a, fold_b, 'A'), (fold_b, fold_a, 'B')):
        best = max(p3, key=lambda s: cumul(rows, s, tr)[0])
        picks.append((tag, best, te))
        print('  %s 折选中 %-16s (训练 Cumul %.4f)' % (
            tag, best, cumul(rows, best, tr)[0]))
    tp = {c: 0 for c in NEW_IDS}
    fp = {c: 0 for c in NEW_IDS}
    fn = {c: 0 for c in NEW_IDS}
    ious = []
    for tag, s, te in picks:
        _, _, (t_, f_, n_) = cumul(rows, s, te)
        for c in NEW_IDS:
            tp[c] += t_[c]
            fp[c] += f_[c]
            fn[c] += n_[c]
        ious += [r['iou'] for r in rows
                 if r['step'] == s and r['g_ep'] in te]
    blind_per = [tp[c] / max(tp[c] + fp[c] + fn[c], 1) for c in NEW_IDS]
    blind_c = float(np.mean(blind_per))
    print('  盲测：Arith %.4f  Cumul %.4f  (dCumul %+.4f)' % (
        float(np.mean(ious)), blind_c, blind_c - base_c))

    print('\n=== 表3 按 Baseline 难度分层（逐集配对差，pp） ===')
    order = sorted(bmap, key=lambda k: bmap[k])
    q = len(order) // 5
    best_p3 = sorted(p3, key=lambda s: -full[s]['cumul'])[:4]
    sel = ['Naive-B', 'P1_cleanB', 'P2_onesided'] + best_p3
    print('%-18s %8s %8s %8s %8s %8s %8s' % (
        'step', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q1-Q3'))
    for s in sel:
        dm = ep_map(rows, s)
        v = []
        for i in range(5):
            seg = order[i * q:(i + 1) * q] if i < 4 else order[4 * q:]
            v.append(np.mean([dm[k] - bmap[k] for k in seg]) * 100)
        q13 = np.mean([dm[k] - bmap[k] for k in order[:3 * q]]) * 100
        print('%-18s %+8.2f %+8.2f %+8.2f %+8.2f %+8.2f %+8.2f' % (s, *v, q13))

    print('\n=== 表4 TP/FP/FN（相对 Baseline） ===')
    _, _, (bt, bf, bn) = cumul(rows, 'Baseline')
    for s in sel:
        _, _, (t_, f_, n_) = cumul(rows, s)
        dt = sum(t_.values()) - sum(bt.values())
        df = sum(f_.values()) - sum(bf.values())
        dn = sum(n_.values()) - sum(bn.values())
        print('%-18s dTP=%+7d  dFP=%+7d  dFN=%+7d' % (s, dt, df, dn))

    print('\n=== 表5 逐类 Cumul（盲测选中的 P3 变体） ===')
    names = [l.strip() for l in open(NAMES_PATH)]
    hdr = ['Baseline', 'Naive-B'] + [picks[0][1], picks[1][1]]
    print('%-14s' % 'class' + ''.join('%16s' % h for h in hdr))
    for i, c in enumerate(NEW_IDS):
        print('%-14s' % names[c] + ''.join(
            '%16.4f' % full[h]['per_class'][i] for h in hdr))

    json.dump(dict(full={k: dict(cumul=v['cumul'], arith=v['arith'],
                                 per_class=v['per_class'])
                         for k, v in full.items()},
                   picks=[(a, b) for a, b, _ in picks],
                   blind=dict(arith=float(np.mean(ious)), cumul=blind_c),
                   n_ep=len(eps_all)),
              open(args.out, 'w'), indent=1)
    print('\n->', args.out)


if __name__ == '__main__':
    main()
