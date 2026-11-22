"""合并全部分片的 ub560 结果：全量 λ 表 + 配对检验 + 2 折交叉验证 + 逐类/分层表。

支持两轮 λ（不同文件名前缀），会自动发现 rows 里出现的所有 λ。
"""
import argparse
import glob
import json
import os
import re

import numpy as np

CACHE = '/data01/jh/data/ocsb_cache'
NAMES_PATH = '/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
BASES = ['Naive-B', 'Oracle-B', 'Oracle-B+C']
PRE_LAM = 0.3
NSHARDS = 4


def load_rows(keep_lams=None):
    """读入所有分片，还原每行的全局 episode 编号 g_ep。

    文件名的数字是分片号 i，`ep` 是分片内的局部序号；因为分片方式是
    `eps[shard::shards]`，所以全局编号 = i + NSHARDS * ep。

    两轮 λ（ub560_* 与 ub560x_*）各自都带一份 Baseline，这里按 g_ep 去重，
    并顺带校验两轮的 Baseline 是否逐集一致——锁种子之后应该完全一致，
    不一致就说明体素化 RNG 又漏了。λ 过滤放在**文件级**：只保留请求内
    的那些轮次，避免 Baseline 被重复计数。
    """
    files = sorted(glob.glob(os.path.join(CACHE, 'ub560*_s*.json')))
    files = [f for f in files if '.partial' not in f]
    rows, seen = [], []
    base_seen = {}
    bad = []
    for f in files:
        mt = re.search(r'ub560\w*_s(\d+)\.json', os.path.basename(f))
        if not mt:
            continue
        i = int(mt.group(1))
        if i >= NSHARDS:
            print('!! 跳过 %s：shard 号 %d >= %d' % (
                os.path.basename(f), i, NSHARDS))
            continue
        d = json.loads(open(f).read())
        # 整文件级别的 λ 过滤：Baseline 的 lam 恒为 0，不能用行级过滤，
        # 否则不同轮次的 Baseline 会重复计数
        if keep_lams:
            fl = {round(float(r['lam']), 2) for r in d['rows']
                  if r['case'] != 'Baseline'}
            if not (fl & set(keep_lams)):
                print('   跳过 %s（λ=%s 不在请求内）' % (
                    os.path.basename(f), sorted(fl)))
                continue
        seen.append((os.path.basename(f), len(d['rows'])))
        for r in d['rows']:
            r['g_ep'] = i + NSHARDS * r['ep']
            if r['case'] == 'Baseline':
                # 两轮 λ 各自都带 Baseline；顺手校验两次是否逐集一致
                prev = base_seen.get(r['g_ep'])
                if prev is not None:
                    if abs(prev - r['iou']) > 1e-4:
                        bad.append((r['g_ep'], prev, r['iou']))
                    continue
                base_seen[r['g_ep']] = r['iou']
            rows.append(r)
    if bad:
        print('!! Baseline 两轮不一致 %d/%d 集，前 5 条：%s' % (
            len(bad), len(base_seen),
            ['g%d %.4f/%.4f' % t for t in bad[:5]]))
    else:
        print('Baseline 两轮逐集一致（%d 集，阈值 1e-4）' % len(base_seen))
    keys = {(r['case'], r['g_ep']) for r in rows}
    if len(keys) != len(rows):
        raise SystemExit('!! 有重复 (case, g_ep)：%d 行 / %d 唯一，'
                         '可能混入了重复分片' % (len(rows), len(keys)))
    return seen, rows


def cumul(rows, case, eps_set=None):
    """累积口径（Cumul）：先把 10 个类的 TP/FP/FN 分别累加，再算每类 IoU 取平均。

    与「逐集 IoU 再平均」（Arith）的区别：Cumul 给每个类别等权，且点数多的
    episode 自然权重大；这是 COSeg 官方表格用的口径，两者不能混着比。
    `eps_set` 用于 2 折交叉验证时只统计某一折。
    """
    tp = {c: 0 for c in NEW_IDS}
    fp = {c: 0 for c in NEW_IDS}
    fn = {c: 0 for c in NEW_IDS}
    for r in rows:
        if r['case'] != case:
            continue
        if eps_set is not None and r['g_ep'] not in eps_set:
            continue
        tp[r['cls']] += r['tp']
        fp[r['cls']] += r['fp']
        fn[r['cls']] += r['fn']
    per = [tp[c] / max(tp[c] + fp[c] + fn[c], 1) for c in NEW_IDS]
    return float(np.mean(per)), per


def ep_map(rows, case):
    """g_ep -> 该 episode 的 IoU，供配对比较使用（同一 key 才可比）。"""
    return {r['g_ep']: r['iou'] for r in rows if r['case'] == case}


def paired(dmap, bmap):
    """逐集配对差：delta = case - Baseline，同一 episode 上相减再统计。

    返回 (均值, 标准误 pSE, t 统计量, 样本数)。配对的意义是把 episode 之间
    的巨大难度差异（IoU 从 0.02 到 0.95）消掉，只看同一集上的变化。
    注意这依赖 Baseline 与 case 在同一个进程里算出——跨进程的 Baseline
    有体素化漂移，会让配对失效。
    """
    ks = sorted(set(dmap) & set(bmap))
    v = np.array([dmap[k] - bmap[k] for k in ks])
    n = len(v)
    mean = float(v.mean())
    pse = float(v.std(ddof=1)) / np.sqrt(n)
    t = mean / pse if pse > 0 else float('nan')
    return mean, pse, t, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lams', default='', help='逗号分隔，只保留这些 λ（空=全部）')
    ap.add_argument('--out', default=os.path.join(CACHE, 'ub560_summary.json'))
    args = ap.parse_args()
    keep = [float(x) for x in args.lams.split(',')] if args.lams else None

    seen, rows = load_rows(keep)
    eps_all = sorted({r['g_ep'] for r in rows})
    half = len(eps_all) // 2
    fold_a, fold_b = set(eps_all[:half]), set(eps_all[half:])
    lams = sorted({round(float(r['lam']), 2) for r in rows
                   if r['case'] != 'Baseline'})
    print('shards (%d): %s' % (len(seen), seen))
    print('rows %d  episodes %d  lams %s  (fold A %d / B %d)' % (
        len(rows), len(eps_all), lams, len(fold_a), len(fold_b)))

    bmap = ep_map(rows, 'Baseline')
    cases = ['Baseline'] + ['%s_l%.1f' % (b, l) for l in lams for b in BASES]

    print('\n=== 表1 全量 %d 集 ===' % len(eps_all))
    print('%-18s %9s %9s %9s %8s %7s' % (
        'case', 'Cumul', 'Arith', 'dCumul', 'dArith', 't'))
    full = {}
    for c in cases:
        cu, per = cumul(rows, c)
        dm = ep_map(rows, c)
        av = float(np.mean(list(dm.values())))
        full[c] = dict(cumul=cu, arith=av, per_class=per)
        if c == 'Baseline':
            print('%-18s %9.4f %9.4f %9s %9s %7s' % (c, cu, av, '-', '-', '-'))
        else:
            m, pse, t, n = paired(dm, bmap)
            print('%-18s %9.4f %9.4f %+9.4f %+9.4f %7.2f' % (
                c, cu, av, cu - full['Baseline']['cumul'], m, t))

    print('\n=== 表2 2 折交叉验证（各折独立选 λ，再合并盲测） ===')
    chosen = {}
    for b in BASES:
        picks = []
        for tr, te, tag in ((fold_a, fold_b, 'A'), (fold_b, fold_a, 'B')):
            vals = [(cumul(rows, '%s_l%.1f' % (b, l), tr)[0], l) for l in lams]
            picks.append((tag, max(vals)[1]))
        chosen[b] = picks
        print('%-14s %s' % (b, picks))

    print('\n%-14s %9s %9s %11s' % ('base', 'Arith', 'Cumul', 'dCumul'))
    blind = {}
    for b, picks in chosen.items():
        tp = {c: 0 for c in NEW_IDS}
        fp = {c: 0 for c in NEW_IDS}
        fn = {c: 0 for c in NEW_IDS}
        ious = []
        for tag, lam in picks:
            fe = fold_a if tag == 'A' else fold_b
            case = '%s_l%.1f' % (b, lam)
            for r in rows:
                if r['case'] != case or r['g_ep'] not in fe:
                    continue
                tp[r['cls']] += r['tp']
                fp[r['cls']] += r['fp']
                fn[r['cls']] += r['fn']
                ious.append(r['iou'])
        per = [tp[c] / max(tp[c] + fp[c] + fn[c], 1) for c in NEW_IDS]
        blind[b] = dict(arith=float(np.mean(ious)), cumul=float(np.mean(per)))
        print('%-14s %9.4f %9.4f %+11.4f' % (
            b, blind[b]['arith'], blind[b]['cumul'],
            blind[b]['cumul'] - full['Baseline']['cumul']))

    print('\n=== 表3 逐类 Cumul（λ=%.1f 预注册） ===' % PRE_LAM)
    names = [l.strip() for l in open(NAMES_PATH)]
    hdr = ['Baseline'] + ['%s_l%.1f' % (b, PRE_LAM) for b in BASES]
    hdr = [h for h in hdr if h in full]
    print('%-14s' % 'class' + ''.join('%18s' % h for h in hdr))
    for i, c in enumerate(NEW_IDS):
        line = '%-14s' % names[c]
        for h in hdr:
            line += '%18.4f' % full[h]['per_class'][i]
        print(line)

    print('\n=== 表4 增益分解（pp，相对 Baseline Cumul） ===')
    base_c = full['Baseline']['cumul']
    for l in lams:
        n = full['Naive-B_l%.1f' % l]['cumul'] - base_c
        ob = full['Oracle-B_l%.1f' % l]['cumul'] - base_c
        obc = full['Oracle-B+C_l%.1f' % l]['cumul'] - base_c
        print('l%.1f  Naive %+6.2f | OracleB %+6.2f (B_neg %+6.2f) | '
              'OBC %+6.2f (C_pos %+6.2f)' % (
                  l, n * 100, ob * 100, (n - ob) * 100,
                  obc * 100, (obc - ob) * 100))

    print('\n=== 表5 按 Baseline 难度分层（逐集配对差，pp） ===')
    print('每个 λ 一张表；最后一列 C_pos 净增 = Oracle-B+C − Oracle-B')
    for lam in [x for x in (0.3, 0.5, 0.7) if x in lams]:
        dmap = {b: ep_map(rows, '%s_l%.1f' % (b, lam)) for b in BASES}
        ks = [k for k in sorted(bmap) if all(k in dmap[b] for b in BASES)]
        order = sorted(ks, key=lambda k: bmap[k])
        q = len(order) // 5
        print('\n-- λ=%.1f --' % lam)
        print('%-5s %5s %9s %9s %9s %9s %10s' % (
            'bin', 'n', 'baseIoU', 'dNaive', 'dOracleB', 'dOracleBC', 'C_pos'))
        for i in range(5):
            seg = order[i * q:(i + 1) * q] if i < 4 else order[4 * q:]
            if not seg:
                continue
            d = {b: np.mean([dmap[b][k] - bmap[k] for k in seg]) for b in BASES}
            print('Q%-4d %5d %9.4f %+9.2f %+9.2f %+9.2f %+10.2f' % (
                i + 1, len(seg), np.mean([bmap[k] for k in seg]),
                d['Naive-B'] * 100, d['Oracle-B'] * 100,
                d['Oracle-B+C'] * 100,
                (d['Oracle-B+C'] - d['Oracle-B']) * 100))

    slice50 = set(range(50))
    if len(eps_all) > 50 and slice50 <= set(eps_all):
        print('\n=== 表6 切片复现：每类前 5 个（g_ep<50，共 50 集） ===')
        print('%-18s %9s %9s %10s' % ('case', 'Arith', 'Cumul', 'dCumul'))
        c0 = cumul(rows, 'Baseline', slice50)[0]
        a0 = float(np.mean([v for k, v in bmap.items() if k in slice50]))
        print('%-18s %9.4f %9.4f %10s' % ('Baseline', a0, c0, '-'))
        for b in BASES:
            for l in lams:
                case = '%s_l%.1f' % (b, l)
                cv = cumul(rows, case, slice50)[0]
                av = float(np.mean([r['iou'] for r in rows
                                    if r['case'] == case
                                    and r['g_ep'] in slice50]))
                print('%-18s %9.4f %9.4f %+10.4f' % (case, av, cv, cv - c0))

    json.dump(dict(full=full, chosen=chosen, blind=blind,
                   n_ep=len(eps_all), lams=lams),
              open(args.out, 'w'), indent=1)
    print('\n-> %s' % args.out)


if __name__ == '__main__':
    main()
