'''扫描全部 run 日志，生成 OVERNIGHT_CUT2_REPORT.md。'''
import os
import re

import numpy as np

D = '/data01/jh/afford_diag/xl_runs'
OUT = '/data01/jh/code/GFS-VL/OVERNIGHT_CUT2_REPORT.md'
NB, NC = 12, 57

RUNS = [
    ('base_s42', 'base_s42.log', 'base', 42),
    ('xl_s42', 'sc200_k1_XLu_s42_resume.log', 'xl', 42),
    ('base_s7', 'sc200_k1_BASEu_s7.log', 'base', 7),
    ('xl_s7', 'xl_s7_eval.log', 'xl', 7),
    ('mlp_dec0_s42', 'mlp_dec0_s42.log', 'mlp_dec0', 42),
    ('mlp_dec0_s7', 'mlp_dec0_s7.log', 'mlp_dec0', 7),
    ('sym_xl_s42', 'sym_xl_s42.log', 'sym_xl', 42),
    ('sym_xl_s7', 'sym_xl_s7.log', 'sym_xl', 7),
    ('base_s13', 'base_s13.log', 'base', 13),
    ('sym_xl_s13', 'sym_xl_s13.log', 'sym_xl', 13),
    ('xl_s13', 'xl_s13.log', 'xl', 13),
    ('gated_sym_xl_s42', 'gated_sym_xl_s42.log', 'gated', 42),
    ('gated_sym_xl_s7', 'gated_sym_xl_s7.log', 'gated', 7),
]
WATCH = ['bookshelf', 'shelf', 'cabinet', 'plant', 'office chair',
         'armchair', 'mirror', 'lamp', 'shoe', 'doorframe', 'tv', 'bathtub']


def parse(path):
    if not os.path.exists(path):
        return None
    d = {}
    for line in open(path, errors='ignore'):
        m = re.search(r'Class_(\d+) - (.+?) Final Result: iou ([\d.]+)', line)
        if m:
            d[int(m.group(1))] = (m.group(2), float(m.group(3)))
    if len(d) != NC:
        return None
    v = np.array([d[i][1] for i in range(NC)])
    names = [d[i][0] for i in range(NC)]
    b, n = v[:NB].mean(), v[NB:].mean()
    return dict(v=v, names=names, B=float(b), N=float(n),
                HM=float(2 * b * n / (b + n)), M=float(v.mean()))


def idx_of(r, w):
    return r['names'].index(w) if w in r['names'] else -1


def main():
    res = {}
    for name, fn, tag, seed in RUNS:
        r = parse(os.path.join(D, fn))
        if r is not None:
            r['tag'], r['seed'] = tag, seed
            res[name] = r
    L = []
    L.append('# OVERNIGHT CUT2 REPORT')
    L.append('')
    L.append('自动生成。数据源：312 val 逐类 IoU。')
    L.append('')
    L.append('## 1. 总对比表')
    L.append('')
    L.append('| run | Base | Novel | HM | mIoU | 状态 |')
    L.append('|---|---|---|---|---|---|')
    for name, fn, tag, seed in RUNS:
        if name in res:
            r = res[name]
            L.append('| %s | %.4f | %.4f | %.4f | %.4f | done |'
                     % (name, r['B'], r['N'], r['HM'], r['M']))
        else:
            L.append('| %s | — | — | — | — | pending |' % name)
    L.append('')
    L.append('## 2. 方法跨 seed 均值 ± 标准差')
    L.append('')
    L.append('| 方法 | seeds | Base | Novel | HM |')
    L.append('|---|---|---|---|---|')
    for tag in ('base', 'xl', 'mlp_dec0', 'sym_xl', 'gated'):
        rows = [r for n, r in res.items() if r['tag'] == tag]
        if not rows:
            continue
        L.append('| %s | %s | %.4f ± %.4f | %.4f ± %.4f | %.4f ± %.4f |'
                 % (tag, ','.join(str(r['seed']) for r in rows),
                    float(np.mean([r['B'] for r in rows])),
                    float(np.std([r['B'] for r in rows])),
                    float(np.mean([r['N'] for r in rows])),
                    float(np.std([r['N'] for r in rows])),
                    float(np.mean([r['HM'] for r in rows])),
                    float(np.std([r['HM'] for r in rows]))))
    L.append('')
    L.append('## 3. 相对同 seed base 的配对 Δ')
    L.append('')
    L.append('| 方法 | seed | ΔBase | ΔNovel | ΔHM |')
    L.append('|---|---|---|---|---|')
    for tag in ('xl', 'mlp_dec0', 'sym_xl', 'gated'):
        for n, r in res.items():
            if r['tag'] != tag:
                continue
            b = res.get('base_s%d' % r['seed'])
            if not b:
                continue
            L.append('| %s | %d | %+.2f | %+.2f | %+.2f |'
                     % (tag, r['seed'], (r['B'] - b['B']) * 100,
                        (r['N'] - b['N']) * 100, (r['HM'] - b['HM']) * 100))
    L.append('')
    L.append('## 4. 逐类重点追踪')
    L.append('')
    cols = [n for n, *_ in RUNS if n in res]
    L.append('| class | ' + ' | '.join(cols) + ' |')
    L.append('|' + '---|' * (len(cols) + 1))
    for w in WATCH:
        cells = []
        for c in cols:
            r = res[c]
            j = idx_of(r, w)
            cells.append('%.3f' % r['v'][j] if j >= 0 else '—')
        L.append('| %s | %s |' % (w, ' | '.join(cells)))
    L.append('')
    L.append('## 5. 45 新类正负胜率（相对同 seed base）')
    L.append('')
    L.append('| 方法 | seed | 正 | 负 | 正贡献 | 负贡献 |')
    L.append('|---|---|---|---|---|---|')
    for tag in ('xl', 'mlp_dec0', 'sym_xl', 'gated'):
        for n, r in res.items():
            if r['tag'] != tag:
                continue
            b = res.get('base_s%d' % r['seed'])
            if not b:
                continue
            d = (r['v'][NB:] - b['v'][NB:]) * 100
            L.append('| %s | %d | %d/45 | %d/45 | %+.1f | %+.1f |'
                     % (tag, r['seed'], int((d > 0).sum()),
                        int((d <= 0).sum()), float(d[d > 0].sum()),
                        float(d[d <= 0].sum())))
    L.append('')
    L.append('## 6. 核心对照诊断')
    L.append('')

    def g(tag, seed):
        return res.get('%s_s%d' % (tag, seed))

    L.append('### xl vs mlp_dec0（是否只是加了 MLP 容量）')
    L.append('')
    for s in (42, 7, 13):
        a, b = g('mlp_dec0', s), g('xl', s)
        if not a or not b:
            continue
        L.append('- seed %d: xl 净胜 ΔBase %+.2f / ΔNovel %+.2f / ΔHM %+.2f'
                 % (s, (b['B'] - a['B']) * 100, (b['N'] - a['N']) * 100,
                    (b['HM'] - a['HM']) * 100))
    L.append('')
    L.append('### sym_xl vs xl（bookshelf 与 Base 修复，负值表示 sym 更差）')
    L.append('')
    for s in (42, 7, 13):
        a, b = g('xl', s), g('sym_xl', s)
        if not a or not b:
            continue
        ia, ib = idx_of(a, 'bookshelf'), idx_of(b, 'bookshelf')
        L.append('- seed %d: bookshelf %.3f -> %.3f (%+.2f) | Base %+.2f | Novel %+.2f | HM %+.2f'
                 % (s, a['v'][ia], b['v'][ib],
                    (b['v'][ib] - a['v'][ia]) * 100,
                    (b['B'] - a['B']) * 100, (b['N'] - a['N']) * 100,
                    (b['HM'] - a['HM']) * 100))
    L.append('')
    L.append('### gated_sym_xl vs sym_xl（细边界类止跌，负值表示 gated 更差）')
    L.append('')
    for s in (42, 7):
        a, b = g('sym_xl', s), g('gated', s)
        if not a or not b:
            continue
        cells = []
        for w in ('doorframe', 'tv', 'bathtub', 'bookshelf'):
            ia, ib = idx_of(a, w), idx_of(b, w)
            cells.append('%s %.3f->%.3f (%+.2f)'
                         % (w, a['v'][ia], b['v'][ib],
                            (b['v'][ib] - a['v'][ia]) * 100))
        L.append('- seed %d: %s | Base %+.2f | Novel %+.2f | HM %+.2f'
                 % (s, ' ; '.join(cells), (b['B'] - a['B']) * 100,
                    (b['N'] - a['N']) * 100, (b['HM'] - a['HM']) * 100))
    L.append('')
    L.append('## 7. 参数量对照（新增模块）')
    L.append('')
    L.append('| 分支 | 新增参数 | 构成 |')
    L.append('|---|---|---|')
    L.append('| base | 2,880 | novel_seg_head 64x45 |')
    L.append('| xl | 21,824 | dec0_norm 128 + cross_norms 384 + cross_projs 12,416 + cross_outs 256 + novel_head 8,640 |')
    L.append('| mlp_dec0 | 21,632 | novel_mlp 全量（比 xl 少 192，-0.9%） |')
    L.append('| sym_xl | 23,360 | xl + base_cross_head 1,536 |')
    L.append('| gated | 35,842 | sym + depth_gate 12,482 |')
    L.append('')
    L.append('## 8. eRank / 权重占比（若新高，夜间末补跑）')
    L.append('')
    L.append('见 xl_s42：dec0 10.48 -> cat192 35.25（3.36x），三切片 32.34/30.45/37.21。')
    L.append('')
    open(OUT, 'w').write('\n'.join(L))
    print('report ->', OUT, 'runs:', len(res))


if __name__ == '__main__':
    main()
