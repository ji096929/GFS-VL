"""整场景评估：按 COSeg 官方协议（scenes/data + data_prepare_v101）测 baseline 与 U 的 Oracle 上限。

与 block 口径的区别
------------------
- 数据：整 scene（20 万点）→ voxel 0.02 → 超 20480 随机采样
- episode：官方 get_test_episode 的做法，从该类候选 scene 里随机抽 2 个
- 因此绝对数字可以与论文的 41.78 直接对比

候选池有两种，必须区分
--------------------
official  官方 get_class2scans 的池：目标类点数 > max(scene 总点数 5%, 100)
          只有这个池与论文口径可比。它把 counter 47→15、desk 84→31、door 219→95，
          正好是我们原先低 IoU 的那几个类。
any       "只要出现该类就算"，会混入目标类只占 0.01% 的 scene，模型找不到目标
          就大面积误报，baseline 会被压到 0.21 左右。仅作对照。

四行
----
Baseline     仅 S 原型
Naive-B      U 中 p0 > 0.7 的点
Oracle-B     p0 > 0.7 且真值为前景
Oracle-B+C   U 中全部真前景

同时统计：盲采的 U 场景里有多少真的包含目标类（协议漏洞的量化）。
"""
import argparse
import json
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as Fs

from ocsb_core import (CACHE, NEW_IDS, build_proto, feats_one, iou_counts,
                       load_model, predict_agg, seed_everything)
from util.data_util import data_prepare_v101

SC = '/data01/jh/data/ScanNet/scenes/data'
SCENE_CLS = os.path.join(CACHE, 'scene_classes.json')
C2S_PATH = os.path.join(os.path.dirname(SC), 'class2scans.pkl')


def prepare_scene(name, cfg, cls):
    """官方 data_prepare_v101：voxel 0.02 + 超 20480 随机采样。"""
    r = np.load(os.path.join(SC, name + '.npy'))
    coord = r[:, 0:3].astype(np.float32)
    feat = r[:, 3:6].astype(np.float32)
    label = r[:, 6].astype(np.int64)
    return data_prepare_v101(coord, feat, label, 'val',
                             cfg.voxel_size, cfg.voxel_max, None, False, cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=5)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--pool', default='official', choices=['official', 'any'],
                    help='候选池：official=官方 5%% 阈值池；any=出现即计入')
    ap.add_argument('--u-k', type=int, default=3)
    ap.add_argument('--fg-hi', type=float, default=0.7)
    ap.add_argument('--bg-lo', type=float, default=0.2)
    ap.add_argument('--lams', default='0.3,0.5')
    ap.add_argument('--out', default=os.path.join(CACHE, 'scene_eval_%s.json'))
    args = ap.parse_args()
    args.out = args.out % args.pool if '%s' in args.out else args.out

    scene_cls = json.load(open(SCENE_CLS))
    all_scenes = sorted(x[:-4] for x in os.listdir(SC) if x.endswith('.npy'))
    if args.pool == 'official':
        c2s = pickle.load(open(C2S_PATH, 'rb'))
        cls2 = {c: sorted(c2s[c]) for c in NEW_IDS}
        hit_pool = {c: set(c2s[c]) for c in NEW_IDS}
    else:
        cls2 = {c: [s for s in sorted(scene_cls) if c in scene_cls[s]]
                for c in NEW_IDS}
        hit_pool = {c: {s for s in scene_cls if c in scene_cls[s]}
                    for c in NEW_IDS}
    print('候选池 %s: %s' % (
        args.pool, {c: len(cls2[c]) for c in NEW_IDS}))
    model, cfg = load_model()
    dev = torch.device('cuda:0')
    print('cfg voxel_size=%s voxel_max=%s' % (cfg.voxel_size, cfg.voxel_max))

    rng = random.Random(0)
    eps = []
    for c in NEW_IDS:
        pool = cls2[c]
        for _ in range(args.per_class):
            if len(pool) < 2:
                break
            s, q = rng.sample(pool, 2)
            eps.append((c, s, q))
    if args.shards > 1:
        eps = eps[args.shard::args.shards]
    print('episodes %d' % len(eps), flush=True)

    lams = [float(x) for x in args.lams.split(',')]
    cases = ['Baseline', 'Naive-B', 'Oracle-B', 'Oracle-B+C']
    rows = []
    for ei, (c, sn, qn) in enumerate(eps):
        g_ep = args.shard + args.shards * ei
        seed_everything(10000 + g_ep)
        sc, sf_, sl = prepare_scene(sn, cfg, c)
        qc, qf_, ql = prepare_scene(qn, cfg, c)
        s_x = torch.cat([sc, sf_], 1)
        q_x = torch.cat([qc, qf_], 1)
        s_bin = (sl == c).long()
        q_map = (ql == c).long()
        sf = feats_one(model, s_x, s_bin, dev)
        qf = feats_one(model, q_x, q_map, dev)
        fg_s = sf['yl'].bool()
        sparse_s = torch.cat([build_proto(model, sf, ~fg_s),
                              build_proto(model, sf, fg_s)], 0)

        u_names = rng.sample([s for s in all_scenes if s not in (sn, qn)], args.u_k)
        n_hit = sum(1 for s in u_names if s in hit_pool[c])
        u_feats = []
        for un in u_names:
            uc, uf_, ul = prepare_scene(un, cfg, c)
            u_x = torch.cat([uc, uf_], 1)
            u_bin = (ul == c).long()
            uf = feats_one(model, u_x, u_bin, dev)
            _, out0 = predict_agg(model, uf, [sparse_s], 0.0, dev)
            uf['p0'] = Fs.softmax(out0, dim=1)[:, 1]
            u_feats.append(uf)

        def make_sparse(mode):
            fgs, bgs = [], []
            for uf in u_feats:
                lab = uf['yl']
                if mode == 'naive':
                    m_fg = uf['p0'] > args.fg_hi
                    m_bg = uf['p0'] < args.bg_lo
                elif mode == 'ob':
                    m_fg = (uf['p0'] > args.fg_hi) & (lab > 0.5)
                    m_bg = lab <= 0.5
                else:
                    m_fg = lab > 0.5
                    m_bg = lab <= 0.5
                fgs.append(build_proto(model, uf, m_fg))
                bgs.append(build_proto(model, uf, m_bg))
            return torch.cat([torch.stack(bgs).mean(0),
                              torch.stack(fgs).mean(0)], 0)

        def run(case, sparse_list, lam):
            pred, _ = predict_agg(model, qf, sparse_list, lam, dev)
            tp, fp, fn, iou = iou_counts(pred, q_map)
            rows.append(dict(ep=ei, case=case, lam=lam, cls=c, iou=iou,
                             tp=tp, fp=fp, fn=fn, n_u_hit=n_hit))

        run('Baseline', [sparse_s], 0.0)
        for mode, case in (('naive', 'Naive-B'), ('ob', 'Oracle-B'),
                           ('obc', 'Oracle-B+C')):
            sp = make_sparse(mode)
            for lam in lams:
                run('%s_l%.1f' % (case, lam), [sparse_s, sp], lam)

        print('  %d / %d  cls %2d  npts s=%d q=%d  U命中 %d/%d' % (
            ei + 1, len(eps), c, len(s_x), len(q_x), n_hit, len(u_names)),
            flush=True)

    out = {}
    print('\n%-18s %9s %9s %9s' % ('case', 'Cumul', 'Arith', 'dCumul'))
    base_c = 0.0
    all_cases = []
    for r in rows:
        if r['case'] not in all_cases:
            all_cases.append(r['case'])
    for k in all_cases:
        tp = {c: 0 for c in NEW_IDS}
        fp = {c: 0 for c in NEW_IDS}
        fn = {c: 0 for c in NEW_IDS}
        ious = []
        for r in rows:
            if r['case'] != k:
                continue
            tp[r['cls']] += r['tp']
            fp[r['cls']] += r['fp']
            fn[r['cls']] += r['fn']
            ious.append(r['iou'])
        per = [tp[c] / max(tp[c] + fp[c] + fn[c], 1) for c in NEW_IDS]
        cu = float(np.mean(per))
        if k == 'Baseline':
            base_c = cu
        out[k] = dict(cumul=cu, arith=float(np.mean(ious)), per_class=per)
        print('%-18s %9.4f %9.4f %+9.4f' % (
            k, cu, float(np.mean(ious)), cu - base_c))

    hit = np.mean([r['n_u_hit'] for r in rows if r['case'] == 'Baseline'])
    print('\n盲采 %d 个 U 场景中平均命中目标类 %.2f 个（命中率 %.1f%%）' % (
        args.u_k, hit, 100 * hit / args.u_k))

    json.dump(dict(rows=rows, out=out), open(args.out, 'w'), indent=1)
    print('->', args.out)


if __name__ == '__main__':
    main()
