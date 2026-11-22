"""免训练对照：块级检索命中率 + 配对分割收益。

每个任务固定同一批 S/Q、同一 500 块候选池、同一权重，只允许送 3 块：
  random3    从池里随机抽 3 块
  retrieve3  用 S 的 fg/bg 子原型给 500 块打分，取 top-3
选完才读 U 真值统计命中率；分割用同一融合器（predict_agg，λ=0.3，naive 原型）。
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, '/data01/jh/code')
import fake_tp  # noqa: F401

sys.path.insert(0, '/data01/jh/code/COSeg')
sys.path.insert(0, '/data01/jh/code')

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ocsb_core import (build_proto, build_u_sparse, feats_one,  # noqa: E402
                       load_model, predict_agg)

CACHE = '/data01/jh/data/retrieval_cache'
DUMP = '/data01/jh/data/sonata_dump'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--out',
                    default='/data01/jh/afford_diag/stage2/retrieval_s%d.json')
    args = ap.parse_args()
    out = args.out % args.shard if '%d' in args.out else args.out
    dev = torch.device('cuda')

    cache = {}
    for f in sorted(glob.glob(os.path.join(CACHE, 'ucache_*.pt'))):
        d = torch.load(f, map_location='cpu')
        for n, v in d.items():
            cache[n] = {k: (t.cuda() if torch.is_tensor(t) else t)
                        for k, t in v.items()}
    names = sorted(cache)
    print('pool', len(names), flush=True)
    fd_all = F.normalize(torch.cat([cache[n]['fd'] for n in names]), dim=1)
    seg_start = [0]
    for n in names:
        seg_start.append(seg_start[-1] + len(cache[n]['fd']))

    model, cfg = load_model()
    files = sorted(glob.glob(os.path.join(DUMP, 'ep*.npz')))
    files = [f for f in files
             if int(os.path.basename(f)[2:5]) % args.shards == args.shard]
    print('tasks', len(files), flush=True)

    rows = {}
    for i, f in enumerate(files):
        d = np.load(f)
        gid = int(os.path.basename(f)[2:5])
        c = int(d['cls'][0])
        sx = torch.cat([
            torch.from_numpy(d['s_coord'].astype(np.float32)),
            torch.from_numpy(d['s_rgb'].astype(np.float32)) / 255.], 1).cuda()
        qx = torch.cat([
            torch.from_numpy(d['q_coord'].astype(np.float32)),
            torch.from_numpy(d['q_rgb'].astype(np.float32)) / 255.], 1).cuda()
        sy = torch.from_numpy(d['s_lab'].astype(np.int64)).cuda()
        qy = torch.from_numpy(d['q_lab'].astype(np.int64)).cuda()
        so = torch.tensor([len(sx)], dtype=torch.int32)
        qo = torch.tensor([len(qx)], dtype=torch.int32)
        gt = d['q_lab'].astype(bool)

        sf = feats_one(model, sx, sy, dev)
        qf = feats_one(model, qx, qy, dev)
        fg = sf['yl'].bool()
        fg_p = build_proto(model, sf, fg, k=100)
        bg_p = build_proto(model, sf, ~fg, k=100)
        sparse_s = torch.cat([bg_p, fg_p], 0)

        # ---- 检索打分：top-10 点的 (max_fg - max_bg) ----
        pf = F.normalize(fg_p, dim=1)
        pb = F.normalize(bg_p, dim=1)
        cf = fd_all @ pf.t()
        cb = fd_all @ pb.t()
        s = cf.max(1).values - cb.max(1).values
        scores = []
        for j in range(len(names)):
            seg = s[seg_start[j]:seg_start[j + 1]]
            scores.append(float(seg.topk(min(10, len(seg))).values.mean()))
        order = np.argsort(scores)[::-1]
        top3 = [int(x) for x in order[:3]]
        rnd3 = random.Random(gid).sample(range(len(names)), 3)

        def counts(p):
            p = p.cpu().numpy().astype(bool)
            return (int((p & gt).sum()), int((p & ~gt).sum()),
                    int((~p & gt).sum()))

        def run_group(idx3):
            uf_list = []
            for j in idx3:
                uf = dict(cache[names[j]])
                _, u_out = predict_agg(model, uf, [sparse_s], 0.0, dev)
                uf['prob'] = F.softmax(u_out, dim=1)[:, 1]
                uf_list.append(uf)
            su, _ = build_u_sparse(model, uf_list, 'naive', 0.7, 0.2)
            pred, _ = predict_agg(model, qf, [sparse_s, su], 0.3, dev)
            return counts(pred.argmax(0))

        def hit_stats(idx3):
            h = h5 = 0
            for j in idx3:
                lab = cache[names[j]]['ulab']
                m = (lab == c)
                h = max(h, int(m.sum() > 0))
                h5 = max(h5, int(m.mean() >= 0.05))
            return h, h5

        pred0, _ = predict_agg(model, qf, [sparse_s], 0.0, dev)
        b = counts(pred0.argmax(0))
        r = run_group(rnd3)
        t = run_group(top3)
        r_h, r_h5 = hit_stats(rnd3)
        t_h, t_h5 = hit_stats(top3)
        rows[gid] = dict(cls=c,
                         base=b, rand=r, retr=t,
                         rand_hit=r_h, rand_hit5=r_h5,
                         retr_hit=t_h, retr_hit5=t_h5,
                         rand_names=[names[j] for j in rnd3],
                         retr_names=[names[j] for j in top3])
        if (i + 1) % 20 == 0:
            print('  %d / %d' % (i + 1, len(files)), flush=True)
    json.dump(rows, open(out, 'w'))
    print('->', out)


if __name__ == '__main__':
    main()
