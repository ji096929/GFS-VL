"""检验B(GPU)：原型采样空间与子原型数量 k 的影响。

对 357 任务，用 Sonata cat(1232 维) 特征比较：
  空间 feat  : 原始特征空间 FPS（与 p2 的 cat_multi/cat_top3 一致）
       nfeat : 先 L2 归一化特征，再 FPS（余弦空间）
       coord : 在 support 的三维坐标上 FPS（空间局部原型）
  k     : 3 / 10 / 30 / 100
  归约  : multi(max 匹配) / top3(Top-3 余弦均值)
另加 mean 均值原型对照。

输出全 query 点分数到 /data01/jh/data/sonata_dump/p3_s*.pt。
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, '/data01/jh/code/sonata')
sys.path.insert(0, '/data01/jh/code')

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import sonata  # noqa: E402
from sonata_probe2 import feat_both, fps_idx  # noqa: E402

CKPT = '/data01/jh/code/sonata/ckpt/sonata.pth'
DUMP = '/data01/jh/data/sonata_dump'
MODES = ('feat', 'nfeat', 'coord')
KS = (3, 10, 30, 100)


def protos_space(sf, sc, mask, mode, k):
    """按指定空间做 FPS + 特征均值池化聚类。点数不足全留，不补零。"""
    f = sf[mask]
    n = f.shape[0]
    if n == 0:
        return f.new_zeros(0, f.shape[1])
    k_eff = min(k, n)
    if mode == 'coord':
        idx = fps_idx(sc[mask], k_eff)
    elif mode == 'nfeat':
        idx = fps_idx(F.normalize(f, dim=1), k_eff)
    else:
        idx = fps_idx(f, k_eff)
    seeds = f[idx]
    assign = torch.cdist(f, seeds).argmin(1)
    out = torch.zeros(k_eff, f.shape[1], device=f.device, dtype=f.dtype)
    for i in range(k_eff):
        m = assign == i
        out[i] = f[m].mean(0) if bool(m.any()) else f[idx[i]]
    return F.normalize(out, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--out', default='/data01/jh/data/sonata_dump/p3_s%d.pt')
    args = ap.parse_args()
    out = args.out % args.shard if '%d' in args.out else args.out

    files = sorted(glob.glob(os.path.join(DUMP, 'ep*.npz')))
    files = [f for f in files
             if int(os.path.basename(f)[2:5]) % args.shards == args.shard]
    print('tasks:', len(files), flush=True)
    if not files:
        return

    cfg = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    model = sonata.load(args.ckpt, custom_config=cfg).cuda().eval()
    print('[load] params %.2fM'
          % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)

    scores, p0s, labs, clss, gids = {}, {}, {}, {}, []
    for i, f in enumerate(files):
        gid = int(os.path.basename(f)[2:5])
        d = np.load(f)
        s_cat, _ = feat_both(model, d['s_coord'], d['s_rgb'])
        q_cat, _ = feat_both(model, d['q_coord'], d['q_rgb'])
        s_lab = torch.from_numpy(d['s_lab'].astype(np.int64)).cuda()
        s_coord = torch.from_numpy(d['s_coord'].astype(np.float32)).cuda()
        fg = s_lab.bool()
        bg = ~fg
        qn = F.normalize(q_cat, dim=1)

        def put(name, cf, cb):
            scores.setdefault(name, {})[gid] = \
                (cf - cb).half().cpu()

        # 均值原型（不依赖 FPS 空间）
        pf = F.normalize(s_cat[fg].mean(0), dim=0)
        pb = F.normalize(s_cat[bg].mean(0), dim=0)
        put('mean', qn @ pf, qn @ pb)

        for mode in MODES:
            for k in KS:
                pf = protos_space(s_cat, s_coord, fg, mode, k)
                pb = protos_space(s_cat, s_coord, bg, mode, k)
                if pf.shape[0] == 0 or pb.shape[0] == 0:
                    continue
                cf = qn @ pf.t()
                cb = qn @ pb.t()
                put('%s_k%d_multi' % (mode, k),
                    cf.max(1).values, cb.max(1).values)
                t3 = min(3, cf.shape[1], cb.shape[1])
                put('%s_k%d_top3' % (mode, k),
                    cf.topk(t3, dim=1).values.mean(1),
                    cb.topk(t3, dim=1).values.mean(1))

        p0s[gid] = d['q_p0'].astype(np.float32)
        labs[gid] = d['q_lab'].astype(np.uint8)
        clss[gid] = int(d['cls'][0])
        gids.append(gid)
        if (i + 1) % 10 == 0:
            print('  %d / %d' % (i + 1, len(files)), flush=True)

    torch.save(dict(gid=gids, scores=scores, p0=p0s, lab=labs, cls=clss), out)
    print('->', out)


if __name__ == '__main__':
    main()
