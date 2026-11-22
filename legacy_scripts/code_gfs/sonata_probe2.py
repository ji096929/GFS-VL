'''Sonata base: 检查1(提纯) + 检查2(零训练分割水位)。

修正的核心 bug
--------------
旧版收集解码链时写成
    while 'pooling_parent' in cur:
        chain.append((cur.pop('pooling_parent'), cur.pop('pooling_inverse')))
循环体里没有 cur = par，所以 chain 永远只有 1 层：特征停在 896 维的
中间分辨率，再拿错层的 'inverse' 去索引，点数就从 2044 缩到 72。

正确的链语义（PTv3 enc，Sonata 只有 enc 没有 dec）：
  base = model(pt) 是最深层（点最少、512 维），它的 pooling_parent 逐级
  指回更浅的层；pooling_inverse 是 cluster（长度=父层点数，值域=子层索引）。
  逐层 cat([par.feat, feat[iv]]) 得到 512->896->1088->1184->1232。
  最后一步才用最顶层 GridSample 的 point['inverse'] 映射回原始输入点数。
'''
import argparse
import glob
import os
import sys

sys.path.insert(0, '/data01/jh/code/sonata')
sys.path.insert(0, '/data01/jh/code')

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import sonata  # noqa: E402

CKPT = '/data01/jh/code/sonata/ckpt/sonata.pth'
DUMP = '/data01/jh/data/sonata_dump'


def estimate_normals(coord, k=30):
    n = len(coord)
    k = min(k, n)
    _, idx = cKDTree(coord).query(coord, k=k)
    nb = coord[idx]
    nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', nb, nb)
    w, v = np.linalg.eigh(cov)
    nrm = v[:, :, 0]
    nrm[nrm[:, 2] < 0] *= -1.0
    return nrm.astype(np.float32)


def feat_both(model, coord, rgb, verbose=False):
    '''一次前向返回 (直接拼接 1232 维, 分尺度 L2 后拼接 1232 维)。

    两者都是原始输入分辨率（行数 == len(coord)）。
    '''
    coord = np.asarray(coord, np.float32)
    rgb = np.asarray(rgb, np.float32)
    pt = dict(coord=coord, color=rgb, normal=estimate_normals(coord))
    pt = sonata.transform.default()(pt)
    with torch.inference_mode():
        for k in list(pt.keys()):
            if isinstance(pt[k], torch.Tensor):
                pt[k] = pt[k].cuda(non_blocking=True)
        base = model(pt)

        # 只读遍历 pooling 链：chain[0] 紧贴最深层，chain[-1] 最顶层
        chain, cur = [], base
        while 'pooling_parent' in cur.keys():
            chain.append((cur['pooling_parent'], cur['pooling_inverse']))
            cur = cur['pooling_parent']
        top = cur
        base_inv = top['inverse']            # GridSample inverse -> 原始 N
        base_feat = base.feat                # (n_deepest, 512)

        def replay(mode):
            feat = base_feat
            for j, (par, iv) in enumerate(chain):
                pf = par.feat
                child = feat[iv]
                if verbose:
                    print('    L%d par %s child %s -> %d'
                          % (j, tuple(pf.shape), tuple(child.shape),
                             pf.shape[1] + child.shape[1]))
                if mode == 'lev':
                    nf = torch.cat([F.normalize(pf, dim=-1),
                                    F.normalize(child, dim=-1)], dim=-1)
                    nf = F.normalize(nf, dim=-1)
                else:
                    nf = torch.cat([pf, child], dim=-1)
                feat = nf
            return feat[base_inv].float()

        f_cat = replay('cat')
        f_lev = replay('lev')
    assert f_cat.shape[0] == len(coord), (f_cat.shape, len(coord))
    assert f_lev.shape[0] == len(coord)
    return f_cat, f_lev


def fps_idx(feat, k, seed=0):
    '''特征空间贪心 FPS。'''
    n = feat.shape[0]
    if n <= k:
        return torch.arange(n, device=feat.device)
    g = torch.Generator(device=feat.device).manual_seed(seed)
    sel = torch.zeros(k, dtype=torch.long, device=feat.device)
    dist = torch.full((n,), 1e10, device=feat.device)
    cur = int(torch.randint(n, (1,), device=feat.device, generator=g).item())
    for i in range(k):
        sel[i] = cur
        d = (feat - feat[cur]).pow(2).sum(1)
        dist = torch.minimum(dist, d)
        cur = int(dist.argmax().item())
    return sel


def protos_multi(feat, k=100):
    '''FPS 种子 + 特征均值池化聚类；点数不足 k 全留，绝不补零。'''
    n = feat.shape[0]
    if n == 0:
        return feat.new_zeros(0, feat.shape[1])
    k_eff = min(k, n)
    idx = fps_idx(feat, k_eff)
    seeds = feat[idx]
    assign = torch.cdist(feat, seeds).argmin(1)
    out = torch.zeros(k_eff, feat.shape[1], device=feat.device,
                      dtype=feat.dtype)
    for i in range(k_eff):
        m = assign == i
        out[i] = feat[m].mean(0) if bool(m.any()) else feat[idx[i]]
    return F.normalize(out, dim=1)


def score_one(qf, sf, s_lab, proto, k=100):
    qn = F.normalize(qf, dim=1)
    fg = s_lab.bool()
    bg = ~fg
    if proto == 'mean':
        if not bool(fg.any()) or not bool(bg.any()):
            return torch.zeros(qf.shape[0], device=qf.device)
        p_fg = F.normalize(sf[fg].mean(0), dim=0)
        p_bg = F.normalize(sf[bg].mean(0), dim=0)
        return qn @ p_fg - qn @ p_bg
    pf = protos_multi(sf[fg], k)
    pb = protos_multi(sf[bg], k)
    if pf.shape[0] == 0 or pb.shape[0] == 0:
        return torch.zeros(qf.shape[0], device=qf.device)
    cf = qn @ pf.t()
    cb = qn @ pb.t()
    if proto == 'multi':
        return cf.max(1).values - cb.max(1).values
    t3 = min(3, cf.shape[1], cb.shape[1])
    return cf.topk(t3, dim=1).values.mean(1) - cb.topk(t3, dim=1).values.mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--k', type=int, default=100)
    ap.add_argument('--only-gid', type=int, default=0)
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--out', default='/data01/jh/data/sonata_dump/p2_s%d.pt')
    args = ap.parse_args()
    out = args.out % args.shard if '%d' in args.out else args.out

    files = sorted(glob.glob(os.path.join(DUMP, 'ep*.npz')))
    files = [f for f in files
             if int(os.path.basename(f)[2:5]) % args.shards == args.shard]
    if args.only_gid:
        files = [f for f in files
                 if int(os.path.basename(f)[2:5]) == args.only_gid]
    print('tasks:', len(files), flush=True)
    if not files:
        return

    cfg = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    model = sonata.load(args.ckpt, custom_config=cfg).cuda().eval()
    print('[load] params %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e6),
          flush=True)

    scores, p0s, labs, clss, gids = {}, {}, {}, {}, []
    for i, f in enumerate(files):
        gid = int(os.path.basename(f)[2:5])
        d = np.load(f)
        s_cat, s_lev = feat_both(model, d['s_coord'], d['s_rgb'],
                                 verbose=(i == 0))
        q_cat, q_lev = feat_both(model, d['q_coord'], d['q_rgb'])
        if i == 0:
            print('  feat: cat %s lev %s  s_lab %d q_lab %d'
                  % (tuple(s_cat.shape), tuple(s_lev.shape),
                     len(d['s_lab']), len(d['q_lab'])), flush=True)
        s_lab = torch.from_numpy(d['s_lab'].astype(np.int64)).cuda()
        for tag, sf, qf in (('cat', s_cat, q_cat), ('lev', s_lev, q_lev)):
            for proto in ('mean', 'multi', 'top3'):
                scores.setdefault('%s_%s' % (tag, proto), {})[gid] =                     score_one(qf, sf, s_lab, proto, args.k).half().cpu()
        p0s[gid] = d['q_p0'].astype(np.float32)
        labs[gid] = d['q_lab'].astype(np.uint8)
        clss[gid] = int(d['cls'][0])
        gids.append(gid)
        if (i + 1) % 20 == 0:
            print('  %d / %d' % (i + 1, len(files)), flush=True)

    torch.save(dict(gid=gids, scores=scores, p0=p0s, lab=labs, cls=clss), out)
    print('->', out)


if __name__ == '__main__':
    main()
