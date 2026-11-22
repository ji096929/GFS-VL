"""疑点 1b/1c：多读出变体 × 采样方式，312 val 一次评估。

变体：
  dec0_u       dec0 + 均匀采样（每类 2000）
  dec0_n       dec0 + 自然分布采样
  dec0_lp      dec0_u 权重 + 评估时加 log 自然先验
  dec01        dec0+dec1（128d，逐层 z-score）
  dec012       dec0+dec1+dec2（256d，去掉粗 dec3）
  multi        dec0..dec3（512d，逐层 z-score）
  multi_dim    同上 + 每层除以 sqrt(D_L)（等贡献）
  multi_l2     同上 + 每层 per-sample L2 归一化

用法: python cut2_variants.py <S1|S2> [shard] [shards]
输出 /data01/jh/data/cut2_feat/{tag}/var_cm_s{i}.npz
"""
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from cut2_common import (DIM, LAYERS, NC, NO, Probe, build_model_and_cfg,
                         make_ds)
from pointcept.datasets import collate_fn  # noqa: E402

FEAT = '/data01/jh/data/cut2_feat'
CAP = 2000
N_NAT = 113508
MULTI = ['dec0', 'dec1', 'dec2', 'dec3']
BS = 4096
NAMES = ['dec0_u', 'dec0_n', 'dec0_lp', 'dec01', 'dec012',
         'multi', 'multi_dim', 'multi_l2']


def load_two(tag):
    files = sorted(glob.glob(os.path.join(FEAT, tag, 'train_s*.npz')))
    assert files
    prior = np.zeros(NC, np.float64)
    for f in files:
        prior += np.bincount(np.load(f)['gt'].astype(np.int64), minlength=NC)
    prior /= prior.sum()
    cu = np.zeros(NC, np.int64)
    cn = np.zeros(NC, np.int64)
    qu = {L: [] for L in LAYERS}
    qn = {L: [] for L in LAYERS}
    gu, gn = [], []
    rng = np.random.RandomState(0)
    for f in files:
        z = np.load(f)
        gt = z['gt'].astype(np.int64)
        iu, inn = [], []
        for c in range(NC):
            w = np.where(gt == c)[0]
            if len(w) == 0:
                continue
            rng.shuffle(w)
            nu = CAP - cu[c]
            if nu > 0:
                a = w[:nu]
                iu.append(a)
                cu[c] += len(a)
            nn = int(N_NAT * prior[c]) - cn[c]
            if nn > 0:
                a = w[:nn]
                inn.append(a)
                cn[c] += len(a)
        iu = np.concatenate(iu) if iu else np.zeros(0, np.int64)
        inn = np.concatenate(inn) if inn else np.zeros(0, np.int64)
        for L in LAYERS:
            fl = z['f_' + L]
            qu[L].append(fl[iu].astype(np.float32))
            qn[L].append(fl[inn].astype(np.float32))
        gu.append(gt[iu])
        gn.append(gt[inn])
        print('  %s  uniform %d / natural %d'
              % (os.path.basename(f), len(iu), len(inn)), flush=True)
    U = {L: np.concatenate(qu[L], 0) for L in LAYERS}
    N = {L: np.concatenate(qn[L], 0) for L in LAYERS}
    return U, np.concatenate(gu), N, np.concatenate(gn), prior


def train_probe(X, y, epochs=40, lr=1e-3, wd=5e-4):
    dev = 'cuda'
    torch.manual_seed(0)
    Xt = torch.tensor(X, device=dev)
    yt = torch.tensor(y, device=dev)
    W = torch.zeros(X.shape[1], NC, requires_grad=True, device=dev)
    b = torch.zeros(NC, requires_grad=True, device=dev)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    cnt = np.bincount(y, minlength=NC).astype(np.float32)
    cw = torch.tensor(np.where(cnt > 0,
                               cnt.sum() / (NC * np.maximum(cnt, 1)),
                               0.0), device=dev)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, BS):
            idx = perm[i:i + BS]
            loss = F.cross_entropy(Xt[idx] @ W + b, yt[idx], weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
    return W.detach().cpu().numpy(), b.detach().cpu().numpy()


def prep(feats, layers):
    """逐层 per-dim z-score，返回 (concat, mus, sds)。"""
    cols, mus, sds = [], [], []
    for L in layers:
        X = feats[L]
        mu = X.mean(0)
        sd = X.std(0) + 1e-6
        cols.append((X - mu) / sd)
        mus.append(mu)
        sds.append(sd)
    return np.concatenate(cols, 1), np.concatenate(mus), np.concatenate(sds)


def main():
    tag = sys.argv[1]
    shard = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    shards = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    out = os.path.join(FEAT, tag, 'var_cm_s%d.npz' % shard)
    if os.path.exists(out):
        print('已存在', out, flush=True)
        return

    U, gu, N, gn, prior = load_two(tag)
    print('uniform %d 点 | natural %d 点 | 先验前6 %s'
          % (len(gu), len(gn), np.round(prior[:6], 3).tolist()), flush=True)

    # 训练 8 个变体
    probes = {}
    # dec0_u
    X, mu, sd = prep(U, ['dec0'])
    W, b = train_probe(X, gu)
    probes['dec0_u'] = (['dec0'], mu, sd, W, b, 0.0)
    # dec0_n
    Xn, mun, sdn = prep(N, ['dec0'])
    Wn, bn = train_probe(Xn, gn)
    probes['dec0_n'] = (['dec0'], mun, sdn, Wn, bn, 0.0)
    # dec0_lp：复用 dec0_u 权重，评估加 log prior
    probes['dec0_lp'] = (['dec0'], mu, sd, W, b, 1.0)
    # 其余读出
    for nm, layers, mode in [('dec01', ['dec0', 'dec1'], 'z'),
                             ('dec012', ['dec0', 'dec1', 'dec2'], 'z'),
                             ('multi', MULTI, 'z'),
                             ('multi_dim', MULTI, 'dim'),
                             ('multi_l2', MULTI, 'l2')]:
        X, m_, s_ = prep(U, layers)
        if mode == 'dim':
            off = 0
            for L in layers:
                d = DIM[L]
                X[:, off:off + d] /= np.sqrt(d)
                off += d
        W, b = train_probe(X, gu)
        probes[nm] = (layers, m_, s_, W, b, 0.0)
        print('  训练 %s 完成 (%d 维)' % (nm, X.shape[1]), flush=True)

    lprior = torch.tensor(np.log(prior + 1e-12), device='cuda',
                          dtype=torch.float32)

    # 评估
    model, cfg = build_model_and_cfg(tag)
    pb = Probe()
    pb.attach(model.backbone)
    ds = make_ds(cfg, 'val')
    ds.data_list = ds.data_list[shard::shards]
    print('%s val 本片 %d 场景' % (tag, len(ds.data_list)), flush=True)
    PAR = {}
    for nm in NAMES:
        layers, m_, s_, W, b, lp = probes[nm]
        PAR[nm] = (layers, torch.tensor(m_, device='cuda'),
                   torch.tensor(s_, device='cuda'),
                   torch.tensor(W, device='cuda'),
                   torch.tensor(b, device='cuda'), lp)

    cms = {nm: np.zeros((NC, NC), np.int64) for nm in NAMES}
    t0 = time.time()
    for i in range(len(ds.data_list)):
        d = ds[i]
        frags = d.pop('fragment_list')
        seg = d.pop('segment')
        gt = np.asarray(seg).astype(np.int64)
        n = len(gt)
        probs = {nm: torch.zeros((n, NC), device='cuda') for nm in NAMES}
        cover = torch.zeros(n, device='cuda')
        for frag in frags:
            inp = collate_fn([frag])
            for k in inp:
                if isinstance(inp[k], torch.Tensor):
                    inp[k] = inp[k].cuda(non_blocking=True)
            ip = inp['index']
            with torch.no_grad():
                model(inp)
                fs = pb.layers()
                for nm in NAMES:
                    layers, mu_, sd_, W_, b_, lp = PAR[nm]
                    if len(layers) == 1:
                        X = fs[layers[0]]
                    else:
                        X = torch.cat([fs[L] for L in layers], 1)
                    X = (X - mu_) / sd_
                    if nm == 'multi_dim':
                        off = 0
                        for L in layers:
                            dd = DIM[L]
                            X[:, off:off + dd] /= float(np.sqrt(dd))
                            off += dd
                    elif nm == 'multi_l2':
                        X = X / (X.norm(dim=1, keepdim=True) + 1e-9)
                    lg = X @ W_ + b_
                    if lp > 0:
                        lg = lg + lprior
                    probs[nm][ip] += torch.softmax(lg.float(), -1)
            cover[ip] += 1
        pred_base = {}
        for nm in NAMES:
            pred_base[nm] = (probs[nm] / cover.clamp(min=1)[:, None]).argmax(1)
        ok = torch.tensor(gt >= 0, device='cuda')
        gt_t = torch.tensor(np.clip(gt, 0, NC - 1), device='cuda')
        for nm in NAMES:
            cms[nm] += torch.bincount(
                gt_t[ok] * NC + pred_base[nm][ok],
                minlength=NC * NC).cpu().numpy().reshape(NC, NC)
        if (i + 1) % 20 == 0:
            print('  %d/%d %.0fs' % (i + 1, len(ds.data_list),
                                     time.time() - t0), flush=True)

    np.savez(out, **cms)
    for nm in NAMES:
        cm = cms[nm]
        d = np.diag(cm).astype(float)
        r = cm.sum(1).astype(float)
        c = cm.sum(0).astype(float)
        iou = d / np.maximum(r + c - d, 1)
        print('%-10s Base %.4f Novel %.4f HM %.4f'
              % (nm, iou[:NO].mean(), iou[NO:].mean(),
                 2 * iou[:NO].mean() * iou[NO:].mean() /
                 max(iou[:NO].mean() + iou[NO:].mean(), 1e-9)), flush=True)


if __name__ == '__main__':
    main()
