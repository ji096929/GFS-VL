"""补测：log-prior 校准对 multi / dec012 / dec01 是否同样有效。

只跑 shard0（104 场景），与 cut2_variants.py 的 dec0_u/dec0_lp 同口径对比。
输出 /data01/jh/data/cut2_feat/{tag}/var_lp_cm_s{i}.npz
"""
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from cut2_common import DIM, LAYERS, NC, NO, Probe, build_model_and_cfg, make_ds
from pointcept.datasets import collate_fn  # noqa: E402

FEAT = '/data01/jh/data/cut2_feat'
CAP = 2000
MULTI = ['dec0', 'dec1', 'dec2', 'dec3']
BS = 4096
NAMES = ['dec0_lp', 'dec01_lp', 'dec012_lp', 'multi_lp', 'multi_dim_lp']


def load_uniform(tag):
    files = sorted(glob.glob(os.path.join(FEAT, tag, 'train_s*.npz')))
    cnt = np.zeros(NC, np.int64)
    acc = {L: [] for L in LAYERS}
    gts = []
    rng = np.random.RandomState(0)
    for f in files:
        z = np.load(f)
        gt = z['gt'].astype(np.int64)
        idx = []
        for c in range(NC):
            w = np.where(gt == c)[0]
            if len(w) == 0:
                continue
            rng.shuffle(w)
            need = CAP - cnt[c]
            if need > 0:
                a = w[:need]
                idx.append(a)
                cnt[c] += len(a)
        if len(idx) == 0:
            # 各类都已采满 CAP，后续文件无需再贡献
            continue
        idx = np.concatenate(idx)
        for L in LAYERS:
            acc[L].append(z['f_' + L][idx].astype(np.float32))
        gts.append(gt[idx])
    return {L: np.concatenate(acc[L], 0) for L in LAYERS}, \
        np.concatenate(gts, 0)


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
    out = os.path.join(FEAT, tag, 'var_lp_cm_s%d.npz' % shard)
    if os.path.exists(out):
        print('已存在', out, flush=True)
        return

    feats, gt = load_uniform(tag)
    files = sorted(glob.glob(os.path.join(FEAT, tag, 'train_s*.npz')))
    prior = np.zeros(NC, np.float64)
    for f in files:
        prior += np.bincount(np.load(f)['gt'].astype(np.int64), minlength=NC)
    prior /= prior.sum()
    print('%s 均匀 %d 点 | 先验前6 %s'
          % (tag, len(gt), np.round(prior[:6], 3).tolist()), flush=True)

    probes = {}
    for nm, layers, mode in [('dec0_lp', ['dec0'], 'z'),
                             ('dec01_lp', ['dec0', 'dec1'], 'z'),
                             ('dec012_lp', ['dec0', 'dec1', 'dec2'], 'z'),
                             ('multi_lp', MULTI, 'z'),
                             ('multi_dim_lp', MULTI, 'dim')]:
        X, m_, s_ = prep(feats, layers)
        if mode == 'dim':
            off = 0
            for L in layers:
                d = DIM[L]
                X[:, off:off + d] /= np.sqrt(d)
                off += d
        W, b = train_probe(X, gt)
        probes[nm] = (layers, m_, s_, W, b)
        print('  训练 %s 完成 (%d 维)' % (nm, X.shape[1]), flush=True)

    lprior = torch.tensor(np.log(prior + 1e-12), device='cuda',
                          dtype=torch.float32)

    model, cfg = build_model_and_cfg(tag)
    pb = Probe()
    pb.attach(model.backbone)
    ds = make_ds(cfg, 'val')
    ds.data_list = ds.data_list[shard::shards]
    print('%s val 本片 %d 场景' % (tag, len(ds.data_list)), flush=True)
    PAR = {}
    for nm in NAMES:
        layers, m_, s_, W, b = probes[nm]
        PAR[nm] = (layers, torch.tensor(m_, device='cuda'),
                   torch.tensor(s_, device='cuda'),
                   torch.tensor(W, device='cuda'),
                   torch.tensor(b, device='cuda'))

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
                    layers, mu_, sd_, W_, b_ = PAR[nm]
                    X = torch.cat([fs[L] for L in layers], 1) \
                        if len(layers) > 1 else fs[layers[0]]
                    X = (X - mu_) / sd_
                    if nm == 'multi_dim_lp':
                        off = 0
                        for L in layers:
                            dd = DIM[L]
                            X[:, off:off + dd] /= float(np.sqrt(dd))
                            off += dd
                    lg = X @ W_ + b_ + lprior
                    probs[nm][ip] += torch.softmax(lg.float(), -1)
            cover[ip] += 1
        ok = torch.tensor(gt >= 0, device='cuda')
        gt_t = torch.tensor(np.clip(gt, 0, NC - 1), device='cuda')
        for nm in NAMES:
            pred = (probs[nm] / cover.clamp(min=1)[:, None]).argmax(1)
            cms[nm] += torch.bincount(gt_t[ok] * NC + pred[ok],
                                      minlength=NC * NC).cpu().numpy() \
                .reshape(NC, NC)
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
        print('%-12s Base %.4f Novel %.4f HM %.4f'
              % (nm, iou[:NO].mean(), iou[NO:].mean(),
                 2 * iou[:NO].mean() * iou[NO:].mean() /
                 max(iou[:NO].mean() + iou[NO:].mean(), 1e-9)), flush=True)


if __name__ == '__main__':
    main()
