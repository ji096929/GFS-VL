"""Cut 2 阶段二：训练 6 组 Linear Probe，并计算 eRank / Base 子空间能量。

用法: python cut2_train.py <S1|S2> [epochs]

6 组 probe: enc4(512) / dec3(256) / dec2(128) / dec1(64) / dec0(64) /
            multi = concat(dec0,dec1,dec2,dec3) = 512
输出 /data01/jh/data/cut2_feat/{tag}/probes.npz
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from cut2_common import DIM, LAYERS, NC, NO

FEAT = '/data01/jh/data/cut2_feat'
CAP = 2000           # eRank / probe 每类最多用多少点
BS = 4096
MULTI = ['dec0', 'dec1', 'dec2', 'dec3']


def load_train(tag, cap):
    """流式读取所有分片，每类最多保留 cap 点（不把全量读进内存）。"""
    files = sorted(glob.glob(os.path.join(FEAT, tag, 'train_s*.npz')))
    assert files, '没有 train 特征，先跑 cut2_extract.py'
    rng = np.random.RandomState(0)
    cnt = np.zeros(NC, dtype=np.int64)
    acc = {L: [] for L in LAYERS}
    gt_acc = []
    for f in files:
        z = np.load(f)
        gt = z['gt'].astype(np.int64)
        idx = []
        for c in range(NC):
            need = cap - cnt[c]
            if need <= 0:
                continue
            w = np.where(gt == c)[0]
            if len(w) > need:
                w = rng.choice(w, need, replace=False)
            idx.append(w)
            cnt[c] += len(w)
        if not idx:
            continue
        idx = [x for x in idx if len(x)]
        if not idx:
            continue
        idx = np.concatenate(idx)
        for L in LAYERS:
            acc[L].append(z['f_' + L][idx].astype(np.float32))
        gt_acc.append(gt[idx])
        print('  %s 取 %d 点 | 累计 %s'
              % (os.path.basename(f), len(idx), cnt.tolist()[:6]), flush=True)
    return {L: np.concatenate(acc[L], 0) for L in LAYERS}, \
        np.concatenate(gt_acc, 0)


def subsample(feats, gt, cap):
    """每类最多 cap 点（随机），返回索引。"""
    rng = np.random.RandomState(0)
    idx = []
    for c in range(NC):
        w = np.where(gt == c)[0]
        if len(w) == 0:
            continue
        if len(w) > cap:
            w = rng.choice(w, cap, replace=False)
        idx.append(w)
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx


def erank(X):
    """有效秩 exp(entropy(eigenvalues))。X [n,D]，内部中心化。"""
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    w = np.linalg.eigvalsh(C.astype(np.float64))
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 0:
        return 0.0
    p = w / s
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def subspace_energy(feats, gt, seg_w, cls_mask):
    """Novel/Base 点在 base 头权重行空间 U_B(64x12) 上的能量占比。"""
    U, S, Vt = np.linalg.svd(seg_w.astype(np.float64), full_matrices=False)
    UB = Vt.T                       # (64, 12)，行空间正交基
    f = feats[cls_mask].astype(np.float64)
    proj = f @ UB
    num = (proj ** 2).sum(1)
    den = (f ** 2).sum(1) + 1e-12
    return (num / den).mean()


def train_probe(X, y, epochs=40, lr=1e-3, wd=5e-4):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.long, device=dev)
    W = torch.zeros(X.shape[1], NC, requires_grad=True, device=dev)
    b = torch.zeros(NC, requires_grad=True, device=dev)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    cnt = np.bincount(y, minlength=NC).astype(np.float32)
    cw = torch.tensor(np.where(cnt > 0,
                               cnt.sum() / (NC * np.maximum(cnt, 1)),
                               0.0), dtype=torch.float32, device=dev)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, BS):
            idx = perm[i:i + BS]
            logits = Xt[idx] @ W + b
            loss = F.cross_entropy(logits, yt[idx], weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sch.step()
        if (ep + 1) % 10 == 0:
            with torch.no_grad():
                acc = (Xt[:20000] @ W + b).argmax(1) \
                    .eq(yt[:20000]).float().mean()
            print('    ep%02d loss %.4f 采样训练准确率 %.3f  [%s]'
                  % (ep + 1, tot / n, acc.item(), dev), flush=True)
    return W.detach().cpu().numpy(), b.detach().cpu().numpy()


def main():
    tag = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    feats, gt = load_train(tag, CAP)
    print('%s: 采样后 %d 点 | 类分布 min=%d max=%d'
          % (tag, len(gt), np.bincount(gt, minlength=NC).min(),
             np.bincount(gt, minlength=NC).max()), flush=True)

    out_dir = os.path.join(FEAT, tag)
    probes = {}
    report = {'tag': tag, 'n_points': int(len(gt)), 'erank': {},
              'subspace': {}}

    # ---------- eRank（每类 2000 点，Base / Novel 分开） ----------
    base_m = gt < NO
    novel_m = gt >= NO
    for L in LAYERS:
        X = feats[L].astype(np.float32)
        eb = erank(X[base_m])
        en = erank(X[novel_m])
        report['erank'][L] = {'base': eb, 'novel': en, 'dim': DIM[L],
                              'base_ratio': eb / DIM[L],
                              'novel_ratio': en / DIM[L]}
        print('eRank %-5s D=%3d  Base %6.2f (%.1f%%)  Novel %6.2f (%.1f%%)'
              % (L, DIM[L], eb, 100 * eb / DIM[L], en, 100 * en / DIM[L]),
              flush=True)
    # Multi-Depth
    Xm = np.concatenate([feats[L].astype(np.float32) for L in MULTI], 1)
    eb, en = erank(Xm[base_m]), erank(Xm[novel_m])
    report['erank']['multi'] = {'base': eb, 'novel': en, 'dim': 512,
                                'base_ratio': eb / 512,
                                'novel_ratio': en / 512}
    print('eRank multi D=512  Base %6.2f (%.1f%%)  Novel %6.2f (%.1f%%)'
          % (eb, 100 * eb / 512, en, 100 * en / 512), flush=True)

    # ---------- Base 子空间能量（用各 ckpt 自己的 12 类 seg_head） ----------
    from cut2_common import CKPTS
    ck = torch.load(CKPTS[tag], map_location='cpu')
    sd = {k[7:] if k.startswith('module.') else k: v
          for k, v in ck['state_dict'].items()}
    seg_w = sd['seg_head.weight'].numpy()
    f0 = feats['dec0'].astype(np.float32)
    report['subspace'] = {
        'novel': subspace_energy(f0, gt, seg_w, novel_m),
        'base': subspace_energy(f0, gt, seg_w, base_m),
        'random_baseline': 12 / 64,
    }
    print('Base 子空间能量(dec0, %s): Novel %.1f%%  Base %.1f%%  (随机基线 18.8%%)'
          % (tag, 100 * report['subspace']['novel'],
             100 * report['subspace']['base']), flush=True)

    # ---------- 训练 6 组 probe ----------
    def prep(X):
        mu = X.mean(0)
        sd = X.std(0) + 1e-6
        return ((X - mu) / sd).astype(np.float32), mu, sd

    for name in LAYERS + ['multi']:
        if name == 'multi':
            cols, mus, sds = [], [], []
            for L in MULTI:
                X, mu, sd = prep(feats[L].astype(np.float32))
                cols.append(X)
                mus.append(mu)
                sds.append(sd)
            X = np.concatenate(cols, 1)
            probes['mu_' + name] = np.concatenate(mus)
            probes['sd_' + name] = np.concatenate(sds)
        else:
            X, mu, sd = prep(feats[name].astype(np.float32))
            probes['mu_' + name] = mu
            probes['sd_' + name] = sd
        t0 = time.time()
        W, b = train_probe(X, gt, epochs=epochs)
        probes['W_' + name] = W.astype(np.float32)
        probes['b_' + name] = b.astype(np.float32)
        with torch.no_grad():
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            Xt = torch.tensor(X, device=dev)
            yt = torch.tensor(gt, device=dev)
            acc = (Xt @ torch.tensor(W, device=dev) +
                   torch.tensor(b, device=dev)).argmax(1).eq(yt) \
                .float().mean().item()
        print('probe %-5s 训练准确率 %.4f  %.0fs'
              % (name, acc, time.time() - t0), flush=True)

    np.savez(os.path.join(out_dir, 'probes.npz'), **probes)
    with open(os.path.join(out_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print('-> %s/probes.npz 与 report.json 已写入' % out_dir, flush=True)


if __name__ == '__main__':
    main()
