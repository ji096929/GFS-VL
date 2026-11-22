"""训练 BridgeScorer：把 10 个相关性空间统计量组合成一个原型权重。

参数量 193（10→16→1），远低于 1000，杜绝过拟合。

监督
----
- 类别平衡 BCE：正样本按 (N_neg / N_pos) 加权，抵消 U 里前景远少于背景
- 排序 margin loss：随机采正负对，要求 w_pos − w_neg >= margin

评估（严格 2 折，按 episode 前后半划分）
- AUC 与 Top-30% 精度，和单信号 R_s (第 7 维) 对比
- 训练折与测试折都报，确认没有过拟合
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

CACHE = '/data01/jh/data/ocsb_cache'
FEAT_NAMES = ['p0', 'sp_mean', 'sp_std', 'sim_fg', 'sim_bg', 'sim_base',
              'R_s', 'rev_fg', 'dist_core', 'sim_core']


class BridgeScorer(nn.Module):
    def __init__(self, d_in=10, h=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_all():
    xs, ys, eps = [], [], []
    for f in sorted(glob.glob(os.path.join(CACHE, 'v3_feat_u_s[0-3].npz'))):
        d = np.load(f)
        xs.append(d['X'])
        ys.append(d['Y'])
        eps.append(d['EP'])
    return (np.concatenate(xs).astype(np.float32),
            np.concatenate(ys).astype(np.float32),
            np.concatenate(eps))


def topk_prec(y, w, frac):
    k = max(int(len(w) * frac), 1)
    idx = np.argsort(-w)[:k]
    return float(y[idx].mean())


def train_one(Xtr, ytr, Xte, yte, seed=0, epochs=300, lr=0.01, margin=0.3):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    net = BridgeScorer().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.from_numpy(Xtr).to(dev)
    yt = torch.from_numpy(ytr).to(dev)
    pos_w = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    wgt = torch.where(yt > 0.5, torch.full_like(yt, pos_w), torch.ones_like(yt))
    n = len(Xt)
    bs = 8192
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x, y = Xt[idx], yt[idx]
            w = net(x)
            bce = F.binary_cross_entropy(w, y, weight=wgt[idx])
            p = w[y > 0.5]
            q = w[y <= 0.5]
            if len(p) and len(q):
                k = min(len(p), len(q), 512)
                pp = p[torch.randperm(len(p))[:k]]
                qq = q[torch.randperm(len(q))[:k]]
                rank = F.relu(margin - (pp - qq)).mean()
            else:
                rank = torch.zeros((), device=w.device)
            loss = bce + 0.5 * rank
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        wtr = net(Xt).cpu().numpy()
        wte = net(torch.from_numpy(Xte).to(dev)).cpu().numpy()
    return net, wtr, wte


def report(tag, y, w, base_w):
    auc = float(roc_auc_score(y, w))
    auc_b = float(roc_auc_score(y, base_w))
    print('%-8s AUC %.4f (R_s 基线 %.4f, %+.4f)  '
          'Top50%% %.4f  Top30%% %.4f  Top10%% %.4f   基线 R_s: %.4f/%.4f/%.4f' % (
              tag, auc, auc_b, auc - auc_b,
              topk_prec(y, w, 0.5), topk_prec(y, w, 0.3), topk_prec(y, w, 0.1),
              topk_prec(y, base_w, 0.5), topk_prec(y, base_w, 0.3),
              topk_prec(y, base_w, 0.1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--out', default=os.path.join(CACHE, 'bridge_scorer.pt'))
    args = ap.parse_args()

    X, y, ep = load_all()
    print('样本 %d  正样本比例 %.4f  episode %d' % (
        len(X), y.mean(), len(np.unique(ep))))

    eps_all = np.unique(ep)
    half = len(eps_all) // 2
    fold_a, fold_b = eps_all[:half], eps_all[half:]
    print('fold A %d eps / fold B %d eps' % (len(fold_a), len(fold_b)))

    base = X[:, 6]          # R_s
    print('\n=== 2 折交叉验证 ===')
    models = []
    for tr_eps, te_eps, tag in ((fold_a, fold_b, 'A->B'), (fold_b, fold_a, 'B->A')):
        mtr = np.isin(ep, tr_eps)
        mte = np.isin(ep, te_eps)
        net, wtr, wte = train_one(X[mtr], y[mtr], X[mte], y[mte],
                                  epochs=args.epochs, lr=args.lr)
        models.append(net)
        # 折内模型：tag='A->B' 表示在 A 折上训练，用于给 B 折生成 pred_U
        torch.save(dict(state=net.state_dict()),
                   os.path.join(CACHE, 'scorer_trained%s.pt' % tag[0]))
        print('\n[%s]  训练 %d 点 / 测试 %d 点' % (
            tag, int(mtr.sum()), int(mte.sum())))
        report('train', y[mtr], wtr, base[mtr])
        report('test', y[mte], wte, base[mte])

    # 全量拟合一份用于后续生成原型
    net_all, w_all, _ = train_one(X, y, X[:10], y[:10], epochs=args.epochs, lr=args.lr)
    torch.save(dict(state=net_all.state_dict(),
                    feat_names=FEAT_NAMES), args.out)
    print('\n=== 全量 ===')
    report('all', y, w_all, base)

    # 各特征单独 AUC，看模型学到了什么
    print('\n=== 单特征 AUC（全量） ===')
    for i, nm in enumerate(FEAT_NAMES):
        v = X[:, i]
        print('  %-10s %.4f' % (nm, float(roc_auc_score(y, v))))

    print('\n->', args.out)


if __name__ == '__main__':
    main()
