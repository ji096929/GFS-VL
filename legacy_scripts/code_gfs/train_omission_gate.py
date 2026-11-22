"""训练 OmissionGate：逐点决定「这一点的预测要不要采信 U」。

输入 11 维 = 3 维点级 + 8 维任务级：
  点级：p0_s(i), p0_u(i), relu(p0_u(i) − p0_s(i))
  任务级：sim_fg_q, frac_q, frac_s, 比值, 熵, margin_p90, rev_fg, valid_ratio

输出 λθ(i) ∈ [0, 0.8)，融合：
  logit_final(i) = (1−λ) · pred_S(i) + λ · pred_U(i)

损失
----
  CE          : 逐点交叉熵（对齐最终分割）
  SoftIoU     : 1 − |p∩g|/|p∪g|，直接对齐 Cumul 的目标形式
  TP 保护项   : pred_S 原本判对的前景点若被翻成背景，额外惩罚——锁死 Q4/Q5 不掉分

评估：严格 2 折（按 episode 前后半），报测试折 Cumul + Q1–Q5 分层。
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE = '/data01/jh/data/ocsb_cache'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]


class OmissionGate(nn.Module):
    def __init__(self, d_in=11, h=16, lam_max=0.8):
        super().__init__()
        self.lam_max = lam_max
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.lam_max * torch.sigmoid(self.net(x).squeeze(-1))


def build_x(rec):
    p0s, p0u = rec['p0s'], rec['p0u']
    d = np.maximum(p0u - p0s, 0.0)
    task = np.tile(rec['x_task'][None, :], (len(p0s), 1))
    return np.concatenate([np.stack([p0s, p0u, d], 1), task], 1).astype(np.float32)


def fuse(net, rec, dev):
    x = torch.from_numpy(build_x(rec)).to(dev)
    lam = net(x)
    ps = torch.from_numpy(rec['pred_s']).to(dev)
    pu = torch.from_numpy(rec['pred_u']).to(dev)
    return (1 - lam)[None, :] * ps + lam[None, :] * pu, lam


def loss_episode(net, rec, dev, w_tp=2.0, w_iou=1.0):
    logit, lam = fuse(net, rec, dev)
    gt = torch.from_numpy(rec['gt']).to(dev)
    ps = torch.from_numpy(rec['pred_s']).to(dev)

    ce = F.cross_entropy(logit.T, gt)
    p = F.softmax(logit, dim=0)[1]
    inter = (p * gt).sum()
    union = p.sum() + gt.sum() - inter
    soft_iou = 1.0 - inter / union.clamp(min=1e-6)

    base_ok = (ps.argmax(0) == 1) & (gt == 1)
    now_bad = logit.argmax(0) == 0
    tp_pen = (base_ok & now_bad).float().sum() / base_ok.float().sum().clamp(min=1.0)
    return ce + w_iou * soft_iou + w_tp * tp_pen, lam


def evaluate(net, recs, dev):
    tp = {c: 0 for c in NEW_IDS}
    fp = {c: 0 for c in NEW_IDS}
    fn = {c: 0 for c in NEW_IDS}
    per_ep = {}
    for rec in recs:
        logit, _ = fuse(net, rec, dev)
        pred = logit.argmax(0).cpu().numpy()
        gt = rec['gt']
        c = rec['cls']
        t_ = int(((pred == 1) & (gt == 1)).sum())
        f_ = int(((pred == 1) & (gt == 0)).sum())
        n_ = int(((pred == 0) & (gt == 1)).sum())
        tp[c] += t_
        fp[c] += f_
        fn[c] += n_
        per_ep[rec['g_ep']] = t_ / max(t_ + f_ + n_, 1)
    per = [tp[c] / max(tp[c] + fp[c] + fn[c], 1) for c in NEW_IDS]
    return float(np.mean(per)), per, per_ep


def cumul_pred_s(recs):
    """基线（只用 pred_S）的 Cumul：按类累加 TP/FP/FN 再平均。"""
    tp = {c: 0 for c in NEW_IDS}
    fp = {c: 0 for c in NEW_IDS}
    fn = {c: 0 for c in NEW_IDS}
    for rec in recs:
        pred = rec['pred_s'].argmax(0)
        gt = rec['gt']
        c = rec['cls']
        tp[c] += int(((pred == 1) & (gt == 1)).sum())
        fp[c] += int(((pred == 1) & (gt == 0)).sum())
        fn[c] += int(((pred == 0) & (gt == 1)).sum())
    per = [tp[c] / max(tp[c] + fp[c] + fn[c], 1) for c in NEW_IDS]
    return float(np.mean(per))


def baseline_ious(recs):
    out = {}
    for rec in recs:
        pred = rec['pred_s'].argmax(0)
        gt = rec['gt']
        tp = int(((pred == 1) & (gt == 1)).sum())
        fp = int(((pred == 1) & (gt == 0)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum())
        out[rec['g_ep']] = tp / max(tp + fp + fn, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--w-tp', type=float, default=2.0)
    ap.add_argument('--out', default=os.path.join(CACHE, 'omission_gate.pt'))
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    recs = []
    for f in sorted(glob.glob(os.path.join(CACHE, 'v3_gate_s[0-3].pt'))):
        recs += torch.load(f)
    recs.sort(key=lambda r: r['g_ep'])
    print('episodes %d  设备 %s' % (len(recs), dev))

    eps_all = [r['g_ep'] for r in recs]
    half = len(eps_all) // 2
    folds = [(set(eps_all[:half]), set(eps_all[half:]), 'A->B'),
             (set(eps_all[half:]), set(eps_all[:half]), 'B->A')]

    base_ious = baseline_ious(recs)
    results = {}
    for tr, te, tag in folds:
        torch.manual_seed(0)
        net = OmissionGate().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
        tr_recs = [r for r in recs if r['g_ep'] in tr]
        te_recs = [r for r in recs if r['g_ep'] in te]
        print('\n[%s] 训练 %d / 测试 %d' % (tag, len(tr_recs), len(te_recs)))
        for ep in range(args.epochs):
            perm = torch.randperm(len(tr_recs))
            tot = 0.0
            for i in perm.tolist():
                loss, _ = loss_episode(net, tr_recs[i], dev, w_tp=args.w_tp)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss)
            if (ep + 1) % 10 == 0:
                cu_te, _, _ = evaluate(net, te_recs, dev)
                print('  ep %3d  loss %.4f  测试 Cumul %.4f' % (
                    ep + 1, tot / len(tr_recs), cu_te), flush=True)
        cu_tr, _, _ = evaluate(net, tr_recs, dev)
        cu_te, per_te, ep_te = evaluate(net, te_recs, dev)
        base_te = cumul_pred_s(te_recs)
        print('  训练 Cumul %.4f   测试 Cumul %.4f   基线 %.4f   dCumul %+.4f' % (
            cu_tr, cu_te, base_te, cu_te - base_te))
        results[tag] = dict(net=net, cu_te=cu_te, base=base_te,
                            per_ep=ep_te, per_class=per_te)

        # 分层
        ks = sorted(base_ious)
        order = sorted([r['g_ep'] for r in te_recs], key=lambda k: base_ious[k])
        order = [k for k in order if k in ep_te]
        q = len(order) // 5
        bins = []
        for i in range(5):
            seg = order[i * q:(i + 1) * q] if i < 4 else order[4 * q:]
            bins.append(np.mean([ep_te[k] - base_ious[k] for k in seg]) * 100)
        print('  分层 Q1..Q5: %s   Q1-Q3 %+.2f' % (
            ' '.join('%+.2f' % b for b in bins),
            float(np.mean(bins[:3]))))

    torch.save(dict(state=results['A->B']['net'].state_dict()), args.out)
    print('\n->', args.out)


if __name__ == '__main__':
    main()
