"""门控信号探针：找能区分"高基线任务"（Q4/Q5）与"低基线任务"（Q1/Q2）的无监督指标。

背景
----
`frac_q`（预测前景占比）在 Q1–Q5 上都是 0.37–0.44，完全无法区分难度——
所有任务都在严重过分割。所以"欠分割门控"的前提不成立，必须换判据。

这里测一组任务级信号，全部只用 S 和 Q，不需要 U，也不需要任何真值
（GT 只用于事后分组算 AUC）。

候选
----
sim_fg_q    预测前景点与 S 前景原型的平均最大余弦
sim_bg_q    预测前景点与 S 背景原型的平均最大余弦
sim_gap     sim_fg_q − sim_bg_q（"预测出来的前景像不像真前景"）
conf_fg     预测前景点的平均前景概率
high_frac   mean(p > 0.7)
entropy     全点二分类熵的均值
fg_mass     全点前景概率之和（比 frac_q 更平滑）
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as Fs
from sklearn.metrics import roc_auc_score

from ocsb_core import (CACHE, build_episodes, build_proto, episode_tensors,
                       feats_one, iou_counts, load_model, predict_agg,
                       prepare_block, seed_everything)


def max_cos(f, protos):
    if protos.shape[0] == 0:
        return torch.zeros(f.shape[0], device=f.device)
    return Fs.cosine_similarity(f[:, None, :], protos[None, :, :], dim=2).max(1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=10)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(CACHE, 'gate_signal.json'))
    args = ap.parse_args()

    eps = build_episodes(args.per_class, args.shard, args.shards)
    model, cfg = load_model()
    dev = torch.device('cuda:0')

    rec = []
    for ei, m in enumerate(eps):
        g_ep = args.shard + args.shards * ei
        seed_everything(10000 + g_ep)
        c = m['class']
        s = prepare_block(m['support'], cfg)
        q = prepare_block(m['query'], cfg)
        s_x, s_bin = episode_tensors(s, c)
        q_x, q_map = episode_tensors(q, c)
        sf = feats_one(model, s_x, s_bin, dev)
        qf = feats_one(model, q_x, q_map, dev)
        fg_s = sf['yl'].bool()
        s_fg = build_proto(model, sf, fg_s)
        s_bg = build_proto(model, sf, ~fg_s)
        sparse_s = torch.cat([s_bg, s_fg], 0)

        pred, out = predict_agg(model, qf, [sparse_s], 0.0, dev)
        pq = Fs.softmax(out, dim=1)[:, 1]        # 降采样侧，和 qf['fd'] 对齐
        _, _, _, base_iou = iou_counts(pred, q_map)

        pred_fg = pq > 0.5
        n_fg = int(pred_fg.sum())
        if n_fg < 10:
            continue
        f = qf['fd'][pred_fg]
        sim_fg = float(max_cos(f, s_fg).mean())
        sim_bg = float(max_cos(f, s_bg).mean())
        conf_fg = float(pq[pred_fg].mean())
        high_frac = float((pq > 0.7).float().mean())
        ent = float((-(pq * pq.clamp(min=1e-8).log()
                        + (1 - pq) * (1 - pq).clamp(min=1e-8).log())).mean())

        rec.append(dict(ep=ei, cls=c, base_iou=base_iou,
                        sim_fg_q=sim_fg, sim_bg_q=sim_bg,
                        sim_gap=sim_fg - sim_bg,
                        conf_fg=conf_fg, high_frac=high_frac,
                        entropy=ent, fg_mass=float(pq.mean())))

    r = {k: np.array([x[k] for x in rec]) for k in rec[0] if k not in ('ep',)}
    iou = r['base_iou']
    print('episodes %d   baseIoU 均值 %.4f' % (len(rec), iou.mean()))

    # 用"高基线(>0.5) vs 低基线(<0.35)"当分组，看哪个信号能分开
    hi = iou > 0.5
    lo = iou < 0.35
    sel = hi | lo
    print('\n分组：高基线 %d（IoU>0.5） vs 低基线 %d（IoU<0.35）' % (
        hi.sum(), lo.sum()))
    print('%-12s %8s %10s %10s %10s' % (
        'signal', 'AUC', '低组均值', '高组均值', '方向'))
    for k in ('sim_fg_q', 'sim_bg_q', 'sim_gap', 'conf_fg',
              'high_frac', 'entropy', 'fg_mass'):
        v = r[k][sel]
        try:
            auc = float(roc_auc_score(hi[sel].astype(int), v))
        except ValueError:
            auc = float('nan')
        print('%-12s %8.4f %10.4f %10.4f %10s' % (
            k, auc, r[k][lo].mean(), r[k][hi].mean(),
            '高基线更大' if auc > 0.5 else '高基线更小'))

    print('\n与 baseIoU 的 Spearman：')
    from scipy.stats import spearmanr
    for k in ('sim_fg_q', 'sim_gap', 'conf_fg', 'high_frac', 'entropy', 'fg_mass'):
        rho, p = spearmanr(r[k], iou)
        print('  %-10s rho=%+.4f  p=%.2e' % (k, rho, p))

    json.dump(rec, open(args.out, 'w'), indent=1)
    print('\n->', args.out)


if __name__ == '__main__':
    main()
