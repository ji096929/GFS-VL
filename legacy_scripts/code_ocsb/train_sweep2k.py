"""在 2000 个零泄漏 train split 任务上训练 W_phi，每 500 步评测 val_560。"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ocsb_faithful import OCSB, ocsb_losses  # noqa: E402
from run_ablation import (base_protos_table, load_ep, per_class, protos_for,
                          run_split, summarize)  # noqa: E402

TRAIN = '/data01/jh/data/ocsb_train2k'
VAL = '/data01/jh/data/ocsb_cache'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default=TRAIN)
    ap.add_argument('--cache', default=VAL)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--w-pos', type=float, default=2.0)
    ap.add_argument('--w-reg', type=float, default=2.0)
    ap.add_argument('--tau-s', type=float, default=1.0)
    ap.add_argument('--gamma', type=float, default=2.0)
    ap.add_argument('--mode', default='omit_select',
                    choices=['omit_select', 'all'])
    ap.add_argument('--lam-o', type=float, default=1.0)
    ap.add_argument('--lam-s', type=float, default=1.0)
    ap.add_argument('--eval-steps', type=int, default=500)
    ap.add_argument('--train-n', type=int, default=0)
    ap.add_argument('--out', default='/data01/jh/data/ocsb_cache/sweep2k.json')
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    # 训练 meta
    metas = []
    for f in sorted(glob.glob(os.path.join(args.train, 'meta_shard*.json'))):
        metas += json.load(open(f))
    metas = [m for m in metas
             if os.path.exists(os.path.join(args.train,
                                            'train_%04d.npz' % m['idx']))]
    if args.train_n:
        metas = metas[:args.train_n]
    tpaths = [os.path.join(args.train, 'train_%04d.npz' % m['idx'])
              for m in metas]
    print('train episodes:', len(tpaths), flush=True)

    # val_560
    vmeta = []
    for f in sorted(glob.glob(os.path.join(args.cache, 'val_meta*.json'))):
        vmeta += json.load(open(f))
    vmeta.sort(key=lambda m: m['idx'])
    seen, uniq = set(), []
    for m in vmeta:
        if m['idx'] in seen:
            continue
        seen.add(m['idx'])
        uniq.append(m)
    vmeta = uniq
    vpaths = [os.path.join(args.cache, 'val_%03d.npz' % m['idx'])
              for m in vmeta]
    for m, p in zip(vmeta, vpaths):
        m['y_full'] = np.load(p)['q_y_full'].astype(np.int64)
    print('val episodes:', len(vpaths), flush=True)

    table = base_protos_table()
    dim = int(np.load(vpaths[0])['s_h'].shape[1])
    m = OCSB(dim, tau=0.1, tau_b=0.6, fusion_mode='A',
             gamma_fuse=args.gamma).cuda()

    from run_ablation import margin_iou
    for mt, p in zip(vmeta, vpaths):
        if 'base_iou' in mt:
            continue
        d = np.load(p)
        ep = {}
        for k in d.files:
            a = d[k]
            ep[k] = torch.from_numpy(
                a.astype(np.float32) if a.dtype.kind == 'f'
                else a.astype(np.int64))
        mt['base_iou'] = margin_iou(
            {k: v.cuda() for k, v in ep.items()}, mt['y_full'])

    opt = torch.optim.Adam([{'params': m.proj.parameters()}], lr=args.lr)
    rng = np.random.RandomState(0)
    step, curve, evals = 0, [], []

    def evaluate():
        recs, preds = run_split(m, vpaths, vmeta, table, 'step5',
                                return_preds=True)
        s = summarize(recs)
        s['step'] = step
        s['per_class'] = per_class(recs)
        tp_c, un_c = {}, {}
        for pred, mt in zip(preds, vmeta):
            c = mt['class']
            y = mt['y_full']
            tp_c[c] = tp_c.get(c, 0) + int((pred & (y > 0)).sum())
            un_c[c] = un_c.get(c, 0) + int((pred | (y > 0)).sum())
        s['cum'] = float(np.mean([tp_c[c] / max(un_c[c], 1) for c in tp_c]))
        evals.append(s)
        print('  step %4d  arith %.4f (d %+.4f)  cum %.4f  Miss %.3f False %.3f'
              ' AUC %.3f' % (step, s['iou'], s['iou'] - s['base'], s['cum'],
                             s['missed_recall'], s['false_rate'], s['auc']),
              flush=True)
        return s

    print('step 0 baseline:')
    evaluate()

    for epoch in range(args.epochs):
        order = rng.permutation(len(tpaths))
        tot, nb = 0.0, 0
        for j in order:
            mt = metas[j]
            ep = load_ep(tpaths[j], protos_for(table, mt['class']))
            out = m(ep, enable_csr=True)
            parts = ocsb_losses(out, ep, model=m, w_pos=args.w_pos,
                                tau_s=args.tau_s)
            if args.mode == 'omit_select':
                loss = parts['omit'] + parts['select']
            else:
                loss = (parts['seg'] + args.lam_o * parts['omit']
                        + args.lam_s * parts['select'])
            if args.w_reg > 0:
                loss = loss + args.w_reg * m.weight_reg()
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
            step += 1
            if step % args.eval_steps == 0:
                evaluate()
        curve.append(tot / max(nb, 1))
        print('epoch %d done  loss %.4f  steps %d' % (epoch + 1, curve[-1], step),
              flush=True)

    final = evaluate()
    out = {'config': vars(args), 'train_n': len(tpaths),
           'curve': curve, 'evals': evals, 'final': final,
           'w_dev': float((m.proj.weight - torch.eye(dim, device='cuda')).norm())}
    json.dump(out, open(args.out, 'w'), indent=1)
    torch.save(m.proj.weight.detach().cpu(), args.out + '.wphi.pt')
    print('->', args.out, '+ .wphi.pt')


if __name__ == '__main__':
    main()
