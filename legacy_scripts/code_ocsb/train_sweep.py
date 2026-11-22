"""Step 5 的 W_phi 训练网格：w_pos / w_reg / lr，逐轮评测。

复用 run_ablation 的缓存加载与评测逻辑。Epoch 0 的评测在更新前做，
用于确认它与 Step 4 的数值完全一致。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ocsb_faithful import OCSB  # noqa: E402
from run_ablation import (base_protos_table, load_ep, per_class, protos_for,
                          run_split, summarize, train_wphi)  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/data01/jh/data/ocsb_cache')
    ap.add_argument('--val-n', type=int, default=100)
    ap.add_argument('--train-n', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--w-pos', type=float, default=1.0)
    ap.add_argument('--w-reg', type=float, default=2.0)
    ap.add_argument('--tau-s', type=float, default=1.0)
    ap.add_argument('--gamma', type=float, default=1.75)
    ap.add_argument('--mode', default='all')
    ap.add_argument('--lam-o', type=float, default=1.0)
    ap.add_argument('--lam-s', type=float, default=1.0)
    ap.add_argument('--out', default='/data01/jh/data/ocsb_cache/sweep5.json')
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    val_meta = []
    for f in sorted(glob.glob(os.path.join(args.cache, 'val_meta*.json'))):
        val_meta += json.load(open(f))
    val_meta.sort(key=lambda m: m['idx'])
    if args.val_n:
        val_meta = val_meta[:args.val_n]
    val_paths = [os.path.join(args.cache, 'val_%03d.npz' % m['idx'])
                 for m in val_meta]
    for m, p in zip(val_meta, val_paths):
        m['y_full'] = np.load(p)['q_y_full'].astype(np.int64)
    print('val episodes:', len(val_paths))

    train_meta = json.load(open(os.path.join(args.cache, 'train_meta.json')))
    train_paths = [os.path.join(args.cache, 'train_%03d.npz' % m['idx'])
                   for m in train_meta]
    if args.train_n:
        train_meta = train_meta[:args.train_n]
        train_paths = train_paths[:args.train_n]
    print('train episodes:', len(train_paths))

    table = base_protos_table()
    dim = int(np.load(val_paths[0])['s_h'].shape[1])

    m = OCSB(dim, tau=0.1, tau_b=0.6, fusion_mode='A',
             gamma_fuse=args.gamma).cuda()

    # Epoch 0：更新前，必须与 Step 4 一致
    st0 = summarize(run_split(m, val_paths, val_meta, table, 'step5'))
    print('epoch 0  IoU %.4f d %+.4f  Miss %.3f False %.3f Cprec %.3f AUC %.3f'
          % (st0['iou'], st0['iou'] - st0['base'], st0['missed_recall'],
             st0['false_rate'], st0['c_prec'], st0['auc']), flush=True)

    eval_fn = (lambda mm: summarize(run_split(
        mm, val_paths, val_meta, table, 'step5')))
    curve, evals = train_wphi(
        m, train_paths, train_meta, table, args.epochs, args.lr,
        args.lam_o, args.lam_s, args.mode, w_reg=args.w_reg,
        w_pos=args.w_pos, tau_s=args.tau_s, eval_fn=eval_fn, eval_every=1)

    recs = run_split(m, val_paths, val_meta, table, 'step5')
    final = summarize(recs)
    final['per_class'] = per_class(recs)

    best = max(evals, key=lambda s: s['iou']) if evals else st0
    out = {'config': vars(args), 'epoch0': st0, 'curve': curve,
           'evals': evals, 'final': final, 'best': best,
           'w_dev': float((m.proj.weight - torch.eye(
               dim, device='cuda')).norm())}
    json.dump(out, open(args.out, 'w'), indent=1)
    print('\nfinal  IoU %.4f d %+.4f  Miss %.3f False %.3f AUC %.3f  w_dev %.2f'
          % (final['iou'], final['iou'] - final['base'], final['missed_recall'],
             final['false_rate'], final['auc'], out['w_dev']))
    print('best   epoch %s  IoU %.4f d %+.4f'
          % (best.get('epoch'), best['iou'], best['iou'] - best['base']))
    print('->', args.out)


if __name__ == '__main__':
    main()
