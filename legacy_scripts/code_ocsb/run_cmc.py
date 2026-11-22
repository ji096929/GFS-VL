"""在 val_100 上评测 CMC 注入（Fix 1 + Fix 2 + Fix 3）。

对每个 gamma 跑 4 种注入：
  baseline  不挂 hook
  true      真实 delta_corr
  mean      任务内均值常数
  shuffle   任务内随机打乱

指标同时给算术平均 IoU 和 Class-Cumulative IoU。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import diag_bc as D  # noqa: E402
from ocsb_cmc import Lin1Injector, OCSBCMC  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
CKPT = '/data01/jh/weights/sc0_1w1s.pth'


def to_t(a):
    if a.dtype == np.float16:
        return torch.from_numpy(a.astype(np.float32)).cuda()
    if a.dtype.kind in 'iu':
        return torch.from_numpy(a.astype(np.int64)).cuda()
    return torch.from_numpy(a.astype(np.float32)).cuda()


def load_block(name, cfg):
    d = np.load(os.path.join(DATA, name + '.npy'))
    return D.data_prepare_seg(d[:, 0:3], d[:, 3:6], d[:, 6], d[:, 7],
                              cfg.voxel_size, cfg.voxel_max)


@torch.no_grad()
def cosseg_margin(model, s, q, cls, dev):
    s_coord, s_feat, s_label, s_seg = s
    q_coord, q_feat, q_label, q_seg = q
    s_bin = (s_label == cls).long()
    q_map = (q_label == cls).long()
    support_x = torch.cat([s_coord, s_feat], 1).to(dev)
    query_x = torch.cat([q_coord, q_feat], 1).to(dev)
    s_off = torch.tensor([s_coord.shape[0]], dtype=torch.int32).to(dev)
    q_off = torch.tensor([q_coord.shape[0]], dtype=torch.int32).to(dev)
    fp, _ = model(s_off, support_x, s_bin.to(dev), q_off, query_x,
                  q_map.to(dev), 100, None, None, np.array([cls]))
    return fp[0][1] - fp[0][0], q_map


def iou_stats(margin, y):
    pred = margin > 0
    y = y.to(pred.device)
    tp = int((pred & (y > 0)).sum())
    un = int((pred | (y > 0)).sum())
    return tp, un


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--m-anchor', type=int, default=3)
    ap.add_argument('--top-m', type=int, default=0)
    ap.add_argument('--mode', default='all',
                    choices=['all', 'baseline', 'true', 'mean', 'shuffle'])
    ap.add_argument('--out', default='/data01/jh/data/ocsb_cache/cmc.json')
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device('cuda:0')

    table = torch.load(CKPT, map_location='cpu')['state_dict'][
        'module.base_prototypes'].float().cuda()

    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))[:args.n]
    eps = []
    for m in metas:
        d = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
        ep = {k: to_t(d[k]) for k in d.files}
        ep['base_protos'] = table
        m['y_full'] = d['q_y_full'].astype(np.int64)
        eps.append(ep)
    print('episodes:', len(eps), flush=True)

    model, cfg = D.build_model()
    blocks = {}
    for m in metas:
        for name in (m['support'], m['query']):
            if name not in blocks:
                blocks[name] = load_block(name, cfg)
    print('blocks loaded:', len(blocks), flush=True)

    # 缓存上算 delta_corr
    cmc = OCSBCMC(int(eps[0]['s_h'].shape[1]), tau=0.1, tau_b=0.6,
                  m_anchor=args.m_anchor).cuda()
    deltas = []
    for ep in eps:
        with torch.no_grad():
            out = cmc(ep, use_csr=True, top_m=(args.top_m or None))
            deltas.append(out['delta_corr'].detach())
    dcat = torch.cat(deltas)
    print('delta_corr: mean %.4f  p50 %.4f  p90 %.4f  frac>0 %.2f'
          % (float(dcat.mean()), float(dcat.median()),
             float(dcat.quantile(0.9)), float((dcat > 0).float().mean())),
          flush=True)

    rng = np.random.RandomState(0)
    injector = Lin1Injector()
    injector.attach(model)

    def run(tag, delta_list, gamma):
        tp_c, un_c, ious = {}, {}, []
        for i, (m, ep) in enumerate(zip(metas, eps)):
            injector.gamma = gamma
            injector.delta = None if delta_list is None else delta_list[i]
            margin, qmap = cosseg_margin(model, blocks[m['support']],
                                         blocks[m['query']], m['class'], dev)
            tp, un = iou_stats(margin, torch.from_numpy(m['y_full']).cuda())
            c = m['class']
            tp_c[c] = tp_c.get(c, 0) + tp
            un_c[c] = un_c.get(c, 0) + un
            ious.append(tp / max(un, 1))
        cum = float(np.mean([tp_c[c] / max(un_c[c], 1) for c in tp_c]))
        return {'arith': float(np.mean(ious)), 'cum': cum, 'n': len(ious)}

    results = {}
    if args.mode in ('all', 'baseline'):
        results['baseline'] = run('baseline', None, 0.0)
        print('baseline  arith %.4f  cum %.4f'
              % (results['baseline']['arith'], results['baseline']['cum']),
              flush=True)

    if args.mode in ('all', 'true', 'mean', 'shuffle'):
        results['true'] = run('true', deltas, args.gamma)
        print('true g=%.2f  arith %.4f  cum %.4f'
              % (args.gamma, results['true']['arith'], results['true']['cum']),
              flush=True)

    if args.mode in ('all', 'mean'):
        means = [torch.full_like(d, d.mean()) for d in deltas]
        results['mean'] = run('mean', means, args.gamma)
        print('mean g=%.2f  arith %.4f  cum %.4f'
              % (args.gamma, results['mean']['arith'], results['mean']['cum']),
              flush=True)

    if args.mode in ('all', 'shuffle'):
        shuf = []
        for d in deltas:
            dn = d.cpu().numpy()
            shuf.append(torch.from_numpy(dn[rng.permutation(len(dn))]).cuda()
                        .to(d.dtype))
        results['shuffle'] = run('shuffle', shuf, args.gamma)
        print('shuffle g=%.2f  arith %.4f  cum %.4f'
              % (args.gamma, results['shuffle']['arith'],
                 results['shuffle']['cum']), flush=True)

    results['config'] = vars(args)
    json.dump(results, open(args.out, 'w'), indent=1)
    print('->', args.out)


if __name__ == '__main__':
    main()
