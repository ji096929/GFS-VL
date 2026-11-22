"""S2（纯加权相似度，中心化）的 CMC 注入评测。

与 run_cmc.py 的区别：注入信号从 delta_corr（S1）换成中心化的 S2。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import diag_bc as D  # noqa: E402
from ocsb_cmc import Lin1Injector, OCSBCMC  # noqa: E402
from run_cmc import cosseg_margin, iou_stats, load_block, to_t  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
CKPT = '/data01/jh/weights/sc0_1w1s.pth'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--gammas', default='1.0,2.0,5.0')
    ap.add_argument('--out', default='/data01/jh/data/ocsb_cache/cmc_s2.json')
    args = ap.parse_args()
    gammas = [float(x) for x in args.gammas.split(',')]

    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device('cuda:0')
    table = torch.load(CKPT, map_location='cpu')['state_dict'][
        'module.base_prototypes'].float().cuda()

    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))[:args.n]
    eps, S2s = [], []
    cmc = None
    for m in metas:
        d = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
        ep = {k: to_t(d[k]) for k in d.files}
        ep['base_protos'] = table
        m['y_full'] = d['q_y_full'].astype(np.int64)
        if cmc is None:
            cmc = OCSBCMC(int(ep['s_h'].shape[1]), tau=0.1, tau_b=0.6,
                          m_anchor=3).cuda()
        with torch.no_grad():
            out = cmc(ep, use_csr=True)
        z_q, M, w = out['z_q'], out['M_feat'], out['M_w']
        if M.shape[0] == 0:
            s2 = torch.zeros(z_q.shape[0], device=z_q.device)
        else:
            wh = w / (w.sum() + 1e-6)
            s2 = (wh.unsqueeze(0) * (z_q @ M.t())).sum(1)
        eps.append(ep)
        S2s.append((s2 - s2.mean()).detach())
    print('episodes:', len(eps), flush=True)
    allc = torch.cat(S2s)
    print('S2 centered: mean %.4f std %.4f p10 %.4f p90 %.4f'
          % (float(allc.mean()), float(allc.std()), float(allc.quantile(0.1)),
             float(allc.quantile(0.9))), flush=True)

    model, cfg = D.build_model()
    blocks = {}
    for m in metas:
        for name in (m['support'], m['query']):
            if name not in blocks:
                blocks[name] = load_block(name, cfg)
    print('blocks:', len(blocks), flush=True)

    injector = Lin1Injector()
    injector.attach(model)

    def run(delta_list, gamma, tag):
        tp_c, un_c, ious = {}, {}, []
        for i, (m, ep) in enumerate(zip(metas, eps)):
            injector.gamma = gamma
            injector.delta = None if delta_list is None else delta_list[i]
            margin, _ = cosseg_margin(model, blocks[m['support']],
                                      blocks[m['query']], m['class'], dev)
            tp, un = iou_stats(margin, torch.from_numpy(m['y_full']).cuda())
            tp_c[m['class']] = tp_c.get(m['class'], 0) + tp
            un_c[m['class']] = un_c.get(m['class'], 0) + un
            ious.append(tp / max(un, 1))
        cum = float(np.mean([tp_c[c] / max(un_c[c], 1) for c in tp_c]))
        print('%-18s g=%-5.2f arith %.4f cum %.4f'
              % (tag, gamma, float(np.mean(ious)), cum), flush=True)
        return {'arith': float(np.mean(ious)), 'cum': cum, 'ious': ious}

    res = {'baseline': run(None, 0.0, 'baseline')}
    base = res['baseline']['ious']
    for g in gammas:
        cur = run(S2s, g, 's2-center')
        d = np.array(cur['ious']) - np.array(base)
        cur['pse'] = float(d.std(ddof=1) / np.sqrt(len(d)))
        print('    pSE %.4f  t %.2f' % (cur['pse'], d.mean() / cur['pse']),
              flush=True)
        res['s2_g%g' % g] = cur
    rng = np.random.RandomState(0)
    shuf = [torch.from_numpy(s.cpu().numpy()[rng.permutation(len(s))]).cuda()
            .to(s.dtype) for s in S2s]
    cur = run(shuf, gammas[0], 's2-shuffle')
    d = np.array(cur['ious']) - np.array(base)
    cur['pse'] = float(d.std(ddof=1) / np.sqrt(len(d)))
    res['shuffle'] = cur
    json.dump(res, open(args.out, 'w'), indent=1)
    print('->', args.out)


if __name__ == '__main__':
    main()
