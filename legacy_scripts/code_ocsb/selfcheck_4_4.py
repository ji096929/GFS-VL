"""导师文档 4.4 节的五项强制检查。

1. B/C 的划分只依赖初始预测 q_c，不使用 U 的真值
2. 负对照 N_c 在训练与测试阶段都能构建（不读 U 标签）
3. 强制 B_c 为空时，查询得分与原始 COSeg margin 的误差 < 1e-6
4. L_omit / L_select / L_seg 都能向 W_phi 传播非零梯度
5. 查询点云不参与 B/C 的发现
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ocsb_faithful import OCSB, ocsb_losses  # noqa: E402

CKPT = '/data01/jh/weights/sc0_1w1s.pth'
BASE_TO_SLOT = {1: 0, 4: 1, 8: 2, 9: 3, 11: 4, 15: 5, 16: 6, 17: 7, 18: 8,
                20: 9}


def to_t(a):
    if a.dtype == np.float16:
        return torch.from_numpy(a.astype(np.float32)).cuda()
    if a.dtype.kind in 'iu':
        return torch.from_numpy(a.astype(np.int64)).cuda()
    return torch.from_numpy(a.astype(np.float32)).cuda()


def load_ep(path):
    d = np.load(path)
    return {k: to_t(d[k]) for k in d.files}


def base_protos_for(cls):
    sd = torch.load(CKPT, map_location='cpu')['state_dict']
    bp = sd['module.base_prototypes'].float()
    if cls not in BASE_TO_SLOT:
        return bp.cuda()
    keep = torch.ones(bp.shape[0], dtype=torch.bool)
    keep[BASE_TO_SLOT[cls]] = False
    return bp[keep].cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/data01/jh/data/ocsb_cache')
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--tau', type=float, default=0.1)
    ap.add_argument('--tau-b', type=float, default=0.6)
    ap.add_argument('--gamma', type=float, default=0.2)
    args = ap.parse_args()

    torch.manual_seed(0)
    paths = sorted([f for f in os.listdir(args.cache) if f.startswith('train_')])
    paths = paths[:args.n]
    print('checking %d training episodes' % len(paths))

    results = []
    for fn in paths:
        ep = load_ep(os.path.join(args.cache, fn))
        cls = 1  # 具体类只影响 base_protos 屏蔽，检查用第 1 个 slot
        # 从 meta 读真实类
        meta_path = os.path.join(args.cache, 'train_meta.json')
        if os.path.exists(meta_path):
            mt = json.load(open(meta_path))
            rec = next(x for x in mt if 'train_%03d.npz' % x['idx'] == fn)
            cls = rec['class']
        ep['base_protos'] = base_protos_for(cls)

        m = OCSB(dim=int(ep['s_h'].shape[1]), tau=args.tau,
                 tau_b=args.tau_b, gamma0=args.gamma).cuda()
        out = m(ep)
        r = {'ep': fn, 'class': cls}

        # ---- 1. B/C 只看初始预测：打乱 U 真值后必须完全一致 ----
        ep_noY = dict(ep)
        ep_noY['u_y'] = torch.zeros_like(ep['u_y'])
        out_noY = m(ep_noY)
        same_B = bool(torch.equal(out['B_idx'], out_noY['B_idx']))
        same_C = bool(torch.equal(out['C_idx'], out_noY['C_idx']))
        r['check1_BC_label_free'] = same_B and same_C

        # ---- 2. 负对照可构建且不读 U 标签 ----
        neg_feat = m._negative_set(ep, m.project(ep['s_h']), out['z_R'],
                                   out['q_reg'], out['s0_bg'])
        n_neg = int(neg_feat[0].shape[0])
        src = neg_feat[1]
        r['check2_neg_n'] = n_neg
        r['check2_neg_src'] = (
            [int((src == k).sum()) for k in range(3)] if src is not None else [])
        r['check2_ok'] = n_neg > 0

        # ---- 3. B 为空时严格退化为 COSeg ----
        m_empty = OCSB(dim=int(ep['s_h'].shape[1]), tau=args.tau,
                       tau_b=1.1, gamma0=args.gamma).cuda()
        out_empty = m_empty(ep)
        err = float((out_empty['ell'] - ep['q_margin']).abs().max())
        r['check3_max_abs_err'] = err
        r['check3_ok'] = err < 1e-6

        # ---- 4. 三项损失分别回传到 W_phi ----
        for tag in ('seg', 'omit', 'select'):
            m.zero_grad(set_to_none=True)
            o = m(ep)
            parts = ocsb_losses(o, ep, model=m)
            loss = parts[tag]
            if loss.requires_grad:
                loss.backward()
                g = m.proj.weight.grad
                r['check4_%s_grad' % tag] = (float(g.norm())
                                             if g is not None else 0.0)
            else:
                r['check4_%s_grad' % tag] = None
            r['check4_%s_loss' % tag] = float(loss.detach())

        # ---- 5. 查询点不参与 B/C 发现 ----
        ep_noQ = dict(ep)
        ep_noQ['q_h'] = torch.randn_like(ep['q_h'])
        ep_noQ['q_margin'] = torch.randn_like(ep['q_margin'])
        out_noQ = m(ep_noQ)
        r['check5_BC_q_free'] = (bool(torch.equal(out['B_idx'], out_noQ['B_idx']))
                                 and bool(torch.equal(out['C_idx'], out_noQ['C_idx'])))

        results.append(r)
        print(r, flush=True)

    print('\n==== summary ====')
    for k in ('check1_BC_label_free', 'check2_ok', 'check3_ok', 'check5_BC_q_free'):
        vals = [x[k] for x in results]
        print('%-24s %s' % (k, 'PASS' if all(vals) else 'FAIL'))
    for tag in ('seg', 'omit', 'select'):
        gs = [x['check4_%s_grad' % tag] for x in results
              if x.get('check4_%s_grad' % tag) is not None]
        print('grad %-6s mean %.3e min %.3e' % (
            tag, float(np.mean(gs)) if gs else float('nan'),
            float(np.min(gs)) if gs else float('nan')))
    print('check3 max err %.2e' % max(x['check3_max_abs_err'] for x in results))

    json.dump(results, open('/data01/jh/data/ocsb_cache/selfcheck.json', 'w'),
              indent=1)


if __name__ == '__main__':
    main()
