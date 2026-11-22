"""生死线检验：Q_bg_pred 中按桥梁得分选 Top-K%，看 Precision 能否超过 I0。

严格阈值：p > I0 = 37.44%（不是 I0/(1+I0) = 27.24%）。
同时报告相对 27.24% 的位置，便于对照。

四种打分：
  S_mean    Σ ŵ_R (z_R·z_q)
  S_max     max_R (z_R·z_q)
  S_B_max   max_B q_c(B) (z_B·z_q)
  S_oracle  max_{R ∈ U 真实前景超点} (z_R·z_q)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import diag_bc as D  # noqa: E402
from ocsb_cmc import OCSBCMC  # noqa: E402
from ocsb_faithful import sp_soft_labels  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
CKPT = '/data01/jh/weights/sc0_1w1s.pth'
KS = [0.01, 0.02, 0.05, 0.10]
NAMES = ('margin', 'mean', 'max', 'B_max', 'oracle')
DEAD = 0.3744
REF = 0.2724


def to_t(a):
    if a.dtype == np.float16:
        return torch.from_numpy(a.astype(np.float32)).cuda()
    if a.dtype.kind in 'iu':
        return torch.from_numpy(a.astype(np.int64)).cuda()
    return torch.from_numpy(a.astype(np.float32)).cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=560)
    ap.add_argument('--out', default='/data01/jh/data/ocsb_cache/death.json')
    ap.add_argument('--w-phi', default='', help='训练后的 W_phi 权重路径')
    args = ap.parse_args()

    torch.manual_seed(0)
    table = torch.load(CKPT, map_location='cpu')['state_dict'][
        'module.base_prototypes'].float().cuda()
    cfg = D.load_cfg_from_cfg_file(D.CFG_PATH)
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))[:args.n]
    print('episodes:', len(metas), flush=True)

    blocks = {}
    for m in metas:
        name = m['query']
        if name not in blocks:
            blocks[name] = np.load(os.path.join(DATA, name + '.npy'))
    print('query blocks:', len(blocks), flush=True)

    qprep, qtree = {}, {}
    cmc = None
    # 点级累计 [dtp, dfp, dfn, dtn]
    pt = {nm: {k: [0, 0, 0, 0] for k in KS} for nm in NAMES}
    # 超点级累计
    sp = {nm: {k: [0, 0, 0, 0] for k in KS} for nm in NAMES}
    base_tp = base_fp = base_fn = 0
    I0_list = []

    for ei, m in enumerate(metas):
        d = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
        ep = {k: to_t(d[k]) for k in d.files}
        ep['base_protos'] = table
        if cmc is None:
            cmc = OCSBCMC(int(ep['s_h'].shape[1]), tau=0.1, tau_b=0.6,
                          m_anchor=3).cuda()
            if args.w_phi:
                w = torch.load(args.w_phi, map_location='cpu')
                cmc.proj.weight.data.copy_(w.to(cmc.proj.weight.dtype))
                print('loaded W_phi from', args.w_phi,
                      'dev %.3f' % float((w - torch.eye(w.shape[0])).norm()),
                      flush=True)
        with torch.no_grad():
            out = cmc(ep, use_csr=True)
        z_q = out['z_q']
        M, w = out['M_feat'], out['M_w']
        y = ep['q_y'] > 0
        margin = ep['q_margin']
        bg = margin <= 0

        base_tp += int((y & (~bg)).sum())
        base_fp += int(((~y) & (~bg)).sum())
        base_fn += int((y & bg).sum())

        scores = {'margin': margin}
        if M.shape[0] > 0:
            sim = z_q @ M.t()
            wh = w / (w.sum() + 1e-6)
            scores['mean'] = (wh.unsqueeze(0) * sim).sum(1)
            scores['max'] = sim.max(1).values
        else:
            scores['mean'] = torch.zeros(z_q.shape[0], device=z_q.device)
            scores['max'] = torch.zeros(z_q.shape[0], device=z_q.device)
        if out['z_B'].shape[0] > 0:
            scores['B_max'] = (out['q_B'].unsqueeze(0)
                               * (z_q @ out['z_B'].t())).max(1).values
        else:
            scores['B_max'] = torch.zeros(z_q.shape[0], device=z_q.device)
        # oracle：U 中真前景超点
        yR = sp_soft_labels(ep['u_y'], ep['u_seg'].long(), out['n_reg'], 2)
        fgR = yR[:, 1] > 0.5
        z_fg = out['z_R'][fgR]
        if z_fg.shape[0] > 0:
            scores['oracle'] = (z_q @ z_fg.t()).max(1).values
        else:
            scores['oracle'] = torch.zeros(z_q.shape[0], device=z_q.device)

        # query 超点 id（downsampled），预处理结果按 block 缓存
        name = m['query']
        if name not in qprep:
            b = blocks[name]
            qprep[name] = D.data_prepare_seg(
                b[:, 0:3], b[:, 3:6], b[:, 6], b[:, 7],
                cfg.voxel_size, cfg.voxel_max)
        q_coord_t, _, _, q_seg_t = qprep[name]
        if name not in qtree:
            qtree[name] = cKDTree(q_coord_t.numpy().astype(np.float64))
        idx = qtree[name].query(
            ep['q_coord'].cpu().numpy().astype(np.float64))[1]
        q_seg = q_seg_t.numpy()[idx]

        for nm in NAMES:
            s = scores[nm].detach()
            # ---- 点级 ----
            n_bg = int(bg.sum())
            if n_bg > 0:
                sc = s[bg]
                order = np.argsort(-sc.cpu().numpy())
                bg_idx = bg.nonzero().flatten()
                yb = y[bg].cpu().numpy()
                for k in KS:
                    kk = max(int(n_bg * k), 1)
                    sel = bg_idx[torch.from_numpy(order[:kk]).cuda()]
                    yy = y[sel]
                    dtp = int(yy.sum())
                    dfp = int((~yy).sum())
                    dfn = int((y & bg).sum()) - dtp
                    dtn = n_bg - dtp - dfp - dfn
                    for j, v in enumerate((dtp, dfp, dfn, dtn)):
                        pt[nm][k][j] += v
            # ---- 超点级 ----
            uniq, inv = np.unique(q_seg, return_inverse=True)
            if len(uniq) > 1:
                inv_t = torch.from_numpy(inv).cuda()
                n_sp = len(uniq)
                m_sp = torch.zeros(n_sp, device=margin.device)
                m_sp.index_add_(0, inv_t, margin)
                cnt = torch.bincount(inv_t, minlength=n_sp).clamp_min(1).float()
                m_sp = m_sp / cnt
                s_sp = torch.zeros(n_sp, device=margin.device)
                s_sp.index_add_(0, inv_t, s)
                s_sp = s_sp / cnt
                bg_sp = m_sp <= 0
                n_bgsp = int(bg_sp.sum())
                if n_bgsp > 0:
                    ids = bg_sp.nonzero().flatten()
                    order = torch.argsort(-s_sp[ids])
                    for k in KS:
                        kk = max(int(n_bgsp * k), 1)
                        keep = ids[order[:kk]]
                        mask = torch.zeros(n_sp, dtype=torch.bool,
                                           device=margin.device)
                        mask[keep] = True
                        sel = mask[inv_t]
                        yy = y[sel]
                        dtp = int(yy.sum())
                        dfp = int((~yy).sum())
                        # 该 episode 的 bg-sp 内真前景总数
                        tot_fn = int((y & bg).sum())
                        dfn = tot_fn - dtp
                        dtn = int(bg.sum()) - dtp - dfp - dfn
                        for j, v in enumerate((dtp, dfp, dfn, dtn)):
                            sp[nm][k][j] += v
        if (ei + 1) % 50 == 0:
            torch.cuda.empty_cache()
            print('  %d / %d' % (ei + 1, len(metas)),
                  'gpu %.1fGB' % (torch.cuda.memory_allocated() / 2**30),
                  flush=True)

    I0 = base_tp / max(base_tp + base_fp + base_fn, 1)
    print('\nbaseline IoU %.4f  (dead line p > %.4f, ref %.4f)'
          % (I0, I0, REF))
    for tag, acc in (('point', pt), ('superpoint', sp)):
        print('\n=== %s level ===' % tag)
        print('%-8s %-6s %10s %10s %10s %8s %8s'
              % ('score', 'top', 'dTP', 'dFP', 'precision', 'recall', 'pass'))
        for nm in NAMES:
            for k in KS:
                dtp, dfp, dfn, _ = acc[nm][k]
                p = dtp / max(dtp + dfp, 1)
                r = dtp / max(dtp + dfn, 1)
                print('%-8s %-6.0f%% %10d %10d %10.4f %8.4f %8s'
                      % (nm, k * 100, dtp, dfp, p, r,
                         'YES' if p > I0 else 'no'))
    json.dump({'I0': I0, 'point': pt, 'superpoint': sp},
              open(args.out, 'w'), indent=1)
    print('->', args.out)


if __name__ == '__main__':
    main()
