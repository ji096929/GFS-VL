"""诊断式(15) 融合后掉点的原因。

打印 s_s（COSeg margin）、超点相似度 sim/τ、ell 的分布，以及预测正例率
与真值正例率的对比。重点看 sim 与 s_s 是否量纲一致。
"""
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ocsb_faithful import OCSB, sp_soft_labels  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'


def to_t(a):
    if a.dtype == np.float16:
        return torch.from_numpy(a.astype(np.float32)).cuda()
    if a.dtype.kind in 'iu':
        return torch.from_numpy(a.astype(np.int64)).cuda()
    return torch.from_numpy(a.astype(np.float32)).cuda()


def pct(name, x):
    x = x.detach().cpu().numpy().ravel()
    if x.size == 0:
        print('%-14s empty' % name)
        return
    q = np.percentile(x, [1, 10, 50, 90, 99])
    print('%-14s n=%7d  p1 %6.2f  p10 %6.2f  p50 %6.2f  p90 %6.2f  p99 %6.2f'
          % (name, x.size, q[0], q[1], q[2], q[3], q[4]))


def main():
    files = sorted(glob.glob(os.path.join(CACHE, 'val_*.npz')))[:5]
    dim = int(np.load(files[0])['s_h'].shape[1])
    m = OCSB(dim, tau=0.1, tau_b=0.6, gamma0=0.2).cuda()

    for f in files:
        d = np.load(f)
        ep = {k: to_t(d[k]) for k in d.files}
        ep['base_protos'] = None
        with torch.no_grad():
            out = m(ep)
        print('\n===== %s =====' % os.path.basename(f))
        pct('q_margin', ep['q_margin'])
        pct('u_p0', ep['u_p0'])
        pct('q_reg', out['q_reg'])
        pct('s0', out['s0'])
        pct('s0_bg', out['s0_bg'])
        if out['B_idx'].numel():
            zq = m.project(ep['q_h'])
            sim = (out['z_B'] @ zq.t()) / m.tau
            pct('sim_BQ', sim)
            pct('w_B', out['w_B'])
            pct('q_B', out['q_B'])
        if out['V'].numel():
            pct('delta', out['delta'])
            pct('eta', out['eta'])
            pct('V', out['V'])
            print('V>0 frac %.3f' % float((out['V'] > 0).float().mean()))
        pct('w_C', out['w_C'])
        pct('ell', out['ell'])
        y = ep['q_y']
        print('pred fg rate %.4f   GT fg rate %.4f   step1 IoU %.4f'
              % (float((out['ell'] > 0).float().mean()), float(y.mean()),
                 float((((out['ell'] > 0) & (y > 0)).sum())
                       / max(int(((out['ell'] > 0) | (y > 0)).sum()), 1))))
        # Step 1 的原始 margin
        print('margin pred rate %.4f' % float((ep['q_margin'] > 0).float().mean()))
        # 只用 B、gamma=1 的融合结果，作对照
        if out['B_idx'].numel():
            from ocsb_faithful import fuse_query_score
            zq = m.project(ep['q_h'])
            ell_g1 = fuse_query_score(ep['q_margin'], out['z_B'], out['q_B'],
                                      zq, m.tau, 1.0)
            print('gamma=1 B-only pred rate %.4f' % float((ell_g1 > 0).float().mean()))


if __name__ == '__main__':
    main()
