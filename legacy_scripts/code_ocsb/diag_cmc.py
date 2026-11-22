"""诊断 delta_corr 是否对准真实漏检点。

分组：TP（margin>0 且真前景）、FN（margin<=0 且真前景，即漏检）、
FP（margin>0 且真背景）、TN（margin<=0 且真背景）。
如果方法有效，delta_corr 在 FN 上应显著高于 FP。
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ocsb_cmc import OCSBCMC  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
CKPT = '/data01/jh/weights/sc0_1w1s.pth'


def to_t(a):
    if a.dtype == np.float16:
        return torch.from_numpy(a.astype(np.float32)).cuda()
    if a.dtype.kind in 'iu':
        return torch.from_numpy(a.astype(np.int64)).cuda()
    return torch.from_numpy(a.astype(np.float32)).cuda()


def main():
    table = torch.load(CKPT, map_location='cpu')['state_dict'][
        'module.base_prototypes'].float().cuda()
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))[:100]
    cmc = None
    agg = {k: [] for k in ('TP', 'FN', 'FP', 'TN')}
    for m in metas:
        d = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
        ep = {k: to_t(d[k]) for k in d.files}
        ep['base_protos'] = table
        if cmc is None:
            cmc = OCSBCMC(int(ep['s_h'].shape[1]), tau=0.1, tau_b=0.6,
                          m_anchor=3).cuda()
        with torch.no_grad():
            out = cmc(ep, use_csr=True)
        dc = out['delta_corr']
        y = ep['q_y'] > 0
        pos = ep['q_margin'] > 0
        groups = {
            'TP': y & pos,
            'FN': y & (~pos),
            'FP': (~y) & pos,
            'TN': (~y) & (~pos),
        }
        for k, mask in groups.items():
            if mask.any():
                agg[k].append(float(dc[mask].mean()))

    print('%-4s %6s %9s %9s' % ('grp', 'n_ep', 'mean', 'median'))
    for k in ('TP', 'FN', 'FP', 'TN'):
        v = np.array(agg[k])
        if len(v):
            print('%-4s %6d %9.4f %9.4f' % (k, len(v), v.mean(), np.median(v)))
    if agg['FN'] and agg['FP']:
        fn = np.array(agg['FN'])
        fp = np.array(agg['FP'])
        print('\nFN - FP = %+.4f  (positive means delta favours missed fg)'
              % (fn.mean() - fp.mean()))


if __name__ == '__main__':
    main()
