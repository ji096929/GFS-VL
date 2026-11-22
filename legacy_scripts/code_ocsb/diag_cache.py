"""检查缓存质量与 OCSB 中间量的分布。

在已缓存的 episode 上跑一次前向，统计 B/C 规模、eta 风险、V 正负、
w_C 恢复比例，以及负对照三个来源的实际数量。用来确认「w_C 全零导致
L_seg 无梯度」不是系统性问题。
"""
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ocsb_faithful import OCSB, sp_soft_labels  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
CKPT = '/data01/jh/weights/sc0_1w1s.pth'
BASE_TO_SLOT = {1: 0, 4: 1, 8: 2, 9: 3, 11: 4, 15: 5, 16: 6, 17: 7, 18: 8,
                20: 9}


def to_t(a):
    if a.dtype == np.float16:
        return torch.from_numpy(a.astype(np.float32)).cuda()
    if a.dtype.kind in 'iu':
        return torch.from_numpy(a.astype(np.int64)).cuda()
    return torch.from_numpy(a.astype(np.float32)).cuda()


def main():
    table = torch.load(CKPT, map_location='cpu')['state_dict'][
        'module.base_prototypes'].float()
    files = sorted(glob.glob(os.path.join(CACHE, 'val_*.npz')))
    print('episodes:', len(files))

    dim = int(np.load(files[0])['s_h'].shape[1])
    m = OCSB(dim, tau=0.1, tau_b=0.6, gamma0=0.2).cuda()

    rows = []
    for f in files:
        d = np.load(f)
        ep = {k: to_t(d[k]) for k in d.files}
        # val 用 novel 类，base_protos 全部保留
        ep['base_protos'] = table.cuda()
        with torch.no_grad():
            out = m(ep)
        yR = sp_soft_labels(ep['u_y'], ep['u_seg'].long(), out['n_reg'], 2)
        fg_reg = yR[:, 1] >= 0.5
        c = out['C_idx']
        r = {
            'n_reg': out['n_reg'],
            'n_B': int(out['B_idx'].numel()),
            'n_C': int(c.numel()),
            'n_Cpos': int(fg_reg[c].sum()) if c.numel() else 0,
            'wC_sum': float(out['w_C'].sum()),
            'wC_nnz': int((out['w_C'] > 1e-8).sum()),
            'wB_sum': float(out['w_B'].sum()) if out['w_B'].numel() else 0.0,
            'V_pos': float((out['V'] > 0).float().mean()) if out['V'].numel() else float('nan'),
            'eta_mean': float(out['eta'].mean()) if out['eta'].numel() else float('nan'),
            'q_hi': float((out['q_reg'] >= 0.6).float().mean()),
            'q_bg': float((out['q_reg'] < 0.05).float().mean()),
            'rec': int(0),
        }
        if c.numel():
            pbar = out['P_bar'][1]
            r['rec'] = int((pbar > 0.5).sum())
        rows.append(r)

    def stat(k):
        v = [x[k] for x in rows if not (isinstance(x[k], float) and np.isnan(x[k]))]
        if not v:
            print('%-10s n/a' % k)
            return
        print('%-10s mean %8.2f  median %8.2f  min %6.2f  max %8.2f'
              % (k, np.mean(v), np.median(v), np.min(v), np.max(v)))

    print('\n== 超点规模 ==')
    for k in ('n_reg', 'n_B', 'n_C', 'n_Cpos'):
        stat(k)
    print('\n== 恢复量 ==')
    for k in ('wC_sum', 'wC_nnz', 'wB_sum', 'rec', 'V_pos', 'eta_mean'):
        stat(k)
    print('\n== 初始概率分布 ==')
    for k in ('q_hi', 'q_bg'):
        stat(k)

    dead = [x for x in rows if x['wC_sum'] <= 1e-8]
    print('\nw_C 全零的 episode: %d / %d (%.0f%%)'
          % (len(dead), len(rows), 100 * len(dead) / max(len(rows), 1)))
    noB = [x for x in rows if x['n_B'] == 0]
    print('B 为空的 episode: %d / %d' % (len(noB), len(rows)))
    json.dump(rows, open(os.path.join(CACHE, 'diag_rows.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
