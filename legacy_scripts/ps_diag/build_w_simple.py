"""把 online_pl_w 的软权重改写成「来源两级权重」：PS keep=1.0，AI/补标=0.45。

依据（实测，同一 Y_b==-1 口径）：
  PS keep 41.4M 点 精度 0.404
  AI 新增 16.3M 点 精度 0.232   ← 1.7 倍差距，是唯一有实测支持的合法信号
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, '/data01/jh/code/ps_diag')
import vlm_common as V  # noqa: E402

NO = 12
DATA = '/data01/jh/data/scannet200_gfs/train'
ONLINE = '/data01/jh/data/online_pl'
PLW = '/data01/jh/data/online_pl_w'
AI_W = float(sys.argv[1]) if len(sys.argv) > 1 else 0.45


def main():
    n_ps = n_ai = 0
    for f in sorted(glob.glob(PLW + '/*_pl.npy')):
        sc = os.path.basename(f)[:-7]
        pl = np.load(f)
        ps = np.load(os.path.join(ONLINE, sc + '.npz'))['ps']
        w = np.ones(len(pl), dtype=np.float32)
        ai = (ps < NO) & (pl >= NO)
        w[ai] = AI_W
        n_ai += int(ai.sum())
        n_ps += int(((ps >= NO) & (pl >= NO)).sum())
        np.save(os.path.join(PLW, sc + '_w.npy'), w)
    print('PS %d  AI/补标 %d (w=%.2f)' % (n_ps, n_ai, AI_W))


if __name__ == '__main__':
    main()
