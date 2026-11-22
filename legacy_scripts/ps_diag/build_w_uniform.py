"""建「均匀权重」目录：_pl.npy 软链接自 online_pl_w，_w.npy 全 1.0。

用途：A2 = 同一份 PL（PS+AI+补标）但权重全 1，用来把 SW3 的效果拆成
      ① ONLINE -> A2  = 补标（43.6 万点）的贡献
      ② A2 -> SW3     = 逐点软权重的贡献
"""
import argparse
import glob
import os

import numpy as np

SRC = '/data01/jh/data/online_pl_w'
DST = '/data01/jh/data/online_pl_w_uniform'


def main():
    global SRC, DST
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--dst', default=DST)
    args = ap.parse_args()
    SRC, DST = args.src, args.dst
    os.makedirs(DST, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(SRC + '/*_pl.npy')):
        sc = os.path.basename(f)[:-7]
        link = os.path.join(DST, sc + '_pl.npy')
        if not os.path.exists(link):
            os.symlink(f, link)
        pl = np.load(f)
        np.save(os.path.join(DST, sc + '_w.npy'),
                np.ones(len(pl), dtype=np.float16))
        n += 1
    print('uniform dir: %d 场景' % n)


if __name__ == '__main__':
    main()
