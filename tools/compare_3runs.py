#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XL-2 第二步：三局 det_eval 结果的 18 类逐类对比。"""
import os
import sys
import numpy as np
import torch

NB = 12
NAMES = ['wall','floor','door','ceiling','table','window','box','ceiling lamp',
         'light switch','cabinet','chair','heater',
         'monitor','whiteboard','office chair','bottle','doorframe','keyboard',
         'window frame','mouse','paper','blinds','trash can','telephone',
         'book','shelf','sink','windowsill','bag','smoke detector']
MARK = {26: '<<<', 25: '<<<', 13: '<<<', 16: '<<<'}

RUNS = [
    ('base',    '/data01/jh/data/det_eval_scpp_base_k5_s42/result/val.pth'),
    ('sym',     '/data01/jh/data/det_eval_scpp_sym_k5_s42/result/val.pth'),
    ('sym_g02', '/data01/jh/data/det_eval_scpp_sym_k5_g02_s42/result/val.pth'),
]

def load(p):
    d = torch.load(p, map_location='cpu', weights_only=False)
    it = d['intersection'].astype(np.float64)
    un = d['union'].astype(np.float64)
    return np.where(un > 0, it / np.maximum(un, 1), np.nan), it, un

def main():
    ious, ok = {}, True
    for tag, p in RUNS:
        if not os.path.exists(p):
            print('[missing] %s: %s' % (tag, p)); ok = False; continue
        ious[tag] = load(p)[0]
    if not ious:
        return
    tags = [t for t, _ in RUNS if t in ious]

    print('=== 整体 ===')
    print('%-10s %8s %8s %8s' % ('run', 'Base', 'Novel', 'HM'))
    for t in tags:
        b = np.nanmean(ious[t][:NB]); n = np.nanmean(ious[t][NB:])
        hm = 2*b*n/(b+n)
        print('%-10s %8.4f %8.4f %8.4f' % (t, b, n, hm))

    print('\n=== 18 novel 类逐类 IoU ===')
    hdr = '%-16s' % 'class'
    for t in tags:
        hdr += ' %10s' % t
    if len(tags) >= 2:
        hdr += ' %10s' % (tags[1] + '-' + tags[0])
    if len(tags) >= 3:
        hdr += ' %10s' % (tags[2] + '-' + tags[0])
    print(hdr)
    for c in range(NB, len(NAMES)):
        line = '%-16s' % NAMES[c]
        for t in tags:
            line += ' %10.4f' % ious[t][c]
        if len(tags) >= 2:
            line += ' %+10.4f' % (ious[tags[1]][c] - ious[tags[0]][c])
        if len(tags) >= 3:
            line += ' %+10.4f' % (ious[tags[2]][c] - ious[tags[0]][c])
        print(line + ' ' + MARK.get(c, ''))

if __name__ == '__main__':
    main()
