"""OCSB 管道的三种校验，合并成一个入口。

    python check.py replicate                 # 复刻的 predict_agg vs COSeg 原生 forward
    python check.py same    --gps 0,2,7,20    # 同进程连跑 3 次，看逐次是否一致
    python check.py cross   --gps 0,2,7,20    # 单次输出；起两个进程比 Baseline

三种校验解决的问题不同：
  replicate  手写的相关性聚合管道有没有漏/错步骤（与原生 logits 逐点比）
  same       进程内是否有随机源（模型状态被污染、dropout 没关等）
  cross      跨进程是否有随机源（体素化随机选点，靠 seed_everything 锁死）

cross 需要人工比对两个进程的输出；其余两种自己给 PASS/FAIL。
"""
import argparse
import json
import sys

import numpy as np
import torch

from ocsb_core import (build_episodes, build_proto, episode_tensors,
                       feats_one, iou_counts, load_model, predict_agg,
                       prepare_block, seed_everything)


def _one_episode(model, cfg, dev, m):
    """跑一个 episode 的 Baseline，返回 (iou, 特征校验和, npos, 原始 logits)。

    这里刻意把 prepare_block 放在函数内部：`same` 模式要求每次都重新
    体素化，才能验证种子是否真的锁住了 RNG；如果复用同一份张量，
    "一致"是假象。
    """
    c = m['class']
    s = prepare_block(m['support'], cfg)
    q = prepare_block(m['query'], cfg)
    s_x, s_bin = episode_tensors(s, c)
    q_x, q_map = episode_tensors(q, c)
    sf = feats_one(model, s_x, s_bin, dev)
    qf = feats_one(model, q_x, q_map, dev)
    fg = sf['yl'].bool()
    sp = torch.cat([build_proto(model, sf, ~fg), build_proto(model, sf, fg)], 0)
    pred, _ = predict_agg(model, qf, [sp], 0.0, dev)
    _, _, _, iou = iou_counts(pred, q_map)
    return iou, float(sf['fd'].abs().sum()), int(pred.argmax(0).sum()), pred, q_map


def cmd_replicate(model, cfg, dev, gps, eps):
    """原生 forward 与 predict_agg 的逐点对比。

    原生返回 [1, 2, N_raw]，复刻返回 [2, N_raw]，比较 argmax 掩码与 logits。
    logit 允许 1e-3 量级差异（ball_query 邻居求和的浮点顺序不同），
    但 argmax 必须几乎完全一致——差 1 个点是可以接受的边界。
    """
    print('%-4s %-14s %9s %9s %10s %11s' % (
        'g', 'class', 'iou_nat', 'iou_rep', 'd_iou', 'max|dlogit|'))
    worst_d, worst_iou = 0.0, 0.0
    for g in gps:
        m = eps[g]
        c = m['class']
        s = prepare_block(m['support'], cfg)
        q = prepare_block(m['query'], cfg)
        s_x, s_bin = episode_tensors(s, c)
        q_x, q_map = episode_tensors(q, c)
        with torch.no_grad():
            model.k_shot = 1
            nat, _ = model(
                torch.tensor([len(s_x)], dtype=torch.int32).cuda(),
                s_x.cuda(), s_bin.cuda(),
                torch.tensor([len(q_x)], dtype=torch.int32).cuda(),
                q_x.cuda(), q_map.cuda(),
                100, None, None, np.array([c]))
            sf = feats_one(model, s_x, s_bin, dev)
            qf = feats_one(model, q_x, q_map, dev)
        fg = sf['yl'].bool()
        sp = torch.cat([build_proto(model, sf, ~fg), build_proto(model, sf, fg)], 0)
        rep, _ = predict_agg(model, qf, [sp], 0.0, dev)

        nat = nat[0]
        d = float((nat - rep).abs().max())
        _, _, _, i_nat = iou_counts(nat, q_map)
        _, _, _, i_rep = iou_counts(rep, q_map)
        worst_d = max(worst_d, d)
        worst_iou = max(worst_iou, abs(i_nat - i_rep))
        print('%-4d %-14d %9.4f %9.4f %10.2e %11.2e' % (
            g, c, i_nat, i_rep, abs(i_nat - i_rep), d), flush=True)
    print('\nmax|dlogit| = %.2e   max|dIoU| = %.2e   -> %s' % (
        worst_d, worst_iou,
        'PASS' if worst_iou < 1e-3 else 'FAIL'))


def cmd_same(model, cfg, dev, gps, eps):
    """同进程连跑 3 次，每次重新 prepare_block。"""
    print('%-4s %-6s %9s %9s %9s %10s' % (
        'g', 'trial', 'iou', 'fd_abs', 'npos', 'd_vs_t0'))
    for g in gps:
        m = eps[g]
        seed_everything(10000 + g)
        base = None
        for t in range(3):
            iou, fd, npos, _, _ = _one_episode(model, cfg, dev, m)
            d = 0.0 if base is None else abs(iou - base)
            base = iou if base is None else base
            print('%-4d %-6d %9.6f %9.1f %9d %10.1e' % (
                g, t, iou, fd, npos, d), flush=True)
        print('', flush=True)


def cmd_cross(model, cfg, dev, gps, eps, tag):
    """单次输出，供两个独立进程逐行比对。"""
    for g in gps:
        seed_everything(10000 + g)
        iou, fd, npos, _, _ = _one_episode(model, cfg, dev, eps[g])
        print('%s g_ep=%3d cls=%2d iou=%.6f fd_abs=%.1f npos=%d' % (
            tag, g, eps[g]['class'], iou, fd, npos), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['replicate', 'same', 'cross'])
    ap.add_argument('--gps', default='0,2,7,20')
    ap.add_argument('--tag', default='A')
    ap.add_argument('--per-class', type=int, default=56)
    args = ap.parse_args()

    gps = [int(x) for x in args.gps.split(',')]
    eps = build_episodes(args.per_class)
    model, cfg = load_model()
    dev = torch.device('cuda:0')

    if args.mode == 'replicate':
        cmd_replicate(model, cfg, dev, gps, eps)
    elif args.mode == 'same':
        cmd_same(model, cfg, dev, gps, eps)
    else:
        cmd_cross(model, cfg, dev, gps, eps, args.tag)


if __name__ == '__main__':
    main()
