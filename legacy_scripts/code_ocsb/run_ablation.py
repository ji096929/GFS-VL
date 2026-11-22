"""导师文档第五部分的五步消融。

Step 1  原始 COSeg（不使用 U）
Step 2  高置信超点 B 直接扩展原型（无 OASB/CSR，不恢复 C，W_phi=I）
Step 3  OASB（eta=0），恢复 C 并按式(13) 反哺 w_B
Step 4  OASB + CSR（扩充负对照）
Step 5  在 200 个 base 类缓存任务上训练 W_phi，再在 val_100 上评测

指标：Query mIoU / Missed-FG Recall / False Recovery Rate / Recovered C
Precision / v_c(R) 的 AUC。
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ocsb_faithful import (OCSB, ocsb_losses, direct_expansion,
                           b_only_region_prob)  # noqa: E402

CKPT = '/data01/jh/weights/sc0_1w1s.pth'
BASE_TO_SLOT = {1: 0, 4: 1, 8: 2, 9: 3, 11: 4, 15: 5, 16: 6, 17: 7, 18: 8,
                20: 9}


def to_t(a):
    if a.dtype == np.float16:
        return torch.from_numpy(a.astype(np.float32)).cuda()
    if a.dtype.kind in 'iu':
        return torch.from_numpy(a.astype(np.int64)).cuda()
    return torch.from_numpy(a.astype(np.float32)).cuda()


def load_ep(path, base_protos=None):
    d = np.load(path)
    ep = {k: to_t(d[k]) for k in d.files}
    ep['base_protos'] = base_protos
    return ep


def base_protos_table():
    sd = torch.load(CKPT, map_location='cpu')['state_dict']
    return sd['module.base_prototypes'].float()


def protos_for(table, cls):
    """novel 类：10 个基类原型全部可用；base 类：屏蔽当前目标类。"""
    if cls not in BASE_TO_SLOT:
        return table.cuda()
    keep = torch.ones(table.shape[0], dtype=torch.bool)
    keep[BASE_TO_SLOT[cls]] = False
    return table[keep].cuda()


def interp(coord_full, coord_down, val, k=3):
    """3-NN 反距离插值，把 downsampled 点的得分送回原始点。"""
    d, idx = cKDTree(coord_down).query(coord_full, k=k)
    d = np.asarray(d)
    idx = np.asarray(idx)
    if d.ndim == 1:
        d = d[:, None]
        idx = idx[:, None]
    w = 1.0 / (d + 1e-8)
    w /= w.sum(1, keepdims=True)
    return (val[idx] * w).sum(1)


def margin_iou(ep, y_full):
    """基线：原始点级 margin 的 IoU（与 Step 1 同口径）。"""
    e = interp(ep['q_coord_full'].cpu().numpy().astype(np.float64),
               ep['q_coord'].cpu().numpy().astype(np.float64),
               ep['q_margin'].detach().cpu().numpy().astype(np.float64))
    return q_iou(e, y_full)


def q_iou(ell_full, y_full):
    pred = ell_full > 0
    tp = int((pred & (y_full > 0)).sum())
    fp = int((pred & (y_full == 0)).sum())
    fn = int((~pred & (y_full > 0)).sum())
    return tp / max(tp + fp + fn, 1)


def region_metrics(P_bar, C_idx, fg_reg):
    """超点级恢复指标。P_bar 可能是全超点 [2,R] 或仅 C 候选 [2,Rc]。"""
    p_fg = P_bar[1]
    if p_fg.numel() == fg_reg.numel():
        p_fg = p_fg[C_idx]
    pred_fg = p_fg > 0.5
    is_pos = fg_reg[C_idx]
    n_pos = int(is_pos.sum())
    n_neg = int((~is_pos).sum())
    rec_pos = int((pred_fg & is_pos).sum())
    rec_neg = int((pred_fg & (~is_pos)).sum())
    missed_recall = rec_pos / max(n_pos, 1)
    false_rate = rec_neg / max(n_neg, 1)
    prec = rec_pos / max(int(pred_fg.sum()), 1)
    return missed_recall, false_rate, prec, n_pos, n_neg


def v_auc(out, fg_reg):
    """v_c(R)=max_B V 区分真遗漏前景与困难负类的 AUC。"""
    V = out['V']
    if V.numel() == 0:
        return float('nan')
    v = V.max(0).values.detach().cpu().numpy()
    c = out['C_idx']
    y = fg_reg[c].cpu().numpy()
    if y.min() == y.max():
        return float('nan')
    order = np.argsort(v)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    # rank-based AUC，避免依赖 sklearn
    ranks = np.empty(len(v), dtype=np.float64)
    ranks[order] = np.arange(1, len(v) + 1)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def eval_episode(model, ep, mode):
    """返回该 episode 的 ell（downsampled）与超点指标。"""
    if mode == 'step1':
        return ep['q_margin'], None

    out = model(ep, enable_csr=(mode in ('step4', 'step5')))
    fg_reg = None
    # 超点真值：用于诊断，不参与前向
    from ocsb_faithful import sp_soft_labels
    yR = sp_soft_labels(ep['u_y'], ep['u_seg'].long(), out['n_reg'], 2)
    fg_reg = yR[:, 1] >= 0.5

    if mode == 'step2':
        ell = direct_expansion(out, ep, model)
        aux = {'out': out, 'fg_reg': fg_reg,
               'P_bar': b_only_region_prob(out['s0'], out['s0_bg'], out['z_R'],
                                           out['B_idx'], out['q_reg'],
                                           model.tau)}
        return ell, aux

    if out['C_idx'].numel() == 0:
        return ep['q_margin'], {'out': out, 'fg_reg': fg_reg, 'P_bar': None}

    return out['ell'], {'out': out, 'fg_reg': fg_reg,
                        'P_bar': out['P_bar']}


def run_split(model, paths, metas, table, mode, return_preds=False):
    ious, recs, preds = [], [], []
    for p, mt in zip(paths, metas):
        bp = protos_for(table, mt['class'])
        ep = load_ep(p, bp)
        with torch.no_grad():
            ell, aux = eval_episode(model, ep, mode)
        ell_np = ell.detach().cpu().numpy().astype(np.float64)
        m_interp = interp(ep['q_coord_full'].cpu().numpy().astype(np.float64),
                          ep['q_coord'].cpu().numpy().astype(np.float64),
                          ell_np)
        iou = q_iou(m_interp, mt['y_full'])
        if 'base_iou' not in mt:
            mt['base_iou'] = margin_iou(ep, mt['y_full'])
        r = {'class': mt['class'], 'iou': iou, 'base_iou': mt['base_iou']}
        if aux is not None and isinstance(aux, dict) and aux.get('P_bar') is not None:
            out = aux['out']
            mr, fr, prec, npos, nneg = region_metrics(
                aux['P_bar'], out['C_idx'], aux['fg_reg'])
            r.update({'missed_recall': mr, 'false_rate': fr,
                      'c_prec': prec, 'n_cpos': npos, 'n_cneg': nneg,
                      'n_B': int(out['B_idx'].numel()),
                      'auc': v_auc(out, aux['fg_reg'])})
        ious.append(iou)
        recs.append(r)
        preds.append(m_interp > 0)
    if return_preds:
        return recs, preds
    return recs


def train_wphi(model, train_paths, train_metas, table, epochs, lr,
               lam_o, lam_s, mode, seed=0, w_reg=0.0, w_pos=1.0,
               tau_s=1.0, eval_fn=None, eval_every=1):
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam([{'params': model.proj.parameters()}], lr=lr)
    curve, evals = [], []
    for ep_i in range(epochs):
        order = rng.permutation(len(train_paths))
        tot, nb = 0.0, 0
        for j in order:
            mt = train_metas[j]
            ep = load_ep(train_paths[j], protos_for(table, mt['class']))
            out = model(ep, enable_csr=True)
            parts = ocsb_losses(out, ep, model=model, w_pos=w_pos,
                                tau_s=tau_s)
            if mode == 'omit_select':
                loss = parts['omit'] + parts['select']
            else:
                loss = parts['seg'] + lam_o * parts['omit'] + lam_s * parts['select']
            if w_reg > 0:
                loss = loss + w_reg * model.weight_reg()
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
        curve.append(tot / max(nb, 1))
        msg = '  epoch %d loss %.4f' % (ep_i + 1, curve[-1])
        if eval_fn is not None and (ep_i + 1) % eval_every == 0:
            st = eval_fn(model)
            st['epoch'] = ep_i + 1
            evals.append(st)
            msg += ('  val IoU %.4f d %+.4f  Miss %.3f False %.3f AUC %.3f'
                    % (st['iou'], st['iou'] - st['base'], st['missed_recall'],
                       st['false_rate'], st['auc']))
        print(msg, flush=True)
    return curve, evals


def summarize(recs):
    keys = ['iou', 'missed_recall', 'false_rate', 'c_prec', 'auc']
    out = {'n': len(recs)}
    for k in keys:
        v = [r[k] for r in recs if k in r]
        out[k] = float(np.nanmean(v)) if v else float('nan')
    bases = [r['base_iou'] for r in recs if 'base_iou' in r]
    out['base'] = float(np.mean(bases)) if bases else float('nan')
    return out


def cum_of_preds(preds, metas):
    tp_c, un_c = {}, {}
    for pred, mt in zip(preds, metas):
        c = mt['class']
        y = mt['y_full']
        tp_c[c] = tp_c.get(c, 0) + int((pred & (y > 0)).sum())
        un_c[c] = un_c.get(c, 0) + int((pred | (y > 0)).sum())
    return float(np.mean([tp_c[c] / max(un_c[c], 1) for c in tp_c]))


def per_class(recs):
    d = {}
    for r in recs:
        d.setdefault(r['class'], []).append(r['iou'] - r['base_iou'])
    return {int(c): {'n': len(v), 'delta': float(np.mean(v))}
            for c, v in sorted(d.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/data01/jh/data/ocsb_cache')
    ap.add_argument('--val-eps', type=int, default=0,
                    help='0 = 使用缓存里全部 episode')
    ap.add_argument('--tau', type=float, default=0.1)
    ap.add_argument('--tau-b', type=float, default=0.6)
    ap.add_argument('--gamma', type=float, default=0.2)
    ap.add_argument('--gamma-fuse', type=float, default=5.0,
                    help='式(15) 修正后的外层叠加系数')
    ap.add_argument('--fusion-mode', default='A', choices=['A', 'B'],
                    help='A=双边增量(式11), B=只取正增量')
    ap.add_argument('--lam-o', type=float, default=1.0)
    ap.add_argument('--lam-s', type=float, default=1.0)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', default='/data01/jh/data/ocsb_cache/ablation.json')
    ap.add_argument('--w-pos', type=float, default=1.0,
                    help='L_select 正类权重')
    ap.add_argument('--tau-s', type=float, default=1.0,
                    help='L_select 的 softplus 温度')
    ap.add_argument('--w-reg', type=float, default=0.0,
                    help='单位阵锚定正则系数')
    ap.add_argument('--eval-every', type=int, default=1,
                    help='每几轮评测一次验证集')
    ap.add_argument('--train-n', type=int, default=0,
                    help='只用前 N 个训练任务，0=全部')
    ap.add_argument('--skip-step5', action='store_true',
                    help='训练缓存还没好时先只跑 Step 1-4')
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    val_meta = []
    for f in sorted(glob.glob(os.path.join(args.cache, 'val_meta*.json'))):
        val_meta += json.load(open(f))
    val_meta.sort(key=lambda m: m['idx'])
    seen, uniq = set(), []
    for m in val_meta:
        if m['idx'] in seen:
            continue
        seen.add(m['idx'])
        uniq.append(m)
    val_meta = uniq
    if args.val_eps:
        val_meta = val_meta[:args.val_eps]
    val_paths = [os.path.join(args.cache, 'val_%03d.npz' % m['idx'])
                 for m in val_meta]
    for m, p in zip(val_meta, val_paths):
        d = np.load(p)
        m['y_full'] = d['q_y_full'].astype(np.int64)
    print('val episodes:', len(val_paths))

    train_meta, train_paths = [], []
    if not args.skip_step5:
        train_meta = json.load(open(os.path.join(args.cache, 'train_meta.json')))
        train_paths = [os.path.join(args.cache, 'train_%03d.npz' % m['idx'])
                       for m in train_meta]
        if args.train_n:
            train_meta = train_meta[:args.train_n]
            train_paths = train_paths[:args.train_n]
        print('train episodes:', len(train_paths))

    table = base_protos_table()
    dim = int(np.load(val_paths[0])['s_h'].shape[1])
    print('feature dim:', dim)

    results = {}

    # ---------------- Step 1-4：W_phi = I ----------------
    for mode, tag in (('step1', 'Step1 COSeg'),
                      ('step2', 'Step2 direct B'),
                      ('step3', 'Step3 OASB'),
                      ('step4', 'Step4 OASB+CSR')):
        m = OCSB(dim, tau=args.tau, tau_b=args.tau_b, gamma0=args.gamma,
                 fusion_mode=args.fusion_mode,
                 gamma_fuse=args.gamma_fuse).cuda()
        t0 = time.time()
        recs, preds = run_split(m, val_paths, val_meta, table, mode,
                                return_preds=True)
        s = summarize(recs)
        s['sec'] = time.time() - t0
        s['cum'] = cum_of_preds(preds, val_meta)
        results[tag] = s
        print('%s  IoU %.4f  cum %.4f  MissRec %.3f  FalseRec %.3f  Cprec %.3f  AUC %.3f'
              % (tag, s['iou'], s['cum'], s['missed_recall'], s['false_rate'],
                 s['c_prec'], s['auc']), flush=True)
        results[tag + '_per_ep'] = recs
        results[tag + '_per_class'] = per_class(recs)

    # ---------------- Step 5：训练 W_phi ----------------
    for mode, tag in (('omit_select', 'Step5a omit+select'),
                      ('all', 'Step5b seg+omit+select')) if not args.skip_step5 else ():
        m = OCSB(dim, tau=args.tau, tau_b=args.tau_b, gamma0=args.gamma,
                 fusion_mode=args.fusion_mode,
                 gamma_fuse=args.gamma_fuse).cuda()
        eval_fn = (lambda mm: summarize(run_split(
            mm, val_paths, val_meta, table, 'step5')))
        curve, evals = train_wphi(
            m, train_paths, train_meta, table, args.epochs, args.lr,
            args.lam_o, args.lam_s, mode, w_reg=args.w_reg,
            w_pos=args.w_pos, tau_s=args.tau_s, eval_fn=eval_fn,
            eval_every=args.eval_every)
        recs, preds = run_split(m, val_paths, val_meta, table, 'step5',
                                return_preds=True)
        s = summarize(recs)
        s['cum'] = cum_of_preds(preds, val_meta)
        s['curve'] = curve
        s['evals'] = evals
        s['per_class'] = per_class(recs)
        s['w_dev'] = float((m.proj.weight - torch.eye(dim, device='cuda')).norm())
        s['gamma'] = float(m.gamma)
        s['cum'] = cum_of_preds(preds, val_meta)
        results[tag] = s
        print('%s  IoU %.4f  cum %.4f  MissRec %.3f  FalseRec %.3f  Cprec %.3f  AUC %.3f'
              % (tag, s['iou'], s['cum'], s['missed_recall'], s['false_rate'],
                 s['c_prec'], s['auc']), flush=True)
        results[tag + '_per_ep'] = recs
        results[tag + '_per_class'] = per_class(recs)

    json.dump(results, open(args.out, 'w'), indent=1)
    print('\n==== table ====')
    print('%-24s %8s %9s %9s %8s %7s' % ('version', 'mIoU', 'MissRec',
                                         'FalseRec', 'Cprec', 'AUC'))
    for tag in ('Step1 COSeg', 'Step2 direct B', 'Step3 OASB',
                'Step4 OASB+CSR', 'Step5a omit+select',
                'Step5b seg+omit+select'):
        if tag not in results:
            continue
        s = results[tag]
        print('%-24s %8.4f %9.3f %9.3f %8.3f %7.3f'
              % (tag, s['iou'], s['missed_recall'], s['false_rate'],
                 s['c_prec'], s['auc']))
    print('saved ->', args.out)


if __name__ == '__main__':
    main()
