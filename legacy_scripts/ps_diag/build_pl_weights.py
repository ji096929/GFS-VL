"""生成最终点级伪标签 + Sonata 软权重。

输入
  /data01/jh/data/online_pl/{scene}.npz      ps / psai（官方在线 PS、PS+AI）
  /data01/jh/data/region_cache + region_index + sonata_region   （novel 预测区）
  /data01/jh/data/base_index  + sonata_base                     （base 预测区）
  /data01/jh/data/support_protos/sc200_regis_1_10_sonata.npy

输出
  {OUT}/{scene}_pl.npy   int16[N]   PS+AI 为主，叠加 reject 池 G3z 补标
  {OUT}/{scene}_w.npy    float32[N] 1.0（GT/NB-Mix）或软权重

权重规则
  PS keep        w = 0.20 + 0.80 * sigmoid(2z)
  AI 新增 / 补标  w = 0.05 + 0.95 * sigmoid(2z)
  GT / ignore / 未标注  w = 1.0（ignore 不参与 loss）
  z = 类别内（PS+AI+补标 合并）对 cos(Sonata, 支持原型) 的标准化
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/ps_diag')
import vlm_common as V  # noqa: E402

NO, NN = 12, 45
DATA = '/data01/jh/data/scannet200_gfs/train'
ONLINE = '/data01/jh/data/online_pl'
REGION_IDX = '/data01/jh/data/region_index'
BASE_IDX = '/data01/jh/data/base_index'
SON_REG = '/data01/jh/data/sonata_region'
SON_BASE = '/data01/jh/data/sonata_base'
PROTO = '/data01/jh/data/support_protos/sc200_regis_1_10_sonata.npy'
REJ = '/data01/jh/data/online_pl_rej_s0.json'
OUT = '/data01/jh/data/online_pl_w'
BUDGET = 2_000_000


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-8 else np.zeros_like(x)


def load_pt(path, cls):
    out = {}
    for f in sorted(glob.glob(path)):
        try:
            d = torch.load(f, map_location='cpu')
        except Exception:
            continue
        for sc, r in d.items():
            out.setdefault(sc, []).extend([r])
    return out


def main():
    global ONLINE, OUT, REJ
    ap = argparse.ArgumentParser()
    ap.add_argument('--proto', default=PROTO)
    ap.add_argument('--online', default=ONLINE,
                    help='在线伪标签 npz 目录')
    ap.add_argument('--out', default=OUT,
                    help='逐场景 _pl/_w 输出目录')
    ap.add_argument('--rej', default=REJ,
                    help='PS 拒绝明细 json')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--ps-lo', type=float, default=0.20)
    ap.add_argument('--ps-hi', type=float, default=1.00)
    ap.add_argument('--ai-lo', type=float, default=0.10)
    ap.add_argument('--ai-hi', type=float, default=1.00)
    ap.add_argument('--ai-global', type=float, default=1.0)
    ap.add_argument('--sharp', type=float, default=2.0)
    args = ap.parse_args()
    ONLINE, OUT, REJ = args.online, args.out, args.rej
    os.makedirs(OUT, exist_ok=True)

    pn = np.load(args.proto).astype(np.float32)
    pn = pn / (np.linalg.norm(pn, axis=1, keepdims=True) + 1e-8)
    rej = json.load(open(REJ))

    # 索引：scene -> region_index / base_index / sonata
    ridx, bidx, sreg, sbase = {}, {}, {}, {}
    for f in glob.glob(REGION_IDX + '/regions_index_train_s*.pt'):
        for sc, r in torch.load(f, map_location='cpu').items():
            ridx[sc] = r
    for f in glob.glob(BASE_IDX + '/base_index_train_s*.pt'):
        for sc, r in torch.load(f, map_location='cpu').items():
            bidx[sc] = r
    for f in glob.glob(SON_REG + '/sonata_regions_train_s*.pt'):
        for sc, r in torch.load(f, map_location='cpu').items():
            sreg[sc] = r['feats']
    for f in glob.glob(SON_BASE + '/sonata_base_train_s*.pt'):
        for sc, r in torch.load(f, map_location='cpu').items():
            sbase[sc] = r['feats']
    print('索引: region %d base %d sonata %d/%d'
          % (len(ridx), len(bidx), len(sreg), len(sbase)), flush=True)

    scenes = sorted(os.listdir(ONLINE))
    if args.limit:
        scenes = scenes[:args.limit]
    print('场景 %d' % len(scenes), flush=True)

    # ---------- 第一遍：算所有 PL 点（PS+AI）的 sonata 分数 ----------
    # 同时选出 reject 池里 top-BUDGET 点（按类内 z，仅 reject 区域内部排名）
    per_scene = {}
    reject_cand = []          # (scene, region_idx_global_key, pts_idx, cls, score)
    for i, fn in enumerate(scenes):
        sc = fn[:-4]
        z = np.load(os.path.join(ONLINE, fn))
        ps, psai = z['ps'], z['psai']
        n = len(ps)
        score = np.full(n, np.nan, dtype=np.float32)
        # novel 区域
        if sc in ridx and sc in sreg:
            r = ridx[sc]
            S = sreg[sc].astype(np.float32)
            S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
            off, idxs = r['off'], r['idx']
            for k in range(len(off) - 1):
                seg = idxs[off[k]:off[k + 1]]
                c = int(r['pred'][off[k]] if len(r['pred']) > off[k] else -1)
                if c < NO or len(seg) == 0:
                    continue
                score[seg] = S[k] @ pn[c - NO]
                if rej.get(sc) and c in rej[sc]:
                    reject_cand.append((sc, k, seg.copy(), c, float(S[k] @ pn[c - NO])))
        # base 区域
        if sc in bidx and sc in sbase:
            r = bidx[sc]
            S = sbase[sc].astype(np.float32)
            S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
            off, idxs = r['off'], r['idx']
            for k in range(len(off) - 1):
                seg = idxs[off[k]:off[k + 1]]
                # 只给已被 AI 赋类的点算分（class 由点决定）
                v = psai[seg]
                ok = v >= NO
                if not ok.any():
                    continue
                for c in np.unique(v[ok]):
                    sel = seg[ok][v[ok] == c]
                    score[sel] = S[k] @ pn[int(c) - NO]
        per_scene[sc] = (ps, psai, score)
        if (i + 1) % 200 == 0:
            print('  第一遍 %d/%d' % (i + 1, len(scenes)), flush=True)

    # ---------- 选 reject 池 top-BUDGET ----------
    print('reject 候选区域 %d' % len(reject_cand), flush=True)
    if reject_cand:
        cls_arr = np.array([x[3] for x in reject_cand])
        sc_arr = np.array([x[4] for x in reject_cand], dtype=np.float32)
        zc = np.zeros(len(sc_arr), dtype=np.float32)
        for c in range(NO, NO + NN):
            m = cls_arr == c
            if m.any():
                zc[m] = zscore(sc_arr[m])
        order = np.argsort(-zc, kind='stable')
        cum, chosen = 0, []
        for j in order:
            if cum >= BUDGET:
                break
            chosen.append(j)
            cum += len(reject_cand[j][2])
        print('选中 %d 区域 / %d 点' % (len(chosen), cum), flush=True)
    else:
        chosen = []

    # 补标标签：scene -> point -> class
    backfill = {}
    for j in chosen:
        sc, k, seg, c, s = reject_cand[j]
        backfill.setdefault(sc, []).append((seg, c))

    # ---------- 第二遍：合并 z、写文件 ----------
    n_ps = n_ai = n_bf = 0
    for i, fn in enumerate(scenes):
        sc = fn[:-4]
        ps, psai, score = per_scene[sc]
        lab = psai.copy()
        is_bf = np.zeros(len(lab), dtype=bool)
        if sc in backfill:
            for seg, c in backfill[sc]:
                sel = seg[lab[seg] < 0]
                lab[sel] = c
                is_bf[sel] = True
        is_ps = (ps >= NO)
        is_ai = (ps < NO) & (psai >= NO)
        n_ps += int(is_ps.sum()); n_ai += int(is_ai.sum())
        n_bf += int((is_bf & ~is_ai).sum())

        s = score.copy()
        miss = ~np.isfinite(s)
        # 无区域覆盖的点：用同类已有点的均值分填充
        for c in range(NO, NO + NN):
            m = (lab == c) & ~miss
            if m.any():
                s[(lab == c) & miss] = s[m].mean()
        s[~np.isfinite(s)] = 0.0

        z = np.zeros(len(lab), dtype=np.float32)
        for c in range(NO, NO + NN):
            m = lab == c
            if m.sum() > 1:
                z[m] = zscore(s[m])

        w = np.ones(len(lab), dtype=np.float32)
        sig = lambda v: 1.0 / (1.0 + np.exp(-args.sharp * v))
        w[is_ps] = args.ps_lo + (args.ps_hi - args.ps_lo) * sig(z[is_ps])
        soft = is_ai | (is_bf & ~is_ai)
        w[soft] = (args.ai_lo + (args.ai_hi - args.ai_lo) * sig(z[soft])) * args.ai_global
        # 未标注点（lab<0）权重仍是 1，但 label=-1 会被 ignore
        np.save(os.path.join(OUT, sc + '_pl.npy'), lab.astype(np.int16))
        np.save(os.path.join(OUT, sc + '_w.npy'), w)
        if (i + 1) % 200 == 0:
            print('  第二遍 %d/%d' % (i + 1, len(scenes)), flush=True)

    print('PS %d  AI %d  补标 %d  合计写入 %.2f M'
          % (n_ps, n_ai, n_bf, (n_ps + n_ai + n_bf) / 1e6))
    print('->', OUT)


if __name__ == '__main__':
    main()
