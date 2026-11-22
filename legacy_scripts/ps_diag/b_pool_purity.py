"""诊断：用基类负例增量 + Sonata 跨视角，对脏 B 池（PS keep 区域）排序提纯。

对每个 novel 类 c，B 池 = PS keep 的区域（pred == c）。这批区域的 GT 精度
已经知道很低（ceiling 0.12 / office chair 0.15 / armchair 0.17），所以直接把
它们当"可信桥梁"是有问题的。这里不动 PS 的 keep 决策，只回答一个问题：

  按某个分数把 B 池从 100% 截断到 10/20/30/50%，精度能拉到多少、召回掉多少？

打分变体
--------
ps     cos(B, proto_c)                                PS 原始分数（无排序信息）
margin cos(B, proto_c) - max_{D in base} cos(B, D)    基类负例增量（D 仅用 base 真值）
sonata cos(B_s, proto_s_c) - max_D cos(B_s, D_s)      Sonata 特征上的同一打分
fuse   z(margin) + z(sonata)                          两个视角的标准化融合

指标（按点数加权）
  precision = 保留区域内 GT 多数类 == c 的点 / 保留区域总点数
  recall    = 保留的正确区域点数 / 全部正确区域点数
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/ps_diag')
import vlm_common as V  # noqa: E402

REGION = '/data01/jh/data/region_cache'
SONATA = '/data01/jh/data/sonata_region'
PROTO = '/data01/jh/data/support_protos'
PS_THRESH = 0.6
NO = 12
NN = 45
CLASS_NAMES = V.CLASS_NAMES
FOCUS = ['armchair', 'office chair', 'ceiling', 'nightstand', 'radiator']


def load_regions():
    scenes = []
    for f in sorted(glob.glob(os.path.join(REGION, 'regions_train_s*.pt'))):
        d = torch.load(f, map_location='cpu')
        for sc, r in sorted(d.items()):
            scenes.append(dict(name=sc,
                               pred=r['pred'].astype(np.int64),
                               sizes=r['sizes'].astype(np.int64),
                               feats=r['feats'].astype(np.float32),
                               hist_n=r['hist_n'].astype(np.int64),
                               hist_b=r['hist_b'].astype(np.int64),
                               n_ign=r['n_ign'].astype(np.int64)))
    return scenes


def load_sonata():
    """{scene: feats float16[R,1232]}；区域顺序与 region_cache 一致。"""
    out = {}
    fs = sorted(glob.glob(os.path.join(SONATA, 'sonata_regions_train_s*.pt')))
    for f in fs:
        d = torch.load(f, map_location='cpu')
        for sc, r in d.items():
            out[sc] = r['feats'].astype(np.float32)
    print('sonata 场景 %d' % len(out))
    return out


def gt_major(sc, i):
    hn, hb, ni = sc['hist_n'][i], sc['hist_b'][i], sc['n_ign'][i]
    bn = int(hn.max()) if len(hn) else 0
    bb = int(hb.max()) if len(hb) else 0
    if ni >= max(bn, bb):
        return -1
    return NO + int(hn.argmax()) if bn >= bb else int(hb.argmax())


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-8 else np.zeros_like(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='sc200_regis_1_10')
    ap.add_argument('--ratios', default='0.1,0.2,0.3,0.5,1.0')
    args = ap.parse_args()

    scenes = load_regions()
    proto = np.load(os.path.join(PROTO, args.split + '.npy')).astype(np.float32)
    pn = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)

    sonata = None
    if os.path.isdir(SONATA) and glob.glob(os.path.join(SONATA, '*.pt')):
        try:
            sonata = load_sonata()
        except Exception as e:  # noqa: BLE001
            print('sonata 读取失败：', e)

    # ---- PS 决策：keep 的才是 B 池 ----
    B = []  # (scene, region_idx, class)
    for si, sc in enumerate(scenes):
        for c in np.unique(sc['pred']):
            c = int(c)
            if c < NO:
                continue
            m = sc['pred'] == c
            w = sc['sizes'][m].astype(np.float64)
            pooled = (sc['feats'][m] * w[:, None]).sum(0) / w.sum()
            pooled /= np.linalg.norm(pooled) + 1e-8
            if float(pooled @ pn[c - NO]) >= PS_THRESH:
                B.extend((si, int(i), c) for i in np.where(m)[0])
    bi = np.array([b[0] for b in B])
    ri = np.array([b[1] for b in B])
    bc = np.array([b[2] for b in B])
    bsz = np.array([scenes[s]['sizes'][i] for s, i, c in B], dtype=np.int64)
    bf = np.stack([scenes[s]['feats'][i] for s, i, c in B])
    bfn = bf / (np.linalg.norm(bf, axis=1, keepdims=True) + 1e-8)
    ok = np.array([gt_major(scenes[s], i) == c for s, i, c in B])
    print('B 池 %d 区域  %d 点  真精度(点加权) %.3f'
          % (len(B), bsz.sum(), float((bsz * ok).sum() / bsz.sum())))

    # ---- D：仅用 base 真值 ----
    Df, D_idx = [], []
    for si, sc in enumerate(scenes):
        gb = sc['hist_b']
        for i in range(len(sc['pred'])):
            vp = int(sc['sizes'][i]) - int(sc['n_ign'][i])
            if vp > 0 and gb[i].sum() > 0 and int(gb[i].max()) >= 0.5 * vp:
                Df.append(sc['feats'][i])
                D_idx.append((si, i))
    D = np.stack(Df)
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    print('D_base %d 区域（只用 base 真值）' % len(D))

    # ---- Sonata 侧 ----
    sf = None
    if sonata is not None:
        rows, miss = [], 0
        for s, i, c in B:
            name = scenes[s]['name']
            arr = sonata.get(name)
            if arr is None or i >= len(arr):
                rows.append(None)
                miss += 1
            else:
                rows.append(arr[i])
        if miss:
            print('!! sonata 缺 %d / %d 个 B 区域，跳过 Sonata 列' % (miss, len(B)))
        else:
            sf = np.stack(rows)
            sD = []
            for s, i in D_idx:
                arr = sonata.get(scenes[s]['name'])
                if arr is not None and i < len(arr):
                    sD.append(arr[i])
            sD = np.stack(sD)
            sDn = sD / (np.linalg.norm(sD, axis=1, keepdims=True) + 1e-8)
    if sf is not None:
        sfn = sf / (np.linalg.norm(sf, axis=1, keepdims=True) + 1e-8)
        # Sonata 侧的 support 原型从 support_protos 目录取同名的 *_sonata.npy
        sp_path = os.path.join(PROTO, args.split + '_sonata.npy')
        if os.path.exists(sp_path):
            sp = np.load(sp_path).astype(np.float32)
            spn = sp / (np.linalg.norm(sp, axis=1, keepdims=True) + 1e-8)
        else:
            spn = None
            print('!! 缺 Sonata 支持原型 %s（需要 build_support_protos_sonata.py）'
                  % sp_path)

    # ---- 打分 ----
    scores = {}
    ps = np.zeros(len(B))
    mg = np.zeros(len(B))
    for c in range(NO, NO + NN):
        m = bc == c
        if not m.any():
            continue
        ps[m] = bfn[m] @ pn[c - NO]
    # C x D 分块
    dmax = np.empty(len(B), dtype=np.float32)
    for st in range(0, len(B), 4096):
        en = min(st + 4096, len(B))
        dmax[st:en] = (bfn[st:en] @ Dn.T).max(1)
    mg = ps - dmax
    scores['ps'] = ps
    scores['margin'] = mg

    if sf is not None and spn is not None:
        ss = np.zeros(len(B))
        for c in range(NO, NO + NN):
            m = bc == c
            if not m.any():
                continue
            ss[m] = sfn[m] @ spn[c - NO]
        sd = np.empty(len(B), dtype=np.float32)
        for st in range(0, len(B), 4096):
            en = min(st + 4096, len(B))
            sd[st:en] = (sfn[st:en] @ sDn.T).max(1)
        scores['sonata'] = ss
        scores['sonata_neg'] = ss - sd
        scores['fuse'] = zscore(mg) + zscore(ss - sd)
    elif sf is not None:
        # 没有 support 原型时，退化成「Sonata 特征上的基类负例增量」：
        # 用 D 的 Sonata 特征（需要 D 的索引，暂缺）——这里只报告相似度分散度
        print('（Sonata 支持原型缺失，跳过 sonata/fuse 列）')

    ratios = [float(x) for x in args.ratios.split(',')]
    print('\n=== 重点类：B 池排序提纯（保留比例 -> 精度/召回）===')
    for c in range(NO, NO + NN):
        if CLASS_NAMES[c] not in FOCUS:
            continue
        m = bc == c
        if not m.any():
            print('\n%s：B 池为空' % CLASS_NAMES[c])
            continue
        idx = np.where(m)[0]
        sz, okc = bsz[idx], ok[idx]
        base_prec = (sz * okc).sum() / sz.sum()
        region_prec = float(okc.mean())
        cls_ok = float((sz * okc).sum())     # 该类正确 B 的点数（类内召回分母）
        print('\n%s  n_B=%d  B精度 %.3f(点加权) / %.3f(区域平均)  正确 B 点数 %d'
              % (CLASS_NAMES[c], len(idx), base_prec, region_prec, int(cls_ok)))
        print('  %-8s' % 'score' + ''.join('%16s' % ('r=%.1f' % r)
                                          for r in ratios))
        for tag, sc_ in scores.items():
            row = ''
            for r in ratios:
                k = max(1, int(round(r * len(idx))))
                sub = idx[np.argsort(-sc_[idx], kind='stable')[:k]]
                p = (bsz[sub] * ok[sub]).sum() / max(bsz[sub].sum(), 1)
                rc = (bsz[sub] * ok[sub]).sum() / max(cls_ok, 1)
                row += '%16s' % ('%.3f/%.2f' % (p, rc))
            print('  %-8s' % tag + row)

    # ---- 全局 ----
    print('\n=== 45 类合计：B 池排序提纯 ===')
    print('  %-8s' % 'score' + ''.join('%14s' % ('r=%.1f' % r)
                                      for r in ratios))
    tot_ok = (bsz * ok).sum()
    for tag, sc_ in scores.items():
        row = ''
        for r in ratios:
            # 每类各取前 r 比例，保证小类不被大类淹没
            sel = []
            for c in range(NO, NO + NN):
                idx = np.where(bc == c)[0]
                if len(idx) == 0:
                    continue
                k = max(1, int(round(r * len(idx))))
                sel.append(idx[np.argsort(-sc_[idx], kind='stable')[:k]])
            sel = np.concatenate(sel)
            p = (bsz[sel] * ok[sel]).sum() / max(bsz[sel].sum(), 1)
            rc = (bsz[sel] * ok[sel]).sum() / max(tot_ok, 1)
            row += '%14s' % ('%.3f/%.2f' % (p, rc))
        print('  %-8s' % tag + row)


if __name__ == '__main__':
    main()
