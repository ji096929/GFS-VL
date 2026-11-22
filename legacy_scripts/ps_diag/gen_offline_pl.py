"""生成点级离线伪标签（最优方案：Sonata z>=0 提纯 + G3z 补 C 池）。

输出 /data01/jh/data/offline_pl/{split}/{scene}.npy
  int16[N]，值为 GFS novel 类 id (12..56) 或 -1（不标）

区域顺序与 region_cache / region_index 完全一致（已验证区域数逐片对齐）。
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
INDEX = '/data01/jh/data/region_index'
SONATA = '/data01/jh/data/sonata_region'
PROTO = '/data01/jh/data/support_protos'
OUT = '/data01/jh/data/offline_pl'
NO, NN = 12, 45
PS_THRESH = 0.6
BUDGET = 1000000


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-8 else np.zeros_like(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='sc200_regis_1_10')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    outdir = args.out or os.path.join(OUT, args.split)
    os.makedirs(outdir, exist_ok=True)

    proto = np.load(os.path.join(PROTO, args.split + '.npy')).astype(np.float32)
    pn = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
    sp = np.load(os.path.join(PROTO, args.split + '_sonata.npy')).astype(np.float32)
    spn = sp / (np.linalg.norm(sp, axis=1, keepdims=True) + 1e-8)

    # ---- 汇总所有区域（保持分片顺序）----
    scene_names, scene_pred, scene_feat, scene_sz = [], [], [], []
    scene_son = []
    son = {}
    for f in sorted(glob.glob(os.path.join(SONATA, 'sonata_regions_train_s*.pt'))):
        d = torch.load(f, map_location='cpu')
        for sc, r in d.items():
            son[sc] = r['feats'].astype(np.float32)

    for f in sorted(glob.glob(os.path.join(REGION, 'regions_train_s*.pt'))):
        d = torch.load(f, map_location='cpu')
        for sc, r in sorted(d.items()):
            n = len(r['pred'])
            if sc not in son or len(son[sc]) != n:
                print('skip %s: cache %d vs sonata %d'
                      % (sc, n, len(son[sc]) if sc in son else -1))
                continue
            scene_names.extend([sc] * n)
            scene_pred.append(r['pred'].astype(np.int64))
            scene_feat.append(r['feats'].astype(np.float32))
            scene_sz.append(r['sizes'].astype(np.int64))
            scene_son.append(son[sc])
    pred = np.concatenate(scene_pred)
    feat = np.concatenate(scene_feat, axis=0)
    sz = np.concatenate(scene_sz)
    sonf = np.concatenate(scene_son, axis=0)
    N = len(pred)
    print('regions %d  sonata %d' % (N, len(sonf)), flush=True)

    fvn = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
    svn = sonf / (np.linalg.norm(sonf, axis=1, keepdims=True) + 1e-8)

    # ---- PS 决策 ----
    # 场景级 pooled cos：按 (scene, class) 聚合
    key = np.array([hash((s, int(c))) for s, c in zip(scene_names, pred)])
    scene_cls = {}
    for i in range(N):
        scene_cls.setdefault((scene_names[i], int(pred[i])), []).append(i)
    keep = np.zeros(N, dtype=bool)
    for (s, c), idxs in scene_cls.items():
        idxs = np.array(idxs)
        w = sz[idxs].astype(np.float64)
        pooled = (feat[idxs] * w[:, None]).sum(0) / w.sum()
        pooled /= np.linalg.norm(pooled) + 1e-8
        if float(pooled @ pn[c - NO]) >= PS_THRESH:
            keep[idxs] = True
    print('keep %d  reject %d' % (int(keep.sum()), int((~keep).sum())))

    # ---- D（仅 base 真值），同时取 VLM 与 Sonata 两套特征 ----
    Dsf = []
    for f in sorted(glob.glob(os.path.join(REGION, 'regions_train_s*.pt'))):
        d = torch.load(f, map_location='cpu')
        for sc, r in d.items():
            if sc not in son:
                continue
            for i in range(len(r['pred'])):
                vp = int(r['sizes'][i]) - int(r['n_ign'][i])
                gb = r['hist_b'][i]
                if vp > 0 and gb.sum() > 0 and int(gb.max()) >= 0.5 * vp:
                    Dsf.append(son[sc][i])
    Dn = np.stack(Dsf)
    Dn = Dn / (np.linalg.norm(Dn, axis=1, keepdims=True) + 1e-8)
    print('D %d (sonata %d 维)' % (len(Dn), Dn.shape[1]))

    # ---- Sonata 侧：B 池 score = s1 - max_D ----
    s1 = np.zeros(N, dtype=np.float32)
    for c in range(NO, NO + NN):
        m = pred == c
        if m.any():
            s1[m] = svn[m] @ spn[c - NO]
    dm = np.empty(N, dtype=np.float32)
    for st in range(0, N, 4096):
        en = min(st + 4096, N)
        dm[st:en] = (svn[st:en] @ Dn.T).max(1)
    mg = s1 - dm
    z = np.zeros(N, dtype=np.float32)
    for c in range(NO, NO + NN):
        m = pred == c
        if m.any():
            z[m] = zscore(mg[m])

    # ---- C 池 G3z ----
    cidx = np.where(~keep)[0]
    # B 池（keep）的 Sonata 特征
    kf = svn[keep]
    kc = pred[keep]
    s2 = np.zeros(len(cidx), dtype=np.float32)
    for c in range(NO, NO + NN):
        m = pred[cidx] == c
        if not m.any():
            continue
        Bm = kf[kc == c]
        s2[m] = (svn[cidx][m] @ Bm.T).max(1) if len(Bm) else -1
    dcm = np.empty(len(cidx), dtype=np.float32)
    for st in range(0, len(cidx), 4096):
        en = min(st + 4096, len(cidx))
        dcm[st:en] = (svn[cidx][st:en] @ Dn.T).max(1)
    zc = np.zeros(len(cidx), dtype=np.float32)
    tmp = s2 - dcm
    for c in range(NO, NO + NN):
        m = pred[cidx] == c
        if m.any():
            zc[m] = zscore(tmp[m])

    # ---- 最终选择 ----
    sel = np.zeros(N, dtype=bool)
    sel[keep] = z[keep] >= 0
    order = np.argsort(-zc, kind='stable')
    cum = np.cumsum(sz[cidx][order])
    k = int(np.searchsorted(cum, BUDGET))
    k = min(k + 1, len(order))
    sel[cidx[order[:k]]] = True
    print('选中 %d 区域 / %d 点' % (int(sel.sum()), int(sz[sel].sum())))

    # ---- 写点级伪标签 ----
    # scene_names 按场景连续排列；用实际加载的顺序建索引（跳过不一致场景后仍对齐）
    per_scene = {}
    i = 0
    while i < len(scene_names):
        sc = scene_names[i]
        j = i
        while j < len(scene_names) and scene_names[j] == sc:
            j += 1
        per_scene[sc] = (i, j - i)
        i = j

    # 读 region_index 得到点索引
    for f in sorted(glob.glob(os.path.join(INDEX, 'regions_index_train_s*.pt'))):
        d = torch.load(f, map_location='cpu')
        for sc, r in d.items():
            if sc not in per_scene:
                continue
            off, n = per_scene[sc]
            if n != len(r['off']) - 1:
                print('!! %s 区域数不一致 %d vs %d' % (sc, n, len(r['off']) - 1))
                continue
            pl = np.full(len(r['pred']), -1, dtype=np.int16)
            for i in range(n):
                if not sel[off + i]:
                    continue
                seg = r['idx'][r['off'][i]:r['off'][i + 1]]
                if len(seg):
                    pl[seg] = pred[off + i]
            np.save(os.path.join(outdir, sc + '.npy'), pl)
    print('->', outdir, 'files', len(os.listdir(outdir)))


if __name__ == '__main__':
    main()
