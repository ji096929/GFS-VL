"""逐类点数对比：A 的离线伪标签 vs PS keep vs base 池 AI 补标。

输出
  ps_full[45]    PS keep 全部区域展开的点数（按 VLM 判的类计）
  base_ai[45]    base 池（Y_b==-1 过滤后）用区域特征 δ>=0.9 判出的点数
  A[45]          实际参与训练的 A 组点级伪标签点数
  B[45]          实际参与训练（并补过 base 池）的 B 组点数

口径说明：ps_full / base_ai 是区域级（整区域展开），A/B 是点级文件精确统计。
"""
import glob
import os

import numpy as np
import torch

REGION = '/data01/jh/data/region_cache'
BASE = '/data01/jh/data/base_cache'
BIDX = '/data01/jh/data/base_index'
PROTO = '/data01/jh/data/support_protos'
PL = '/data01/jh/data/offline_pl'
NO, NN = 12, 45
PS_THRESH = 0.6
AI_THRESH = 0.9


def load_regions():
    scenes = []
    for f in sorted(glob.glob(os.path.join(REGION, 'regions_train_s*.pt'))):
        d = torch.load(f, map_location='cpu')
        for sc, r in sorted(d.items()):
            scenes.append(dict(name=sc,
                               pred=r['pred'].astype(np.int64),
                               sizes=r['sizes'].astype(np.int64),
                               feats=r['feats'].astype(np.float32)))
    return scenes


def main():
    proto = np.load(os.path.join(PROTO, 'sc200_regis_1_10.npy')).astype(np.float32)
    pn = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)

    # ---- PS keep ----
    scenes = load_regions()
    ps_cnt = np.zeros(NN, dtype=np.int64)
    for sc in scenes:
        for c in np.unique(sc['pred']):
            c = int(c)
            if c < NO:
                continue
            m = sc['pred'] == c
            w = sc['sizes'][m].astype(np.float64)
            pooled = (sc['feats'][m] * w[:, None]).sum(0) / w.sum()
            pooled /= np.linalg.norm(pooled) + 1e-8
            if float(pooled @ pn[c - NO]) >= PS_THRESH:
                ps_cnt[c - NO] += int(sc['sizes'][m].sum())
    print('region_cache 场景 %d 个；PS keep 合计 %.2f M 点'
          % (len(scenes), ps_cnt.sum() / 1e6), flush=True)

    # ---- base 池 AI（区域级）----
    F, SZ, HN, IGN = [], [], [], []
    for f in sorted(glob.glob(os.path.join(BASE, 'base_train_s*.pt'))):
        d = torch.load(f, map_location='cpu')
        for sc, r in d.items():
            F.append(r['feats'].astype(np.float32))
            SZ.append(r['sizes'].astype(np.int64))
            HN.append(r['hist_n'].astype(np.int32))
            IGN.append(r['n_ign'].astype(np.int64))
    F = np.concatenate(F, axis=0)
    SZ = np.concatenate(SZ)
    HN = np.concatenate(HN, axis=0)
    IGN = np.concatenate(IGN)
    unlab = HN.sum(1) + IGN
    sel = unlab >= 0.5 * SZ
    print('base 池 %d 区 %.2f M 点 -> Y_b==-1 过滤 %d 区 %.2f M 点'
          % (len(F), SZ.sum() / 1e6, int(sel.sum()), SZ[sel].sum() / 1e6), flush=True)
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    best = np.full(len(F), -1.0, dtype=np.float32)
    bestc = np.full(len(F), -1, dtype=np.int32)
    for st in range(0, len(F), 8192):
        en = min(st + 8192, len(F))
        s1 = Fn[st:en] @ pn.T
        best[st:en] = s1.max(1)
        bestc[st:en] = s1.argmax(1) + NO
    ai_sel = (best >= AI_THRESH) & sel
    base_cnt = np.bincount(bestc[ai_sel] - NO, minlength=NN).astype(np.int64)
    # 点数按区域 size 累加
    base_cnt = np.zeros(NN, dtype=np.int64)
    for c, sz in zip(bestc[ai_sel] - NO, SZ[ai_sel]):
        base_cnt[c] += int(sz)
    print('base 池 AI 选中 %d 区 %.2f M 点' % (int(ai_sel.sum()), base_cnt.sum() / 1e6), flush=True)

    # ---- A / B 点级统计 ----
    def pl_cnt(tag):
        out = np.zeros(NN, dtype=np.int64)
        files = glob.glob(os.path.join(PL, 'sc200_regis_1_10_' + tag, '*.npy'))
        for f in files:
            a = np.load(f)
            v = a[a >= NO]
            if len(v):
                out += np.bincount(v - NO, minlength=NN)[:NN]
        return out, len(files)
    cnt_a, na = pl_cnt('A')
    cnt_b, nb = pl_cnt('B')
    print('A: %d 文件 %.2f M 点 | B: %d 文件 %.2f M 点'
          % (na, cnt_a.sum() / 1e6, nb, cnt_b.sum() / 1e6), flush=True)

    NAMES = ['trash can', 'ceiling', 'doorframe', 'object', 'shelf', 'sink',
             'picture', 'backpack', 'couch', 'box', 'pillow', 'radiator',
             'mirror', 'whiteboard', 'lamp', 'toilet', 'book', 'monitor',
             'towel', 'tv', 'clothes', 'coffee table', 'office chair',
             'nightstand', 'bag', 'dresser', 'toilet paper', 'recycling bin',
             'kitchen cabinet', 'bathtub', 'telephone', 'plant', 'stool',
             'keyboard', 'shoe', 'jacket', 'shower curtain', 'armchair',
             'microwave', 'computer tower', 'bathroom vanity', 'kitchen counter',
             'shower wall', 'paper towel dispenser', 'file cabinet']
    print('\n%-24s %10s %10s %10s %10s %9s'
          % ('class', 'ps_full', 'base_ai', 'A', 'B', 'A-B'))
    for i in range(NN):
        print('%-24s %10d %10d %10d %10d %+9d'
              % (NAMES[i], ps_cnt[i], base_cnt[i], cnt_a[i], cnt_b[i],
                 cnt_a[i] - cnt_b[i]))
    print('\n合计  ps_full %.2fM  base_ai %.2fM  A %.2fM  B %.2fM'
          % (ps_cnt.sum() / 1e6, base_cnt.sum() / 1e6,
             cnt_a.sum() / 1e6, cnt_b.sum() / 1e6))


if __name__ == '__main__':
    main()
