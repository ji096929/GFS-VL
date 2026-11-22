"""Task 2：在【真实在线 PS+AI 底盘】上验证两个改动。

(a) Sonata + D_base 对真实底盘写入点的排序区分度
    - 用 PS 保留区域（kept）做评估集：纯度 = hist_n[c] / (size - hist_b)
    - score = max_cos(sonata, 同类 B 池) - max_D，类内 z
    - 输出 AUC 与 z 分桶精度
(b) 在真实 PS 拒绝区上叠加 G3z
    - reject 集合来自 online_pl_rej_s0.json（在线 PS 逐场景拒绝的类）
    - 选 top-K 点（按类内 z），加到 PS 或 PS+AI 的统计上，重算 mIoU-N

区域级近似说明
  region_cache / sonata 都是连通分量级；写入点数按 size - hist_b（Y_b>=0 的点不写），
  命中数按 hist_n[c]。这与点级口径有 <2% 的差异。
"""
import glob
import json
import os
import sys

import numpy as np
import torch  # noqa: E402

sys.path.insert(0, '/data01/jh/code/ps_diag')
import vlm_common as V  # noqa: E402

REGION = '/data01/jh/data/region_cache'
SONATA = '/data01/jh/data/sonata_region'
PROTO = '/data01/jh/data/support_protos'
REJ = '/data01/jh/data/online_pl_rej_s0.json'
SHARD = '/data01/jh/data/online_pl_shard0.npz'
NO, NN = 12, 45
NAMES = V.CLASS_NAMES[NO:]


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-8 else np.zeros_like(x)


def miou(P, I, G):
    ious = []
    for c in range(NN):
        p, it, g = P[c], I[c], G[c]
        if g == 0 and p == 0:
            continue
        u = p + g - it
        ious.append(it / u if u > 0 else 0.0)
    return float(np.mean(ious)), len(ious)


def main():
    rej = json.load(open(REJ))
    print('被 PS 拒绝过的场景数 %d' % len(rej), flush=True)

    scenes = {}
    for f in sorted(glob.glob(os.path.join(REGION, 'regions_train_s*.pt'))):
        d = torch.load(f, map_location='cpu')
        for sc, r in d.items():
            scenes[sc] = r
    son = {}
    for f in sorted(glob.glob(os.path.join(SONATA, 'sonata_regions_train_s*.pt'))):
        d = torch.load(f, map_location='cpu')
        for sc, r in d.items():
            son[sc] = r['feats'].astype(np.float32)
    print('region_cache %d 场景, sonata %d 场景' % (len(scenes), len(son)), flush=True)

    sp = np.load(os.path.join(PROTO, 'sc200_regis_1_10_sonata.npy')).astype(np.float32)
    spn = sp / (np.linalg.norm(sp, axis=1, keepdims=True) + 1e-8)

    # 收集区域级信息
    R = dict(sc=[], cls=[], sz=[], hn=[], hb=[], fn=[], sn=[])
    for sc, r in scenes.items():
        if sc not in son:
            continue
        S = son[sc]
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
        for i in range(len(r['pred'])):
            R['sc'].append(sc)
            R['cls'].append(int(r['pred'][i]))
            R['sz'].append(int(r['sizes'][i]))
            R['hn'].append(r['hist_n'][i])
            R['hb'].append(r['hist_b'][i])
            R['fn'].append(r['feats'][i])
            R['sn'].append(S[i])
    for k in ('sz', 'cls'):
        R[k] = np.array(R[k])
    for k in ('hn', 'hb', 'fn', 'sn'):
        R[k] = np.stack(R[k])
    n = len(R['cls'])
    print('区域 %d 个' % n, flush=True)

    # 在线 PS 的 keep/reject（按场景+类）
    is_rej = np.zeros(n, dtype=bool)
    for i in range(n):
        rc = rej.get(R['sc'][i])
        if rc is not None and R['cls'][i] in rc:
            is_rej[i] = True
    print('区域级 reject %d 个 / %.2f M 点（点级 reject 3.98 M，差值是 <20 点碎片）'
          % (int(is_rej.sum()), R['sz'][is_rej].sum() / 1e6), flush=True)

    # D_base：base 真值占多数的区域
    hb_sum = R['hb'].sum(1)
    Dn = R['sn'][R['hb'].max(1) >= 0.5 * np.maximum(R['sz'], 1)]
    Dn = Dn / (np.linalg.norm(Dn, axis=1, keepdims=True) + 1e-8)
    print('D_base %d 区域' % len(Dn), flush=True)

    # 每个区域的打分：同类 keep 池最近邻 - max_D
    K = ~is_rej
    s = np.empty(n, dtype=np.float32)
    for c in range(NO, NO + NN):
        m = R['cls'] == c
        if not m.any():
            continue
        kf = R['sn'][m & K]
        dmax = (R['sn'][m] @ Dn.T).max(1) if len(Dn) else 0
        if len(kf) == 0:
            s[m] = -1
        else:
            s[m] = (R['sn'][m] @ kf.T).max(1)
        s[m] = s[m] - dmax
    z = np.zeros(n, dtype=np.float32)
    for c in range(NO, NO + NN):
        m = R['cls'] == c
        if m.any():
            z[m] = zscore(s[m])

    # ---- (a) Sonata 区分度：kept 区域里，纯度 >= 0.5 为正样本 ----
    pur = np.array([R['hn'][i][R['cls'][i] - NO] / max(R['sz'][i] - hb_sum[i], 1)
                    for i in range(n)])
    ok = K & (R['sz'] - hb_sum > 0)
    y = (pur >= 0.5).astype(int)
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y[ok], z[ok])
    except Exception as e:  # noqa: BLE001
        auc = float('nan')
        print('AUC 失败', e)
    print('\n(a) Sonata+D_base 在真实 keep 区域上的区分度')
    print('    样本 %d 区域，正例率 %.3f，AUC %.4f' % (int(ok.sum()), y[ok].mean(), auc))
    qs = np.quantile(z[ok], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for i in range(5):
        m = ok & (z >= qs[i]) & (z <= qs[i + 1])
        if m.any():
            print('    z 分桶 %d/%d: 区域 %6d  平均纯度 %.3f'
                  % (i + 1, 5, int(m.sum()), pur[m].mean()))

    # ---- (b) 在 reject 池上补 G3z ----
    d = np.load(SHARD)
    G = d['gt']
    print('\n(b) 在真实 PS reject 池上补 G3z')
    for base_tag, P0, I0 in (('online PS', d['ps_cnt'], d['ps_hit']),
                             ('online PS+AI', d['psai_cnt'], d['psai_hit'])):
        m0, _ = miou(P0, I0, G)
        print('  底盘 %-14s mIoU-N %.4f  写入 %.2f M'
              % (base_tag, m0, P0.sum() / 1e6))
        order = np.argsort(-z[is_rej], kind='stable')
        idx = np.where(is_rej)[0][order]
        cum = 0
        for budget in (0.5e6, 1.0e6, 2.0e6):
            P, I = P0.astype(np.int64).copy(), I0.astype(np.int64).copy()
            tot = 0
            for j in idx:
                if tot >= budget:
                    break
                add = int(R['sz'][j] - hb_sum[j])
                if add <= 0:
                    continue
                P[R['cls'][j] - NO] += add
                I[R['cls'][j] - NO] += int(R['hn'][j][R['cls'][j] - NO])
                tot += add
            m1, _ = miou(P, I, G)
            print('    + G3z %5.1f 万点 (实际 %.2f M): mIoU-N %.4f (%+0.4f)'
                  % (budget / 1e4, tot / 1e6, m1, m1 - m0))


if __name__ == '__main__':
    main()
