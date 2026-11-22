"""逐场景 dump 官方原版在线 PS + 逐点 AI 的点级伪标签。

口径（对齐 pointcept/engines/train.py 的 target_calibrate）
-----------------------------------------------------------
* 整场景（不做训练时的随机 SphereCrop，所以是「全集」版本）
* PS：场景内每个 VLM 预测类 c
    - c < 12（基类预测）→ 全部置 -1
    - c >= 12 → proto = 该场景内被判为 c 的所有点的【原始】特征均值，
                cos(proto, 支持原型[c]) < 0.6 则整类置 -1
* AI：mask = (pred == -1) & (label == -1)
    - 每场景内，对背景区（label == -1）中 pred==c 的点算【原始】特征均值作为当前原型
    - 缺则该类退回支持原型
    - 逐点 cos >= 0.9 赋类，否则保持 -1
* 只统计「写入 Y_b == -1 的点」

输出
----
/data01/jh/data/online_pl/{scene}.npz       ps int16[N] / psai int16[N]
/data01/jh/data/online_pl_shard{s}.npz      逐类计数与混淆统计
/data01/jh/data/online_pl_rej_s{s}.json     每场景被 PS 拒绝的类（供 G3z 用）
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/data01/jh/code/ps_diag')
import vlm_common as V  # noqa: E402

DATA = '/data01/jh/data/scannet200_gfs'
PROTO = '/data01/jh/data/support_protos/sc200_regis_1_10.npy'
OUT = '/data01/jh/data/online_pl'
NO, NN = 12, 45
PS_THRESH = 0.6
AI_THRESH = 0.9


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def process(model, cfg, pn, scene):
    d = os.path.join(DATA, 'train', scene)
    coord = np.load(os.path.join(d, 'coord.npy')).astype(np.float32)
    color = np.load(os.path.join(d, 'color.npy'))
    seg200 = np.load(os.path.join(d, 'segment200.npy')).reshape(-1)
    gfs = V.convert_to_gfs(seg200)                    # GT：0..11 base, 12.. novel, -1 ignore
    lab = np.where((gfs >= 0) & (gfs < NO), gfs, -1)  # 训练标签：novel 一律 -1

    feats, raw = V.vlm_forward(model, cfg, coord, color)
    fn = unit(feats)

    # ---------- PS（原始特征求均值，与 torch.cosine_similarity 等价）----------
    pred = raw.astype(np.int16).copy()
    rej_cls = []
    for c in np.unique(pred):
        c = int(c)
        m = pred == c
        if c < NO:
            pred[m] = -1
            continue
        proto = unit(feats[m].mean(0))
        if float(proto @ pn[c - NO]) < PS_THRESH:
            pred[m] = -1
            rej_cls.append(c)
    ps = pred.copy()
    reject_pts = int(((raw >= NO) & (ps == -1)).sum())

    # ---------- AI ----------
    bg = lab == -1
    bg_pred = ps[bg]
    bg_feats = feats[bg]
    cur = {}
    for c in np.unique(bg_pred):
        c = int(c)
        if c < NO:
            continue
        cur[c] = bg_feats[bg_pred == c].mean(0)
    protos = np.empty((NN, feats.shape[1]), dtype=np.float32)
    for i in range(NN):
        p = cur.get(i + NO)
        protos[i] = p if p is not None else pn[i]
    protos = unit(protos)

    mask = (ps == -1) & bg
    psai = ps.copy()
    if mask.any():
        sim = fn[mask] @ protos.T
        best = sim.max(1)
        cls = sim.argmax(1) + NO
        psai[mask] = np.where(best >= AI_THRESH, cls, -1).astype(np.int16)

    np.savez_compressed(os.path.join(OUT, scene + '.npz'),
                        ps=ps.astype(np.int16), psai=psai.astype(np.int16))

    # ---------- 统计（只统计写入 Y_b==-1 的点）----------
    out = {}
    for tag, pl in (('ps', ps), ('psai', psai)):
        v = pl[bg]
        ok = v >= NO
        vv = v[ok]
        out[tag + '_cnt'] = np.bincount(vv - NO, minlength=NN)[:NN].astype(np.int64)
        g = gfs[bg][ok]
        hit = np.zeros(NN, dtype=np.int64)
        m2 = (g >= NO) & (g == vv)
        hit += np.bincount(vv[m2] - NO, minlength=NN)[:NN]
        out[tag + '_hit'] = hit
    add = (ps == -1) & (psai >= NO) & bg
    v = psai[add]
    out['ai_cnt'] = np.bincount(v - NO, minlength=NN)[:NN].astype(np.int64)
    g = gfs[add]
    m2 = (g >= NO) & (g == v)
    out['ai_hit'] = np.bincount(v[m2] - NO, minlength=NN)[:NN].astype(np.int64)
    out['gt'] = np.bincount(gfs[gfs >= NO] - NO, minlength=NN)[:NN].astype(np.int64)
    out['reject_pts'] = np.array([reject_pts], dtype=np.int64)
    return out, rej_cls


def main():
    global PROTO, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--proto', default=PROTO,
                    help='支持集原型 npy（多 support 种子各一份）')
    ap.add_argument('--out', default=OUT,
                    help='逐场景 npz 输出目录')
    ap.add_argument('--tag', default='',
                    help='分片统计与拒绝明细文件名后缀，如 _s20')
    args = ap.parse_args()
    PROTO, OUT = args.proto, args.out
    os.makedirs(OUT, exist_ok=True)

    scenes = sorted(os.listdir(os.path.join(DATA, 'train')))
    mine = scenes[args.shard::args.shards]
    print('shard %d/%d: %d 场景' % (args.shard, args.shards, len(mine)), flush=True)

    pn = unit(np.load(PROTO).astype(np.float32))
    cfg, model, _, _, _ = V.build_vlm()
    print('[vlm] ready', flush=True)

    keys = ['ps_cnt', 'ps_hit', 'psai_cnt', 'psai_hit', 'ai_cnt', 'ai_hit', 'gt']
    agg = {k: np.zeros(NN, dtype=np.int64) for k in keys}
    agg['reject_pts'] = np.zeros(1, dtype=np.int64)
    rej = {}
    for i, sc in enumerate(mine):
        try:
            out, rc = process(model, cfg, pn, sc)
        except Exception as e:  # noqa: BLE001
            print('FAIL %s %s' % (sc, e), flush=True)
            continue
        for k in agg:
            agg[k] = agg[k] + out[k]
        if rc:
            rej[sc] = rc
        if (i + 1) % 50 == 0:
            print('  %d/%d' % (i + 1, len(mine)), flush=True)

    np.savez('/data01/jh/data/online_pl_shard%s%d.npz' % (args.tag, args.shard), **agg)
    with open('/data01/jh/data/online_pl_rej%s_s%d.json' % (args.tag, args.shard), 'w') as f:
        json.dump(rej, f)
    print('-> shard %d done, reject_pts=%d'
          % (args.shard, int(agg['reject_pts'][0])), flush=True)


if __name__ == '__main__':
    main()
