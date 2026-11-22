"""Exp1-low ep1 的推理时 U 消融 + 训练前后 U 上的桥梁质量。

对 357 个任务，Exp1-low ep1 模型分别在：
  official  官方 forward，不传 U（= 之前 0.4153 的口径）
  su_l0     forward_su 传 U 但 λ=0（U 前向但融合权重为 0）
  su_l03    forward_su 传真 U，λ=0.3
  su_noise  forward_su 传高斯噪声 U，λ=0.3
另在 U 块上用 S 原型预测，算 B={p>0.7} 的纯度和池内 AUC，
对比训练前（官方权重）与训练后（Exp1-low ep1）。
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, '/data01/jh/code')
import fake_tp  # noqa: F401

sys.path.insert(0, '/data01/jh/code/COSeg')
sys.path.insert(0, '/data01/jh/code')

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

from util import config  # noqa: E402
from util.data_util import data_prepare_v101  # noqa: E402
from model.coseg import COSeg  # noqa: E402
from coseg_su import COSegSU  # noqa: E402

CFG = '/data01/jh/code/COSeg/config/scannetv2_COSeg_fs.yaml'
WEIGHT = '/data01/jh/weights/sc0_1w1s.pth'
ST2 = '/data01/jh/afford_diag/stage2'
BLK = '/data01/jh/data/scannet_train_v2/blocks_bs1.0_s1.0/data'
DUMP = '/data01/jh/data/sonata_dump'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
BLOCKS = sorted(os.listdir(BLK))


def build_cfg():
    cfg = config.load_cfg_from_cfg_file(CFG)
    cfg.cvfold = 0
    cfg.n_way = 1
    cfg.k_shot = 1
    cfg.n_queries = 1
    cfg.save_path = os.path.join(ST2, 'tmp_u')
    cfg.ngpus_per_node = 1
    cfg.rank = 0
    cfg.multiprocessing_distributed = False
    cfg.distributed = False
    os.makedirs(cfg.save_path, exist_ok=True)
    return cfg


def load_model(path, su=False):
    m = (COSegSU if su else COSeg)(build_cfg()).cuda()
    ck = torch.load(path, map_location='cpu')
    sd = ck.get('model', ck.get('state_dict', ck))
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def make_u(gid, noise=False):
    rng = random.Random(100000 + gid)
    name = rng.choice(BLOCKS)
    r = np.load(os.path.join(BLK, name))
    coord, feat, lab = data_prepare_v101(
        r[:, :3].astype(np.float32), r[:, 3:6].astype(np.float32),
        r[:, 6].astype(np.int64), 'train', 0.02, 20480, None, True, 0)
    x = torch.cat([coord, feat], 1)
    if noise:
        x = torch.randn_like(x) * x.std(0) + x.mean(0)
    return x.cuda(), torch.tensor([x.shape[0]], dtype=torch.int32), lab


@torch.no_grad()
def predict(m, so, sx, sy, qo, qx, qy, c):
    with torch.cuda.amp.autocast():
        out, _ = m(so, sx, sy, qo, qx, qy, 5,
                   support_base_y=None, query_base_y=None,
                   sampled_classes=np.array([c]))
    return F.softmax(out[0].t().float(), dim=1)[:, 1].cpu().numpy()


@torch.no_grad()
def predict_su(m, so, sx, sy, uo, ux, qo, qx, qy, c, lam):
    with torch.cuda.amp.autocast():
        out, _ = m.forward_su(so, sx, sy, uo, ux, qo, qx, qy, 5,
                              support_base_y=None, query_base_y=None,
                              sampled_classes=np.array([c]), lam=lam)
    return F.softmax(out[0].t().float(), dim=1)[:, 1].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(ST2, 'uinfer_s%d.json'))
    args = ap.parse_args()
    out = args.out % args.shard if '%d' in args.out else args.out

    files = sorted(glob.glob(os.path.join(DUMP, 'ep*.npz')))
    files = [f for f in files
             if int(os.path.basename(f)[2:5]) % args.shards == args.shard]
    print('tasks', len(files), flush=True)

    model = load_model(os.path.join(ST2, 'exp1_lowlr', 'ep1.pth'), su=True)
    base = load_model(WEIGHT, su=False)
    res = {}
    for i, f in enumerate(files):
        d = np.load(f)
        gid = int(os.path.basename(f)[2:5])
        c = int(d['cls'][0])
        sx = torch.cat([
            torch.from_numpy(d['s_coord'].astype(np.float32)),
            torch.from_numpy(d['s_rgb'].astype(np.float32)) / 255.0], 1).cuda()
        qx = torch.cat([
            torch.from_numpy(d['q_coord'].astype(np.float32)),
            torch.from_numpy(d['q_rgb'].astype(np.float32)) / 255.0], 1).cuda()
        sy = torch.from_numpy(d['s_lab'].astype(np.int64)).cuda()
        qy = torch.from_numpy(d['q_lab'].astype(np.int64)).cuda()
        so = torch.tensor([len(sx)], dtype=torch.int32)
        qo = torch.tensor([len(qx)], dtype=torch.int32)
        gt = d['q_lab'].astype(bool)

        ux, uo, ulab = make_u(gid, noise=False)
        nx, no, _ = make_u(gid, noise=True)

        def rec(p):
            pred = p > 0.5
            return dict(cls=c,
                        tp=int((pred & gt).sum()),
                        fp=int((pred & ~gt).sum()),
                        fn=int((~pred & gt).sum()),
                        iou=float((pred & gt).sum()
                                  / max(int((pred | gt).sum()), 1)),
                        predfg=float(pred.mean()))

        row = dict(gid=gid, cls=c)
        row['official'] = rec(predict(model, so, sx, sy, qo, qx, qy, c))
        row['su_l0'] = rec(predict_su(model, so, sx, sy, uo, ux, qo, qx, qy,
                                      c, 0.0))
        row['su_l03'] = rec(predict_su(model, so, sx, sy, uo, ux, qo, qx, qy,
                                       c, 0.3))
        row['su_noise'] = rec(predict_su(model, so, sx, sy, no, nx, qo, qx, qy,
                                         c, 0.3))

        # U 块上的桥梁质量（统一用官方 forward：S 原型直接预测 U）
        ug = (ulab == c).cpu().numpy()
        ug_t = torch.from_numpy(ug.astype(np.int64)).cuda()
        for tag, m in (('u_base', base), ('u_exp1', model)):
            pu = predict(m, so, sx, sy, uo, ux, ug_t, c)
            B = pu > 0.7
            npos = int(ug[B].sum())
            nneg = int((~ug[B]).sum())
            row[tag] = dict(
                Bn=int(B.sum()),
                pure=float(ug[B].mean()) if B.any() else -1.0,
                auc=float(mannwhitneyu(
                    pu[B][ug[B]], pu[B][~ug[B]],
                    alternative='greater').statistic / (npos * nneg))
                if npos > 0 and nneg > 0 else -1.0)
        res[gid] = row
        if (i + 1) % 20 == 0:
            print('  %d / %d' % (i + 1, len(files)), flush=True)
    json.dump(res, open(out, 'w'))
    print('->', out)


if __name__ == '__main__':
    main()
