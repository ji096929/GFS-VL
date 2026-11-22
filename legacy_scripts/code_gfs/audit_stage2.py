"""Stage II 审计：ep0 基线、逐 checkpoint 逐集指标（TP/FP/FN、B 池纯度/AUC）。"""
import argparse
import glob
import json
import os
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
from model.coseg import COSeg  # noqa: E402

CFG = '/data01/jh/code/COSeg/config/scannetv2_COSeg_fs.yaml'
WEIGHT = '/data01/jh/weights/sc0_1w1s.pth'
DUMP = '/data01/jh/data/sonata_dump'
ST2 = '/data01/jh/afford_diag/stage2'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]


def build_cfg():
    cfg = config.load_cfg_from_cfg_file(CFG)
    cfg.cvfold = 0
    cfg.n_way = 1
    cfg.k_shot = 1
    cfg.n_queries = 1
    cfg.save_path = '/data01/jh/afford_diag/stage2/tmp_audit'
    cfg.ngpus_per_node = 1
    cfg.rank = 0
    cfg.multiprocessing_distributed = False
    cfg.distributed = False
    os.makedirs(cfg.save_path, exist_ok=True)
    return cfg


def load_model(path):
    m = COSeg(build_cfg()).cuda()
    ck = torch.load(path, map_location='cpu')
    sd = ck.get('model', ck.get('state_dict', ck))
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


CKPTS = [('base', WEIGHT)]
for tag in ('exp0_lowlr', 'exp1_lowlr', 'exp0_s', 'exp1_su_l03'):
    for ep in (1, 3, 5, 7, 9):
        p = os.path.join(ST2, tag, 'ep%d.pth' % ep)
        if os.path.exists(p):
            CKPTS.append(('%s_ep%d' % (tag, ep), p))


@torch.no_grad()
def eval_one(m, files):
    rows = {}
    for f in files:
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
        with torch.cuda.amp.autocast():
            out, _ = m(so, sx, sy, qo, qx, qy, 5,
                       support_base_y=None, query_base_y=None,
                       sampled_classes=np.array([c]))
        p = F.softmax(out[0].t().float(), dim=1)[:, 1].cpu().numpy()
        gt = d['q_lab'].astype(bool)
        pred = p > 0.5
        m_old = d['q_p0'] > 0.7          # 原始 COSeg 的高置信池
        m_new = p > 0.7                  # 训练后模型自己的高置信池
        row = dict(gid=gid, cls=c,
                   tp=int((pred & gt).sum()),
                   fp=int((pred & ~gt).sum()),
                   fn=int((~pred & gt).sum()),
                   iou=float((pred & gt).sum()
                             / max(int((pred | gt).sum()), 1)),
                   predfg=float(pred.mean()), gtfg=float(gt.mean()),
                   gold_n=int(m_old.sum()),
                   gold_pure=float(gt[m_old].mean()) if m_old.any() else -1.0,
                   gnew_n=int(m_new.sum()),
                   gnew_pure=float(gt[m_new].mean()) if m_new.any() else -1.0,
                   oldauc=-1.0, newauc=-1.0)
        if m_old.any() and (~gt[m_old]).any() and gt[m_old].any():
            u = mannwhitneyu(p[m_old][gt[m_old]], p[m_old][~gt[m_old]],
                             alternative='greater').statistic
            row['oldauc'] = float(u / (gt[m_old].sum() * (~gt[m_old]).sum()))
        if m_new.any() and (~gt[m_new]).any() and gt[m_new].any():
            u = mannwhitneyu(p[m_new][gt[m_new]], p[m_new][~gt[m_new]],
                             alternative='greater').statistic
            row['newauc'] = float(u / (gt[m_new].sum() * (~gt[m_new]).sum()))
        rows[gid] = row
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(ST2, 'audit_s%d.json'))
    args = ap.parse_args()
    out = args.out % args.shard if '%d' in args.out else args.out

    files = sorted(glob.glob(os.path.join(DUMP, 'ep*.npz')))
    files = [f for f in files
             if int(os.path.basename(f)[2:5]) % args.shards == args.shard]
    print('tasks', len(files), 'ckpts', len(CKPTS), flush=True)

    res = {}
    for name, path in CKPTS:
        m = load_model(path)
        res[name] = eval_one(m, files)
        del m
        torch.cuda.empty_cache()
        print('  %s done' % name, flush=True)
    json.dump(res, open(out, 'w'))
    print('->', out)


if __name__ == '__main__':
    main()
