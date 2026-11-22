"""Stage II 训练：Exp 0（纯 S->Q 微调）与 Exp 1（S+U 可微软原型）。

用法:
  CUDA_VISIBLE_DEVICES=2 python train_su.py --mode s  --epochs 10 --tag exp0
  CUDA_VISIBLE_DEVICES=3 python train_su.py --mode su --lam 0.3 --epochs 10 --tag exp1

评估：每 eval-freq 个 epoch 在 357 个官方 5% 任务上跑一次（复用 p2 的 npz），
口径与之前 COSeg 基线 39.39% 完全一致。
"""
import argparse
import glob
import json
import os
import random
import sys
import time

sys.path.insert(0, '/data01/jh/code')
import fake_tp  # noqa: F401

sys.path.insert(0, '/data01/jh/code/COSeg')
sys.path.insert(0, '/data01/jh/code')

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from util import config  # noqa: E402
from util.scannet_v2_fs import Scannetv2_FS  # noqa: E402
from util.data_util import (collate_fn_limit_fs_train,  # noqa: E402
                            data_prepare_v101)
from model.coseg import COSeg  # noqa: E402
from coseg_su import COSegSU  # noqa: E402

CFG = '/data01/jh/code/COSeg/config/scannetv2_COSeg_fs.yaml'
TRAIN_ROOT = '/data01/jh/data/scannet_train_v2/blocks_bs1.0_s1.0/data'
WEIGHT = '/data01/jh/weights/sc0_1w1s.pth'
DUMP = '/data01/jh/data/sonata_dump'
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
NAMES = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter',
         'curtain', 'desk', 'door', 'floor']


def load_weight(model, path):
    ck = torch.load(path, map_location='cpu')
    sd = ck.get('state_dict', ck.get('model', ck))
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print('[weight] loaded %d keys, missing %d, unexpected %d'
          % (len(sd), len(missing), len(unexpected)), flush=True)
    if missing:
        print('  missing sample:', missing[:5])
    if unexpected:
        print('  unexpected sample:', unexpected[:5])


def make_u(blocks):
    f = random.choice(blocks)
    r = np.load(os.path.join(TRAIN_ROOT, f))
    coord, feat, _ = data_prepare_v101(
        r[:, :3].astype(np.float32), r[:, 3:6].astype(np.float32),
        r[:, 6].astype(np.int64), 'train', 0.02, 20480, None, True, 0)
    x = torch.cat([coord, feat], 1).cuda()
    off = torch.tensor([x.shape[0]], dtype=torch.int32)
    return x, off


@torch.no_grad()
def eval357(model, cfg):
    """在 357 个官方任务上算 Cumul/Arith（与 p0 基线同口径）。"""
    files = sorted(glob.glob(os.path.join(DUMP, 'ep*.npz')))
    tp = {c: 0 for c in NEW_IDS}
    fp = {c: 0 for c in NEW_IDS}
    fn = {c: 0 for c in NEW_IDS}
    ious = []
    model.eval()
    for f in files:
        d = np.load(f)
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
            out, _ = model(so, sx, sy, qo, qx, qy, 5,
                           support_base_y=None, query_base_y=None,
                           sampled_classes=np.array([c]))
        p = F.softmax(out[0].t().float(), dim=1)[:, 1]
        pred = (p > 0.5).cpu().numpy().astype(bool)
        gt = d['q_lab'].astype(bool)
        t = int((pred & gt).sum()); ff = int((pred & ~gt).sum())
        n = int((~pred & gt).sum())
        tp[c] += t; fp[c] += ff; fn[c] += n
        ious.append(t / max(t + ff + n, 1))
    per = [tp[c] / max(tp[c] + fp[c] + fn[c], 1) for c in NEW_IDS]
    model.train()
    return float(np.mean(per)), float(np.mean(ious)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='su', choices=['s', 'su'])
    ap.add_argument('--lam', type=float, default=0.3)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--episodes', type=int, default=400)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--eval-freq', type=int, default=5)
    ap.add_argument('--tag', default='exp1')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--freeze', default='none', choices=['none', 'last'])
    ap.add_argument('--out', default='/data01/jh/afford_diag/stage2')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = config.load_cfg_from_cfg_file(CFG)
    cfg.cvfold = 0
    cfg.n_way = 1
    cfg.k_shot = 1
    cfg.n_queries = 1
    cfg.save_path = os.path.join(args.out, args.tag)
    cfg.ngpus_per_node = 1
    cfg.rank = 0
    cfg.multiprocessing_distributed = False
    cfg.distributed = False
    os.makedirs(cfg.save_path, exist_ok=True)

    model = (COSegSU(cfg) if args.mode == 'su' else COSeg(cfg)).cuda()
    load_weight(model, WEIGHT)
    if args.freeze == 'last':
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.encoder.layers[-1].parameters():
            p.requires_grad = True
        for p in model.encoder.classifier.parameters():
            p.requires_grad = True
        for p in model.bk_ffn.parameters():
            p.requires_grad = True
        # 冻结部分必须 eval：关掉 DropPath(0.3) 与 FastBatchNorm1d 统计漂移
        model.encoder.eval()
    print('[model] %s  lam=%.2f  trainable %.2fM'
          % (args.mode, args.lam,
             sum(p.numel() for p in model.parameters()
                 if p.requires_grad) / 1e6), flush=True)

    ds = Scannetv2_FS(split='train', data_root=TRAIN_ROOT, voxel_size=0.02,
                      voxel_max=20480, transform=None, shuffle_index=True,
                      loop=1, cvfold=0, num_episode=args.episodes,
                      n_way=1, k_shot=1, n_queries=1)
    loader = DataLoader(ds, batch_size=1, shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        collate_fn=collate_fn_limit_fs_train,
                        drop_last=True)
    blocks = sorted(os.listdir(TRAIN_ROOT))
    print('[data] %d episodes/epoch, U pool %d blocks' % (len(ds), len(blocks)),
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler()

    hist = []
    for ep in range(args.epochs):
        model.train()
        if args.freeze == 'last':
            model.encoder.eval()
            model.encoder.layers[-1].train()
        t0 = time.time()
        losses = []
        for i, batch in enumerate(loader):
            (sx, sby, sy, so, qx, qby, qy, qo, cls) = batch
            sx = sx.cuda(non_blocking=True)
            sy = sy.cuda(non_blocking=True)
            sby = sby.cuda(non_blocking=True)
            qx = qx.cuda(non_blocking=True)
            qy = qy.cuda(non_blocking=True)
            qby = qby.cuda(non_blocking=True)
            so = so.int()
            qo = qo.int()
            with torch.cuda.amp.autocast():
                if args.mode == 'su':
                    ux, uo = make_u(blocks)
                    out, loss = model.forward_su(
                        so, sx, sy, uo, ux, qo, qx, qy, ep,
                        support_base_y=sby, query_base_y=qby,
                        sampled_classes=cls, lam=args.lam)
                else:
                    out, loss = model(
                        so, sx, sy, qo, qx, qy, ep,
                        support_base_y=sby, query_base_y=qby,
                        sampled_classes=cls)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
            if (i + 1) % 50 == 0:
                print('  ep %d  %d/%d  loss %.4f  %.1fs'
                      % (ep, i + 1, len(loader),
                         float(np.mean(losses[-50:])), time.time() - t0),
                      flush=True)
        rec = dict(ep=ep, loss=float(np.mean(losses)),
                   sec=time.time() - t0)
        if (ep + 1) % args.eval_freq == 0 or ep == args.epochs - 1:
            cum, arith, per = eval357(model, cfg)
            rec.update(cumul=cum, arith=arith, per_class=per)
            print('  [eval] ep %d  Cumul %.4f  Arith %.4f  (%.0fs)'
                  % (ep, cum, arith, time.time() - t0), flush=True)
            torch.save(dict(cfg=vars(cfg), model=model.state_dict(),
                            ep=ep, mode=args.mode, lam=args.lam),
                       os.path.join(cfg.save_path, 'ep%d.pth' % ep))
        hist.append(rec)
        json.dump(hist, open(os.path.join(cfg.save_path, 'hist.json'), 'w'),
                  indent=1)
        print('[epoch %d] loss %.4f  %.0fs' % (ep, rec['loss'], rec['sec']),
              flush=True)


if __name__ == '__main__':
    main()
