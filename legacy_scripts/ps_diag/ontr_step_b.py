# -*- coding: utf-8 -*-
"""Track 2 Step B：Stage-1 正交零空间续训（ONTR）。

目的：在 Stage-1 从源头阻止 dec0 的隧道坍塌 —— 把预训练高秩 dec2 的几何结构
写进基类分类头 W_b 的 52 维正交零空间 ker(W_b)，做到 Base 零损伤、
推理零额外参数/零 hook，同时保留 dec0 的 2cm 锐利边界。

冻结：seg_head、enc、dec3、dec2（教师绝对不漂移）
只训：dec1、dec0

损失：
  L_base = CE(W_b h0, y) + Lovasz(W_b h0, y)     仅 12 类 Base
  L_null = SmoothL1(z, t)                        全部点（含 -1 未标区域）
      z = h0 @ Q_perp                            (N, 52) 零空间投影坐标
      t = LayerNorm((g2 - mu2) @ V52)            (N, 52) 冻结 dec2 主成分目标
  L = L_base + lam * L_null
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/data01/jh/code/GFS-VL")
sys.path.insert(0, "/data01/jh/code/GFS-VL/pointcept/models/PLA")

from pointcept.datasets import build_dataset, collate_fn  # noqa: E402
from pointcept.models import build_model  # noqa: E402
from pointcept.models.losses import LovaszLoss  # noqa: E402
from pointcept.models.utils.structure import Point  # noqa: E402
from pointcept.utils.config import Config  # noqa: E402

ROOT = "/data01/jh/code/GFS-VL"
CFG = ROOT + "/configs/scannet200/semseg-pt-v3m1-0-gfspretrain.py"
INIT = ROOT + "/exp/scannet200/ptv3_bk/model/model_best.pth"
BASIS = "/data01/jh/data/ontr_basis.pt"
OUT = ROOT + "/exp/scannet200/ptv3_bk_ontr.pth"
CROSS_LAYER = 2


def erank(x):
    """特征的有效秩：exp(谱熵)，x 为 (N, D)。"""
    x = x.float()
    x = x - x.mean(0, keepdim=True)
    c = (x.T @ x) / max(x.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(c).clamp_min(1e-12)
    p = ev / ev.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--epoch", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--num-worker", type=int, default=8)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    cfg = Config.fromfile(CFG)
    cfg.model.backbone.enable_flash = False   # 环境无 flash_attn

    # ---------- 模型 ----------
    model = build_model(cfg.model)
    ck = torch.load(INIT, map_location="cpu", weights_only=False)
    sd = {(k[7:] if k.startswith("module.") else k): v
          for k, v in ck["state_dict"].items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print("init load: missing=%d unexpected=%d" % (len(miss), len(unexp)), flush=True)
    model.cuda()

    # ---------- 冻结：只训 dec1 与 dec0 ----------
    for p in model.parameters():
        p.requires_grad = False
    n_train = 0
    # 实际参数名是 backbone.dec.dec0.* / backbone.dec.dec1.*（不是 backbone.dec.0.*）
    TRAINABLE = ("backbone.dec.dec0.", "backbone.dec.dec1.")
    FROZEN = ("backbone.embedding.", "backbone.enc.", "backbone.dec.dec2.",
              "backbone.dec.dec3.", "seg_head.")
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(TRAINABLE)
        if p.requires_grad:
            n_train += p.numel()
    assert n_train > 0, "没有可训练参数，前缀写错了"
    print("可训练参数: %.2fM（仅 dec0 + dec1）" % (n_train / 1e6), flush=True)
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert name.startswith(TRAINABLE), "意外的可训练参数: %s" % name
        else:
            assert name.startswith(FROZEN), "意外的未冻结参数: %s" % name
    print("冻结校验通过：embedding / enc / dec3 / dec2 / seg_head 全部冻结", flush=True)

    # ---------- dec2 特征 hook ----------
    buf = {}

    def inv_hook(s):
        key = "inv%d" % s

        def h(module, inputs):
            buf[key] = inputs[0]["pooling_inverse"]

        return h

    def feat_hook(s):
        key = "f%d" % s

        def h(module, inputs, output):
            buf[key] = output.feat

        return h

    for s in range(len(model.backbone.dec._modules)):
        up = model.backbone.dec._modules["dec%d" % s]._modules["up"]
        up.register_forward_pre_hook(inv_hook(s))
    model.backbone.dec._modules["dec%d" % CROSS_LAYER].register_forward_hook(
        feat_hook(CROSS_LAYER))

    # ---------- 零空间基与目标 PCA 基 ----------
    basis = torch.load(BASIS, map_location="cuda")
    Q_perp = basis["Q_perp"].cuda().float()
    V52 = basis["V52"].cuda().float()
    mu2 = basis["mu2"].cuda().float()
    print("Q_perp %s  V52 %s" % (tuple(Q_perp.shape), tuple(V52.shape)), flush=True)
    Wb = model.seg_head.weight.detach()
    print("max|W_b @ Q_perp| = %.3e" % (Wb @ Q_perp).abs().max().item(), flush=True)

    # ---------- 数据 ----------
    cfg.data.train.data_root = "/data01/jh/data/scannet200_gfs"
    ds = build_dataset(cfg.data.train)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_worker, pin_memory=True, collate_fn=collate_fn)
    print("训练场景 %d，每 epoch %d iter" % (len(ds), len(loader)), flush=True)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.epoch * len(loader), 1))
    scaler = torch.cuda.amp.GradScaler()
    lov = LovaszLoss(mode="multiclass", ignore_index=-1, loss_weight=1.0)

    model.train()
    for ep in range(args.epoch):
        t0 = time.time()
        agg = {"base": 0.0, "null": 0.0, "n": 0}
        last_h0 = None
        for it, inp in enumerate(loader):
            for k in inp:
                if isinstance(inp[k], torch.Tensor):
                    inp[k] = inp[k].cuda(non_blocking=True)
            seg = inp["segment"]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                out = model.backbone(Point(inp))
                h0 = out.feat if hasattr(out, "feat") else out
                seg_logits = model.seg_head(h0)
            # 损失全部在 fp32 下计算，避免 fp16 影响 Lovasz 与零空间目标的数值
            h0f = h0.float()
            sl = seg_logits.float()
            l_base = F.cross_entropy(sl, seg, ignore_index=-1)
            n_pt = sl.shape[0]
            l_base = l_base + lov(sl.view(n_pt, -1, 1, 1), seg.view(n_pt, 1, 1))

            g2 = buf["f%d" % CROSS_LAYER].float()
            for t in range(CROSS_LAYER - 1, -1, -1):
                g2 = g2[buf["inv%d" % t]]
            z = h0f @ Q_perp
            raw = (g2 - mu2) @ V52
            tgt = F.layer_norm(raw, (raw.shape[1],))
            l_null = F.smooth_l1_loss(z, tgt)

            loss = l_base + args.lam * l_null
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            agg["base"] += float(l_base)
            agg["null"] += float(l_null)
            agg["n"] += 1
            last_h0 = h0f.detach()
            if it % 20 == 0:
                print("Ep%d [%d/%d] L_base=%.4f L_null=%.4f loss=%.4f lr=%.2e"
                      % (ep, it, len(loader), float(l_base), float(l_null),
                         float(loss), sched.get_last_lr()[0]), flush=True)

        nb = max(agg["n"], 1)
        with torch.no_grad():
            idx = torch.randperm(last_h0.shape[0], device=last_h0.device)[:8192]
            er = erank(last_h0[idx])
        print("== Epoch %d 完成  %.1fs  L_base=%.4f L_null=%.4f  dec0 eRank(末 batch)=%.2f"
              % (ep, time.time() - t0, agg["base"] / nb, agg["null"] / nb, er), flush=True)

        torch.save(
            {"state_dict": model.state_dict(), "epoch": ep + 1,
             "ontr": {"lam": args.lam, "erank_last": er}},
            args.out)
        print("  已保存 %s" % args.out, flush=True)

    print("DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
