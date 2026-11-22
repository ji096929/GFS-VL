# -*- coding: utf-8 -*-
"""Track 2 Step A：预计算正交零空间基与 dec2 目标 PCA 基。

产出 /data01/jh/data/ontr_basis.pt
  Q_perp (64, 52)   基类分类头 W_b 的精确正交零空间基（W_b @ Q_perp ≈ 0）
  V52    (128, 52)  冻结 dec2 全分辨率特征的前 52 个主成分
  mu2    (128,)     dec2 特征均值

口径：100 个训练场景全点（含 segment == -1 的背景/新类点），每场景随机采 4096 点。
"""
import copy
import sys

import numpy as np
import torch

sys.path.insert(0, "/data01/jh/code/GFS-VL")
sys.path.insert(0, "/data01/jh/code/GFS-VL/pointcept/models/PLA")

from pointcept.datasets import build_dataset, collate_fn  # noqa: E402
from pointcept.models import build_model  # noqa: E402
from pointcept.utils.config import Config  # noqa: E402

CFG = "/data01/jh/code/GFS-VL/configs/scannet200/semseg-pt-v3m1-0-gfspretrain.py"
CKPT = "/data01/jh/code/GFS-VL/exp/scannet200/ptv3_bk/model/model_best.pth"
OUT = "/data01/jh/data/ontr_basis.pt"

N_SCENE = 100
N_POINT = 4096
D2 = 128
NB = 12
CROSS_LAYER = 2


def main():
    cfg = Config.fromfile(CFG)
    # 环境未安装 flash_attn，Stage-2 全部配置都用非 flash 路径。
    # 二者只影响 attention 的实现与数值精度，参数结构完全一致。
    cfg.model.backbone.enable_flash = False

    # ---------- 1. 基类头的正交零空间 ----------
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in ck["state_dict"].items()
    }
    Wb = sd["seg_head.weight"].float()
    print("W_b shape:", tuple(Wb.shape), flush=True)

    U, S, Vh = torch.linalg.svd(Wb.T, full_matrices=True)
    Q_b = U[:, :NB]                    # (64, 12) 行空间
    Q_perp = U[:, NB:]                 # (64, 52) 零空间
    resid = (Wb @ Q_perp).abs().max().item()
    print("Q_perp shape:", tuple(Q_perp.shape), flush=True)
    print("max|W_b @ Q_perp| = %.3e" % resid, flush=True)
    assert resid < 1e-5, "零空间基不正交，W_b @ Q_perp 应为 0"
    print("奇异值前 14 个:", S[:14].tolist(), flush=True)

    # ---------- 2. 冻结 ptv3_bk 提取 dec2 全分辨率特征 ----------
    model = build_model(cfg.model)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print("load: missing=%d unexpected=%d" % (len(miss), len(unexp)), flush=True)
    model.eval().cuda()

    store = {}
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
    del store

    # train 场景 + val 口径 transform（无随机增广）
    val_cfg = copy.deepcopy(cfg.data.val)
    val_cfg.split = "train"
    val_cfg.data_root = "/data01/jh/data/scannet200_gfs"
    ds = build_dataset(val_cfg)
    print("场景总数:", len(ds), flush=True)

    rng = np.random.RandomState(42)
    chunks = []
    used = 0
    for i in range(min(N_SCENE, len(ds))):
        d = ds[i]
        d.pop("fragment_list", None)
        inp = collate_fn([d])
        for k in inp:
            if isinstance(inp[k], torch.Tensor):
                inp[k] = inp[k].cuda(non_blocking=True)
        with torch.no_grad():
            model(inp)
        f = buf["f%d" % CROSS_LAYER]
        for t in range(CROSS_LAYER - 1, -1, -1):
            f = f[buf["inv%d" % t]]
        f = f.float()
        if f.shape[0] > N_POINT:
            idx = rng.choice(f.shape[0], N_POINT, replace=False)
            f = f[idx]
        chunks.append(f.cpu())
        used += 1
        if (i + 1) % 20 == 0:
            print("  %d/%d 场景" % (i + 1, min(N_SCENE, len(ds))), flush=True)

    X = torch.cat(chunks, dim=0)
    print("采样点总数:", tuple(X.shape), flush=True)

    # ---------- 3. 中心化 + 协方差 eigh 取前 52 主成分 ----------
    mu2 = X.mean(0)
    Xc = X - mu2
    C = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    evals, evecs = torch.linalg.eigh(C)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]
    V52 = evecs[:, :52]                # (128, 52)
    total = evals.sum().item()
    explained = (evals[:52].sum() / total).item()
    print("前 52 主成分解释方差: %.4f" % explained, flush=True)

    # eRank 参考：dec2 原始空间的谱
    p = (evals / total).clamp_min(1e-12)
    erank = float(torch.exp(-(p * p.log()).sum()))
    print("dec2 全分辨率 eRank = %.2f / %d" % (erank, D2), flush=True)

    torch.save(
        {
            "Q_perp": Q_perp,
            "V52": V52,
            "mu2": mu2,
            "svd_S": S,
            "explained": explained,
            "erank_dec2": erank,
            "n_points": int(X.shape[0]),
        },
        OUT,
    )
    print("已保存:", OUT, flush=True)


if __name__ == "__main__":
    main()
