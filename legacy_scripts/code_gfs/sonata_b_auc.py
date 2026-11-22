"""用 Sonata 独立三维特征判别 B_pos / B_neg。

B 的定义与 OCSB 一致：query 上 COSeg 初始预测 p0 > thresh 的点。
B_pos = B 中真值为前景，B_neg = B 中真值为背景。

score = cos(sonata_feat, support_fg_proto) - cos(sonata_feat, support_bg_proto)
输出全局 AUC（Mann-Whitney），以及 B 的头部（按 score 排序 Top 10/20/30%）精度。
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, "/data01/jh/code/sonata")
sys.path.insert(0, "/data01/jh/code")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

import sonata  # noqa: E402

CKPT = "/data01/jh/code/sonata/ckpt/sonata_small.pth"
DUMP = "/data01/jh/data/sonata_dump"


def estimate_normals(coord, k=30):
    n = len(coord)
    k = min(k, n)
    _, idx = cKDTree(coord).query(coord, k=k)
    nb = coord[idx]
    nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    w, v = np.linalg.eigh(cov)
    nrm = v[:, :, 0]
    nrm[nrm[:, 2] < 0] *= -1.0
    return nrm.astype(np.float32)


def feat_of(model, coord, rgb):
    point = dict(coord=coord.astype(np.float32),
                 color=rgb.astype(np.float32),
                 normal=estimate_normals(coord))
    point = sonata.transform.default()(point)
    with torch.inference_mode():
        for k in list(point.keys()):
            if isinstance(point[k], torch.Tensor):
                point[k] = point[k].cuda(non_blocking=True)
        point = model(point)
        # PTv3 解码链：每一层都是 U-Net skip-connection concat。
        # base 有 4 层（896→1088→1184→1232），small 有 2 层（→896）。
        # 之前写死 2 层 + while 覆盖，会让 base 最后只剩 48 维。
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inv = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inv]], dim=-1)
            point = parent
        # 最后一跳：GridSample 的逆映射，回到原始输入分辨率
        return point.feat[point.inverse].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--thresh", type=float, default=0.7)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default="/data01/jh/data/sonata_dump/auc_s%d.npz")
    args = ap.parse_args()
    out = args.out % args.shard

    files = sorted(glob.glob(os.path.join(DUMP, "ep*.npz")))
    files = [f for f in files if int(os.path.basename(f)[2:5]) % args.shards == args.shard]
    print("tasks:", len(files), flush=True)

    _cfg = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    model = sonata.load(args.ckpt, custom_config=_cfg).cuda().eval()
    print("[load] params %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6),
          flush=True)

    scores, labels, tasks = [], [], []
    for i, f in enumerate(files):
        d = np.load(f)
        sf = feat_of(model, d["s_coord"], d["s_rgb"])
        qf = feat_of(model, d["q_coord"], d["q_rgb"])
        s_lab = torch.from_numpy(d["s_lab"]).bool()
        if int(s_lab.sum()) < 5 or int((~s_lab).sum()) < 5:
            continue
        p_fg = F.normalize(sf[s_lab].mean(0), dim=0)
        p_bg = F.normalize(sf[~s_lab].mean(0), dim=0)
        qn = F.normalize(qf, dim=1)
        B = torch.from_numpy(d["q_p0"] > args.thresh)
        nB = int(B.sum())
        if nB < 5:
            continue
        sc = (qn[B] @ p_fg - qn[B] @ p_bg).cpu().numpy()
        lb = torch.from_numpy(d["q_lab"]).bool()[B].numpy().astype(np.uint8)
        scores.append(sc)
        labels.append(lb)
        tasks.append(np.full(len(sc), int(os.path.basename(f)[2:5]), dtype=np.int32))
        if (i + 1) % 20 == 0:
            print("  %d / %d  B=%d  pos=%d" % (i + 1, len(files), nB, int(lb.sum())),
                  flush=True)

    sc = np.concatenate(scores)
    lb = np.concatenate(labels)
    tk = np.concatenate(tasks)
    pos, neg = sc[lb == 1], sc[lb == 0]
    u, p = mannwhitneyu(pos, neg, alternative="greater")
    auc = u / (len(pos) * len(neg))
    print("\n=== Sonata 独立特征 ===")
    print("B 点数 %d (pos %d / neg %d)  正例率 %.1f%%"
          % (len(sc), len(pos), len(neg), 100 * len(pos) / len(sc)))
    print("AUC = %.4f   (p=%.3g)" % (auc, p))
    order = np.argsort(-sc)
    for frac in (0.1, 0.2, 0.3):
        k = max(int(len(sc) * frac), 1)
        prec = lb[order[:k]].mean()
        print("  Top %2d%%: precision = %.4f" % (int(frac * 100), prec))
    np.savez_compressed(out, score=sc, label=lb, task=tk)
    print("->", out)


if __name__ == "__main__":
    main()
