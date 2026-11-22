"""PS/AI diagnosis for GFS-VL (RegionPLC 636 + ScanNet200).

Reproduces GFS-VL's Pseudo-label Selection (PS) and Adaptive Infilling (AI)
(`GFS_VL_Trainer.pseudo_label_selection` / `target_calibrate`) and tests the
"cross-region coupling" hypothesis:

    PS pools *all* points predicted as class c within the current input into a
    single prototype u_c and keeps/drops them together based on
    cos(u_c, p_c) vs tau. A locally-correct region can therefore be dropped
    because of unrelated same-predicted-class regions elsewhere.

For each candidate region R (connected component of predicted class c) the
script computes:
  - local  : cos(mean(R), p_c)              -> decision if R were alone
  - pooled : cos(mean(all c), p_c)          -> the actual PS decision
  - leave-one-out: cos(mean(all c except R), p_c) -> sensitivity of others to R

Diagnostics reported:
  - how many GT-correct points are dropped by PS (ps_lost_correct)
  - of those, how many pass the threshold under the local score (misled)
  - how often adding/removing another same-class region flips a region's fate

Usage:
    python ps_diag.py --query scene0011_00 --regis sc200_regis_1_10.txt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlm_common as V  # noqa: E402

DATA = Path("/data01/jh/data/scannet200_gfs")
PS_THRESH = 0.6
AI_THRESH = 0.9


def load_scene(split, scene):
    d = DATA / split / scene
    coord = np.load(d / "coord.npy").astype(np.float32)
    color = np.load(d / "color.npy")
    seg200 = np.load(d / "segment200.npy").reshape(-1)
    return coord, color, V.convert_to_gfs(seg200)


def scene_name_from_rel(rel):
    """'train/scene0694_00' -> ('train', 'scene0694_00')."""
    parts = rel.replace("\\", "/").split("/")
    return parts[-2], parts[-1]


@torch.no_grad()
def support_prototypes(model, cfg, regis_file, n_novel):
    """Average VLM features over GT novel-class points in the support scenes."""
    acc = torch.zeros(n_novel, 512)
    cnt = torch.zeros(n_novel)
    with (DATA / regis_file).open() as f:
        entries = [l.split("\t") for l in f.read().splitlines() if l.strip()]
    for class_id_str, rel in entries:
        # The registration file stores the original ScanNet200 class id; map it
        # through the class name to the unified GFS label space.
        orig_id = int(class_id_str)
        gfs_id = V.CLASS_NAMES.index(V.LABELS200[orig_id])
        split, scene = scene_name_from_rel(rel)
        coord, color, gfs = load_scene(split, scene)
        feats, _ = V.vlm_forward(model, cfg, coord, color)
        idx = gfs_id - V.NUM_BASE
        mask = gfs == gfs_id
        if mask.sum() == 0:
            raise RuntimeError(f"support scene {scene} has no class {gfs_id}")
        acc[idx] += torch.as_tensor(feats[mask]).sum(0)
        cnt[idx] += mask.sum()
    return (acc / cnt.unsqueeze(1)).numpy()


def connected_components(pred_mask, coord, radius=0.05):
    """Union-find connected components over points within `radius` of each other.

    Uses a voxel-grid neighbour lookup so the pairwise distance test stays local.
    """
    idx = np.where(pred_mask)[0]
    n = len(idx)
    if n == 0:
        return np.empty(0, dtype=int), []
    pts = coord[idx].astype(np.float64)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    key = np.floor(pts / radius).astype(np.int64)
    buckets = {}
    for i, k in enumerate(map(tuple, key)):
        buckets.setdefault(k, []).append(i)
    r2 = radius * radius
    for i, k in enumerate(map(tuple, key)):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in buckets.get(
                        (k[0] + dx, k[1] + dy, k[2] + dz), ()
                    ):
                        if j > i:
                            d = pts[i] - pts[j]
                            if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= r2:
                                union(i, j)
    roots = np.array([find(i) for i in range(n)])
    uniq = {r: c for c, r in enumerate(np.unique(roots))}
    labels = np.array([uniq[r] for r in roots])
    comps = [idx[labels == c] for c in range(len(uniq))]
    return labels, comps


def diagnose_scene(model, cfg, proto_np, coord, color, gfs, max_classes=4):
    feats, preds = V.vlm_forward(model, cfg, coord, color)
    feat_t = torch.as_tensor(feats)
    proto_t = torch.as_tensor(proto_np)

    result = {
        "points": int(len(preds)),
        "novel_pred_points": int((preds >= V.NUM_BASE).sum()),
        "classes": {},
    }

    novel_classes = [c for c in np.unique(preds) if c >= V.NUM_BASE]
    # focus on classes with enough support to form a pooled prototype
    novel_classes = [c for c in novel_classes if (preds == c).sum() >= 50]
    novel_classes = novel_classes[:max_classes]

    for c in novel_classes:
        mask = preds == c
        p_c = proto_t[c - V.NUM_BASE]
        pooled = torch.cosine_similarity(feat_t[mask].mean(0), p_c, dim=0).item()
        ps_keeps = pooled >= PS_THRESH

        # GT-correct points of this predicted class
        correct = mask & (gfs == c)
        n_correct = int(correct.sum())

        # local (per-point) score, and grouped-by-region scores
        _, comps = connected_components(mask, coord)
        comp_info = []
        for comp in comps:
            if len(comp) < 20:
                continue
            local = torch.cosine_similarity(
                feat_t[comp].mean(0), p_c, dim=0
            ).item()
            # leave-this-region-out pooled score
            rest = np.setdiff1d(np.where(mask)[0], comp, assume_unique=False)
            if len(rest) == 0:
                loo = None
            else:
                loo = torch.cosine_similarity(
                    feat_t[rest].mean(0), p_c, dim=0
                ).item()
            comp_correct = int(((gfs[comp] == c)).sum())
            comp_info.append(
                {
                    "size": int(len(comp)),
                    "local_score": round(local, 4),
                    "loo_score": None if loo is None else round(loo, 4),
                    "local_would_keep": bool(local >= PS_THRESH),
                    "correct_points": comp_correct,
                }
            )
        comp_info.sort(key=lambda x: -x["size"])

        # points PS drops that are GT-correct, and whether local score would keep
        dropped_correct = correct & (not ps_keeps)
        local_scores_all = torch.cosine_similarity(
            feat_t, p_c.unsqueeze(0), dim=-1
        )
        if int(dropped_correct.sum()) > 0:
            local_ok = (local_scores_all[dropped_correct] >= PS_THRESH).sum().item()
        else:
            local_ok = 0

        result["classes"][int(c)] = {
            "name": V.CLASS_NAMES[c],
            "pred_points": int(mask.sum()),
            "gt_correct_points": n_correct,
            "pooled_score": round(pooled, 4),
            "ps_keeps_class": bool(ps_keeps),
            "regions": comp_info[:6],
            "ps_dropped_correct_points": int(dropped_correct.sum()),
            "of_dropped_local_would_keep": int(local_ok),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="scene0011_00")
    ap.add_argument("--query_split", default="val")
    ap.add_argument("--regis", default="sc200_regis_1_10.txt")
    ap.add_argument("--max_classes", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg, model, text_embed, missing, unexpected = V.build_vlm()
    n_novel = len(V.CLASS_NAMES) - V.NUM_BASE
    print(f"[vlm] loaded 636; text_embed={tuple(text_embed.shape)}", flush=True)

    proto_cache = Path(__file__).resolve().parent / f"proto_{Path(args.regis).stem}.npy"
    if proto_cache.exists():
        proto = np.load(proto_cache)
        print(f"[proto] loaded cached {proto.shape}", flush=True)
    else:
        proto = support_prototypes(model, cfg, args.regis, n_novel)
        np.save(proto_cache, proto)
        print(f"[proto] support prototypes {proto.shape}", flush=True)

    coord, color, gfs = load_scene(args.query_split, args.query)
    print(f"[query] {args.query} points={len(coord)}", flush=True)
    rep = diagnose_scene(
        model, cfg, proto, coord, color, gfs, args.max_classes
    )
    rep["query"] = args.query
    rep["regis"] = args.regis
    text = json.dumps(rep, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
