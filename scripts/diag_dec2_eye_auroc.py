import argparse, logging, os
import numpy as np
import torch
import torch.nn.functional as F

from pointcept.datasets import build_dataset, collate_fn
from pointcept.engines.train import GFS_VL_Trainer
from pointcept.models import build_model
from pointcept.models.utils.structure import Point
from pointcept.utils.config import Config


def det_transforms(cfg_ds):
    """Keep only deterministic preprocessing (no random flip/rotate/jitter)."""
    keep = {"CenterShift", "GridSample", "NormalizeColor", "ToTensor", "Collect"}
    tf = []
    for t in cfg_ds["transform"]:
        if t.get("type") in keep:
            t2 = dict(t)
            if t2["type"] == "GridSample":
                t2["mode"] = "train"
            tf.append(t2)
    out = dict(cfg_ds)
    out["transform"] = tf
    out["loop"] = 1
    return out


def build_stage1_ptv3(cfg):
    mcfg = dict(cfg.model.backbone)
    mcfg["enable_flash"] = False
    bb = build_model(mcfg).cuda().eval()
    sd = torch.load(cfg.backbone_weight, map_location="cpu")["state_dict"]
    bb_sd = {}
    for k, v in sd.items():
        k2 = k.replace("module.", "")
        if k2.startswith("backbone."):
            bb_sd[k2[len("backbone."):]] = v
    miss, unexp = bb.load_state_dict(bb_sd, strict=True)
    assert len(miss) == 0 and len(unexp) == 0, f"miss={miss} unexp={unexp}"

    cache = {}
    for s in range(len(bb.dec._modules)):
        bb.dec._modules[f"dec{s}"]._modules["up"].register_forward_pre_hook(
            lambda m, inp, s=s: cache.__setitem__(f"inv{s}", inp[0]["pooling_inverse"])
        )
    for s in (0, 1, 2):
        bb.dec._modules[f"dec{s}"].register_forward_hook(
            lambda m, inp, out, s=s: cache.__setitem__(f"f{s}", out.feat)
        )

    def extract(inp):
        cache.clear()
        pt = bb(Point(inp))
        res = {}
        for s in (0, 1, 2):
            f = cache[f"f{s}"].float()
            for t in range(s - 1, -1, -1):
                f = f[cache[f"inv{t}"]]
            res[f"dec{s}"] = F.normalize(f, dim=-1)
        cache.clear()
        return res

    return bb, extract


def build_vlm(cfg, ds_train):
    tr = GFS_VL_Trainer.__new__(GFS_VL_Trainer)
    tr.cfg = cfg
    tr.logger = logging.getLogger("diag")
    tr.train_loader = type("L", (), {"dataset": ds_train})()
    tr.build_zero_shot_model()
    tr.VLM_3D.eval()
    return tr


def cmd_protos(args):
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config.fromfile(args.config)
    ds_r = build_dataset(det_transforms(cfg.data.regis1))
    _, extract = build_stage1_ptv3(cfg)

    dims = {0: 64, 1: 64, 2: 128}
    sums = {s: np.zeros((30, dims[s]), dtype=np.float64) for s in (0, 1, 2)}
    cnts = np.zeros(30, dtype=np.int64)

    tr_b = det_transforms(cfg.data.train)
    tr_b["full_gt"] = False
    tr_b["nb_mix_blks"] = 0
    ds_b = build_dataset(tr_b)

    with torch.no_grad():
        for i in range(min(60, len(ds_b))):
            d = ds_b[i]
            inp = collate_fn([d])
            for k, v in inp.items():
                if isinstance(v, torch.Tensor):
                    inp[k] = v.cuda(non_blocking=True)
            feats = extract(inp)
            seg = inp["segment"].cpu().numpy()
            for c in np.unique(seg):
                if c < 0 or c >= 12:
                    continue
                m = seg == c
                cnts[c] += int(m.sum())
                for s in (0, 1, 2):
                    sums[s][c] += feats[f"dec{s}"][m].cpu().numpy().sum(0)

        for i in range(len(ds_r)):
            d = ds_r[i]
            inp = collate_fn([d])
            for k, v in inp.items():
                if isinstance(v, torch.Tensor):
                    inp[k] = v.cuda(non_blocking=True)
            feats = extract(inp)
            seg = inp["segment"].cpu().numpy()
            for c in np.unique(seg):
                if c < 0 or c >= 30:
                    continue
                m = seg == c
                cnts[c] += int(m.sum())
                for s in (0, 1, 2):
                    sums[s][c] += feats[f"dec{s}"][m].cpu().numpy().sum(0)

    protos = {}
    for s in (0, 1, 2):
        p = sums[s].copy()
        nrm = np.linalg.norm(p, axis=1, keepdims=True)
        p = (p / np.maximum(nrm, 1e-8)).astype(np.float32)
        protos[f"p{s}"] = p
    np.savez(os.path.join(args.out_dir, "protos_k5.npz"), cnts=cnts, **protos)
    print("Saved protos_k5.npz | min_cnt:", int(cnts.min()), "novel_min:", int(cnts[12:].min()), "cnts[12:]:", cnts[12:].tolist())


def cmd_scan(args):
    cfg = Config.fromfile(args.config)
    # Build two paired datasets? No: regis1 train has nb_mix_blks=3; for clean original chunks use regis1 with nb_mix_blks=0!
    tr_regis = cfg.data.train
    tr_full = det_transforms(tr_regis)
    tr_full["full_gt"] = True
    tr_full["nb_mix_blks"] = 0
    ds_full = build_dataset(tr_full)

    tr_mask = det_transforms(tr_regis)
    tr_mask["full_gt"] = False
    tr_mask["nb_mix_blks"] = 0
    ds_mask = build_dataset(tr_mask)

    _, extract = build_stage1_ptv3(cfg)
    vlm = build_vlm(cfg, ds_mask)
    W = vlm.VLM_3D.task_head.cls_head.weight.float()

    pdata = np.load(os.path.join(args.out_dir, "protos_k5.npz"))
    P = {s: torch.tensor(pdata[f"p{s}"], device="cuda", dtype=torch.float32) for s in (0, 1, 2)}

    sig_names = [
        "VLM_S1",
        "dec0_proto", "dec0_m_nv", "dec0_m_all",
        "dec1_proto", "dec1_m_nv", "dec1_m_all",
        "dec2_proto", "dec2_m_nv", "dec2_m_all",
    ]
    CAP = 192
    rng = np.random.default_rng(1000 + args.shard)
    samp_vals = []
    samp_cls = []
    samp_pol = []
    n_pos_true = np.zeros(18, dtype=np.int64)
    n_neg_true = np.zeros(18, dtype=np.int64)

    idxs = list(range(args.shard, len(ds_full), args.num_shards))
    if getattr(args, "max_chunks", 0) > 0:
        idxs = idxs[:args.max_chunks]
    with torch.no_grad():
        for step, idx in enumerate(idxs):
            np.random.seed(10000 + idx)
            d_gt = ds_full[idx]
            np.random.seed(10000 + idx)
            d_mk = ds_mask[idx]
            inp_gt = collate_fn([d_gt])
            inp_mk = collate_fn([d_mk])
            # Verify identical grid points
            if inp_gt["coord"].shape != inp_mk["coord"].shape:
                continue
            for k, v in inp_gt.items():
                if isinstance(v, torch.Tensor):
                    inp_gt[k] = v.cuda(non_blocking=True)
            bg_mask = (inp_mk["segment"].cuda() == -1)
            gt = inp_gt["segment"]

            # 1. VLM
            vd = vlm.construct_3D_vlm_input(inp_gt)
            ret = vlm.VLM_3D(vd)
            f_vlm = F.normalize(ret["all_feats"].float(), dim=-1)
            cos_nv = f_vlm @ W[12:].t()
            pred = cos_nv.argmax(dim=-1) + 12
            log_p = F.log_softmax(cos_nv, dim=-1)
            vlm_s1 = (log_p.exp() * log_p).sum(dim=-1)

            # 2. Stage-1 PTv3
            f3d = extract(inp_gt)

            # 3. Compute all 10 signals on bg_mask
            m = bg_mask
            if not m.any():
                continue
            p_m = pred[m] - 12  # 0..17
            gt_m = gt[m]
            is_pos = (gt_m == pred[m])

            cols = [vlm_s1[m]]
            for s in (0, 1, 2):
                sim = f3d[f"dec{s}"][m] @ P[s].t()  # (M, 30)
                spred = sim.gather(1, (p_m + 12).unsqueeze(1)).squeeze(1)
                sim_nv = sim[:, 12:].clone()
                sim_nv.scatter_(1, p_m.unsqueeze(1), -1e9)
                m_nv = spred - sim_nv.max(dim=1).values
                sim_all = sim.clone()
                sim_all.scatter_(1, (p_m + 12).unsqueeze(1), -1e9)
                m_all = spred - sim_all.max(dim=1).values
                cols.extend([spred, m_nv, m_all])

            mat = torch.stack(cols, dim=1).cpu().numpy().astype(np.float32)
            p_np = p_m.cpu().numpy()
            pos_np = is_pos.cpu().numpy()

            for c in range(18):
                mc = (p_np == c)
                if not mc.any():
                    continue
                for pol, arr_cnt in [(1, n_pos_true), (0, n_neg_true)]:
                    mcp = np.where(mc & (pos_np == (pol == 1)))[0]
                    cnt = len(mcp)
                    if cnt == 0:
                        continue
                    arr_cnt[c] += cnt
                    if cnt > CAP:
                        mcp = rng.choice(mcp, size=CAP, replace=False)
                    samp_vals.append(mat[mcp])
                    samp_cls.append(np.full(len(mcp), c, dtype=np.int16))
                    samp_pol.append(np.full(len(mcp), pol, dtype=np.int8))

    out_p = os.path.join(args.out_dir, f"shard{args.shard}.npz")
    np.savez(
        out_p,
        sig_names=np.array(sig_names),
        vals=np.concatenate(samp_vals, axis=0),
        cls=np.concatenate(samp_cls, axis=0),
        pol=np.concatenate(samp_pol, axis=0),
        n_pos_true=n_pos_true,
        n_neg_true=n_neg_true,
    )
    print(f"Shard {args.shard} saved: {out_p} | pos={int(n_pos_true.sum())} neg={int(n_neg_true.sum())}")


def weighted_auroc(scores, pol, w_neg):
    """Exact weighted Mann-Whitney U AUROC (ties=0.5)."""
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    p_sorted = pol[order]
    w = np.where(p_sorted == 1, 1.0, w_neg)
    # handle ties
    n = len(s_sorted)
    pos_w_tot = w[p_sorted == 1].sum()
    neg_w_tot = w[p_sorted == 0].sum()
    if pos_w_tot == 0 or neg_w_tot == 0:
        return np.nan
    auc_num = 0.0
    cum_neg = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        w_pos_grp = w[i:j][p_sorted[i:j] == 1].sum()
        w_neg_grp = w[i:j][p_sorted[i:j] == 0].sum()
        auc_num += w_pos_grp * (cum_neg + 0.5 * w_neg_grp)
        cum_neg += w_neg_grp
        i = j
    return float(auc_num / (pos_w_tot * neg_w_tot))


def weighted_p_at_q(scores, pol, w_neg, q=0.10):
    w = np.where(pol == 1, 1.0, w_neg)
    order = np.argsort(-scores, kind="mergesort")
    w_ord = w[order]
    p_ord = pol[order]
    cum_w = np.cumsum(w_ord)
    cutoff = q * cum_w[-1]
    k = int(np.searchsorted(cum_w, cutoff, side="right"))
    k = max(1, min(k, len(w_ord)))
    tp_w = w_ord[:k][p_ord[:k] == 1].sum()
    tot_w = w_ord[:k].sum()
    pos_all = w[pol == 1].sum()
    return float(tp_w / tot_w), float(tp_w / max(pos_all, 1e-8))


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def cmd_analyze(args):
    cfg = Config.fromfile(args.config)
    names = list(cfg.data.names)[12:]
    vals_l, cls_l, pol_l = [], [], []
    n_pos_true = np.zeros(18, dtype=np.int64)
    n_neg_true = np.zeros(18, dtype=np.int64)
    sig_names = None
    for i in range(args.num_shards):
        z = np.load(os.path.join(args.out_dir, f"shard{i}.npz"))
        sig_names = list(z["sig_names"])
        vals_l.append(z["vals"])
        cls_l.append(z["cls"])
        pol_l.append(z["pol"])
        n_pos_true += z["n_pos_true"]
        n_neg_true += z["n_neg_true"]

    vals = np.concatenate(vals_l, axis=0)
    cls = np.concatenate(cls_l, axis=0)
    pol = np.concatenate(pol_l, axis=0)

    # Global z-score joint signals
    z_s1 = (vals[:, 0] - vals[:, 0].mean()) / (vals[:, 0].std() + 1e-8)
    joints = []
    for jname, jidx in [("S1+dec0_m_all", 3), ("S1+dec1_m_all", 6), ("S1+dec2_proto", 7), ("S1+dec2_m_nv", 8), ("S1+dec2_m_all", 9)]:
        zj = (vals[:, jidx] - vals[:, jidx].mean()) / (vals[:, jidx].std() + 1e-8)
        joints.append((jname, (z_s1 + zj).astype(np.float32)))
        sig_names.append(jname)
    if joints:
        vals = np.concatenate([vals, np.stack([j[1] for j in joints], axis=1)], axis=1)

    w_neg_cls = np.ones(18, dtype=np.float64)
    w_pt = np.ones(len(pol), dtype=np.float64)
    for c in range(18):
        mc = (cls == c)
        np_s = int((mc & (pol == 1)).sum())
        nn_s = int((mc & (pol == 0)).sum())
        if np_s > 0 and nn_s > 0 and n_pos_true[c] > 0 and n_neg_true[c] > 0:
            wn = (n_neg_true[c] / nn_s) / (n_pos_true[c] / np_s)
            w_neg_cls[c] = wn
            w_pt[mc & (pol == 0)] = wn
        # scale class weight so global sum matches true counts
        if np_s > 0 and n_pos_true[c] > 0:
            scale_p = n_pos_true[c] / np_s
            w_pt[mc & (pol == 1)] = scale_p
            w_pt[mc & (pol == 0)] = n_neg_true[c] / max(nn_s, 1)

    print(f"Total BG points: pos={int(n_pos_true.sum()):,} neg={int(n_neg_true.sum()):,} base_rate={n_pos_true.sum()/(n_pos_true.sum()+n_neg_true.sum()):.4f}")
    print("\n=== OVERALL SIGNAL SUMMARY (18 Novel Classes, exact Mann-Whitney U) ===")
    print(f"{'Signal':18s} | {'AUROC_mean':>10s} {'AUROC_med':>9s} | {'Spearman_S1':>11s} | {'wP@10%':>7s} | {'GlobP@5%':>8s} {'GlobR@5%':>8s}")
    aucs_by_cls = {}
    for si, sname in enumerate(sig_names):
        sc = vals[:, si]
        aucs = []
        p10s = []
        spms = []
        for c in range(18):
            mc = (cls == c)
            if (mc & (pol == 1)).sum() < 5 or (mc & (pol == 0)).sum() < 5:
                continue
            aucs.append(weighted_auroc(sc[mc], pol[mc], w_neg_cls[c]))
            p10, _ = weighted_p_at_q(sc[mc], pol[mc], w_neg_cls[c], q=0.10)
            p10s.append(p10)
            spms.append(spearman(sc[mc], vals[mc, 0]))
        # Global P@5% using true-count weights w_pt
        order = np.argsort(-sc, kind="mergesort")
        w_ord = w_pt[order]; p_ord = pol[order]
        cw = np.cumsum(w_ord)
        k5 = max(1, int(np.searchsorted(cw, 0.05 * cw[-1], side="right")))
        gp5 = float(w_ord[:k5][p_ord[:k5] == 1].sum() / w_ord[:k5].sum())
        gr5 = float(w_ord[:k5][p_ord[:k5] == 1].sum() / w_pt[pol == 1].sum())
        aucs_by_cls[sname] = aucs
        print(f"{sname:18s} | {np.mean(aucs):10.4f} {np.median(aucs):9.4f} | {np.mean(spms):11.4f} | {np.mean(p10s):7.4f} | {gp5:8.4f} {gr5:8.4f}")

    # Per-class table for key signals
    key_sigs = ["VLM_S1", "dec0_m_all", "dec1_m_all", "dec2_proto", "dec2_m_all", "S1+dec2_m_all"]
    kidx = [sig_names.index(k) for k in key_sigs]
    print("\n=== PER-CLASS AUROC (18 Novel Classes) ===")
    print(f"{'Class':18s} | {'nPos':>7s} | " + " ".join(f"{k[:10]:>10s}" for k in key_sigs) + " | Spm(S1,d2)")
    for c in range(18):
        mc = (cls == c)
        if (mc & (pol == 1)).sum() < 5:
            print(f"{c+12:2d} {names[c]:15s} | {n_pos_true[c]:7d} | (insufficient pos)")
            continue
        vals_str = " ".join(f"{weighted_auroc(vals[mc, j], pol[mc], w_neg_cls[c]):10.3f}" for j in kidx)
        spm = spearman(vals[mc, 0], vals[mc, sig_names.index("dec2_m_all")])
        print(f"{c+12:2d} {names[c]:15s} | {n_pos_true[c]:7d} | {vals_str} | {spm:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("protos", "scan", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/scannetpp/semseg-pt-v3m1-0-gfsregistrain_k5_geq02.py")
        p.add_argument("--out-dir", default="/data01/jh/data/dec2_eye_audit")
        if name == "scan":
            p.add_argument("--shard", type=int, required=True)
            p.add_argument("--num-shards", type=int, default=4)
            p.add_argument("--max-chunks", type=int, default=0)
        if name == "analyze":
            p.add_argument("--num-shards", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "protos":
        cmd_protos(args)
    elif args.cmd == "scan":
        cmd_scan(args)
    else:
        cmd_analyze(args)
