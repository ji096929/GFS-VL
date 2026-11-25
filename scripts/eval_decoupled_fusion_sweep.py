import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F

from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.config import Config
from pointcept.utils.misc import intersection_and_union


def build_grid():
    grid = [
        ("z0_only", dict(mode="z0_only")),
        ("z2_only", dict(mode="z2_only")),
        ("raw_sum", dict(mode="raw_sum")),
    ]
    for T0 in (1.0, 1.5, 2.0):
        for T2 in (1.0, 1.5):
            for a in (0.2, 0.35, 0.5):
                grid.append((f"pmix_T0={T0}_T2={T2}_a={a}", dict(mode="prob_mix", T0=T0, T2=T2, a=a)))
    for T0 in (1.0, 1.5, 2.0, 3.0):
        for T2 in (1.0, 1.5, 2.0):
            for a in (0.5, 1.0, 1.5, 2.0):
                grid.append((f"entp_T0={T0}_T2={T2}_a={a}", dict(mode="ent_prob", T0=T0, T2=T2, a=a)))
    for T0 in (1.0, 1.5, 2.0):
        for a in (0.5, 1.0, 1.5):
            grid.append((f"entl_T0={T0}_a={a}", dict(mode="ent_logit", T0=T0, a=a)))
    return grid


def apply_fusion(z0, z2, spec):
    m = spec["mode"]
    if m == "z0_only":
        return z0
    if m == "z2_only":
        return z2
    if m == "raw_sum":
        return z0 + z2
    if m == "prob_mix":
        p0 = torch.softmax(z0 / spec["T0"], dim=-1)
        p2 = torch.softmax(z2 / spec["T2"], dim=-1)
        return (1.0 - spec["a"]) * p0 + spec["a"] * p2
    if m == "ent_prob":
        p0 = torch.softmax(z0 / spec["T0"], dim=-1)
        p2 = torch.softmax(z2 / spec["T2"], dim=-1)
        H = -(p0 * p0.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        g = spec["a"] * (1.0 - torch.exp(-H))
        return p0 + g * p2
    if m == "ent_logit":
        z0s = z0 / spec["T0"]
        p0 = torch.softmax(z0s, dim=-1)
        H = -(p0 * p0.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        g = spec["a"] * (1.0 - torch.exp(-H))
        s0 = z0s.std(dim=-1, keepdim=True).clamp_min(1e-6)
        s2 = z2.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return z0s + g * z2 * (s0 / s2)
    raise ValueError(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config.fromfile(os.path.join(args.exp_dir, "config.py"))
    cfg.model.backbone.enable_flash = False
    model = build_model(cfg.model).cuda().eval()
    ckpt_path = args.ckpt or os.path.join(args.exp_dir, "model", "regis1_model_last.pth")
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)

    test_dataset = build_dataset(cfg.data.test)
    num_classes = cfg.model.num_base_classes + cfg.model.num_novel_classes
    ignore_index = cfg.data.ignore_index
    n_base = cfg.model.num_base_classes

    grid = build_grid()
    inter = {name: np.zeros(num_classes, dtype=np.float64) for name, _ in grid}
    union = {name: np.zeros(num_classes, dtype=np.float64) for name, _ in grid}
    target = {name: np.zeros(num_classes, dtype=np.float64) for name, _ in grid}

    # Capture z0 and z2 cleanly via forward hooks on base_cross_head / novel_cross_head and seg_head / novel_seg_head
    buf = {}
    def _cap(key):
        def hk(mod, inp, out):
            buf[key] = out.detach().float()
        return hk
    h1 = model.seg_head.register_forward_hook(_cap("b0"))
    h2 = model.novel_seg_head.register_forward_hook(_cap("n0"))
    h3 = model.base_cross_head.register_forward_hook(_cap("bc"))
    h4 = model.novel_cross_head.register_forward_hook(_cap("nc"))

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            data_dict = test_dataset[idx]
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            n_pts = segment.shape[0]
            if "origin_segment" in data_dict:
                inv = data_dict["inverse"]
                seg_eval = data_dict["origin_segment"]
            else:
                inv = None
                seg_eval = segment

            z0_sum = torch.zeros((n_pts, num_classes), dtype=torch.float32, device="cuda")
            z2_sum = torch.zeros((n_pts, num_classes), dtype=torch.float32, device="cuda")
            cnt = torch.zeros((n_pts, 1), dtype=torch.float32, device="cuda")

            for frag in fragment_list:
                inp = collate_fn([frag])
                idx_part = inp.pop("index").cuda(non_blocking=True)
                for k, v in inp.items():
                    if isinstance(v, torch.Tensor):
                        inp[k] = v.cuda(non_blocking=True)
                buf.clear()
                _ = model(inp)
                z0_p = torch.cat([buf["b0"], buf["n0"]], dim=-1)
                z2_p = torch.cat([buf["bc"], buf["nc"]], dim=-1)
                bs = 0
                for be in inp["offset"]:
                    z0_sum[idx_part[bs:be]] += z0_p[bs:be]
                    z2_sum[idx_part[bs:be]] += z2_p[bs:be]
                    cnt[idx_part[bs:be]] += 1.0
                    bs = be

            z0_avg = z0_sum / cnt.clamp_min(1.0)
            z2_avg = z2_sum / cnt.clamp_min(1.0)

            for name, spec in grid:
                score = apply_fusion(z0_avg, z2_avg, spec)
                pred = score.argmax(dim=-1).cpu().numpy()
                if inv is not None:
                    pred = pred[inv]
                i_c, u_c, t_c = intersection_and_union(pred, seg_eval, num_classes, ignore_index)
                inter[name] += i_c
                union[name] += u_c
                target[name] += t_c

            if (idx + 1) % 10 == 0 or (idx + 1) == len(test_dataset):
                print(f"[{idx+1}/{len(test_dataset)}] done", flush=True)

    for h in (h1, h2, h3, h4):
        h.remove()

    results = {}
    for name, _ in grid:
        iou_c = inter[name] / (union[name] + 1e-10)
        b = float(np.mean(iou_c[:n_base]))
        n = float(np.mean(iou_c[n_base:]))
        hm = 2 * b * n / (b + n + 1e-10)
        results[name] = dict(Base=round(b, 4), Novel=round(n, 4), HM=round(hm, 4), mIoU=round(float(np.mean(iou_c)), 4), novel_ious=[round(float(x), 4) for x in iou_c[n_base:]])

    out_path = args.out or os.path.join(args.exp_dir, "fusion_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved:", out_path)
    for k in ("z0_only", "z2_only", "raw_sum"):
        print(f"  {k:24s}: {results[k]['Base']:.4f} / {results[k]['Novel']:.4f} / {results[k]['HM']:.4f}")
    best = max(results.items(), key=lambda kv: kv[1]["HM"])
    print(f"  BEST {best[0]:20s}: {best[1]['Base']:.4f} / {best[1]['Novel']:.4f} / {best[1]['HM']:.4f}")


if __name__ == "__main__":
    main()
