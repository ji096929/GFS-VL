"""训练后诊断：读出支路 eRank + novel_head 三切片范数占比。

用法: python xl_erank.py <run> [shard shards]
run: base_s42 / xl_s42 / base_s7 / xl_s7
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/data01/jh/code/GFS-VL")
from pointcept.datasets import build_dataset, collate_fn  # noqa: E402
from pointcept.engines.defaults import (  # noqa: E402
    default_config_parser, default_setup)
from pointcept.models import build_model  # noqa: E402

ROOT = "/data01/jh/code/GFS-VL"
CFG_BASE = ROOT + "/configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1.py"
CFG_XL = ROOT + "/configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_xl.py"
CFG_REP3 = ROOT + "/configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_rep3.py"
RUNS = {
    "base_s42": (CFG_BASE, ROOT + "/exp/scannet200/sc200_k1_BASEu_s42/model/regis1_model_last.pth"),
    "xl_s42": (CFG_XL, ROOT + "/exp/scannet200/sc200_k1_XLu_s42/model/regis1_model_last.pth"),
    "base_s7": (CFG_BASE, ROOT + "/exp/scannet200/sc200_k1_BASEu_s7/model/regis1_model_last.pth"),
    "xl_s7": (CFG_XL, ROOT + "/exp/scannet200/sc200_k1_XLu_s7/model/regis1_model_last.pth"),
    "test_online": (CFG_BASE, ROOT + "/exp/scannet200/sc200_k1_ONLINE_s42/model/regis1_model_last.pth"),
    "rep3_s42": (CFG_REP3, ROOT + "/exp/scannet200/rep3_s42/model/regis1_model_last.pth"),
}
CAP = 1500
NC = 57


def erank(X):
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    w = np.linalg.eigvalsh(C.astype(np.float64))
    w = np.maximum(w, 0.0)
    p = w / w.sum()
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


class Collector:
    def __init__(self):
        self.f, self.inv, self.h = {}, {}, []

    def attach(self, model):
        bb = model.backbone
        for s in range(4):
            up = bb.dec._modules["dec%d" % s]._modules["up"]
            self.h.append(up.register_forward_pre_hook(
                lambda m, i, s=s: self.inv.__setitem__(
                    s, i[0]["pooling_inverse"])))
        for s in (0, 1, 2):
            d = bb.dec._modules["dec%d" % s]
            self.h.append(d.register_forward_hook(
                lambda m, i, o, s=s: self.f.__setitem__(s, o.feat.detach())))
        if getattr(model, "cross_layers", ()):
            self.h.append(model.dec0_norm.register_forward_hook(
                lambda m, i, o: self.f.__setitem__("bn0", o.detach())))
            for s in model.cross_layers:
                self.h.append(model.cross_outs[str(s)].register_forward_hook(
                    lambda m, i, o, s=s: self.f.__setitem__(
                        "bp%d" % s, o.detach())))

    def cols(self, has_cross):
        out = {"dec0": self.f[0].float()}
        if has_cross:
            out["bn0"] = self.f["bn0"].float()
            names = ["bp%d" % s for s in sorted(
                int(k[2:]) for k in self.f
                if isinstance(k, str) and k.startswith("bp"))]
            for k in names:
                out[k] = self.f[k].float()
            out["cat192"] = torch.cat([out["bn0"]] + [out[k] for k in names],
                                      dim=1)
        return out

    def clear(self):
        self.f.clear()


def main():
    run = sys.argv[1]
    shard = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    shards = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    cfgp, ckp = RUNS[run]
    cfg = default_config_parser(cfgp, {})
    cfg.save_path = "/tmp/xl_erank_%s" % run
    cfg = default_setup(cfg)
    model = build_model(cfg.model).cuda().eval()
    ck = torch.load(ckp, map_location="cpu")
    sd = {k[7:] if k.startswith("module.") else k: v
          for k, v in ck["state_dict"].items()}
    m, u = model.load_state_dict(sd, strict=False)
    print("%s: epoch=%s missing=%d unexpected=%d"
          % (run, ck.get("epoch"), len(m), len(u)))
    model.backbone.shuffle_orders = False
    has_cross = bool(getattr(model, "cross_layers", ()))
    if has_cross:
        W = model.novel_seg_head.weight.detach()
        norms = []
        print("novel_head %s" % (tuple(W.shape),))
        for i in range(0, W.shape[1], 64):
            n = float(W[:, i:i + 64].norm())
            norms.append(n)
            print("  slice[%3d:%3d] |W|_F=%.4f" % (i, i + 64, n))
        print("  share=%s" % [round(n / sum(norms), 4) for n in norms])
    else:
        print("seg_head %s" % (tuple(model.seg_head.weight.shape),))

    col = Collector()
    col.attach(model)
    ds = build_dataset(cfg.data.test)
    ds.data_list = ds.data_list[shard::shards]
    print("val shard %d/%d: %d scenes"
          % (shard, shards, len(ds.data_list)), flush=True)

    acc = {}
    gt_acc = []
    cnt = np.zeros(NC, np.int64)
    rng = np.random.RandomState(0)
    for i in range(len(ds.data_list)):
        d = ds[i]
        frags = d.pop("fragment_list")
        gt = np.asarray(d["segment"]).astype(np.int64)
        per = {}
        cov = np.zeros(len(gt), np.int64)
        for frag in frags:
            inp = collate_fn([frag])
            for k in inp:
                if isinstance(inp[k], torch.Tensor):
                    inp[k] = inp[k].cuda(non_blocking=True)
            ip = inp["index"].cpu().numpy()
            with torch.no_grad():
                model(inp)
                cs = col.cols(has_cross)
            for k, v in cs.items():
                per.setdefault(k, []).append(
                    (ip, v.cpu().numpy().astype(np.float32)))
            cov[ip] += 1
            col.clear()
        for k in per:
            dim = per[k][0][1].shape[1]
            full = np.zeros((len(gt), dim), np.float32)
            for ip, arr in per[k]:
                full[ip] = arr
            per[k] = full
        for c in range(NC):
            w = np.where((gt == c) & (cov > 0))[0]
            if len(w) == 0:
                continue
            rng.shuffle(w)
            need = CAP - cnt[c]
            if need <= 0:
                continue
            a = w[:need]
            cnt[c] += len(a)
            for k in per:
                acc.setdefault(k, []).append(per[k][a])
            gt_acc.append(gt[a])
        if (i + 1) % 40 == 0:
            print("  %d/%d cnt=%d cov=%d"
                  % (i + 1, len(ds.data_list), cnt.sum(), (cov > 0).sum()),
                  flush=True)

    dump = os.environ.get("ERANK_DUMP", "")
    if dump:
        np.savez(dump, gt=np.concatenate(gt_acc, 0),
                 **{k: np.concatenate(acc[k], 0) for k in acc})
        print("dumped", dump, {k: np.concatenate(acc[k], 0).shape
                               for k in acc})
        return
    print("== %s eRank (%d points) ==" % (run, cnt.sum()))
    for k in sorted(acc):
        X = np.concatenate(acc[k], 0)
        print("  %-8s D=%3d eRank=%.2f" % (k, X.shape[1], erank(X)))


if __name__ == "__main__":
    main()
