"""Pre-flight 校验：mlp / sym / gated 三种分支。"""
import sys

import torch

sys.path.insert(0, "/data01/jh/code/GFS-VL")
from pointcept.datasets import build_dataset, collate_fn  # noqa: E402
from pointcept.engines.defaults import (  # noqa: E402
    default_config_parser, default_setup)
from pointcept.models import build_model  # noqa: E402
from pointcept.utils.optimizer import build_optimizer  # noqa: E402

ROOT = "/data01/jh/code/GFS-VL/configs/scannet200/"
MODES = {
    "mlp": ROOT + "semseg-pt-v3m1-0-gfsregistrain_k1_mlp.py",
    "sym": ROOT + "semseg-pt-v3m1-0-gfsregistrain_k1_sym.py",
    "gated": ROOT + "semseg-pt-v3m1-0-gfsregistrain_k1_gated.py",
}
NEW = ("cross_", "dec0_norm", "base_cross", "depth_gate", "novel_mlp")


def check(name, cfgp):
    print("\n===== %s =====" % name)
    cfg = default_config_parser(cfgp, {})
    cfg.save_path = "/tmp/pf_%s" % name
    cfg = default_setup(cfg)
    model = build_model(cfg.model).cuda()
    if model.novel_seg_head is None:
        print("novel head: MLP -> %s" % (tuple(model.novel_mlp[-1].weight.shape),))
    else:
        print("novel head: %s" % (tuple(model.novel_seg_head.weight.shape),))

    ck = torch.load(cfg.backbone_weight, map_location="cpu")
    sd = {k[7:] if k.startswith("module.") else k: v
          for k, v in ck["state_dict"].items()}
    mi, un = model.load_state_dict(sd, strict=False)
    bad = [k for k in mi if not k.startswith(NEW)]
    print("load: missing=%d unexpected=%d 非新模块missing=%s"
          % (len(mi), len(un), bad))

    ds = build_dataset(cfg.data.test)
    d = ds[0]
    frag = d["fragment_list"][0]
    inp = collate_fn([frag])
    for k in inp:
        if isinstance(inp[k], torch.Tensor):
            inp[k] = inp[k].cuda(non_blocking=True)
    model.eval()

    rec = {}
    if getattr(model, "base_cross", False):
        model.seg_head.register_forward_hook(
            lambda m, i, o: rec.__setitem__("base", o.detach()))
    if getattr(model, "depth_gate", False):
        model.depth_gate.register_forward_hook(
            lambda m, i, o: rec.__setitem__("dg", o.detach()))

    with torch.no_grad():
        out = model(inp)
    print("logits %s" % (tuple(out["seg_logits"].shape),))
    if "base" in rec:
        diff = (out["seg_logits"][:, :12] - rec["base"]).abs().max()
        print("Step0 base logits max|diff| = %.3e" % float(diff))
    if "dg" in rec:
        sg = torch.sigmoid(rec["dg"])
        print("depth_gate sigmoid mean=%.4f min=%.4f max=%.4f"
              % (float(sg.mean()), float(sg.min()), float(sg.max())))
    print("cache after fwd = %d" % len(model._cr))

    opt = build_optimizer(cfg.optimizer, model, cfg.param_dicts)
    name_of = {id(p): n for n, p in model.named_parameters()}
    for gi, g in enumerate(opt.param_groups):
        hit = [name_of[id(p)] for p in g["params"]
               if name_of[id(p)].startswith(NEW)]
        print("group%d lr=%.4f total=%d new=%s"
              % (gi, g["lr"], len(g["params"]), hit))

    model.train()
    seg = inp.get("segment")
    if seg is None:
        seg = torch.randint(0, 57, (inp["coord"].shape[0],), device="cuda")
    with torch.cuda.amp.autocast(enabled=True):
        out = model(inp)
        loss = model.criteria(out["seg_logits"], seg)
    loss.backward()
    print("loss=%.4f" % float(loss))
    for nm, p in model.named_parameters():
        if nm.startswith(NEW):
            g = None if p.grad is None else float(p.grad.norm())
            print("  grad %-26s %s" % (nm, "None" if g is None else "%.3e" % g))


for n, p in MODES.items():
    try:
        check(n, p)
    except Exception as e:
        print("[%s] FAILED: %r" % (n, e))
