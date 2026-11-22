"""启动前验证：cross_layers 分支的 shape / 尺度 / 参数分组 / 梯度 / 缓存。"""
import sys

import torch

sys.path.insert(0, "/data01/jh/code/GFS-VL")
from pointcept.datasets import build_dataset, collate_fn  # noqa: E402
from pointcept.engines.defaults import (  # noqa: E402
    default_config_parser, default_setup)
from pointcept.models import build_model  # noqa: E402
from pointcept.utils.optimizer import build_optimizer  # noqa: E402

CFG = ("/data01/jh/code/GFS-VL/configs/scannet200/"
       "semseg-pt-v3m1-0-gfsregistrain_k1_xl.py")


def main():
    cfg = default_config_parser(CFG, {})
    cfg.save_path = "/tmp/verify_cross"
    cfg = default_setup(cfg)
    model = build_model(cfg.model).cuda()
    print("cross_layers=%s" % (model.cross_layers,))
    print("novel_seg_head %s | dec0_norm %s"
          % (tuple(model.novel_seg_head.weight.shape),
             tuple(model.dec0_norm.normalized_shape)))

    ck = torch.load(cfg.backbone_weight, map_location="cpu")
    sd = {k[7:] if k.startswith("module.") else k: v
          for k, v in ck["state_dict"].items()}
    bb = {k: v for k, v in sd.items() if k.startswith("backbone.")}
    m, u = model.backbone.load_state_dict(bb, strict=False)
    print("backbone load: missing=%d unexpected=%d" % (len(m), len(u)))
    model.eval()

    ds = build_dataset(cfg.data.test)
    d = ds[0]
    frag = d["fragment_list"][0]
    inp = collate_fn([frag])
    for k in inp:
        if isinstance(inp[k], torch.Tensor):
            inp[k] = inp[k].cuda(non_blocking=True)
    print("points=%d keys=%s" % (inp["coord"].shape[0], sorted(inp.keys())))

    rec = {}
    model.dec0_norm.register_forward_hook(
        lambda mm, i, o: rec.__setitem__(
            "dec0", float(o.detach().norm(dim=1).mean())))
    for s in model.cross_layers:
        model.cross_outs[str(s)].register_forward_hook(
            lambda mm, i, o, s=s: rec.__setitem__(
                "p%d" % s, float(o.detach().norm(dim=1).mean())))

    with torch.no_grad():
        out = model(inp)
    print("logits %s" % (tuple(out["seg_logits"].shape),))
    print("branch L2 mean: %s"
          % " | ".join("%s=%.3f" % (k, rec[k]) for k in sorted(rec)))
    print("ratio dec0/p1=%.2fx dec0/p2=%.2fx"
          % (rec["dec0"] / rec["p1"], rec["dec0"] / rec["p2"]))
    print("cache after fwd=%d" % len(model._cr))
    with torch.no_grad():
        model(inp)
    print("cache after 2nd fwd=%d" % len(model._cr))

    opt = build_optimizer(cfg.optimizer, model, cfg.param_dicts)
    name = {id(p): nm for nm, p in model.named_parameters()}
    for gi, g in enumerate(opt.param_groups):
        ns = [name[id(p)] for p in g["params"]]
        hit = [x for x in ns if "cross_" in x or "dec0_norm" in x]
        print("group%d lr=%.4f n=%d new=%s"
              % (gi, g["lr"], len(ns), hit))

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
        if "cross_projs" in nm or "dec0_norm" in nm or "cross_outs" in nm:
            g = None if p.grad is None else float(p.grad.norm())
            print("  grad %-30s %s"
                  % (nm, "None" if g is None else "%.3e" % g))
    bbg = sum(float(p.grad.norm()) for nm, p in model.named_parameters()
              if nm.startswith("backbone.") and p.grad is not None)
    print("backbone grad total norm=%.3e" % bbg)


if __name__ == "__main__":
    main()
