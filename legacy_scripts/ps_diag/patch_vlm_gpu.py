# -*- coding: utf-8 -*-
"""给 vlm_forward 加 return_gpu 选项：让调用方自己决定是否把 400MB 特征搬回 CPU。

默认 False，行为完全不变（dump_online_pl.py 等旧调用不受影响）。
"""
import io

P = "/data01/jh/code/ps_diag/vlm_common.py"
src = io.open(P, encoding="utf-8").read()
if "return_gpu" in src:
    print("已支持 return_gpu，跳过")
    raise SystemExit(0)

old_sig = 'def vlm_forward(model, cfg, coord, color_u8):\n'
new_sig = ('def vlm_forward(model, cfg, coord, color_u8, return_gpu=False):\n'
           '    """return_gpu=True 时返回 GPU tensor，避免 400MB 的 H2D/D2H 往返。"""\n')
assert src.count(old_sig) == 1
src = src.replace(old_sig, new_sig)

old_ret = '    out = model(data)\n    return out["all_feats"].cpu().numpy(), out["seg_preds"].cpu().numpy()\n'
new_ret = ('    out = model(data)\n'
           '    if return_gpu:\n'
           '        return out["all_feats"], out["seg_preds"]\n'
           '    return out["all_feats"].cpu().numpy(), out["seg_preds"].cpu().numpy()\n')
assert src.count(old_ret) == 1, "返回锚点 %d" % src.count(old_ret)
src = src.replace(old_ret, new_ret)

io.open(P, "w", encoding="utf-8").write(src)
print("已给 vlm_forward 加 return_gpu")
