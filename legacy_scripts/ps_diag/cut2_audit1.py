"""疑点 1a：各层特征在 val 上的 L2 范数量级。

判据：若 dec3/enc4 的范数是 dec0 的 3 倍以上，multi 的裸 concat 就会被高维粗糙层主导。
"""
import sys

import numpy as np
import torch

from cut2_common import LAYERS, Probe, build_model_and_cfg, make_ds
from pointcept.datasets import collate_fn  # noqa: E402


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'S2'
    nscene = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    model, cfg = build_model_and_cfg(tag)
    pb = Probe()
    pb.attach(model.backbone)
    ds = make_ds(cfg, 'val')
    ds.data_list = ds.data_list[:nscene]
    acc = {L: [] for L in LAYERS}
    for i in range(len(ds.data_list)):
        d = ds[i]
        frags = d.pop('fragment_list')
        d.pop('segment')
        for frag in frags:
            inp = collate_fn([frag])
            for k in inp:
                if isinstance(inp[k], torch.Tensor):
                    inp[k] = inp[k].cuda(non_blocking=True)
            with torch.no_grad():
                model(inp)
                fs = pb.layers()
            for L in LAYERS:
                acc[L].append(fs[L].norm(dim=1).cpu().numpy())
    print('== %s 各层逐点 L2 范数（%d 场景）==' % (tag, nscene))
    print('%-6s %8s %8s %8s %8s %10s' %
          ('层', 'mean', 'std', 'p5', 'p95', '相对dec0'))
    base = np.concatenate(acc['dec0']).mean()
    for L in LAYERS:
        n = np.concatenate(acc[L])
        print('%-6s %8.2f %8.2f %8.2f %8.2f %9.2fx' %
              (L, n.mean(), n.std(), np.percentile(n, 5),
               np.percentile(n, 95), n.mean() / base))


if __name__ == '__main__':
    main()
