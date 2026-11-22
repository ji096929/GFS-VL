'''分析 gated 模型的逐点门控值：细边界类 vs 复杂几何类。'''
import sys

import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/GFS-VL')
from pointcept.datasets import build_dataset, collate_fn  # noqa: E402
from pointcept.datasets.preprocessing.scannet.meta_data import (  # noqa: E402
    scannet200_constants as const)
from pointcept.engines.defaults import (  # noqa: E402
    default_config_parser, default_setup)
from pointcept.models import build_model  # noqa: E402

ROOT = '/data01/jh/code/GFS-VL'
CFG = ROOT + '/configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_gated.py'
CK = ROOT + '/exp/scannet200/gated_sym_xl_s42/model/regis1_model_last.pth'
NAMES = list(const.CLASS_LABELS_BASE_NOVEL)


def main():
    cfg = default_config_parser(CFG, {})
    cfg.save_path = '/tmp/gate_an'
    cfg = default_setup(cfg)
    model = build_model(cfg.model).cuda().eval()
    ck = torch.load(CK, map_location='cpu')
    sd = {k[7:] if k.startswith('module.') else k: v
          for k, v in ck['state_dict'].items()}
    m, u = model.load_state_dict(sd, strict=False)
    print('load missing=%d unexpected=%d' % (len(m), len(u)))
    model.backbone.shuffle_orders = False

    rec = {}
    model.depth_gate.register_forward_hook(
        lambda mm, i, o: rec.__setitem__('g', o.detach()))
    ds = build_dataset(cfg.data.test)
    acc = {}
    for i in range(len(ds.data_list)):
        d = ds[i]
        frags = d.pop('fragment_list')
        gt = np.asarray(d['segment']).astype(np.int64)
        g_full = np.full((len(gt), 2), np.nan, np.float32)
        for frag in frags:
            inp = collate_fn([frag])
            for k in inp:
                if isinstance(inp[k], torch.Tensor):
                    inp[k] = inp[k].cuda(non_blocking=True)
            ip = inp['index'].cpu().numpy()
            with torch.no_grad():
                model(inp)
            g_full[ip] = torch.sigmoid(rec['g']).float().cpu().numpy()
        ok = ~np.isnan(g_full[:, 0])
        for c in range(57):
            sel = (gt == c) & ok
            if sel.sum() > 0:
                acc.setdefault(c, []).append(g_full[sel].mean(0))
        if (i + 1) % 60 == 0:
            print(i + 1, flush=True)

    print('%-24s %8s %8s' % ('class', 'g1_mean', 'g2_mean'))
    for c in range(57):
        if c in acc:
            v = np.mean(np.stack(acc[c]), 0)
            print('%-24s %8.4f %8.4f' % (NAMES[c], v[0], v[1]))


if __name__ == '__main__':
    main()
