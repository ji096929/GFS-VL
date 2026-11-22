"""零训练剖析 v2：按 tester 的【分片推理】跑 312 val，输出混淆矩阵拆分 + δ 扫描。

用法:
  python val_cm_probe2.py <exp_config.py> <weight.pth>
"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, '/data01/jh/code/GFS-VL')

from pointcept.datasets import build_dataset  # noqa: E402
from pointcept.datasets.utils import collate_fn  # noqa: E402
from pointcept.engines.defaults import (  # noqa: E402
    default_config_parser, default_setup)
from pointcept.models import build_model  # noqa: E402

NO, NC = 12, 57
DELTAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def metrics(cm):
    diag = np.diag(cm).astype(np.float64)
    row = cm.sum(1).astype(np.float64)
    col = cm.sum(0).astype(np.float64)
    iou = diag / np.maximum(row + col - diag, 1)
    base = float(iou[:NO].mean())
    nov = float(iou[NO:].mean())
    hm = 2 * base * nov / max(base + nov, 1e-9)
    return base, nov, hm


def main():
    cfg = default_config_parser(sys.argv[1], {})
    cfg.save_path = '/tmp/cm_probe'
    cfg.weight = sys.argv[2]
    cfg = default_setup(cfg)

    model = build_model(cfg.model).cuda().eval()
    ck = torch.load(cfg.weight, map_location='cpu')
    sd = {k[7:] if k.startswith('module.') else k: v
          for k, v in ck['state_dict'].items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print('weight %s | missing %d unexpected %d | epoch %s'
          % (cfg.weight, len(miss), len(unexp), ck.get('epoch')), flush=True)

    ds = build_dataset(cfg.data.test)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1,
                        collate_fn=lambda b: b)
    print('val scenes %d' % len(ds), flush=True)

    cms = {d: np.zeros((NC, NC), dtype=np.int64) for d in DELTAS}
    bmask = torch.zeros(NC, device='cuda')
    bmask[:NO] = 1.0
    ignore = cfg.data.ignore_index

    t0 = time.time()
    for idx, data_dict in enumerate(loader):
        if isinstance(data_dict, (list, tuple)):
            data_dict = data_dict[0]
        frags = data_dict.pop("fragment_list")
        segment = data_dict.pop("segment")
        n = int(segment.size)
        pred = {d: torch.zeros((n, NC), device='cuda') for d in DELTAS}
        for i in range(len(frags)):
            inp = collate_fn(frags[i:i + 1])
            for k in inp:
                if isinstance(inp[k], torch.Tensor):
                    inp[k] = inp[k].cuda(non_blocking=True)
            ip = inp["index"]
            with torch.no_grad():
                logits = model(inp)["seg_logits"]
                for d in DELTAS:
                    p = F.softmax(logits - d * bmask, -1)
                    bs = 0
                    for be in inp["offset"]:
                        pred[d][ip[bs:be], :] += p[bs:be]
                        bs = be
        seg = segment
        if "origin_segment" in data_dict:
            seg = data_dict["origin_segment"]
        for d in DELTAS:
            pr = pred[d].max(1)[1]
            if "inverse" in data_dict:
                pr = pr[data_dict["inverse"]]
            pr = pr.cpu().numpy()
            sg = seg.cpu().numpy() if torch.is_tensor(seg) else np.asarray(seg)
            ok = sg != ignore
            cms[d] += np.bincount((sg[ok] * NC + pr[ok]),
                                  minlength=NC * NC).reshape(NC, NC)
        if (idx + 1) % 50 == 0:
            print('  %d/%d  %.0fs' % (idx + 1, len(ds), time.time() - t0), flush=True)

    cm = cms[0.0]
    diag = np.diag(cm).astype(np.float64)
    tn = float(cm[NO:].sum())
    corr = float(diag[NO:].sum())
    to_base = float(cm[NO:, :NO].sum())
    oth = float(cm[NO:, NO:].sum() - diag[NO:].sum())
    print('\n=== 45 个新类真值点去向（%d 点）===' % int(tn))
    print('  判对             %9d  %5.1f%%' % (corr, 100 * corr / tn))
    print('  错判成 12 个基类  %9d  %5.1f%%' % (to_base, 100 * to_base / tn))
    print('  错判成其他新类    %9d  %5.1f%%' % (oth, 100 * oth / tn))

    pn = float(cm[:, NO:].sum())
    fp = pn - corr
    fpb = float(cm[:NO, NO:].sum())
    fpo = float(cm[NO:, NO:].sum() - diag[NO:].sum())
    print('\n=== 新类假阳性来源（%d 点）===' % int(fp))
    print('  GT 是 12 个基类   %9d  %5.1f%%' % (fpb, 100 * fpb / max(fp, 1)))
    print('  GT 是其他新类     %9d  %5.1f%%' % (fpo, 100 * fpo / max(fp, 1)))

    print('\n=== δ 扫描（基类 logits 整体减 δ）===')
    print('%6s %8s %8s %8s' % ('delta', 'Base', 'Novel', 'HM'))
    for d in DELTAS:
        b, nv, h = metrics(cms[d])
        print('%6.1f %8.4f %8.4f %8.4f' % (d, b, nv, h))


if __name__ == '__main__':
    main()
