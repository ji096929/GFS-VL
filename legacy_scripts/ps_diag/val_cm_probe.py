"""零训练剖析：312 val 混淆矩阵拆分 + 基类 logit 偏置 δ 扫描。

用法:
  python val_cm_probe.py <exp_config.py> [weight]

输出
  ① 45 个新类真值点的错误去向：错判成 12 基类 / 另外 44 新类 的百分比
  ② 新类假阳性的来源：来自基类真值点 / 来自其他新类真值点 的百分比
  ③ δ 扫描：把 12 个基类 logits 整体减 δ 后的 Novel / HM
"""
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/data01/jh/code/GFS-VL')

from pointcept.datasets import build_dataset  # noqa: E402
from pointcept.datasets.utils import collate_fn  # noqa: E402
from pointcept.engines.defaults import (  # noqa: E402
    default_config_parser, default_setup)
from pointcept.models import build_model  # noqa: E402

NO, NC = 12, 57
DELTAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]


def metrics(cm):
    diag = np.diag(cm).astype(np.float64)
    row = cm.sum(1).astype(np.float64)
    col = cm.sum(0).astype(np.float64)
    iou = diag / np.maximum(row + col - diag, 1)
    base = float(iou[:NO].mean())
    nov = float(iou[NO:].mean())
    hm = 2 * base * nov / max(base + nov, 1e-9)
    return base, nov, hm, iou


def main():
    cfg_path = sys.argv[1]
    cfg = default_config_parser(cfg_path, {})
    cfg.save_path = '/tmp/cm_probe'
    if len(sys.argv) > 2:
        cfg.weight = sys.argv[2]
    cfg = default_setup(cfg)

    model = build_model(cfg.model).cuda().eval()
    ck = torch.load(cfg.weight, map_location='cpu')
    sd = {k[7:] if k.startswith('module.') else k: v
          for k, v in ck['state_dict'].items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print('weight %s | missing %d unexpected %d | epoch %s'
          % (cfg.weight, len(miss), len(unexp), ck.get('epoch')), flush=True)

    ds = build_dataset(cfg.data.val)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                        collate_fn=collate_fn)
    print('val scenes %d' % len(ds), flush=True)

    cms = {d: np.zeros((NC, NC), dtype=np.int64) for d in DELTAS}
    bmask = torch.zeros(NC, device='cuda')
    bmask[:NO] = 1.0

    for i, b in enumerate(loader):
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                b[k] = v.cuda(non_blocking=True)
        with torch.no_grad(), torch.cuda.amp.autocast():
            out = model(b)
        logits = out['seg_logits'].float()
        gt = b['segment']
        valid = gt >= 0
        if valid.sum() == 0:
            continue
        g = gt[valid]
        for d in DELTAS:
            pred = (logits - d * bmask).argmax(1)[valid]
            idx = (g * NC + pred).cpu().numpy()
            cms[d] += np.bincount(idx, minlength=NC * NC).reshape(NC, NC)
        if (i + 1) % 100 == 0:
            print('  %d/%d' % (i + 1, len(ds)), flush=True)

    cm = cms[0.0]
    diag = np.diag(cm).astype(np.float64)
    truth_novel = cm[NO:].sum()
    correct = diag[NO:].sum()
    to_base = cm[NO:, :NO].sum()
    other_novel = (cm[NO:, NO:].sum() - diag[NO:].sum())
    print('\n=== 45 个新类真值点的去向（共 %d 点）===' % truth_novel)
    print('  判对            %9d  %.1f%%' % (correct, 100 * correct / truth_novel))
    print('  错判成 12 个基类 %9d  %.1f%%' % (to_base, 100 * to_base / truth_novel))
    print('  错判成其他新类   %9d  %.1f%%' % (other_novel, 100 * other_novel / truth_novel))

    pred_novel = cm[:, NO:].sum()
    fp = pred_novel - correct
    fp_from_base = cm[:NO, NO:].sum()
    fp_from_novel = cm[NO:, NO:].sum() - diag[NO:].sum()
    fp_from_ign = 0  # ignore 点不进混淆矩阵
    print('\n=== 新类假阳性来源（共 %d 点）===' % fp)
    print('  GT 是 12 个基类  %9d  %.1f%%' % (fp_from_base, 100 * fp_from_base / max(fp, 1)))
    print('  GT 是其他新类    %9d  %.1f%%' % (fp_from_novel, 100 * fp_from_novel / max(fp, 1)))
    print('  （ignore 点不进混淆矩阵，故不在此表）')

    print('\n=== 基类 logit 偏置 δ 扫描 ===')
    print('%6s %8s %8s %8s' % ('delta', 'Base', 'Novel', 'HM'))
    for d in DELTAS:
        b, n, h, _ = metrics(cms[d])
        print('%6.1f %8.4f %8.4f %8.4f' % (d, b, n, h))


if __name__ == '__main__':
    main()
