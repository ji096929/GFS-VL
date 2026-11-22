"""Minimal reproducibility probe: same manifest episode, same GPU, two runs.

If two separate processes on the SAME gpu give different IoU, the model build
path is nondeterministic. If they agree, the earlier mismatch came from mixing
A6000 and 6000 Ada cards.
"""
import json
import random
import sys

import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code')
import diag_bc as H  # noqa: E402

# device is chosen by CUDA_VISIBLE_DEVICES; inside the process it is always 0
gpu = 0
# user hypothesis: a per-episode seed reset removes the cross-process drift
if len(sys.argv) > 1 and sys.argv[1] == 'seed':
    np.random.seed(10000)
    torch.manual_seed(10000)
    random.seed(10000)

model, cfg = H.build_model()
blocks = H.scan_blocks('/data01/jh/data/ScanNet/blocks_bs1_s1/data')
by = {b['name']: b for b in blocks}
man = json.load(open('/data01/jh/data/manifest_fold0.json'))['episodes'][:8]


def load(info):
    d = np.load(info['path'])
    return H.data_prepare_seg(d[:, 0:3], d[:, 3:6], d[:, 6], d[:, 7],
                              cfg.voxel_size, cfg.voxel_max)


dev = torch.device('cuda:0')
out = []
for ep_i, spec in enumerate(man):
    if len(sys.argv) > 1 and sys.argv[1] == 'seed':
        np.random.seed(10000 + ep_i)
        torch.manual_seed(10000 + ep_i)
        random.seed(10000 + ep_i)
    s, q, c = by[spec['support']], by[spec['query']], spec['class']
    sc, sf, sl, _ = load(s)
    qc, qf, ql, _ = load(q)
    sx = torch.cat([sc, sf], 1).to(dev)
    qx = torch.cat([qc, qf], 1).to(dev)
    so = torch.tensor([sc.shape[0]], dtype=torch.int32).to(dev)
    qo = torch.tensor([qc.shape[0]], dtype=torch.int32).to(dev)
    sb = (sl == c).long().to(dev)
    qm = (ql == c).long().to(dev)
    with torch.no_grad():
        p, _ = model(so, sx, sb, qo, qx, qm, 100, None, None, np.array([c]))
    lg = p[0, 1]
    pred = (lg > 0).long()
    tp = ((pred == 1) & (qm == 1)).sum().item()
    fp = ((pred == 1) & (qm == 0)).sum().item()
    fn = ((pred == 0) & (qm == 1)).sum().item()
    out.append(tp / max(tp + fp + fn, 1))

print('gpu=%d  %s' % (gpu, ' '.join('%.6f' % v for v in out)))
print('mean %.6f' % float(np.mean(out)))
