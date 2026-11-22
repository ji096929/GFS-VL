'''任务 1：sym_xl 的 eRank + base_cross_head 范数 + 频次校准扫描。'''
import sys

import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/GFS-VL')
from pointcept.datasets import build_dataset, collate_fn  # noqa: E402
from pointcept.engines.defaults import (  # noqa: E402
    default_config_parser, default_setup)
from pointcept.models import build_model  # noqa: E402

seed = sys.argv[1]
shard = int(sys.argv[2])
shards = int(sys.argv[3])
CFG = 'configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_sym.py'
CK = './exp/scannet200/sym_xl_s%s/model/regis1_model_last.pth' % seed
OUT = '/data01/jh/afford_diag/xl_runs/task1_sym_s%s_shard%d.npz' % (seed, shard)
NC = 57
TAUS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]
CAP = 1500

cfg = default_config_parser(CFG, {})
cfg.save_path = '/tmp/task1_%s_%d' % (seed, shard)
cfg = default_setup(cfg)
model = build_model(cfg.model).cuda().eval()
ck = torch.load(CK, map_location='cpu')
sd = {k[7:] if k.startswith('module.') else k: v
      for k, v in ck['state_dict'].items()}
m, u = model.load_state_dict(sd, strict=False)
print('load missing=%d unexpected=%d' % (len(m), len(u)), flush=True)
model.backbone.shuffle_orders = False

Wb = model.base_cross_head.weight.detach().cpu().numpy()
base_norms = np.linalg.norm(Wb, axis=1)

cnt_pl = np.load('/data01/jh/afford_diag/xl_runs/pl_counts.npy')
logNc = np.log(cnt_pl[12:].astype(np.float64) + 1e-6)
logNc = logNc - logNc.mean()
lg_t = torch.tensor(logNc, device='cuda', dtype=torch.float32)

rec = {}
model.dec0_norm.register_forward_hook(
    lambda mm, i, o: rec.__setitem__('n0', o.detach()))
model.cross_outs['1'].register_forward_hook(
    lambda mm, i, o: rec.__setitem__('p1', o.detach()))
model.cross_outs['2'].register_forward_hook(
    lambda mm, i, o: rec.__setitem__('p2', o.detach()))

ds = build_dataset(cfg.data.test)
ds.data_list = ds.data_list[shard::shards]
print('scenes %d' % len(ds.data_list), flush=True)

cms = {t: np.zeros((NC, NC), np.int64) for t in TAUS}
acc = []
cnt = np.zeros(NC, np.int64)
rng = np.random.RandomState(0)

for i in range(len(ds.data_list)):
    d = ds[i]
    frags = d.pop('fragment_list')
    gt = np.asarray(d['segment']).astype(np.int64)
    for frag in frags:
        inp = collate_fn([frag])
        for k in inp:
            if isinstance(inp[k], torch.Tensor):
                inp[k] = inp[k].cuda(non_blocking=True)
        ip = inp['index'].cpu().numpy()
        seg = gt[ip]
        with torch.no_grad():
            out = model(inp)
            logits = out['seg_logits'].float()
            feat = torch.cat([rec['n0'], rec['p1'], rec['p2']], 1)
            for t in TAUS:
                lg = logits.clone()
                lg[:, 12:] = lg[:, 12:] - t * lg_t
                pred = lg.argmax(1).cpu().numpy()
                ok = seg >= 0
                cms[t] += np.bincount(seg[ok] * NC + pred[ok],
                                      minlength=NC * NC).reshape(NC, NC)
            f = feat.cpu().numpy().astype(np.float32)
        for c in range(NC):
            w = np.where((seg == c))[0]
            if len(w) == 0:
                continue
            rng.shuffle(w)
            need = CAP - cnt[c]
            if need > 0:
                a = w[:need]
                acc.append((ip[a], f[a]))
                cnt[c] += len(a)
    if (i + 1) % 20 == 0:
        print('  %d/%d cnt=%d' % (i + 1, len(ds.data_list), cnt.sum()),
              flush=True)

feat_full = np.zeros((0, 192), np.float32)
if acc:
    idxs = np.concatenate([a[0] for a in acc])
    arrs = np.concatenate([a[1] for a in acc], 0)
    feat_full = arrs
np.savez(OUT, base_norms=base_norms, cnt_pl=cnt_pl,
         feat=feat_full, **{'cm_%s' % t: cms[t] for t in TAUS})
print('saved', OUT, 'feat', feat_full.shape, flush=True)
