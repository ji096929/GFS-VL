'''抓 agglayers[0] 的输入，对比 corr 与 base_g。'''
import glob, os, sys
sys.path.insert(0, '/data01/jh/code'); import fake_tp
sys.path.insert(0, '/data01/jh/code/COSeg'); sys.path.insert(0, '/data01/jh/code')
import numpy as np, torch, torch.nn.functional as F
from ocsb_core import load_model, feats_one, build_proto, predict_agg

f = sorted(glob.glob('/data01/jh/data/sonata_dump/ep*.npz'))[2]
d = np.load(f); c = int(d['cls'][0])
m, cfg = load_model(); dev = torch.device('cuda')
sx = torch.cat([torch.from_numpy(d['s_coord'].astype(np.float32)),
                torch.from_numpy(d['s_rgb'].astype(np.float32))/255.], 1).cuda()
qx = torch.cat([torch.from_numpy(d['q_coord'].astype(np.float32)),
                torch.from_numpy(d['q_rgb'].astype(np.float32))/255.], 1).cuda()
sy = torch.from_numpy(d['s_lab'].astype(np.int64)).cuda()
qy = torch.from_numpy(d['q_lab'].astype(np.int64)).cuda()
so = torch.tensor([len(sx)], dtype=torch.int32)
qo = torch.tensor([len(qx)], dtype=torch.int32)

cap = []
h = m.agglayers[0].register_forward_pre_hook(
    lambda mod, inp: cap.append((inp[0].detach().clone(),
                                 None if inp[1] is None else inp[1].detach().clone())))
with torch.no_grad(), torch.cuda.amp.autocast():
    m(so, sx, sy, qo, qx, qy, 5, support_base_y=None, query_base_y=None,
      sampled_classes=np.array([c]))
sf = feats_one(m, sx, sy, dev); qf = feats_one(m, qx, qy, dev)
fg = sf['yl'].bool()
sp = torch.cat([build_proto(m, sf, ~fg), build_proto(m, sf, fg)], 0)
with torch.no_grad():
    predict_agg(m, qf, [sp], 0.0, dev)
h.remove()
print('n calls', len(cap))
(a1, g1), (a2, g2) = cap[0], cap[1]
print('corr diff mean %.6f max %.6f' % ((a1-a2).abs().mean(), (a1-a2).abs().max()))
print('base_g: official', float(g1.min()), float(g1.max()), float(g1.mean()))
print('base_g: agg     ', float(g2.min()), float(g2.max()), float(g2.mean()))
print('base_g diff mean %.6f max %.6f' % ((g1-g2).abs().mean(), (g1-g2).abs().max()))
print('base_g equal:', torch.allclose(g1, g2, atol=1e-5))
print('agglayers[0]:', m.agglayers[0])
