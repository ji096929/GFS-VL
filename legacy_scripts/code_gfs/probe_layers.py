'''逐层对比官方 forward 与 predict_agg：抓 lin1 的输入。'''
import glob, os, sys
sys.path.insert(0, '/data01/jh/code'); import fake_tp
sys.path.insert(0, '/data01/jh/code/COSeg'); sys.path.insert(0, '/data01/jh/code')
import numpy as np, torch, torch.nn.functional as F
from ocsb_core import load_model, feats_one, build_proto, predict_agg

DUMP = '/data01/jh/data/sonata_dump'
f = sorted(glob.glob(os.path.join(DUMP, 'ep*.npz')))[2]
d = np.load(f); c = int(d['cls'][0])
m, cfg = load_model()
dev = torch.device('cuda')

sx = torch.cat([torch.from_numpy(d['s_coord'].astype(np.float32)),
                torch.from_numpy(d['s_rgb'].astype(np.float32))/255.], 1).cuda()
qx = torch.cat([torch.from_numpy(d['q_coord'].astype(np.float32)),
                torch.from_numpy(d['q_rgb'].astype(np.float32))/255.], 1).cuda()
sy = torch.from_numpy(d['s_lab'].astype(np.int64)).cuda()
qy = torch.from_numpy(d['q_lab'].astype(np.int64)).cuda()
so = torch.tensor([len(sx)], dtype=torch.int32)
qo = torch.tensor([len(qx)], dtype=torch.int32)

cap = {}
hooks = []
def mk(name):
    def fn(mod, inp, out):
        cap.setdefault(name, []).append(inp[0].detach())
    return fn
hooks.append(m.lin1.register_forward_hook(mk('lin1_in')))
hooks.append(m.class_reduce.register_forward_hook(mk('cr_in')))

with torch.no_grad(), torch.cuda.amp.autocast():
    out, _ = m(so, sx, sy, qo, qx, qy, 5, support_base_y=None,
               query_base_y=None, sampled_classes=np.array([c]))
p_off = F.softmax(out[0].t().float(), 1)[:, 1].cpu().numpy()

sf = feats_one(m, sx, sy, dev)
qf = feats_one(m, qx, qy, dev)
fg = sf['yl'].bool()
sp = torch.cat([build_proto(m, sf, ~fg), build_proto(m, sf, fg)], 0)
with torch.no_grad():
    pred, _ = predict_agg(m, qf, [sp], 0.0, dev)
p_agg = F.softmax(pred, 0)[1].cpu().numpy()

for h in hooks: h.remove()
print('tasks', os.path.basename(f), 'cls', c)
print('lin1_in shapes:', [tuple(t.shape) for t in cap['lin1_in']])
print('cr_in shapes:', [tuple(t.shape) for t in cap['cr_in']])
a, b = cap['lin1_in'][0], cap['lin1_in'][1]
print('lin1_in diff: mean %.5f max %.5f' % ((a-b).abs().mean(), (a-b).abs().max()))
ca, cb = cap['cr_in'][0], cap['cr_in'][1]
print('cr_in diff: mean %.5f max %.5f' % ((ca-cb).abs().mean(), (ca-cb).abs().max()))
print('official iou %.3f  agg iou %.3f  p0 cos %.4f' % (
    float(((p_off>0.5)&(d['q_lab']>0)).sum()/max(int(((p_off>0.5)|(d['q_lab']>0)).sum()),1)),
    float(((p_agg>0.5)&(d['q_lab']>0)).sum()/max(int(((p_agg>0.5)|(d['q_lab']>0)).sum()),1)),
    float(np.dot(p_off,p_agg)/max(np.linalg.norm(p_off)*np.linalg.norm(p_agg),1e-8))))
# 检查 build_proto vs 官方 getPrototypes
fgm = sf['yl'].bool()
sp_off = torch.cat([m.getPrototypes([sf['cd']], [sf['fd']], [~fgm], k=100),
                    m.getPrototypes([sf['cd']], [sf['fd']], [fgm], k=100)], 0)
print('sparse diff: mean %.6f max %.6f' % ((sp-sp_off).abs().mean(), (sp-sp_off).abs().max()))
print('base_prototypes nonzero:', int((m.base_prototypes.abs().sum(1) > 0).sum()))
