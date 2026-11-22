'''验证 predict_agg 的差异是否来自 fp16 autocast。'''
import glob, os, sys
sys.path.insert(0, '/data01/jh/code'); import fake_tp
sys.path.insert(0, '/data01/jh/code/COSeg'); sys.path.insert(0, '/data01/jh/code')
import numpy as np, torch, torch.nn.functional as F
from ocsb_core import load_model, feats_one, build_proto, predict_agg

files = sorted(glob.glob('/data01/jh/data/sonata_dump/ep*.npz'))[:20]
m, cfg = load_model(); dev = torch.device('cuda')

def run(with_amp):
    ious=[]; cos=[]
    for f in files:
        d=np.load(f); c=int(d['cls'][0])
        sx=torch.cat([torch.from_numpy(d['s_coord'].astype(np.float32)),
                      torch.from_numpy(d['s_rgb'].astype(np.float32))/255.],1).cuda()
        qx=torch.cat([torch.from_numpy(d['q_coord'].astype(np.float32)),
                      torch.from_numpy(d['q_rgb'].astype(np.float32))/255.],1).cuda()
        sy=torch.from_numpy(d['s_lab'].astype(np.int64)).cuda()
        qy=torch.from_numpy(d['q_lab'].astype(np.int64)).cuda()
        so=torch.tensor([len(sx)],dtype=torch.int32); qo=torch.tensor([len(qx)],dtype=torch.int32)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=with_amp):
            out,_=m(so,sx,sy,qo,qx,qy,5,support_base_y=None,query_base_y=None,
                    sampled_classes=np.array([c]))
            sf=feats_one(m,sx,sy,dev); qf=feats_one(m,qx,qy,dev)
            fg=sf['yl'].bool()
            sp=torch.cat([build_proto(m,sf,~fg),build_proto(m,sf,fg)],0)
            pred,_=predict_agg(m,qf,[sp],0.0,dev)
        p_off=F.softmax(out[0].t().float(),1)[:,1].cpu().numpy()
        p_agg=F.softmax(pred.float(),0)[1].cpu().numpy()
        gt=d['q_lab'].astype(bool)
        ious.append((float(((p_agg>0.5)&gt).sum()/max(int(((p_agg>0.5)|gt).sum()),1)),))
        cos.append(float(np.dot(p_off,p_agg)/max(np.linalg.norm(p_off)*np.linalg.norm(p_agg),1e-8)))
    return np.mean(ious), np.mean(cos)

for amp in (False, True):
    iou, cs = run(amp)
    print('predict_agg autocast=%s: mean IoU %.4f  p0 cos %.4f' % (amp, iou, cs))
