'''定位官方 forward 与 ocsb_core.predict_agg 的基线差异（20 个任务）。'''
import glob, os, sys
sys.path.insert(0, '/data01/jh/code'); import fake_tp
sys.path.insert(0, '/data01/jh/code/COSeg'); sys.path.insert(0, '/data01/jh/code')
import numpy as np, torch, torch.nn.functional as F
from ocsb_core import load_model, feats_one, build_proto, predict_agg
from util import config
from model.coseg import COSeg

DUMP = '/data01/jh/data/sonata_dump'
files = sorted(glob.glob(os.path.join(DUMP, 'ep*.npz')))[:20]

m1, cfg = load_model()          # diag_bc 构建（含 pretrain_backbone + weight）
cfg2 = config.load_cfg_from_cfg_file('/data01/jh/code/COSeg/config/scannetv2_COSeg_fs.yaml')
cfg2.cvfold=0; cfg2.n_way=1; cfg2.k_shot=1; cfg2.n_queries=1
cfg2.save_path='/data01/jh/afford_diag/stage2/tmp_gap'
cfg2.ngpus_per_node=1; cfg2.rank=0; cfg2.multiprocessing_distributed=False; cfg2.distributed=False
os.makedirs(cfg2.save_path, exist_ok=True)
m2 = COSeg(cfg2).cuda()
ck = torch.load('/data01/jh/weights/sc0_1w1s.pth', map_location='cpu')
sd = ck.get('state_dict', ck)
sd = {k.replace('module.',''): v for k,v in sd.items()}
r = m2.load_state_dict(sd, strict=False)
print('load missing', len(r.missing_keys), 'unexpected', len(r.unexpected_keys))
m2.eval()
# 权重是否一致
same = all(torch.equal(m1.state_dict()[k], m2.state_dict()[k]) for k in m1.state_dict() if k in m2.state_dict())
print('state_dict identical:', same)

def rows(m, tag):
    ious=[]; ps=[]
    for f in files:
        d=np.load(f); c=int(d['cls'][0])
        sx=torch.cat([torch.from_numpy(d['s_coord'].astype(np.float32)),
                      torch.from_numpy(d['s_rgb'].astype(np.float32))/255.],1).cuda()
        qx=torch.cat([torch.from_numpy(d['q_coord'].astype(np.float32)),
                      torch.from_numpy(d['q_rgb'].astype(np.float32))/255.],1).cuda()
        sy=torch.from_numpy(d['s_lab'].astype(np.int64)).cuda()
        qy=torch.from_numpy(d['q_lab'].astype(np.int64)).cuda()
        so=torch.tensor([len(sx)],dtype=torch.int32); qo=torch.tensor([len(qx)],dtype=torch.int32)
        with torch.no_grad(), torch.cuda.amp.autocast():
            out,_=m(so,sx,sy,qo,qx,qy,5,support_base_y=None,query_base_y=None,
                    sampled_classes=np.array([c]))
        p=F.softmax(out[0].t().float(),1)[:,1].cpu().numpy()
        gt=d['q_lab'].astype(bool)
        ious.append(float(((p>0.5)&gt).sum()/max(int(((p>0.5)|gt).sum()),1)))
        ps.append(p)
    return ious, ps

iou1, p1 = rows(m1, 'diag_bc')
iou2, p2 = rows(m2, 'audit')
print('diag_bc mean iou %.4f  audit mean iou %.4f' % (np.mean(iou1), np.mean(iou2)))
for i in range(5):
    print('  task%d iou %.3f vs %.3f  p0 maxdiff %.4f' % (i, iou1[i], iou2[i], np.abs(p1[i]-p2[i]).max()))
# predict_agg 的 p0
ious3=[]; diff=[]
for f,pref in zip(files,p1):
    d=np.load(f); c=int(d['cls'][0])
    s={'coord':torch.from_numpy(d['s_coord'].astype(np.float32)),
       'feat':torch.from_numpy(d['s_rgb'].astype(np.float32))/255.,
       'label':torch.from_numpy(d['s_lab'].astype(np.int64))}
    q={'coord':torch.from_numpy(d['q_coord'].astype(np.float32)),
       'feat':torch.from_numpy(d['q_rgb'].astype(np.float32))/255.,
       'label':torch.from_numpy(d['q_lab'].astype(np.int64))}
    s_x=torch.cat([s['coord'],s['feat']],1).cuda(); s_y=(s['label']==c).long().cuda()
    q_x=torch.cat([q['coord'],q['feat']],1).cuda(); q_map=(q['label']==c).long().cuda()
    sf=feats_one(m1,s_x,s_y,torch.device('cuda'))
    qf=feats_one(m1,q_x,q_map,torch.device('cuda'))
    fg=sf['yl'].bool()
    sp=torch.cat([build_proto(m1,sf,~fg),build_proto(m1,sf,fg)],0)
    pred,_=predict_agg(m1,qf,[sp],0.0,torch.device('cuda'))
    p0=F.softmax(pred,0)[1].cpu().numpy()
    gt=d['q_lab'].astype(bool)
    ious3.append(float(((p0>0.5)&gt).sum()/max(int(((p0>0.5)|gt).sum()),1)))
    diff.append(float(np.abs(p0-pref).max()))
print('predict_agg mean iou %.4f  maxdiff vs official %.4f' % (np.mean(ious3), max(diff)))
