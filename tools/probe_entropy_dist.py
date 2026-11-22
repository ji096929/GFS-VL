import sys; sys.path.insert(0,'/data01/jh/code/GFS-VL')
import torch
from pointcept.engines.defaults import default_config_parser
from pointcept.datasets import build_dataset
from pointcept.engines.train import GFS_VL_Trainer
from pointcept.utils.logger import get_root_logger
class Mini(GFS_VL_Trainer):
    def __init__(self,cfg,logger,ds):
        self.cfg=cfg; self.logger=logger
        class L: pass
        self.train_loader=L(); self.train_loader.dataset=ds
        self.build_zero_shot_model()
cfg=default_config_parser('configs/scannetpp/semseg-pt-v3m1-0-gfsregistrain_k5.py',None)
lg=get_root_logger(log_file=None); ds=build_dataset(cfg.data.train)
nb=int(cfg.data.num_bases); nc=int(cfg.data.num_base_novels)
print('logit_scale =', float(mini_vl) if False else 'check below')
mini=Mini(cfg,lg,ds)
ls = float(mini.VLM_3D.task_head.logit_scale)
print('logit_scale = %.4f' % ls)
import numpy as np
for i in range(8):
    d=ds[i]
    gt0=d['segment'].view(-1)
    nv=int(((gt0>=nb)&(gt0<nc)).sum())
    if nv<300: continue
    for k,v in list(d.items()):
        if isinstance(v,torch.Tensor): d[k]=v.cuda()
    gt=d['segment'].view(-1)
    out=mini.VLM_3D(mini.construct_3D_vlm_input(d))
    sc=out['seg_scores']
    p=torch.softmax(sc.float(),dim=-1)
    ent=-(p*(p+1e-8).log()).sum(-1)
    valid=(gt>=nb)&(gt<nc)
    ev=ent[valid]
    q=torch.quantile
    print('sample %d: novel=%d ent min %.4f p10 %.4f med %.4f p90 %.4f max %.4f | <0.01 占比 %.3f' % (
        i, int(valid.sum()), float(ev.min()), float(q(ev,0.1)), float(q(ev,0.5)),
        float(q(ev,0.9)), float(ev.max()), float((ev<0.01).float().mean())))
    # 类内：熵最低 10% 的阈值 与 熵=0 的点占比
    pred=sc.argmax(-1)
    for c in pred[valid].unique()[:3]:
        m=pred==c
        e=ent[m]
        print('    cls %d n=%d 熵=0占比 %.3f | q10=%.4f | 唯一值数 %d' % (
            int(c), int(m.sum()), float((e<1e-6).float().mean()), float(q(e,0.1)), int(torch.unique(e).numel())))
    break
