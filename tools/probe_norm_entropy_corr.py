import sys; sys.path.insert(0,'/data01/jh/code/GFS-VL')
import torch, torch.nn.functional as F
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
cfg=default_config_parser('configs/scannettpp/x.py',None) if False else default_config_parser('configs/scannetpp/semseg-pt-v3m1-0-gfsregistrain_k5.py',None)
lg=get_root_logger(log_file=None); ds=build_dataset(cfg.data.train)
mini=Mini(cfg,lg,ds)
th=mini.VLM_3D.task_head
print('feat_norm =', getattr(th,'feat_norm','?'))
W=th.cls_head.weight.detach()
print('cls_head W shape', tuple(W.shape), 'row norm mean %.4f' % float(W.norm(dim=1).mean()))
nb=int(cfg.data.num_bases); nc=int(cfg.data.num_base_novels)
def ent_of(logits):
    p=torch.softmax(logits.float(),dim=-1)
    return -(p*(p+1e-8).log()).sum(-1)
import numpy as np
for i in range(6):
    d=ds[i]
    gt0=d['segment'].view(-1)
    if int(((gt0>=nb)&(gt0<nc)).sum())<300: continue
    for k,v in list(d.items()):
        if isinstance(v,torch.Tensor): d[k]=v.cuda()
    gt=d['segment'].view(-1)
    out=mini.VLM_3D(mini.construct_3D_vlm_input(d))
    sc=out['seg_scores']; feats=out['all_feats']
    pred=sc.argmax(-1); valid=(gt>=nb)&(gt<nc)
    m=valid & (pred>=nb)                      # novel 预测点
    if int(m.sum())<100: continue
    fn=feats.float().norm(dim=-1)
    e_raw=ent_of(sc)
    sc_n=F.normalize(feats.float(),dim=-1) @ W.T  # 余弦相似度（归一化）
    e_cos=ent_of(sc_n*14.3)                   # CLIP 温度 1/0.07
    # Spearman 相关（手算秩相关）
    a=fn[m]; b=e_raw[m]
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    r=float(torch.corrcoef(torch.stack([ra,rb]))[0,1])
    rc=b.new_tensor(0)
    c=fn[m]; dd=e_cos[m]
    rc2=float(torch.corrcoef(torch.stack([c.argsort().argsort().float(), dd.argsort().argsort().float()]))[0,1])
    print('s%d: novel_pred=%d | ||f|| p10/med/p90 %.1f/%.1f/%.1f' % (i,int(m.sum()),
        float(torch.quantile(fn[m],0.1)),float(torch.quantile(fn[m],0.5)),float(torch.quantile(fn[m],0.9))))
    print('   熵(原始) min/med/max %.3f/%.3f/%.3f | Spearman(||f||,ent)=%+.3f' % (
        float(e_raw[m].min()),float(e_raw[m].median()),float(e_raw[m].max()),r))
    print('   熵(余弦*14.3) min/med/max %.3f/%.3f/%.3f | Spearman=%+.3f' % (
        float(e_cos[m].min()),float(e_cos[m].median()),float(e_cos[m].max()),rc2))
    print('   余弦口径 top1@novel acc %.4f vs 原始 %.4f' % (
        float((sc_n[m].argmax(-1)==gt[m]).float().mean()), float((pred[m]==gt[m]).float().mean())))
    break
