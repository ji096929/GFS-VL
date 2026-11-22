"""生成软权重伪标签：A 组全保留 + 每点权重 w=0.2+0.8*sigmoid(z)，小类保底 1.0。"""
import os, sys, glob
sys.path.insert(0, '/data01/jh/code/ps_diag')
import numpy as np, torch, vlm_common as V

REGION='/data01/jh/data/region_cache'
INDEX='/data01/jh/data/region_index'
SONATA='/data01/jh/data/sonata_region'
PROTO='/data01/jh/data/support_protos'
NO,NN=12,45; PS=0.6; WMIN=0.2; MIN_PTS=50000

def zscore(x):
    s=x.std(); return (x-x.mean())/s if s>1e-8 else np.zeros_like(x)

def main():
    split='sc200_regis_1_10'
    proto=np.load(os.path.join(PROTO,split+'.npy')).astype(np.float32)
    pn=proto/(np.linalg.norm(proto,axis=1,keepdims=True)+1e-8)
    sp=np.load(os.path.join(PROTO,split+'_sonata.npy')).astype(np.float32)
    spn=sp/(np.linalg.norm(sp,axis=1,keepdims=True)+1e-8)
    son={}
    for f in sorted(glob.glob(os.path.join(SONATA,'sonata_regions_train_s*.pt'))):
        d=torch.load(f,map_location='cpu')
        for sc,r in d.items(): son[sc]=r['feats'].astype(np.float32)

    names=[]; pred=[]; feat=[]; sz=[]; sonf=[]
    for f in sorted(glob.glob(os.path.join(REGION,'regions_train_s*.pt'))):
        d=torch.load(f,map_location='cpu')
        for sc,r in sorted(d.items()):
            n=len(r['pred'])
            if sc not in son or len(son[sc])!=n: continue
            names.extend([sc]*n); pred.append(r['pred'].astype(np.int64))
            feat.append(r['feats'].astype(np.float32)); sz.append(r['sizes'].astype(np.int64))
            sonf.append(son[sc])
    pred=np.concatenate(pred); feat=np.concatenate(feat,axis=0); sz=np.concatenate(sz)
    sonf=np.concatenate(sonf,axis=0); N=len(pred)
    svn=sonf/(np.linalg.norm(sonf,axis=1,keepdims=True)+1e-8)

    groups={}
    for i in range(N): groups.setdefault((names[i],int(pred[i])),[]).append(i)
    keep=np.zeros(N,dtype=bool)
    for (s,c),idx in groups.items():
        idx=np.array(idx); w=sz[idx].astype(np.float64)
        pooled=(feat[idx]*w[:,None]).sum(0)/w.sum(); pooled/=np.linalg.norm(pooled)+1e-8
        if float(pooled@pn[c-NO])>=PS: keep[idx]=True

    Dsf=[]
    for f in sorted(glob.glob(os.path.join(REGION,'regions_train_s*.pt'))):
        d=torch.load(f,map_location='cpu')
        for sc,r in d.items():
            if sc not in son: continue
            for i in range(len(r['pred'])):
                vp=int(r['sizes'][i])-int(r['n_ign'][i]); gb=r['hist_b'][i]
                if vp>0 and gb.sum()>0 and int(gb.max())>=0.5*vp: Dsf.append(son[sc][i])
    Dn=np.stack(Dsf); Dn/=np.linalg.norm(Dn,axis=1,keepdims=True)+1e-8

    s1=np.zeros(N,dtype=np.float32)
    for c in range(NO,NO+NN):
        m=pred==c
        if m.any(): s1[m]=svn[m]@spn[c-NO]
    dm=np.empty(N,dtype=np.float32)
    for st in range(0,N,4096):
        en=min(st+4096,N); dm[st:en]=(svn[st:en]@Dn.T).max(1)
    mg=s1-dm
    z=np.zeros(N,dtype=np.float32)
    for c in range(NO,NO+NN):
        m=pred==c
        if m.any(): z[m]=zscore(mg[m])
    w=WMIN+(1-WMIN)/(1+np.exp(-np.clip(z,-20,20)))

    # 小类保底：该类保留区域总点数 < MIN_PTS 时权重 1.0
    for c in range(NO,NO+NN):
        m=keep&(pred==c)
        if m.any() and sz[m].sum()<MIN_PTS: w[m]=1.0
        elif not m.any(): 
            m2=pred==c
            if m2.any() and sz[m2].sum()<MIN_PTS: w[m2]=1.0

    outdir='/data01/jh/data/offline_pl/%s_SW'%split
    os.makedirs(outdir,exist_ok=True)
    per={}
    i=0
    while i<len(names):
        sc=names[i]; j=i
        while j<len(names) and names[j]==sc: j+=1
        per[sc]=(i,j-i); i=j
    nf=0
    for f in sorted(glob.glob(os.path.join(INDEX,'regions_index_train_s*.pt'))):
        d=torch.load(f,map_location='cpu')
        for sc,r in d.items():
            if sc not in per: continue
            off,n=per[sc]
            if n!=len(r['off'])-1: continue
            pl=np.full(len(r['pred']),-1,dtype=np.int16)
            wt=np.ones(len(r['pred']),dtype=np.float16)
            for k in range(n):
                if not keep[off+k]: continue
                seg=r['idx'][r['off'][k]:r['off'][k+1]]
                if len(seg): pl[seg]=pred[off+k]; wt[seg]=w[off+k]
            o1=os.path.join(outdir,sc+'_pl.npy'); o2=os.path.join(outdir,sc+'_w.npy')
            np.save(o1,pl); np.save(o2,wt); nf+=1
    print('wrote %d scenes, keep %d regions, mean w %.3f'%(nf,int(keep.sum()),float(w[keep].mean())))

if __name__ == '__main__':
    main()
