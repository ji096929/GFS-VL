"""把 AI 在 base 池的补标写入 B 组点级伪标签。"""
import os, sys, glob
sys.path.insert(0, '/data01/jh/code/ps_diag')
import numpy as np, torch, vlm_common as V

BASE='/data01/jh/data/base_cache'
BIDX='/data01/jh/data/base_index'
PROTO='/data01/jh/data/support_protos'
PL='/data01/jh/data/offline_pl/sc200_regis_1_10_B'
NO,NN=12,45

def main():
    proto=np.load(os.path.join(PROTO,'sc200_regis_1_10.npy')).astype(np.float32)
    pn=proto/(np.linalg.norm(proto,axis=1,keepdims=True)+1e-8)

    for s in [0, 2, 3, 4, 5, 6, 7]:
        bp=os.path.join(BASE,'base_train_s%d.pt'%s)
        ip=os.path.join(BIDX,'base_index_train_s%d.pt'%s)
        if not (os.path.exists(bp) and os.path.exists(ip)):
            print('s%d missing'%s); continue
        bd=torch.load(bp,map_location='cpu'); id_=torch.load(ip,map_location='cpu')
        n_add=0; n_pt=0
        for sc,r in bd.items():
            if sc not in id_: 
                print('  %s no index'%sc); continue
            ri=id_[sc]
            n=len(r['pred'])
            if n != len(ri['off'])-1:
                print('  %s region mismatch %d vs %d'%(sc,n,len(ri['off'])-1)); continue
            F=r['feats'].astype(np.float32)
            Fn=F/(np.linalg.norm(F,axis=1,keepdims=True)+1e-8)
            s1=Fn@pn.T
            best=s1.max(1); cls=s1.argmax(1)+NO
            unlab=(r['hist_n'].sum(1)+r['n_ign'])>=0.5*r['sizes']
            sel=(best>=0.9)&unlab
            if not sel.any(): continue
            p=os.path.join(PL,sc+'.npy')
            if not os.path.exists(p):
                # 场景不在 B 组（无 novel 区域），仍然写：AI 补的点
                pl=np.full(len(ri['pred']),-1,dtype=np.int16)
            else:
                pl=np.load(p)
                if len(pl)!=len(ri['pred']):
                    print('  %s point mismatch'%sc); continue
            for k in np.where(sel)[0]:
                seg=ri['idx'][ri['off'][k]:ri['off'][k+1]]
                if len(seg):
                    m=pl[seg]<0
                    pl[seg[m]]=cls[k]
                    n_pt+=int(m.sum()); n_add+=1
            np.save(p,pl)
        print('s%d: AI 补 %d 区域 / %d 点'%(s,n_add,n_pt))
    print('done')

if __name__ == '__main__':
    main()
