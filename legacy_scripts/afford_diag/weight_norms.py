# -*- coding: utf-8 -*-
import torch, numpy as np

RUNS = [
    ('sym_k1', '/data01/jh/code/GFS-VL/exp/scannet200/sym_xl_s42/model/regis1_model_last.pth'),
    ('sym_k5', '/data01/jh/code/GFS-VL/exp/scannet200/sym_xl_k5_s42/model/regis1_model_last.pth'),
]
BASE = ['refrigerator','desk','curtain','bookshelf','bed','table',
        'window','cabinet','door','chair','floor','wall']
NOVEL = ['trash can','ceiling','doorframe','object','shelf','sink','picture',
         'backpack','couch','box','pillow','radiator','mirror','whiteboard',
         'lamp','toilet','book','monitor','towel','tv','clothes','coffee table',
         'office chair','nightstand','bag','dresser','toilet paper',
         'recycling bin','kitchen cabinet','bathtub','telephone','plant',
         'stool','keyboard','shoe','jacket','shower curtain','armchair',
         'microwave','computer tower','bathroom vanity','kitchen counter',
         'shower wall','paper towel dispenser','file cabinet']

out = {}
for tag, path in RUNS:
    sd = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
    out[tag] = dict(
        seg=np.linalg.norm(sd['seg_head.weight'].float().numpy(), axis=1),
        bcross=np.linalg.norm(sd['base_cross_head.weight'].float().numpy(), axis=1),
        n_dec0=np.linalg.norm(sd['novel_seg_head.weight'][:, :64].float().numpy(), axis=1),
        n_proj=np.linalg.norm(sd['novel_seg_head.weight'][:, 64:].float().numpy(), axis=1),
    )

print('%-15s %-22s %-22s' % ('', 'sym_k1', 'sym_k5'))
print('%-15s %10s %10s %10s %10s' % ('class', 'bcross', 'novel_proj', 'bcross', 'novel_proj'))
print('-' * 62)
for i, name in enumerate(BASE):
    print('%-15s %10.3f %10s %10.3f %10s'
          % (name, out['sym_k1']['bcross'][i], '-',
             out['sym_k5']['bcross'][i], '-'))
print()
focus = ['shelf', 'book', 'office chair', 'armchair', 'stool']
print('%-15s %10s %10s %10s %10s' % ('novel class', 'k1_dec0', 'k1_proj', 'k5_dec0', 'k5_proj'))
print('-' * 62)
for name in focus:
    j = NOVEL.index(name)
    print('%-15s %10.3f %10.3f %10.3f %10.3f'
          % (name, out['sym_k1']['n_dec0'][j], out['sym_k1']['n_proj'][j],
             out['sym_k5']['n_dec0'][j], out['sym_k5']['n_proj'][j]))

print()
for tag in ('sym_k1', 'sym_k5'):
    o = out[tag]
    print('%s: seg_head mean=%.3f | base_cross mean=%.3f | novel_dec0 mean=%.3f | novel_proj mean=%.3f'
          % (tag, o['seg'].mean(), o['bcross'].mean(), o['n_dec0'].mean(), o['n_proj'].mean()))
