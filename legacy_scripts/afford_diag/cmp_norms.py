# -*- coding: utf-8 -*-
import torch, numpy as np
RUNS = [
    ('sym_k5_adam', '/data01/jh/code/GFS-VL/exp/scannet200/sym_xl_k5_s42/model/regis1_model_last.pth'),
    ('cos_sym_k5', '/data01/jh/code/GFS-VL/exp/scannet200/cos_sym_k5_s42/model/regis1_model_last.pth'),
    ('detach_sym_k5', '/data01/jh/code/GFS-VL/exp/scannet200/detach_sym_k5_s42/model/regis1_model_last.pth'),
]
print('run              seg_head  base_cross  novel_dec0  novel_proj  bookshelf_bc  chair_bc')
for tag, p in RUNS:
    try:
        sd = torch.load(p, map_location='cpu', weights_only=False)['state_dict']
    except Exception as e:
        print(tag, 'FAIL', e)
        continue
    seg = np.linalg.norm(sd['seg_head.weight'].float().numpy(), axis=1)
    bc = np.linalg.norm(sd['base_cross_head.weight'].float().numpy(), axis=1)
    nd = np.linalg.norm(sd['novel_seg_head.weight'][:, :64].float().numpy(), axis=1)
    npj = np.linalg.norm(sd['novel_seg_head.weight'][:, 64:].float().numpy(), axis=1)
    print(f'{tag:<16} {seg.mean():9.3f} {bc.mean():11.3f} {nd.mean():11.3f} {npj.mean():11.3f} {bc[3]:13.3f} {bc[9]:9.3f}')
