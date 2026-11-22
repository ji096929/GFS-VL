# -*- coding: utf-8 -*-
p = '/data01/jh/code/GFS-VL/tools/gen_offline_pl.py'
s = open(p, encoding='utf-8').read()

old = """        input_dict = {
            'coord': torch.from_numpy(coord).unsqueeze(0).cuda(),
            'color': torch.from_numpy(color).unsqueeze(0).cuda(),
            'segment': torch.from_numpy(seg).unsqueeze(0).cuda(),
            'offset': torch.tensor([coord.shape[0]], device='cuda'),
        }
        new = trainer.target_calibrate(input_dict)
        pl = torch.where(
            (input_dict['segment'][0] == -1) & (new[0] >= 0),
            new[0], torch.full_like(new[0], -1))"""

new = """        input_dict = {
            'coord': torch.from_numpy(coord).cuda(),
            'color': torch.from_numpy(color).cuda(),
            'segment': torch.from_numpy(seg).cuda(),
            'offset': torch.tensor([coord.shape[0]], device='cuda'),
        }
        new = trainer.target_calibrate(input_dict)
        pl = torch.where(
            (input_dict['segment'] == -1) & (new >= 0),
            new, torch.full_like(new, -1))"""

assert s.count(old) == 1, 'anchor=%d' % s.count(old)
s = s.replace(old, new)
open(p, 'w', encoding='utf-8').write(s)
print('fixed: removed batch dim')
