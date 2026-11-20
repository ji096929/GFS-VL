# -*- coding: utf-8 -*-
"""生成离线伪标签（原始点数版）。

关键：不能用 dataset.get_data()，它含 NB-Mix 会改变点数。
这里直接读原始 npy，复现 transform 里的 CenterShift + NormalizeColor，
再走与训练完全一致的 VLM 校准流程（PS + AI）。

用法: python tools/gen_offline_pl.py <config> <out_dir>
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/GFS-VL')
sys.path.insert(0, '/data01/jh/code/GFS-VL/pointcept/models/PLA')

from pointcept.engines.defaults import default_config_parser
from pointcept.engines.train import GFS_VL_Trainer
from pointcept.datasets import build_dataset, collate_fn

cfg_file, out_dir = sys.argv[1], sys.argv[2]

cfg = default_config_parser(cfg_file, {})
cfg.num_worker = cfg.num_worker or 4
cfg.num_worker_per_gpu = cfg.num_worker
cfg.batch_size_per_gpu = cfg.batch_size
cfg.batch_size_val_per_gpu = 1
cfg.batch_size_test_per_gpu = 1
cfg.save_path = '/data01/jh/afford_diag/gen_pl_tmp'
os.makedirs(cfg.save_path, exist_ok=True)

cfg.data.train.k_shot = cfg.k_shot
cfg.data.train.seed = cfg.data.regis1['seed']
cfg.data.train.nb_mix_blks = cfg.nb_mix_blks

trainer = GFS_VL_Trainer(cfg)
print('model built', flush=True)

val_data = build_dataset(cfg.data.regis1)
regis_loader = torch.utils.data.DataLoader(
    val_data, batch_size=1, shuffle=False, num_workers=2,
    pin_memory=True, sampler=None, collate_fn=collate_fn)
trainer.vlm_novel_proto = trainer.extract_3d_vlm_proto(regis_loader)
print('prototypes extracted (k_shot=%d)' % cfg.k_shot, flush=True)

train_data = build_dataset(cfg.data.train)
data_root = cfg.data.train.data_root
split = cfg.data.train.split
lookup = train_data.regis_lookup_array

os.makedirs(out_dir, exist_ok=True)
trainer.model.eval()
n = 0
tot_pts = 0
tot_pos = 0
with torch.no_grad():
    for i in range(len(train_data)):
        name = train_data.get_data_name(i)
        scene_dir = os.path.join(data_root, split, name)
        try:
            coord = np.load(os.path.join(scene_dir, 'coord.npy')).astype(np.float32)
            color = np.load(os.path.join(scene_dir, 'color.npy')).astype(np.float32)
            seg200 = np.load(os.path.join(scene_dir, 'segment200.npy'))
        except FileNotFoundError:
            continue

        # segment200 -> 57 类索引（非 57 类映射为 -1）
        seg = np.full(seg200.shape, -1, dtype=np.int64)
        valid = seg200 >= 0
        seg[valid] = lookup[seg200[valid]]

        # CenterShift(apply_z=False)
        x_min, y_min, _ = coord.min(axis=0)
        x_max, y_max, _ = coord.max(axis=0)
        coord[:, 0] -= (x_min + x_max) / 2
        coord[:, 1] -= (y_min + y_max) / 2
        # NormalizeColor
        color = color / 127.5 - 1

        input_dict = {
            'coord': torch.from_numpy(coord).cuda(),
            'color': torch.from_numpy(color).cuda(),
            'segment': torch.from_numpy(seg).cuda(),
            'offset': torch.tensor([coord.shape[0]], device='cuda'),
        }
        new = trainer.target_calibrate(input_dict)
        pl = torch.where(
            (input_dict['segment'] == -1) & (new >= 0),
            new, torch.full_like(new, -1))
        pl_np = pl.cpu().numpy().astype(np.int16)
        np.save(os.path.join(out_dir, name + '.npy'), pl_np)

        n += 1
        tot_pts += pl_np.size
        tot_pos += int((pl_np >= 0).sum())
        if n % 100 == 0:
            print('scenes %d  labeled %.2f%%' % (n, 100.0 * tot_pos / max(tot_pts, 1)), flush=True)

print('done %d scenes, labeled %.2f%%' % (n, 100.0 * tot_pos / max(tot_pts, 1)), flush=True)
