# -*- coding: utf-8 -*-
p = '/data01/jh/code/GFS-VL/tools/gen_offline_pl.py'
s = open(p, encoding='utf-8').read()

old = 'cfg = default_config_parser(cfg_file, {})'
new = '''cfg = default_config_parser(cfg_file, {})
# 等价于 default_setup 的派生字段（脚本不走 launch）
cfg.num_worker = cfg.num_worker or 4
cfg.num_worker_per_gpu = cfg.num_worker
cfg.batch_size_per_gpu = cfg.batch_size
cfg.batch_size_val_per_gpu = 1
cfg.batch_size_test_per_gpu = 1
cfg.save_path = '/data01/jh/afford_diag/gen_pl_tmp'
os.makedirs(cfg.save_path, exist_ok=True)'''

assert s.count(old) == 1
s = s.replace(old, new)
open(p, 'w', encoding='utf-8').write(s)
print('fixed')
