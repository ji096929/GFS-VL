# -*- coding: utf-8 -*-
p = '/data01/jh/code/GFS-VL/tools/gen_offline_pl.py'
s = open(p, encoding='utf-8').read()

# 在 import 区补 point_collate_fn
old_imp = 'from pointcept.datasets import build_dataset, collate_fn'
new_imp = 'from pointcept.datasets import build_dataset, collate_fn, point_collate_fn\nfrom functools import partial'
assert s.count(old_imp) == 1
s = s.replace(old_imp, new_imp)

# 替换整个生成循环
start = s.index('os.makedirs(out_dir, exist_ok=True)')
head = s[:start]

tail = '''os.makedirs(out_dir, exist_ok=True)

# shuffle=False 重建 loader，才能按索引对齐场景名
train_data = build_dataset(cfg.data.train)
loader = torch.utils.data.DataLoader(
    train_data, batch_size=cfg.batch_size, shuffle=False,
    num_workers=4, drop_last=True, pin_memory=True,
    collate_fn=partial(point_collate_fn, mix_prob=0))

trainer.model.eval()
n = 0
bs = cfg.batch_size
with torch.no_grad():
    for bi, input_dict in enumerate(loader):
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                input_dict[k] = v.cuda(non_blocking=True)
        orig = input_dict['segment'].clone()
        new = trainer.target_calibrate(input_dict)
        offsets = input_dict['offset'].tolist()
        start_i = 0
        for j, end in enumerate(offsets):
            idx = bi * bs + j
            name = train_data.get_data_name(idx)
            o = orig[start_i:end]
            nw = new[start_i:end]
            pl = torch.where((o == -1) & (nw >= 0), nw,
                             torch.full_like(nw, -1))
            np.save(os.path.join(out_dir, name + '.npy'),
                    pl.cpu().numpy().astype(np.int8))
            start_i = end
            n += 1
        if bi % 20 == 0:
            print('batch', bi, 'scenes', n, flush=True)

print('done total', n, flush=True)
'''

open(p, 'w', encoding='utf-8').write(head + tail)
print('rewritten')
