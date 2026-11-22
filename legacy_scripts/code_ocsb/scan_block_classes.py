"""一次性扫描所有 block 的类别，缓存为 json。"""
import glob
import json
import os

import numpy as np

DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
OUT = '/data01/jh/data/ocsb_cache/block_classes.json'

out = {}
for f in sorted(glob.glob(os.path.join(DATA, '*.npy'))):
    d = np.load(f, mmap_mode='r')
    out[os.path.basename(f)[:-4]] = sorted(
        set(np.asarray(d[:, 6]).astype(int).tolist()))
json.dump(out, open(OUT, 'w'))
print('blocks:', len(out), '->', OUT)
