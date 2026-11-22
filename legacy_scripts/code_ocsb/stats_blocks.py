"""Gate-3 check: block count and superpoints per block."""
import glob
import os
import sys
import numpy as np

data_dir = sys.argv[1] if len(sys.argv) > 1 else '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
files = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
print('blocks:', len(files))
if not files:
    sys.exit(0)

ns, sps, cls = [], [], []
scenes = set()
for f in files:
    d = np.load(f, mmap_mode='r')
    n = int(d.shape[0])
    seg = np.asarray(d[:, 7])
    lab = np.asarray(d[:, 6]).astype(int)
    ns.append(n)
    sps.append(len(np.unique(seg)))
    cls.append(len(np.unique(lab)))
    scenes.add(os.path.basename(f).split('_block_')[0])

ns = np.array(ns)
sps = np.array(sps)
print('scenes with at least one block:', len(scenes))
print('points per block: mean %.0f  median %.0f  min %d  max %d' % (
    ns.mean(), np.median(ns), ns.min(), ns.max()))
print('superpoints per block: mean %.1f  median %.1f  min %d  max %d' % (
    sps.mean(), np.median(sps), sps.min(), sps.max()))
print('blocks with <5 superpoints: %d (%.1f%%)' % (
    (sps < 5).sum(), 100.0 * (sps < 5).mean()))
print('classes per block: mean %.1f' % np.mean(cls))
