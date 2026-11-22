import sys, os
sys.path.insert(0, '/data01/jh/code/sonata')
import numpy as np, torch
import sonata

print('=== transform ops ===')
tf = sonata.transform.default()
for t in tf.transforms:
    d = dict(t.__dict__)
    d.pop('logger', None)
    print(' ', type(t).__name__, {k: v for k, v in d.items() if not k.startswith('_')})

print('=== coord stats + GridSample 保留率 ===')
for gid in [1, 2, 3, 4, 5]:
    f = '/data01/jh/data/sonata_dump/ep%03d.npz' % gid
    if not os.path.exists(f):
        continue
    d = np.load(f)
    for k, rk in (('s_coord', 's_rgb'), ('q_coord', 'q_rgb')):
        c = d[k].astype(np.float32)
        mn, mx = c.min(0), c.max(0)
        pt = dict(coord=c, color=d[rk].astype(np.float32),
                  normal=np.zeros_like(c))
        out = tf(pt)
        n1 = len(out['coord'])
        print('ep%03d %s N=%d extent=%s gridkeep=%d (%.1f%%)' % (
            gid, k, len(c), np.round(mx - mn, 3), n1, 100.0 * n1 / len(c)))
        # 重复坐标统计
        uq = np.unique(np.round(c / 0.02).astype(np.int64), axis=0)
        print('         unique 0.02 voxels = %d  top-dup = %d' % (
            len(uq), len(c) - len(uq)))

print('=== model enc_mode / dec ===')
cfg_extra = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
m = sonata.load('/data01/jh/code/sonata/ckpt/sonata.pth', custom_config=cfg_extra).cuda().eval()
print('enc_mode =', m.enc_mode, ' dec modules =', len(m.dec),
      ' dec names =', [n for n, _ in m.dec.named_children()][:8])
d = np.load('/data01/jh/data/sonata_dump/ep001.npz')
pt = dict(coord=d['s_coord'].astype(np.float32), color=d['s_rgb'].astype(np.float32),
          normal=np.zeros((len(d['s_coord']), 3), np.float32))
pt = tf(pt)
for k in list(pt.keys()):
    if isinstance(pt[k], torch.Tensor):
        pt[k] = pt[k].cuda()
with torch.inference_mode():
    out = m(pt)
print('model out keys:', sorted(out.keys()), 'feat', tuple(out.feat.shape))
cur = out
n = 0
while 'pooling_parent' in cur:
    print('  L%d feat=%s keys=%s' % (n, tuple(cur.feat.shape), sorted(cur.keys())))
    cur = cur['pooling_parent']
    n += 1
print('  top L%d feat=%s keys=%s' % (n, tuple(cur.feat.shape), sorted(cur.keys())))
print('chain len =', n)
