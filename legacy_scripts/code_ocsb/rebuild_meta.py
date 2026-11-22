"""从 manifest + 已存在的 npz 重建 val_meta（不依赖分段进程写文件）。"""
import json
import os

CACHE = '/data01/jh/data/ocsb_cache'
MANIFEST = '/data01/jh/data/manifest_fold0.json'

spec = json.load(open(MANIFEST))['episodes']
metas = []
for i, e in enumerate(spec):
    if os.path.exists(os.path.join(CACHE, 'val_%03d.npz' % i)):
        metas.append({'split': 'val', 'idx': i, 'class': int(e['class']),
                      'support': e['support'], 'query': e['query'],
                      'u': e.get('u_cross', [])})
json.dump(metas, open(os.path.join(CACHE, 'val_meta_all.json'), 'w'), indent=1)
print('rebuilt', len(metas), 'of', len(spec))
