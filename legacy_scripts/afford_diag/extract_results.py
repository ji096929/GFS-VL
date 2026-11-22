# -*- coding: utf-8 -*-
"""从所有训练日志提取最终结果（汇总 + 57 类），固化到 results_all.json
即使日志后来被删/覆盖，结果仍可重建。"""
import os, re, json, glob

LOG_DIR = '/data01/jh/afford_diag/xl_runs'
OUT = '/data01/jh/afford_diag/results_all.json'

res = {}
for path in sorted(glob.glob(os.path.join(LOG_DIR, '*.log'))):
    name = os.path.basename(path)[:-4]
    try:
        lines = open(path, encoding='utf-8', errors='ignore').read().splitlines()
    except Exception:
        continue

    idx = None
    for i, l in enumerate(lines):
        if 'Eval result: Final' in l:
            idx = i
    if idx is None:
        continue

    m = re.search(
        r'mIoU: ([0-9.]+), BASE: ([0-9.]+), NOVEL: ([0-9.]+), hm_mIoU: ([0-9.]+)',
        lines[idx])
    if not m:
        continue
    summary = dict(zip(['mIoU', 'base', 'novel', 'hm'],
                       [float(x) for x in m.groups()]))

    cls = {}
    for l in lines[idx:]:
        mm = re.search(r'Class_(\d+) - (.+?) Final Result: iou ([0-9.]+)', l)
        if mm:
            cls[int(mm.group(1))] = {'name': mm.group(2), 'iou': float(mm.group(3))}

    res[name] = {
        'summary': summary,
        'n_classes': len(cls),
        'classes': {str(k): v for k, v in sorted(cls.items())},
        'log': path,
    }

json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('extracted %d runs -> %s' % (len(res), OUT))
for k, v in sorted(res.items()):
    s = v['summary']
    print('  %-28s mIoU=%.4f base=%.4f novel=%.4f hm=%.4f  classes=%d'
          % (k, s['mIoU'], s['base'], s['novel'], s['hm'], v['n_classes']))
