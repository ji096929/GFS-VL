"""汇总全量 560 eps 的 Step 1-4 与 Step 5 网格结果。"""
import glob
import json
import os
import sys

import numpy as np

CACHE = '/data01/jh/data/ocsb_cache'
ORDER = ['Step1 COSeg', 'Step2 direct B', 'Step3 OASB', 'Step4 OASB+CSR']
KEYS = ['iou', 'base', 'missed_recall', 'false_rate', 'c_prec', 'auc']


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def main():
    print('=' * 100)
    print('全量 Step 1-4（不同 gamma）')
    print('=' * 100)
    header = ('%-6s %-18s %8s %8s %9s %8s %7s' %
              ('gamma', 'version', 'mIoU', 'base', 'MissRec', 'FalseRec',
               'AUC'))
    for gtag in ('g150', 'g175', 'g200'):
        d = load(os.path.join(CACHE, 'ablation_full_%s.json' % gtag))
        if d is None:
            print('  %s: not found' % gtag)
            continue
        print('\n' + header)
        for tag in ORDER:
            s = d.get(tag)
            if not s:
                continue
            print('%-6s %-18s %8.4f %8.4f %9.3f %8.3f %7.3f'
                  % (gtag, tag, s['iou'], s['base'], s['missed_recall'],
                     s['false_rate'], s['auc']))
        # per-class
        pc = d.get('Step4 OASB+CSR_per_class')
        if pc:
            txt = ' '.join('%d:%+.3f' % (int(k), v['delta'])
                           for k, v in sorted(pc.items(), key=lambda x: int(x[0])))
            print('  Step4 per-class delta: %s' % txt)

    print('\n' + '=' * 100)
    print('Step 5 网格')
    print('=' * 100)
    print('%-28s %8s %8s %8s %8s %8s %8s' %
          ('config', 'IoU', 'd', 'Miss', 'False', 'AUC', 'w_dev'))
    for p in sorted(glob.glob(os.path.join(CACHE, 'sweep5_*.json'))):
        d = load(p)
        if d is None:
            continue
        c = d['config']
        tag = ('wpos%.1f_reg%.1f_lr%.0e' %
               (c['w_pos'], c['w_reg'], c['lr']))
        f = d['final']
        print('%-28s %8.4f %8.4f %8.3f %8.3f %8.3f %8.2f'
              % (tag, f['iou'], f['iou'] - f['base'], f['missed_recall'],
                 f['false_rate'], f['auc'], d['w_dev']))
        ev = d.get('evals') or []
        if ev:
            best = max(ev, key=lambda s: s['iou'])
            print('    epoch curve: ' + ' '.join(
                'e%d:%.4f' % (s.get('epoch', i + 1), s['iou'])
                for i, s in enumerate(ev)))
            print('    best epoch %s IoU %.4f (d %+.4f)'
                  % (best.get('epoch'), best['iou'], best['iou'] - best['base']))
        e0 = d.get('epoch0')
        if e0:
            print('    epoch0 IoU %.4f (d %+.4f)' %
                  (e0['iou'], e0['iou'] - e0['base']))


if __name__ == '__main__':
    main()
