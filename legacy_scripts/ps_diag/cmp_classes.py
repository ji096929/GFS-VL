import re

def parse(p):
    d = {}
    for line in open(p):
        m = re.search(r'Class_(\d+) - (.*?) Final Result: iou ([0-9.]+)', line)
        if m:
            d[int(m.group(1))] = (m.group(2), float(m.group(3)))
    return d

base = parse('/data01/jh/afford_diag/gfs_build/test_regis1_v3.log')
b = parse('/data01/jh/afford_diag/gfs_build/test_B.log')
a = parse('/data01/jh/afford_diag/gfs_build/test_A.log')
print('类数 base=%d B=%d A=%d' % (len(base), len(b), len(a)))
rows = []
for c in sorted(a):
    name = a[c][0]
    ib, ic, ia = base.get(c, ('', 0))[1], b.get(c, ('', 0))[1], a[c][1]
    rows.append((c, name, ib, ic, ia, ia - ic, ia - ib))
print('%-4s %-26s %8s %8s %8s %8s %8s' % ('id', 'name', 'base', 'B', 'A', 'A-B', 'A-base'))
for r in sorted(rows, key=lambda x: -x[6]):
    print('%-4d %-26s %8.4f %8.4f %8.4f %+8.4f %+8.4f' % r)
print()
print('novel(>=12) 平均: base %.4f  B %.4f  A %.4f'
      % (sum(r[2] for r in rows if r[0] >= 12) / max(len([r for r in rows if r[0] >= 12]), 1),
         sum(r[3] for r in rows if r[0] >= 12) / max(len([r for r in rows if r[0] >= 12]), 1),
         sum(r[4] for r in rows if r[0] >= 12) / max(len([r for r in rows if r[0] >= 12]), 1)))
print('A 涨(>0.5pp) %d 类, A 跌(<-0.5pp) %d 类 (相对 B)'
      % (sum(1 for r in rows if r[5] > 0.005), sum(1 for r in rows if r[5] < -0.005)))
