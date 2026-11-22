# -*- coding: utf-8 -*-
p = '/data01/jh/code/GFS-VL/CUT2_EXT_REPORT.md'
s = open(p, encoding='utf-8').read()

old = '''结论：bookshelf 的范数是 12 个基类里最小的之一，不是预期的最大。bookshelf 的 10-11pp 恢复不来自 base_cross_head 加权，更可能来自 novel 头对 shelf 的预测变准、减少抢点。'''

new = '''结论（已按 312 val 修正）：bookshelf 的范数是 12 个基类里最小的之一
（s42 0.262 / s7 0.291），但 312 val 上 bookshelf 是
1-shot 0.5187 -> 0.4890（-2.97）、5-shot 0.4317 -> 0.3014（-13.03），
两种设置下都是跌的，范数大小不能解释跌幅方向。

5-shot 下同族类同步下跌：shelf -3.36 / book -6.94 / stool -4.61，
而父类 bookshelf -13.03 / chair -7.61 跌得更多。
父子类一起塌说明不是“novel 头抢父类的点”（那样 novel 侧应当上涨），
而是 dec1/dec2 提供的多尺度结构特征整体被污染。

待验证：实验 A（detach_cross + seg_head lr=1e-4）若使整族同步回血，
即确认梯度倒灌；若只回血平均 Base 而整族无变化，则另有原因。'''

assert s.count(old) == 1, 'anchor=%d' % s.count(old)
s = s.replace(old, new)
open(p, 'w', encoding='utf-8').write(s)
print('attribution fixed')
