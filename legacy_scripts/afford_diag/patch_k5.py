# -*- coding: utf-8 -*-
p = '/data01/jh/code/GFS-VL/CUT2_EXT_REPORT.md'
s = open(p, encoding='utf-8').read()

old = '''| base_k5_s42 | 0.3811 | 0.6740 | 0.3030 | 0.4181 |
| sym_xl_k5_s42 | 待补 | | | |'''

new = '''| base_k5_s42 | 0.3811 | 0.6740 | 0.3030 | 0.4181 |
| sym_xl_k5_s42 | 0.3874 | 0.6570 | 0.3155 | 0.4263 |

### 配对 Δ（sym - base）

| 设置 | dBase | dNovel | dHM | dmIoU |
|---|---|---|---|---|
| 1-shot (s42) | +0.06 | +3.98 | +3.96 | +3.15 |
| 5-shot (s42) | -1.70 | +1.25 | +0.82 | +0.63 |

### 天花板效应（直接证据）

Novel 增益从 +3.98pp 收窄到 +1.25pp，下降 69%。support 从 1 增到 5 后，
模块所补偿的跨层上下文缺口被 support 自身填补，边际收益急剧衰减。
同时 Base 从 +0.06 变成 -1.70：额外容量在 support 充足时开始挤占 Base 表示。

sym_xl_k5 的 Novel 0.3155 是全部实验中唯一突破 0.310 的结果，
但代价是 Base -1.70pp，HM 仅 +0.82。

结论：模块的作用是补偿 support 稀缺，而不是提升表示能力上限。
这一定位比“稳定涨点”更准确，也解释了 1-shot 下 8 次实验锁死在 0.305 的现象。'''

assert s.count(old) == 1, 'anchor=%d' % s.count(old)
s = s.replace(old, new)
open(p, 'w', encoding='utf-8').write(s)
print('k5 results written')
