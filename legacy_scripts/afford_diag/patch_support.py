# -*- coding: utf-8 -*-
p = '/data01/jh/code/GFS-VL/CUT2_EXT_REPORT.md'
s = open(p, encoding='utf-8').read()

if '## 任务 4' not in s:
    s += '''

## 任务 4：纯 3D Support-Only（无 VLM 伪标签）

GFS_OFFLINE_PL 指向空目录，其余口径与 1-shot 完全一致
（k=1, regis1, point_max=65536, bs=12）。目的：隔离 VLM 伪标签的贡献。

| run | mIoU | Base | Novel | HM |
|---|---|---|---|---|
| base（有伪标签） | 0.3566 | 0.6871 | 0.2685 | 0.3861 |
| support_base（无伪标签） | 0.1768 | 0.6319 | 0.0554 | 0.1018 |
| support_sym（无伪标签） | 待补 | | | |

Novel 从 0.2685 掉到 0.0554（-21.3pp）。没有伪标签时，3D backbone 单靠
1 个 support 样本几乎无法建立新类的判别边界。

意义：伪标签是 Novel 能力的必要条件；跨层模块的作用是更充分地利用
伪标签提供的监督，而不是独立产生新信息。这也解释了 Novel 上限受
伪标签自身质量（mIoU 0.2646）约束。
'''
    open(p, 'w', encoding='utf-8').write(s)
    print('task4 appended')
else:
    print('task4 exists')
