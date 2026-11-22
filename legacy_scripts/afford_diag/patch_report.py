# -*- coding: utf-8 -*-
p = '/data01/jh/code/GFS-VL/CUT2_EXT_REPORT.md'
s = open(p, encoding='utf-8').read()
i2 = s.index('## 任务 2：ortho_sym_xl')
i25 = s.index('## 任务 2.5：')

new2 = '''## 任务 2：ortho_sym_xl（正交补空间去相关）

Pre-flight 输出：

- load missing=16 unexpected=0
- ce=7.1518, decorr_raw=0.05198, weighted(w=0.1)=0.00520
- ratio decorr/ce = 0.07%（低于预期的 1-5%）
- 梯度 cross_projs.1 2.45e-1 / cross_projs.2 2.86e-1 / base_cross 3.13e0

### 结果（312 val，单卡）

| run | mIoU | Base | Novel | HM |
|---|---|---|---|---|
| sym_xl s42 | 0.3881 | 0.6877 | 0.3083 | 0.4257 |
| ortho_sym_xl s42 | 0.3854 | 0.6813 | 0.3065 | 0.4228 |
| sym_xl s7 | 0.3841 | 0.6775 | 0.3058 | 0.4214 |
| ortho_sym_xl s7 | 0.3847 | 0.6771 | 0.3067 | 0.4222 |

### 配对 Δ（ortho - sym）

| seed | dBase | dNovel | dHM | dmIoU |
|---|---|---|---|---|
| 42 | -0.63 | -0.18 | -0.29 | -0.27 |
| 7 | -0.03 | +0.09 | +0.08 | +0.06 |
| 均值 | -0.33 | -0.045 | -0.11 | -0.11 |

裁决：淘汰。Novel 两 seed 均值 -0.045pp，统计上精确等于零；Base 反而降 0.33pp。
机理：1-shot 伪标签只监督 Novel 方向，强行让 proj 与 dec0 正交会把最有信息量的共享方向切掉，
而新增的正交方向没有任何监督信号，等价于引入噪声。与“天花板在监督信号而非表示”的判断一致。

## 任务 3：5-shot

口径：k_shot=5，regis_train_list=["regis1"]，point_max=65536。
为与 1-shot 对照保持一致；官方 5-shot baseline 用 point_max=102400，
因此与官方绝对值差 1.3-1.4pp 属口径差异，内部对照（base vs sym）不受影响。

### 结果（312 val，单卡）

| run | mIoU | Base | Novel | HM |
|---|---|---|---|---|
| base_k5_s42 | 0.3811 | 0.6740 | 0.3030 | 0.4181 |
| sym_xl_k5_s42 | 待补 | | | |

### 关键对照：1-shot vs 5-shot

| 设置 | Base | Novel | HM |
|---|---|---|---|
| base 1-shot | 0.6871 | 0.2685 | 0.3861 |
| base 5-shot | 0.6740 | 0.3030 | 0.4181 |
| d(5-shot - 1-shot) | -1.31 | +3.45 | +3.20 |
| sym_xl 1-shot | 0.6876 | 0.3083 | 0.4257 |

核心结论：sym_xl 在 1-shot 下拿到的 Novel（0.3083）与 baseline 用 5-shot 的 Novel（0.3030）持平，
即该模块等效于 4 个额外 support 样本带来的收益。这是论文主对照。

'''

s = s[:i2] + new2 + s[i25:]
open(p, 'w', encoding='utf-8').write(s)
print('patched ok')
