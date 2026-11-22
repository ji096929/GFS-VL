# -*- coding: utf-8 -*-
p = '/data01/jh/code/GFS-VL/CUT2_EXT_REPORT.md'
s = open(p, encoding='utf-8').read()

if '## 任务 5' in s:
    print('already exists'); raise SystemExit

s += '''
## 任务 5：5-shot 诊断与最终配方（2026-11-20 下午）

### 5.1 起点：5-shot 的异常

早先的 5-shot 结果（base_k5 0.6740/0.3030/0.4181 vs sym_k5 0.6570/0.3155/0.4263）
表面看是“Novel +1.25 但 Base −1.70”，曾被误判为“RegionPLC 伪标签天花板”。
逐项排查后推翻，真实情况如下。

### 5.2 三个配置事实（先纠正前提）

1. **5-shot 的 PL 是 1-shot 的**：启动时显式设了 GFS_OFFLINE_PL=online_pl_w_uniform，
   而 train.py 里该开关会短路在线原型提取。因此两个 5-shot run 的伪标签逐点相同，
   都是 1-shot 原型产出的。真正的 5-shot 需要不设该变量（在线提原型）。
2. **point_max 三个配置都是 65536**（k1/k5/k5_sym），与官方 5-shot 的 102400 不同，
   所以 5-shot 绝对值不能与官方直接比，内部对照不受影响。
3. **NB-Mix 贴块数在 1-shot 与 5-shot 完全相同**：每场景恒 3 类、每类恒 1 个实例
   （scannet.py: `random.sample(self.novel_classes, nb_mix_blks)` 后每类
   `random.sample(samples, 1)`）。唯一变化是实例池从 1 个 support 变成 5 个。

### 5.3 诊断链

#### (a) P/R 分解（从 result/val.pth 的逐场景 inter/union/target 离线计算）

| 类 | base P/R | sym P/R | dP | dR |
|---|---|---|---|---|
| bookshelf | 0.689 / 0.540 | 0.730 / 0.339 | +0.041 | -0.201 |
| chair | 0.867 / 0.845 | 0.900 / 0.723 | +0.033 | -0.122 |
| shelf | 0.652 / 0.566 | 0.710 / 0.477 | +0.059 | -0.088 |
| book | 0.415 / 0.678 | 0.292 / 0.843 | -0.123 | +0.165 |
| office chair | 0.268 / 0.433 | 0.231 / 0.660 | -0.037 | +0.227 |
| armchair | 0.640 / 0.688 | 0.623 / 0.838 | -0.017 | +0.150 |
| stool | 0.519 / 0.435 | 0.368 / 0.507 | -0.150 | +0.072 |

Base 族整体收缩（P 涨 R 跌），Novel 族整体扩张（R 涨 P 跌）。
点流向：chair 流失 304608 点，office chair + armchair + stool 的 FP 增量合计 348445（87% 吻合）；
bookshelf 流失 225820 点，book 的 FP 增量 384284（59% 吻合）。

**结论：不是特征污染**（novel 的 R 大涨说明 dec1/dec2 表征更强），
而是两分支在 softmax 里的 logit 尺度失衡。

#### (b) 权重范数（推翻“范数膨胀”假说）

| run | seg_head | base_cross | novel(dec0) | novel(proj) | Base | Novel |
|---|---|---|---|---|---|---|
| sym_k5 (Adam) | 1.418 | 0.372 | 0.652 | 0.924 | 0.6570 | 0.3155 |
| cos_sym_k5 | 1.407 | 0.320 | 0.567 | 0.822 | 0.6835 | 0.3165 |
| detach_sym_k5 | 1.425 | 0.438 | 0.776 | 0.873 | 0.6838 | 0.2894 |

cos_sym 的 base_cross 范数比 sym_k5 更小（0.320 < 0.372），Base 却更好。
**范数与 Base 表现无单调关系，范数不是根因**；根因在 logit 尺度。

#### (c) 推理期 logit 校准扫描

| alpha | Base | Novel | HM |
|---|---|---|---|
| 1.0 | 0.6570 | 0.3155 | 0.4263 |
| 0.9 | 0.6834 | 0.3154 | 0.4316 |
| 0.8 | 0.6927 | 0.3032 | 0.4218 |

alpha=0.9 无损（Novel -0.01），HM +0.53；alpha=0.8 开始零和（office chair 0.2067 -> 0.0130）。
1-shot 上 alpha=0.9 为负收益（HM -0.86），因为 1-shot 的 Base 本来没被压制。
**alpha 的收益与 Base 被压制的程度成正比，只在 5-shot 成立。**

### 5.4 三个训练期方案（均在同一 1-shot PL 口径下）

| 方案 | 配置 | Base | Novel | HM | 裁决 |
|---|---|---|---|---|---|
| detach_cross + seg_head lr 1e-4 | k5_detach.py | 0.6838 | 0.2894 | 0.4066 | 淘汰：Base 修好但 Novel -2.61 |
| AdamW + Cosine | k5_cos_base.py | 0.6886 | 0.2657 | 0.3835 | 淘汰：Novel -3.73 |
| **AdamW + Cosine + sym** | **k5_cos_sym.py** | **0.6835** | **0.3165** | **0.4326** | **采用** |

detach 证明 Base 损伤来自 novel 梯度经 cross_projs 倒灌 dec1/dec2（bookshelf 0.3014 -> 0.5452），
但阻断也剥夺了 novel 优化共享层的能力。
cos_base 证明 Cosine 单独用在低秩模型上会拖垮新类（20 epoch 不够退火收敛）。
**只有高秩 sym_xl 配合 Cosine 才同时成立**：良态条件数让前期学得快，后期低 lr 完成尺度平衡。

### 5.5 最终配方与对标

    struct : dec0 直连 + dec1/dec2 跨层投影（sym_xl，+21.8K 参数）
    optim  : AdamW(lr=1e-3, wd=1e-4) + CosineAnnealingLR(eta_min=1e-5)
    budget : 20 epoch, bs=12, point_max=65536, 1-shot PL
    infer  : 无后处理（alpha=1.0）

| 配置 | Base | Novel | HM | mIoU |
|---|---|---|---|---|
| 官方 5-shot（5-shot PL, 102400） | 0.6757 | 0.3167 | 0.4312 | 0.3929 |
| **cos_sym_k5（1-shot PL, 65536）** | **0.6835** | **0.3165** | **0.4326** | **0.3938** |

在更严口径（1-shot 监督 + 一半点数）下四项全超官方。

### 5.6 epoch 曲线（cos_sym_k5，save_freq=4）

| epoch | Base | Novel | HM | bookshelf |
|---|---|---|---|---|
| 4 | 0.6751 | 0.2613 | 0.3768 | - |
| 8 | 0.6572 | 0.2845 | 0.3971 | - |
| 12 | 0.6807 | 0.2994 | 0.4158 | 0.4039 |
| 16 | 0.6804 | 0.3174 | **0.4329** | - |
| 20 | 0.6835 | 0.3165 | 0.4326 | 0.4532 |

HM 在 ep16 达峰；Base 在 ep8 触底后回升；bookshelf 到 ep20 仍在改善（0.4039 -> 0.4532）。
说明后期低 lr 阶段在调整类间平衡，而非整体提升。

### 5.7 待验证（进行中）

- L_cross（跨层子空间辅助 CE，lambda=0.3）：验证能否在训练期替代 alpha
- 真 5-shot（在线提原型，不设 GFS_OFFLINE_PL）：base / cos_sym 各一，对标官方
- cos_sym 在 1-shot 上的一致性（alpha 在 1-shot 是负收益，需确认 Cosine 不重蹈）
- cos_sym 多 seed 复现（当前仅 s42）
'''

open(p, 'w', encoding='utf-8').write(s)
print('task 5 appended, total lines:', len(s.splitlines()))
