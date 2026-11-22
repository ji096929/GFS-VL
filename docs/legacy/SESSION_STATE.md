# 会话状态快照（2026-11-21 00:25 更新）

> 跨会话恢复上下文用。先读这份，再看 `CUT2_EXT_REPORT.md`。

## 一、命名规范（强制）

**禁止使用不明所以的缩写**。统一用完整名称：

| 旧缩写（禁止） | 完整名称 |
|---|---|
| `sym_xl` | **跨层残差双头** |
| `xl` | **跨层读出（无残差）** |
| `mlp_dec0` | **末层双层 MLP** |
| `base` | **末层基线** |
| `rep3` | **末层三重复制** |
| `cos_sym` | **跨层残差双头 + 余弦退火** |
| `ln_only` | **仅末层归一化** |

磁盘目录名与实验名绑定，不再改动，只用于定位。

## 二、方法定义

**跨层残差双头**（当前主方法）：

```
dec0（全分辨率，64 维）→ seg_head（Stage-1 预训练）→ Base logits
                       → LayerNorm → ln0
dec1（1/2，64 维）  → LN→Linear(64→64)→GELU→LN → proj1
dec2（1/4，128 维） → LN→Linear(128→64)→GELU→LN → proj2

新类头：novel_seg_head([ln0, proj1, proj2]) = 192 → 45
基类头：seg_head(dec0) + base_cross_head([proj1, proj2])  ← 零初始化残差
```

新增参数 23,360。PTv3 内部结构完全不动。

**末层三重复制**（结构对照）：

同样的拼接/头结构，但 `proj1`/`proj2` 都从 **dec0** 独立投影而来（不读 dec1/dec2）。
用于剥离"结构容量"与"跨层信息"的贡献。

## 三、1-shot 实验矩阵（312 val，65536 点，1-shot 伪标签）

### 3.1 三 seed 完整（HM）

| 方法 | s42 | s7 | s13 | 均值±std |
|---|---|---|---|---|
| 末层基线 | 0.3861 | 0.4046 | 0.4116 | 0.4008±0.0105 |
| 末层双层 MLP | 0.4078 | 0.4125 | 0.3991 | 0.4065±0.0068 |
| **末层三重复制** | 0.4204 | 0.4047 | 0.4191 | 0.4147±0.0068 |
| **跨层残差双头** | 0.4257 | 0.4214 | 0.4194 | **0.4222±0.0026** |

### 3.2 配对 Δ（相对同 seed 基线）

| 方法 | ΔBase | ΔNovel | ΔHM | ΔHM 跨 seed std |
|---|---|---|---|---|
| 末层双层 MLP | +0.93 | +0.39 | +0.57 | — |
| **末层三重复制** | +0.16 | +1.63 | **+1.40** | **1.49** |
| **跨层残差双头** | +0.45 | +2.09 | **+2.14** | **1.30** |

**逐 seed 的 ΔHM**：

| 方法 | s42 | s7 | s13 |
|---|---|---|---|
| 末层三重复制 | +3.43 | **+0.01** | +0.75 |
| 跨层残差双头 | +3.96 | +1.68 | +0.78 |

**关键结论**：
- 三重复制在 **s7 上完全失效（+0.01）**，方差 1.49
- 跨层残差双头**三 seed 全正**，方差 1.30
- **跨层净贡献（sym − rep3）= +0.74 HM**（不是早期估计的 +1.57）

### 3.3 缺 seed

| 方法 | 已有 | 缺 |
|---|---|---|
| 跨层读出（无残差） | s42, s7, **s13 跑着** | — |
| 仅末层归一化（128 参数） | **s42 跑着** | s7, s13 |

## 四、纯 3D 实验（无 VLM 伪标签，1-shot，s42）

| 方法 | Base | Novel | HM |
|---|---|---|---|
| 末层基线 | 0.6319 | 0.0554 | 0.1018 |
| **末层三重复制** | 0.6401 | 0.0726 | 0.1304 |
| **跨层残差双头** | 0.6370 | **0.0845** | **0.1492** |

配对 Δ：

| 方法 | ΔBase | ΔNovel | ΔHM |
|---|---|---|---|
| 末层三重复制 | +0.82 | +1.72 | +2.86 |
| 跨层残差双头 | +0.51 | **+2.91** | **+4.74** |
| **跨层净贡献** | −0.31 | **+1.19** | **+1.88** |

## 五、增益分解（这是当前最重要的结论）

### 纯 3D（最干净的判据）

```
总增益 +4.74 HM
├─ 结构 + 容量（三重复制）:  +2.86  (60%)
└─ 跨层真实信息:            +1.88  (40%)
```

### 1-shot

```
总增益 +2.14 HM
├─ 结构 + 容量（三重复制）:  +1.40  (65%)
└─ 跨层真实信息:            +0.74  (35%)
```

**规律：监督越弱，跨层的份额越大（40% → 35%）。**

**正确的论文表述**：

> "跨层读出贡献了总增益的 35–40%（纯 3D 下 +1.88 HM），其余来自配套的结构改进（LayerNorm + 宽并行非线性 + 残差直连）。跨层的价值随监督强度下降而上升。"

**禁止的表述**：
- ❌ "跨层读出补足了末层缺失的信息"（1-shot 下只有 +0.74）
- ❌ "增益主要来自跨层特征"（实际 60–65% 来自结构）

## 六、已证伪清单（不要重复尝试）

| # | 假设 | 证据 | 结论 |
|---|---|---|---|
| 1 | 5-shot 原型单均值坍塌 | 多原型 Max-Sim：覆盖率 31.19%→31.25%，训练 0.4105→0.4108 | ❌ 无效 |
| 2 | 离线 PL 可复现在线 | 同 1-shot 原型下 28.09% vs 74.43% | ❌ 流程不等价 |
| 3 | 在线 PL 有 rgb_norm bug | 实测 47.53%（正常范围） | ❌ 无 bug |
| 4 | Omission Recovery | PTv3 dec1 AUC 0.913 < RegionPLC 0.932；阈值下 Precision 3% | ❌ 不成立 |
| 5 | L_cross 是通用增益 | 1-shot: −0.27 HM vs 无 L_cross（但相对基线 +3.69） | ⚠️ 只在 5-shot 有效 |
| 6 | 推理期 α 是通用校准 | 1-shot: −0.86；5-shot: +0.53 | ❌ 只在 5-shot 有效 |
| 7 | 余弦退火是通用配方 | 1-shot: −0.56 HM vs 原版 Adam | ❌ 只在 5-shot 有效 |
| 8 | detach 阻断跨层梯度 | Base +2.68 但 Novel −2.61 | ❌ 伤共享特征 |
| 9 | Cosine + 低秩模型 | Novel −3.73 | ❌ 退火即停学 |
| 10 | 逐点深度门控 | 57 类门控值全饱和 0.92-1.00 | ❌ 退化为恒等 |
| 11 | 正交补去相关 | 双 seed ΔNovel = −0.045 | ❌ 精确为零 |
| 12 | 伪标签频次校准 | 全 tau 扫描 HM 增益 ≤ +0.28 | ❌ 过拟合 |

## 七、5-shot 结果（65536，单 seed s42）

| 配置 | PL 来源 | Base | Novel | HM |
|---|---|---|---|---|
| 末层基线 | 固定 1-shot PL | 0.6740 | 0.3030 | 0.4181 |
| 跨层残差双头 | 固定 1-shot PL | 0.6570 | 0.3155 | 0.4263 |
| 跨层残差双头+余弦退火 | 固定 1-shot PL | 0.6835 | 0.3165 | **0.4326** |
| 末层基线 | 在线 5-shot PL | 0.6815 | 0.2937 | 0.4105 |
| 跨层残差双头+余弦退火 | 在线 5-shot PL | 0.6788 | 0.3136 | 0.4290 |

官方 5-shot（5-shot PL + 102400）：0.6757 / 0.3167 / 0.4312

**PL 覆盖率对比**：
- 固定 1-shot PL：背景区 66.07%
- 固定 5-shot PL：背景区 73.64%
- 在线动态 5-shot PL：背景区 47.53%（含 NB-Mix 贴块污染分母）

## 八、关键辅助数据

### 8.1 解码器有效维度（同 64 维对比）

| 特征 | 维度 | eRank | eRank/D |
|---|---|---|---|
| dec0（末层） | 64 | 10.24 | **16.0%** |
| dec1（中间） | 64 | 21.22 | **33.2%** |
| dec2（中间） | 64 | 25.32 | **39.6%** |
| 拼接 cat192 | 192 | 32.36 | 16.9% |

**注意**：cat192 的 eRank 32.36 是 dec0 的 3.16 倍，但**维度也涨了 3 倍**，归一化后几乎没变（16.9% vs 16.0%）。**不能说"扩大 3.4 倍"。**

### 8.2 40 轮秩不涨

| 轮数 | dec0 eRank |
|---|---|
| 20 | 10.30 |
| 30 | 9.42 |
| 40 | 10.33 |

Stage-2 以 lr=1e-4 微调 backbone 40 轮，秩不涨反微降 → 坍塌是结构性的。

### 8.3 Omission Recovery 探针（3D-OCSB，已砍）

50 场景，B=ps已标，C=真该类但ps漏检，Neg=其他-1点：

| 特征空间 | AUC | AUC+Base拒斥 |
|---|---|---|
| RegionPLC（基线） | **0.9323** | 0.9162 |
| PTv3 dec0 | 0.8444 | 0.9262 |
| PTv3 dec1 | 0.9133 | 0.9478 |
| PTv3 dec2 | 0.9222 | 0.9433 |

阈值化后 Precision：3.1–3.3%（多类 argmax，thr=0.9）。**AUC 高但 Precision 不可用。**

## 九、待办

1. **`ln_only_s42`** 在跑（约 01:10）—— 测 LN 单独贡献
2. **`xl_s13`** 在跑（约 01:10）—— 补全跨层读出（无残差）的 3 seed
3. **`rep3_s42` 的 eRank** 在跑 —— 看 dec0 是否被撑开
4. **5-shot 固定 PL 版**：`base_k5_fixed` / `sym_k5_fixed`（可选）
5. **ScanNet++ 下载**：41/325 GB，速度 0.83 MB/s，还需 ~95 小时
6. **5-split 均值**（regis1-5）
7. **第二骨干验证**（SCN）

## 十、关键文件

| 内容 | 路径 |
|---|---|
| 主报告 | `CUT2_EXT_REPORT.md`（任务 1-5） |
| 结果固化 | `exp_reports/results_all.json`（所有 run 的 57 类 IoU） |
| 资产清单 | `exp_reports/ASSET_README.md` |
| 启动手册 | `exp_reports/RUNBOOK.md` |
| 代码 | `pointcept/models/default.py` |
| | `pointcept/engines/train.py` |
| | `pointcept/engines/test.py` |
| 探针 | `ps_diag/probe_omission.py`、`ps_diag/probe_rlp_auc.py` |

## 十一、写论文时的四个坑

1. **eRank 不能说"扩大 3.4 倍"** —— 维度也涨了 3 倍，必须用 eRank/D
2. **不能说"超越官方 5-shot"** —— 65536 vs 102400，口径不同
3. **`dec012_lp` 的 0.331 是全监督线性探测**（每类 2000 点伪标签），不是 few-shot 方法
4. ~~**不能说"跨层是增益主体"** —— 结构占 60–65%，跨层占 35–40%~~
   → **本条已被 2026-11-21 的 3-seed 数据推翻，见 12.10**

---

## 十二、2026-11-21 白天新增（重要）### 12.1 三个代码语义陷阱（已核实）

| 陷阱 | 事实 |
|---|---|
| `epoch` 参数语义 | 只用来算数据重复倍数 `loop = epoch // eval_epoch`（`defaults.py:127`），**不控制训练轮数** |
| 真正的训练轮数 | `max_epoch = cfg.eval_epoch`（`train.py:170`） |
| 总迭代数 | `epoch × 场景数 / batch_size`，与 `eval_epoch` 无关 |

所以 `gfspretrain.py` 的 `epoch=800, eval_epoch=100` 实际是：训 100 个 data loop、
每 loop 重复 8 遍数据，总 24 万 iter。

### 12.2 多 split 训练的致命缺陷（已修）

`GFS_VL_Trainer.train()` 原来只在有 per-split checkpoint 时 resume；没有 checkpoint
就**直接沿用上一个 split 训完的权重继续训**。这样 split2–5 是在"已见过 novel 类
20 个 epoch"的模型上继续，5-split 均值会被系统性抬高。

已修：无 checkpoint 且 `regis_train_list` 长度 > 1 时，重建模型 → 重注入
`BackboneLoader` 的 Stage-1 骨干 → 重建优化器/调度器/scaler。

### 12.3 5-split 的真实含义

`sc200_regis_{k_shot}_{seed}.txt` 里的第二个数字是 **support 抽样种子**，不是 split
编号。官方"5-split 平均"= 5 个 support 抽样（10/20/30/40/50）的平均。

已就绪的伪标签目录（各 2402 文件 = 1201 场景 × `_pl`/`_w`）：

| 种子 | 目录 | 新类点覆盖（PS+AI） |
|---|---|---|
| 10 | `pl_k1_s10`（软链到 `online_pl_w_uniform`） | 66.07%（背景区口径） |
| 20 | `pl_k1_s20` | 60.72% |
| 30 | `pl_k1_s30` | 62.55% |
| 40 | `pl_k1_s40` | 56.88% |
| 50 | `pl_k1_s50` | 61.56% |

启动器 `afford_diag/run_5split.sh`，靠 `GFS_OFFLINE_PL_TMPL=/data01/jh/data/pl_k1_s{seed}`
在 trainer 循环里自动切换。

### 12.4 正交零空间防坍塌（ONTR）的 Step A 数字

| 量 | 值 |
|---|---|
| `seg_head.weight` $W_b$ | (12, 64) |
| $Q_\perp$ | (64, 52)，`max|W_b @ Q_perp| = 1.508e-07` |
| dec2 采样 | 409,600 点（100 场景 × 4096，含 `-1` 背景） |
| 前 52 主成分解释方差 | **92.82%** |
| **dec2 全分辨率 eRank** | **39.13 / 128** |
| 对照 dec0 eRank（同口径） | 8.44 / 64（GridSample 0.02 下采样后） |

基已固化在 `/data01/jh/data/ontr_basis.pt`（`Q_perp` / `V52` / `mu2`）。

### 12.5 SpUNet / SCN 权重的事实

- Pointcept 官方 model zoo **只发布 PTv3 系列**，没有 SpUNet 权重
- HF 上 `Pointcept` 组织只有 4 个仓库：PointTransformerV3 / Sonata / Concerto / Utonia
- 本地 `pretrain_weights/sparseunet32_636.pth` 是 **3D 视觉语言模型的 MinkowskiEngine 版**
  SpUNet32 骨干（命名 `unet.u.blocks.block0.conv_branch`），与我们 spconv 版
  `SpUNet-v1m1-backbone`（命名 `enc.enc0.block0.conv1`）体系与层级都不同，不能直接加载
- 且 GFS-VL 的 Stage-1 是 **12 类 Base** 口径，任何公开权重都不是这个定义

### 12.6 下载带宽是出口 IP 级硬约束

实测五条通道（各 10–12 秒）：

| 通道 | 线程 | 总吞吐 |
|---|---|---|
| hf-mirror 直连 | 1 | 301 KB/s |
| hf-mirror 直连 | 8 | 471 KB/s |
| HF 官方 + 本机代理 7890 | 1 | 679 KB/s |
| HF 官方 + 本机代理 | 8 | 846 KB/s |
| HF 官方 + 本机代理 | 16 | 247 KB/s（代理限并发） |
| hf-mirror + 代理 | 8 | 0（308 重定向） |

`huggingface.co` 直连不通（http 000）。ModelScope 没有这个数据集。加线程最多 3 倍。
ScanNet++ 剩 281 GB，按 1.05 MB/s 约 3.1 天，只能后台挂 7 分片并行续传。

### 12.7 两个实现 bug（写代码时踩过）

1. PTv3 decoder 的参数名前缀是 `backbone.dec.dec0.*`，**不是** `backbone.dec.0.*`
2. `backbone.embedding.stem`（patch embedding）也要一起冻结，它不属于 `enc.` 前缀
3. `Compose` 只接受 dict 配置，不接受 transform 实例

### 12.8 共享机器调度纪律（2026-11-21 实测）

**CPU 现实**：机器 176 核；`xzy` 的 6 个 `python multi_process.py` 长期各占约 20 核
（合计约 120 核，已运行 13 天）；多个用户的 `codex` 进程各 180 线程（多数空闲但
推高线程总数）。我们实际可用 CPU 约 50 核。

**单线程才是真正的瓶颈**：
- `cut2_extract.py` 没有 DataLoader worker，单进程只用 1 核
- `pt_data_worker` 是 PyTorch 单个 worker，本身单线程，上限 1 核
- 所以"停掉别的任务"并不会让 extract 变快 —— 它从来只用了 1 核

**规范**：
1. 所有启动脚本统一 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`
2. 训练统一 `num_worker=2`
3. **诊断探针禁止全量 1201 场景多片推理**：一律抽样 100–200 场景 + 单次体素前向，
   3 分钟出结论
4. 需要快速提取特征时用**多进程分片**，不要单进程跑全量
5. 并发训练进程数 ≤ 3（不含关键路径上的提取任务）
6. 观察进程真实 CPU 占用时，`ps` 的 `%CPU` 是生命周期平均值，会严重误导；
   要用 `/proc/PID/stat` 的 utime/stime 增量配合 `vmstat` 的 us/id 判断

### 12.10 【重大修正】3-seed 归因翻转：跨层才是主体

补齐 `rep3_lovasz_s7/s13` 后，单 seed s42 的结论被彻底推翻：

| 方法 | s42 | s7 | s13 | 均值 | std | 配对 ΔHM | 逐 seed |
|---|---|---|---|---|---|---|---|
| `base_lovasz` | 0.4122 | 0.4303 | 0.4308 | 0.4244 | 0.0087 | — | — |
| `rep3_lovasz` | 0.4379 | 0.4242 | 0.4202 | 0.4274 | 0.0076 | **+0.0030** | `[+0.0257, −0.0062, −0.0106]` |
| **`sym_lovasz`** | 0.4414 | 0.4438 | 0.4379 | **0.4410** | **0.0024** | **+0.0166** | `[+0.0292, +0.0135, +0.0071]` |

**分解（总 +1.66）**：
- 结构容量 `rep3 − base`：**+0.30（18%），2/3 seed 为负 → 噪声**
- 真跨层 `sym − rep3`：**+1.36（82%），3/3 全正，std 最小**

**物理解释**：纯 CE 时长尾优化受压制，rep3 的 21 K 容量能临时救优化（+1.40，65%）；
补回 Lovász 修好优化后，在 `eRank=10.24` 的低秩 dec0 上堆参数退化为过拟合（2/3 seed 掉点），
只有引入 dec1/dec2 高秩新信息的 `sym` 稳定拿到 +1.36。

**结论**：**跨层残差双头是标准 GFS 设定（VLM + 全解冻）的主力**，
1-shot +1.36、5-shot HM **0.4553**（超官方 +2.41，四项全超）。
"结构占 60–65%"的旧结论**作废**。

### 12.11 ONTR 的定位与三场景规律

ONTR = 正交零空间防坍塌：Stage-1 续训时把冻结 `dec2` 的主成分写进 `dec0` 的
`ker(W_b)`（`|W_b Q_perp| = 1.5e-07`），零额外参数、零 hook、推理不变。

**三场景一致性（监督越弱，价值越大）**：

| 场景 | 监督强度 | ONTR 相对基线 | ONTR vs rep3 |
|---|---|---|---|
| 冻结骨干（只训头 3,660 参数） | 最弱 | **+10.75（+78%）** | **+1.54（赢）** |
| 纯 3D 无 VLM（裸线性） | 弱 | **+3.36（+33%）** | **+0.55（赢）** |
| **全解冻 + VLM（3-seed 均值）** | 强 | **−0.55（噪声）** | — |

**全解冻 + VLM 的 3-seed 真值**（2026-11-21 晚补齐）：

| 方法 | s42 | s7 | s13 | 均值 | 配对 ΔHM |
|---|---|---|---|---|---|
| `base_lovasz` | 0.4122 | 0.4303 | 0.4308 | 0.4244 | — |
| `rep3_lovasz` | 0.4379 | 0.4242 | 0.4202 | 0.4274 | **+0.0030** |
| `ontr_base_lovasz` | 0.4236 | 0.4150 | 0.4181 | 0.4189 | **−0.0055** |
| **`sym_lovasz`** | 0.4414 | 0.4438 | 0.4379 | **0.4410** | **+0.0166** |

**单 seed s42 上 `ontr_base` 的 +1.14、`rep3` 的 +2.57 都是幸运种子，3-seed 均值分别掉到 −0.55 和 +0.30。**
**只有 `sym` 是 3/3 全正且 std 最小（0.0024）。**

**机制证据**：`dec0_lp` 探针 **0.0721 → 0.1374（+90.6%）**；`eRank(h0@Q_perp)` 10.74 → 19.81（+84%）；
权威 Base mIoU **0.8057 → 0.8014（−0.43pp）**。

**冻结态 2×2 因子（HM）**：

| | 裸线性 | rep3 宽头 |
|---|---|---|
| 原权重 | 0.1370 | 0.2291 |
| **ONTR** | **0.2445** | **0.2955** |

- LN 只值 +2.10（且掉 Base −1.51），ONTR 值 +10.75（且涨 Base +2.13）→ **不是 LayerNorm**
- 零参数（3,660）打赢三万参数（32,972）
- **冻结态可叠加（+6.64）**，全解冻态不叠加（−0.06）

### 12.12 【致命陷阱】纯 3D 必须用 empty_pl，不能 unset

`target_calibrate()` 的逻辑：

```python
if os.environ.get("GFS_OFFLINE_PL", ""):
    return input_dict["segment"].clone()   # 用离线 PL，跳过 VLM
data_dict = self.construct_3D_vlm_input(input_dict)
ret_dict = self.VLM_3D(data_dict)          # ← 否则跑在线 RegionPLC VLM
```

**`unset GFS_OFFLINE_PL` 会触发在线 VLM**。纯 3D 的正确开关是
`GFS_OFFLINE_PL=/data01/jh/data/empty_pl`（空目录，0 文件）。

**验证判据（不用改代码）**：训练 loss 量级
- 纯 3D ≈ **0.35–0.47**
- VLM 版 ≈ **2.79–2.99**
- 差 8 倍

`support_ontr_s42_BUG_vlm` / `support_xl_s42_BUG_vlm` 就是踩了这个坑的归档。

### 12.9 Infra 性能优化：单样本 21.9 s → 0.45 s（49×）

`profile_loader.py` 逐段拆解发现三座大山（当时系统还在 swap 抖动，绝对值偏大）：

| 环节 | 优化前 | 优化后 | 加速 | 根因与修法 |
|---|---|---|---|---|
| `get_data`（NB-Mix） | 10.849 s | **0.097 s** | 112× | `_get_instances_from_class` 每个训练样本都重新 `np.load` 该类**全部** support 场景，5-shot 下白读 4/5。改为 `self._inst_cache` 懒加载缓存（全集仅 45/225 样本） |
| `ElasticDistortion` | 6.884 s | **0.177 s** | 39× | `scipy.interpolate.RegularGridInterpolator` 逐点 Python 调用。换成纯 NumPy 三线性插值，5 万点对比**最大误差 6.2e-15**。**增广概率保持 `random()<0.95` 不变** |
| `GridSample` | 3.894 s | **0.049 s** | 79× | 主要是 swap 抖动的放大；内存恢复后自然回落 |
| 合计 | 21.883 s | **0.448 s** | 49× | batch 12：112.6 s → 5.4 s |

**备份文件**：`pointcept/datasets/scannet.py.bak_infra`、
`pointcept/datasets/transform.py.bak_infra`。

**配套的环境事实**：
- `/data01` 是 **HDD**（`rotational=1`，ST20000NM007D），`/data02` 同；
  系统盘 `sda` 是 894 GB SATA SSD，根分区 98% 满（`du` 只找到 47 G，`df` 说 755 G，
  差 708 G 查不到实体文件，不归我们管）
- 训练数据 8.4 GB + 离线 PL 671 MB 已复制到 `/dev/shm`（120 GB tmpfs），
  随机读从 128 ms/场景降到 1.8 ms（71×）；但按 profile 它只占原总耗时的 2%，
  **只有在 dataloader 优化后它的价值才体现出来**
- **`cut2_extract.py` 会把全部分片特征累积在内存**（400 场景 ≈ 54 GB/进程），
  三片并行时曾把内存吃到只剩 1.5 GB、swap 100% 满，导致所有进程进入 D 状态。
  **以后最多两片并行，或改成分段落盘**

### 12.13 GFS-VL 的 `max_epoch` bug（2026-11-21 夜）

`pointcept/engines/train.py:169` 原文：

```python
self.max_epoch = cfg.eval_epoch      # cfg.epoch 根本没被读取
```

**后果**：想跑 N 轮必须写 `eval_epoch=N`，否则实际轮数被 `eval_epoch` 截断
（第一次跑 Stage-1a 写 `epoch=40 eval_epoch=10`，日志显示 `[1/10]`）。

已改为 `self.max_epoch = cfg.epoch`。历史配置全是 `epoch == eval_epoch`，
**不影响任何历史结果的可比性**。备份 `train.py.bak_maxepoch`。

### 12.14 `shuffle_orders=True` 在 eval 模式下仍然随机（重要）

`pointcept/models/utils/structure.py:94-98` 与
`point_transformer_v3m1_base.py:432-436`：

```python
if shuffle_orders:
    perm = torch.randperm(code.shape[0])
```

**都没有 `self.training` 判断**。实测同一模型、同一输入、`eval()` 模式，
两次前向 enc4 特征 max 差 **4.89**。

**影响**：过去所有 GFS 评估都带这个抖动，单 seed 间差异里有一部分不是种子差异。
**处置**：正式评估时强制 `shuffle_orders=False`（训练保持 True，与历史同口径）。

### 12.15 Sonata-Small 接入正式管线（详见 `SONATA_SMALL_INTEGRATION.md`）

- `sonata_small.pth` = **38.65M 纯 encoder**（`enc_mode=True`），
  `enc_channels=(32,64,128,256,512)`、`enc_depths=(2,2,2,6,2)` **与我们 PTv3 完全相同**
- 我们的 `backbone.enc` 也是 **38.65M**，`backbone.dec` 7.50M，总计 46.17M
- **接入后 46.14M，精确同参数量**（不是以大欺小）
- 必须改：`in_channels 6→9`、linear embedding、`norm_all_ln=True`
  （sonata 的 state_dict 里 `running_mean` 键数量为 0，全是 LayerNorm；
  我们原版 `embedding.stem.norm` 与 `enc.*.down.norm.0` 是 BatchNorm1d，
  `strict=False` 不会报警但前向会算废）
- 验收：embedding 0 误差、enc0 **3.5e-07**、enc1 按 `pooling_inverse` 展开 **2e-06**
- enc2 以上对不齐是**排序语义差异**（pointcept 按 Z-order，sonata 按字典序），
  不是权重问题，也不是 bug
- Stage-1a 实测：**1 epoch = 400 iter × 0.45 s ≈ 3 分钟**（Data 只要 6 ms，
  dataloader 的 49× 优化在这里全额兑付）

### 12.16 【重要纠正】多卡评估其实是精确的，不必单卡重评

之前记录「`GFSSemSegTester.one_regis_test` 没有跨 rank all-reduce，3 卡只统计 1/N 场景，
结果不可比」，**这条是误判**。

代码实证（`pointcept/engines/test.py:439-590`）：

```python
logger.info("Syncing ...")
comm.synchronize()
record_sync = comm.gather(record, dst=0)
if comm.is_main_process():
    record = {}
    for _ in range(len(record_sync)):
        r = record_sync.pop()
        record.update(r)                      # 按场景名合并
    intersection = np.sum([m["intersection"] for _, m in record.items()], axis=0)
    union = np.sum([m["union"] for _, m in record.items()], axis=0)
    target = np.sum([m["target"] for _, m in record.items()], axis=0)
```

**这是精确的全局聚合。** 日志里的 `[28/104]` 只是每卡自己的进度条（104 = 312/3），
不是结果范围。

Stage-1 的 `SemSegEvaluator` 也有 `dist.all_reduce`（`evaluator.py:157`）。

**结论：3 卡可直接评估，不需要单卡重评，每次省 8 分钟。**
AGENTS.md 里那条已同步改正。

### 12.17 Stage-1（12 类）的真实对标值是 0.8057

不是 0.6870（那是 57 类 Stage-2 的 Base mIoU，45 个新类会抢点）。

来源：`/data01/jh/data/ontr_test_bk.log`
`Val result: mIoU/mAcc/allAcc 0.8057/0.8760/0.9392`

sonata_small Stage-1a 进度（3 卡，每轮 300 iter ≈ 2.5 分钟含 eval）：

| epoch | mIoU | allAcc | 达成率 |
|---|---|---|---|
| 1 | 0.5863 | 0.8825 | 72.8% |
| 2 | 0.6836 | 0.9041 | 84.8% |
| 3 | 0.6947 | 0.9071 | 86.2% |
| 4 | 0.7230 | 0.9153 | 89.7% |

### 12.18 GFS-VL 的 `loop` 机制（400 iter 的真正来源）

`pointcept/engines/defaults.py:127`：

```python
cfg.data.train.loop = cfg.epoch // cfg.eval_epoch
```

所以：

| 配置 | loop | len(ds) | 单卡 bs12 | 3 卡 bs4 |
|---|---|---|---|---|
| `epoch=40, eval_epoch=10` | 4 | 4804 | **400 iter** | — |
| `epoch=15, eval_epoch=5` | 3 | 3603 | 900 iter | **300 iter** |
| `epoch=X, eval_epoch=X`（历史全部） | 1 | 1201 | 100 iter | — |

**这解释了为什么单独 `build_dataset` 时看到 `loop=1`** —— 那条路径不走 `default_setup`，
`loop` 是在 `default_setup` 里注入的。

配套的 `train.py:355`：

```python
total_steps = len(train_loader) * cfg.eval_epoch   # 原始
total_steps = len(train_loader) * cfg.epoch        # 修正
```

原版只覆盖**一个 eval 周期**的步数。历史配置 `epoch == eval_epoch` 时
`loop = 1` 且 `total_steps` 恰好等于总步数，两处互相掩盖；一旦拆开就会在
第 `eval_epoch + 1` 轮开头崩：

```
ValueError: Tried to step 1501 times. The specified number of total steps is 1500
```

**结论：想同时要「多轮 + 中途 eval」就必须两个 bug 一起修（已修）。
想省事就保持 `epoch == eval_epoch`。**

### 12.19 【血亏教训】分片下载不能用 `--retry` + `--max-time`

2026-11-22 凌晨实测：`curl --retry 5 --retry-all-errors --max-time 3600 -r start-end -o seg.part`
是一个**无限清零死循环**。

原因：
- `--max-time 3600` 每小时**强杀**一次
- `--retry` 让 curl **在内部重试**，重试时 `-r` 起点固定、`-o` 覆盖写 → **每次从 0 重下**
- shell 层的 `for` 循环永远等不到 curl 退出，`cat .part >> seg$i` 的续传逻辑**形同虚设**

实测后果：5 路各下到 786 MB 就被清零，两小时白烧 7.8 GB 代理流量，`dl.log` 里
`round` 记录数为 **0**（循环体从未执行）。

**正确写法**（`dl_scanpp_parallel.sh`）：
```bash
curl -sSL -r $((start + have))-${end} \
     --connect-timeout 30 --speed-time 180 --speed-limit 1024 \
     -o "$f.part" "$URL"          # 无 --retry / 无 --max-time
```
让 curl 单次跑到失败或极慢就退出，由 shell 循环按 `stat -c%s $f` 续传。

**另两个排查陷阱**：
- `du -ch seg[0-9]` 匹配不到 `seg0.part`（通配符不含 `.part`），会误判成"下载为 0"
- `xargs kill` 的 grep 模式如果写 `ScanNetpp.tar.gz`，会连带杀掉分片 curl（其命令行含同一 URL）

**验证过的正常速率**：5 路并发 1106 KB/s（单路 170 KB/s，**6.5×**）；瓶颈在本机代理单连接限速。

