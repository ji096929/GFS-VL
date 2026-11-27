# 白天并行实验报告（2026-11-21）

> 三条 Track 并行推进。所有训练严格 `regis_train_list=["regis1"]`、原版 Adam、
> 单卡完整 312 val 场景，除 Track 2 的 Stage-1 续训外不加任何额外结构。

---

## 一、Track 1：主表收口

### 1.1 跨层辅助损失 L_cross 归档（已完成）

| 口径 | 有 L_cross | 无 L_cross | ΔHM |
|---|---|---|---|
| 1-shot（65536） | `sym_lcross01_k1_s42` 0.4234 | `sym_s42` 0.4257 | **−0.23** |
| 5-shot（102400） | `sym_lcross01_k5_102400_s42` 0.4406 | `sym_k5_102400_s42` 0.4410 | **−0.04** |

**结论：5-shot 上是纯噪声，1-shot 上直接掉点。不能作为 1/5-shot 统一配方，归档。**

### 1.2 `末层基线 + Lovász` 的 Base 真值核查

原报告 4.1 节写 `0.6701`、4.2 节写 `ΔBase = −0.31`，都是错的。从三个 `result/val.pth`
重新核算：

| seed | Base | Novel | HM |
|---|---|---|---|
| s42 | 0.6862 | 0.2946 | 0.4122 |
| s7 | 0.6754 | 0.3157 | 0.4303 |
| s13 | 0.6680 | 0.3179 | 0.4308 |
| **均值** | **0.6765** | **0.3084** | **0.4244** |

同口径基线三 seed 的 Base 均值是 0.6787，所以 **ΔBase = −0.21**（不是 −0.86，
也不是 −0.31）。已修正 `OVERNIGHT_ICCV_REPORT.md`。

### 1.3 满血 5-shot + Lovász 主对照（已完成）

| 配置 | Base | Novel | HM | mIoU |
|---|---|---|---|---|
| `base_lovasz_k5_102400_s42`（末层基线 + Lovász） | 0.6715 | 0.3385 | 0.4501 | 0.4086 |
| **`sym_lovasz_k5_102400_s42`（跨层残差双头 + Lovász）** | **0.6835** | **0.3413** | **0.4553** | **0.4134** |
| 官方 Tab.2 | 0.6757 | 0.3167 | 0.4312 | 0.3929 |
| **Δ vs 官方** | **+0.78** | **+2.46** | **+2.41** | **+2.05** |

**HM 0.4553 落进目标区间 0.455–0.460，四项全超官方。**

两条值得记录的规律：

1. **Lovász 在满血 5-shot 上依然有效**：基线 0.4216 → 0.4501（**+2.85**），
   方法 0.4410 → 0.4553（**+1.43**）。Lovász 是 1/5-shot 统一配方，成立。
2. **配对 Δ 从 +1.94 缩到 +0.52 HM**。原因是基线吃 Lovász 的增益更大
   （+2.85 vs +1.43）—— Lovász 与跨层读出在"补 Base"上有功能重叠，不可简单相加。

### 1.4 5-split（5 个 support 抽样种子）准备就绪

发现 `sc200_regis_{k_shot}_{seed}.txt` 里的第二个数字是 **support 抽样种子**，
不是 split 编号。官方口径的"5-split 平均"实际是 5 个不同 support 抽样（10/20/30/40/50）。

| 种子 | 伪标签目录 | 文件数 | 新类点覆盖（PS+AI） |
|---|---|---|---|
| 10 | `pl_k1_s10`（指向 `online_pl_w_uniform`） | 2402 | 66.07%（背景区口径） |
| 20 | `pl_k1_s20` | 2402 | 60.72% |
| 30 | `pl_k1_s30` | 2402 | 62.55% |
| 40 | `pl_k1_s40` | 2402 | 56.88% |
| 50 | `pl_k1_s50` | 生成中 | 61.56% |

四个新种子的覆盖率跨度 56.9–62.6%，内部一致，没有异常种子。

**同时修掉一个会让 5-split 结果作废的缺陷**：`GFS_VL_Trainer.train()` 原来只在有
per-split checkpoint 时 resume，没有 checkpoint 时会带着上一个 split 的微调权重
继续训 20 epoch。已改为多 split 时重建模型、重注入 Stage-1 骨干、重建优化器。

---

## 二、Track 2：正交零空间防坍塌（ONTR）

### 2.1 设计要点

Stage-1 从源头阻止 `dec0` 的隧道坍塌：把预训练高秩 `dec2` 的几何结构写进基类
分类头 $W_b$ 的 52 维正交零空间 $\ker(W_b)$。Base 零损伤、推理零额外参数、零 hook。

### 2.2 Step A：基预计算（已完成）

| 量 | 值 |
|---|---|
| $W_b$ | (12, 64) |
| $Q_\perp$ | (64, 52) |
| $\max\|W_b Q_\perp\|$ | **1.508e-07**（精确正交） |
| $W_b$ 奇异值 | 2.496, 2.062, 1.639, ..., 0.519（12 个，无近零） |
| dec2 采样点 | 409,600（100 场景 × 4096 点，含 `-1` 背景） |
| 前 52 主成分解释方差 | **92.82%** |
| **dec2 全分辨率 eRank** | **39.13 / 128** |
| 对照：dec0 eRank | 10.24 / 64 |

**这是整个探针的可行性基础**：零空间能容纳的秩上限约 39，是 dec0 现有 10.24 的
近 4 倍，且 52 个主成分已覆盖 dec2 的 92.8% 方差。

### 2.3 Step B：零空间续训（运行中）

- 冻结：`backbone.embedding` / `backbone.enc.enc0-4` / `backbone.dec.dec2` / `dec.dec3` / `seg_head`
- 只训：`backbone.dec.dec0` + `backbone.dec.dec1`
- 损失：`L = CE + Lovász + 1.0 × SmoothL1(z, t)`，全部在 fp32 下算
- 20 个 epoch × 300 iter = 6000 iter

| epoch | L_base | L_null | dec0 eRank（末 batch） |
|---|---|---|---|
| 0 | 0.1815 | 0.2568 | 9.66 |
| 3 | 0.1757 | 0.1650 | 12.14 |
| 5 | 0.1727 | 0.1377 | 13.57 |
| 7 | 0.1705 | 0.1185 | 14.63 |
| 9 | 0.1662 | 0.1064 | **15.52** |

两个损失同量级且同步下降，`L_null` 从 0.49 降到 0.11，零空间目标在被拟合；
`L_base` 从 0.18 降到 0.17，基类没有退化。**10 个 epoch 内 eRank 涨了 61%。**

### 2.5 生死判据：Stage-1 冻结 dec0 的 45 类线性探针

从 `cut2_feat/S1` 的 57×57 混淆矩阵算出的基线（新类 45 类，点数 3,170,251）：

| 特征层 | 线性探针 mIoU |
|---|---|
| **dec0（单层，ONTR 直接作用对象）** | **0.0988** |
| dec1 | 0.1284 |
| dec2 | 0.1741 |
| multi（512 维拼接） | 0.2217 |

**ONTR 成功的标志是 dec0 单层从 0.099 往 0.16–0.20 靠**，也就是"用零空间把 dec2 的秩搬进 dec0"这件事
在真实下游可分性上兑现，而不只是 eRank 这个内部指标好看。

### 2.5.1 一个必须澄清的指标陷阱

`eRank(h0)` 会被基类子空间的大奇异值压住，看不到零空间里发生了什么：

- 基类子空间 `h0 @ Q_b` 的方差约 10（12 个方向）
- 零空间目标 `t = LN(...)` 的每维方差被归一化到 1.0
- 两者混在一起算熵，前 12 个大奇异值主导结果

所以 Step C 额外打印：

| 诊断量 | 意义 |
|---|---|
| `eRank(h0 @ Q_perp) / 52` | **纯零空间秩**，这才是 ONTR 真正改了的地方 |
| `eRank(h0 @ Q_b) / 12` | 基类子空间是否被破坏 |
| `eRank(LN(h0))` | 消掉尺度差后的总秩 |
| `RMS(h0 @ Q_b)` vs `RMS(h0 @ Q_perp)` | 两个子空间的尺度比，决定要不要补一个 `LN(dec0)` |

### 2.6 一个口径错误（已修）

第一版体检脚本在 `GridSample` 降采样后的点上直接算 IoU，得到 Base mIoU **0.5034**，
与训练时的 0.79 低一大截 —— 分母点集不同。已改为
`GridSample(return_inverse=True)` + 预测 `pred_voxel[inverse]` 散射回原点云，
与 `SemSegTester` 同口径。同时用 `tools/test.py` 跑一份权威数字做交叉验证。

### 2.5 两个实现 bug（已修）

1. **decoder 参数名前缀**：实际是 `backbone.dec.dec0.*`，不是 `backbone.dec.0.*`，
   最初写的冻结逻辑匹配不到任何参数（可训练参数 0.00 M）。
2. **`backbone.embedding.stem`** 也要冻结（patch embedding 属于 enc 前端）。

---

## 三、Track 3：基础设施

### 3.1 ScanNet++ 下载：带宽是硬约束

实测五条通道（各 10-12 秒）：

| 通道 | 线程 | 总吞吐 |
|---|---|---|
| hf-mirror 直连 | 1 | 301 KB/s |
| hf-mirror 直连 | 8 | 471 KB/s |
| HF 官方 + 本机代理（7890） | 1 | 679 KB/s |
| HF 官方 + 本机代理 | 8 | 846 KB/s |
| HF 官方 + 本机代理 | 16 | 247 KB/s（代理限并发，多数连接被拒） |
| hf-mirror + 本机代理 | 8 | 0（308 重定向） |

**限速在出口 IP 级别，不在连接级别**：加线程最多 3 倍，不能线性扩展。
`huggingface.co` 直连不通（http 000），必须走代理。

结论：
- ModelScope 没有这个数据集（只有 5 GB 的 `lyy715/ScanNetPP`，不是同一份）
- HF 页面只提供 325 GB 单包，无网盘/直链替代
- **281 GB 剩余，按 0.95 MB/s 约需 3.4 天**，只能后台挂着
- 已重启 7 分片并行续传（`dl_scannetpp_parallel.sh`）

### 3.2 SpUNet Stage-1

**官方没有发布 SpUNet 权重**：Pointcept 的 model zoo 只提供 PTv3 系列，
HF 上 `Pointcept` 组织只有 4 个仓库（PointTransformerV3 / Sonata / Concerto / Utonia），
README 里 SpUNet 只有训练命令和配置名。

本地 `pretrain_weights/sparseunet32_636.pth` 不能用：它是 3D 视觉语言模型的
MinkowskiEngine 版 SpUNet32 骨干（命名 `unet.u.blocks.block0.conv_branch`），
与我们 spconv 版 `SpUNet-v1m1-backbone`（命名 `enc.enc0.block0.conv1`）体系不同，
层级结构也不同（30.12 M vs 39.16 M）。唯一契合点是 `input_conv.0.weight` 是
`(32,3,3,3,6)`，in_channels=6 与我们的 color+normal 一致。

而且 GFS-VL 的 Stage-1 是 **12 类 Base** 口径，任何公开权重（ScanNet 20 类 /
ScanNet200 198 类）都不是这个定义。所以自训必要。

**进度：5/100 loop，约 0.25 s/iter，总迭代 800×1201/4 = 24 万，约 16 小时。**

### 3.3 三个代码语义陷阱（已核实并记录）

| 陷阱 | 事实 |
|---|---|
| `epoch` 参数的语义 | 只用来算数据重复倍数 `loop = epoch // eval_epoch`，**不控制训练轮数** |
| 真正的训练轮数 | `max_epoch = cfg.eval_epoch`（`train.py:170`） |
| 总迭代数 | `epoch × 场景数 / batch_size`，与 `eval_epoch` 无关 |

所以 `gfspretrain.py` 的 `epoch=800, eval_epoch=100` 实际是：训 100 个 data loop、
每 loop 重复 8 遍数据，总 24 万 iter。

---

## 五、Infra 性能优化（21.9 s → 0.45 s/样本，49 倍）

### 5.1 Profile：三座大山

在系统 swap 100% 满、内存只剩 1.5 GB 的状态下逐段计时（所以绝对值偏大）：

| 环节 | 耗时 | 占比 | 真因 |
|---|---|---|---|
| `get_data`（NB-Mix） | **10.849 s** | 50% | `_get_instances_from_class` 每个训练样本都重新 `np.load` 该类**全部** support 场景，5-shot 下白读 4/5 的盘 |
| `ElasticDistortion` | **6.884 s** | 31% | `scipy.interpolate.RegularGridInterpolator` 是 Python 类，对 40-50 万点逐点调用 `__call__` |
| `GridSample` | **3.894 s** | 18% | **不是算法问题**，是 swap 抖动把所有内存分配放大 |
| 其余 15 个 transform | 0.25 s | 1% | — |

单样本 **21.883 s** → batch 12 = **112.6 s**，而 GPU 前向只要 0.5 s。这就是 GPU util 长期 0% 的直接原因。

### 5.2 两处外科手术（数学等价，增广分布不变）

| 改动 | 文件 | 做法 |
|---|---|---|
| NB-Mix 缓存 | `pointcept/datasets/scannet.py` | `self._inst_cache` 懒加载；support 全集仅 45（1-shot）/225（5-shot）个，几百 MB |
| 三线性插值 | `pointcept/datasets/transform.py` | `RegularGridInterpolator` → 纯 NumPy 8 角点加权；5 万点对比 **最大误差 6.2e-15** |

**没有降低任何增广概率**：`ElasticDistortion` 仍是 `random() < 0.95`，两个 granularity 都跑。

### 5.3 效果

| 环节 | 优化前 | 优化后 | 加速 |
|---|---|---|---|
| `get_data` | 10.849 s | **0.097 s** | **112×** |
| `ElasticDistortion` | 6.884 s | **0.177 s** | **39×** |
| `GridSample` | 3.894 s | **0.049 s** | **79×**（未改代码，纯内存恢复）|
| 合计 | 21.883 s | **0.448 s** | **49×** |

**batch 12：112.6 s → 5.4 s。**
`cut2_extract` 同步从 18 s/场景降到 **1.65 s/场景**，内存从 54 GB/进程降到 **7.3 GB/进程**。

### 5.4 环境事实（务必记住）

| 项 | 事实 |
|---|---|
| `/data01` | **HDD**（`rotational=1`，ST20000NM007D），`/data02` 同 |
| 系统盘 `sda` | 894 GB **SATA SSD**，根分区 98% 满；`du` 只找到 47 G，`df` 说 755 G，**708 G 查不到实体文件**（非我们所有，不动） |
| **全机无 NVMe** | `lspci` 只有 3 个 SATA 控制器 |
| `/dev/shm` | 120 GB tmpfs；已放 8.4 GB 训练数据 + 671 MB PL（随机读 128 ms → 1.8 ms，71×）|
| `cut2_extract.py` | 全部分片特征累积在内存，**三片并行曾把内存吃到只剩 1.5 GB、swap 100% 满**，所有进程进 D 状态。**以后最多两片并行** |
| `xzy` 的 6 个 `multi_process.py` | 13 天，top 实测 5 个各占 26-33 核（约 155 核），性质是正常 CPU 密集计算（`min_flt` 持续增长、R 线程数匹配 CPU 占用），**不是死锁/孤儿** |
| 我们的可用 CPU | 约 **21 核 / 176** |

### 5.5 中间状态（12:18）

| 任务 | 进度 |
|---|---|
| `ontr_base_lovasz` / `ontr_sym_lovasz` | 12/20（1.7 s/iter，每轮 2.8 分钟）|
| `ontr_ln_lovasz` | 10/20 |
| extract s0b / s1b / s2b | 重跑中，1.65 s/场景，预计 12:40 |
| `rep3_lovasz_s7` / `s13` | 刚在 GPU1 / GPU4 启动 |
| 5-split（1-shot） | 保持暂停，等 ONTR 裁决 |

- Track 1.3：满血 5-shot + Lovász 三组结果
- Track 2 Step C：Base mIoU + dec0 eRank 体检
- Track 2 Step D：`ontr_base_lovasz_s42` / `ontr_sym_lovasz_s42`
- 5-split 五个种子的均值
