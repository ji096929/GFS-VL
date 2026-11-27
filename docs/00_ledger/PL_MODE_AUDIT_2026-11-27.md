# 伪标签模式审计：在线 / 离线口径盘点（2026-11-27）

> 起因：核对 ScanNet200 1-shot 基线该用哪一套时，发现台账里 85 个实验走的是离线伪标签缓存，
> 而基线和方法之间存在「在线 vs 离线」混比。本文档固化判据、清点家底、定下不混比的纪律。

---

## §1 判据（10 秒定性，不再靠目录名猜）

**看 `train.log` 里有没有这一行：**

```
==> Loading parameters from checkpoint ./pretrain_weights/sparseunet32_636.pth to GPU
```

| 有 | 无 |
|---|---|
| **在线**：`build_zero_shot_model()` 被调用，VLM 加载 | **离线**：`train.py:373-378` 直接把 `VLM_3D = None` |

**为什么可靠**：`train.py:373` 的 `if not os.environ.get("GFS_OFFLINE_PL", "")` 是唯一的分叉点，
离线时整个 3D VLM 都不构建，日志里不可能出现 `network_template.py` 的任何输出。

**辅助判据（不单独使用，仅交叉验证）**：

| 判据 | 在线 | 离线 |
|---|---|---|
| 每迭代耗时 | 高（每 batch 多算一次 VLM 前向 + 相似度） | 低 |
| config 里的 `pl_weight` | 无 | 软权重缓存有、均匀缓存无 |

**`s/iter` 不能单独用**：`num_worker` 从 2 到 24 有 9 种取值，数据加载速度差异能到 2 倍，会掩盖 VLM 的开销。

---

## §2 家底清点

| 数据集 | 有 `train.log` 的 run | 在线 | 离线 |
|---|---|---|---|
| **ScanNet++** | 97 | **97** | **0** |
| **ScanNet200** | 106 | **21** | **85** |

### 2.1 ScanNet++ 物理上不支持离线

`pointcept/datasets/scannetpp.py` 里**没有任何 `GFS_OFFLINE_PL` 分支**，只有 `scannet.py:422-448` 有。
所以 ScanNet++ 的每一局都必须现场跑 VLM。

**结论：ScanNet++ 的 12 格全部可用，不需要任何重跑。**

### 2.2 ScanNet200 的 21 个在线实验

| run | 结构 / 用途 |
|---|---|
| `sc200_k1_ONLINE_s42 / s7 / s13` | 1-shot base，三 seed |
| `sc200_k1_sym_gtproj_s7` | 1-shot sym（gtproj 变体） |
| `sc200_k1_m5_cleancon_s7` | M5 + cleancon |
| `sc200_k1_m5_cleancon_v2_s7` | M5 + cleancon v2 |
| `sc200_k1_m5_cleancon_v2_softent_s7` | M5 + cleancon softent |
| `sc200_k1_m5_elr_s7` | M5 + ELR（发散崩，已归档） |
| `sc200_k1_m5_entq10_s7` | M5 + entq10 |
| `sc200_k1_m5_gce07_s7` | M5 + gce07 |
| `sc200_k1_m5_m1vote_s7` | M5 + m1vote |
| `sc200_k1_m5_psai_d2r10_s7` | M5 + d2r10 |
| `sc200_k1_m5_psai_fullent10_s7` | M5 + fullent10 |
| `sc200_k1_m5_psai_fullent30_pp_s7` | M5 + fullent30 + 先验守恒 |
| `base_k5_true_s42` / `base_k5_truemulti_s42` | 5-shot base（真 5-shot 变体） |
| `cos_sym_k5_true_s42` | 5-shot sym + 余弦退火 |
| `sc200_k5_m5_m1vote_s7` | 5-shot M5 + m1vote |
| `sym_lovasz_5split_s42` | 5 折 + Lovász |
| `support_ontr_s42_BUG_vlm` / `support_xl_s42_BUG_vlm` | 已作废的 support 实验 |

### 2.3 唯一真正的跨折违规

| run | 问题 | 处理 |
|---|---|---|
| **`sc200_k1_offline`** | config 写 5 折，实际跑完 regis1 + regis2，**最终 0.4038 是 regis2 的** | **作废** |

**注意**：`sc200_k1_BASEu_s7` / `sc200_k1_XLu_s42` 的 config 也写了 5 折，但都在 regis1 结束就中断，
**最终 Eval 是 regis1 的，结果干净**。config 写 5 折不等于结果用了 5 折，必须看最终 Eval 落在哪个 regis。

---

## §3 离线缓存的合法性核查

### 3.1 生成逻辑（`tools/gen_offline_pl.py`）

```python
cfg.data.train.k_shot  = cfg.k_shot
cfg.data.train.seed    = cfg.data.regis1['seed']
val_data = build_dataset(cfg.data.regis1)
trainer.vlm_novel_proto = trainer.extract_3d_vlm_proto(regis_loader)   # 原型来自 regis1 support

new = trainer.target_calibrate(input_dict)          # 与训练完全相同的 PS + AI
pl  = torch.where((segment == -1) & (new >= 0), new, -1)   # 只填背景区
```

| 检查项 | 结论 |
|---|---|
| 跨折泄漏 | **无**。原型来自 `regis1`，与训练同源 |
| 标签泄漏 | **无**。只填 `segment == -1` 的背景区，不碰真值 |
| 算法一致性 | **一致**。`target_calibrate` 内部完全走 VLM 前向（`self.VLM_3D`），不依赖训练模型预测 |

### 3.2 实测

| 项 | 结果 |
|---|---|
| `online_pl_w` vs `online_pl_w_uniform` 的 `_pl.npy` | 前 200 场景 **200/200 完全一致** |
| 唯一区别 | `_w.npy`：uniform 版全 1.0，软权重版 0.05–1.0 |
| 标注覆盖率 | 32.80% |

### 3.3 两个真实瑕疵

**① 缓存比脚本早一天**

| 项 | 时间 |
|---|---|
| `online_pl_w` | 11-19 13:08 |
| `online_pl_w_uniform` | 11-19 14:09 |
| `gen_offline_pl.py` 首次 git 提交 | **11-20 16:31** |

**缓存的生成器不可追溯**——可能是 11-19 未提交的工作区版本，也可能是另一个已删除的脚本。

**② k_shot 绑定**

缓存是用 **1-shot 原型**生成的。所以：

| 用途 | 是否匹配 |
|---|---|
| 1-shot 训练读这份缓存 | ✅ 匹配 |
| **5-shot 训练读这份缓存** | ❌ **等于白扔 4 个 support** |

`CUT2_EXT_REPORT.md:224` 早就抓到过这一点。

---

## §4 不混比纪律（硬约束）

> **同一张表里：离线只和离线比，在线只和在线比。绝不拿在线基线去算离线方法的涨幅。**

### 4.1 各口径的可用基线

| 数据集 | shot | 口径 | 基线 | 状态 |
|---|---|---|---|---|
| ScanNet++ | 1 | 在线 | `scpp_base_k1_s42/s7/s13` = 0.2843/0.2673/0.3125 | ✅ 齐 |
| ScanNet++ | 5 | 在线 | `scpp_base_k5_s42/s7/s13` = 0.3060/0.3189/0.3047 | ✅ 齐 |
| ScanNet200 | 1 | **在线** | `sc200_k1_ONLINE_s42/s7/s13` = 0.3855/0.3904/0.3823（均值 **0.3861**） | ✅ 齐 |
| ScanNet200 | 1 | **离线** | `BASEu_s7` 0.4046 / `base_s13` 0.4116 / **s42 待补** | ⚠️ 缺 1 |
| ScanNet200 | 5 | **离线** | `base_k5_102400_s42/s7/s13` = 0.4216/0.4346/0.4275（均值 **0.4279**） | ✅ 齐 |

### 4.2 已知的「假涨幅」（必须改写）

| 原结论 | 错误原因 | 正确值 |
|---|---|---|
| ScanNet200 1-shot sym **+3.61** | 用在线基线 0.3861 比离线方法 0.4222 | 离线内部：0.4222 vs 约 0.405 → **约 +1.7** |
| ScanNet++ 加 Lovász **掉 3.00** | **`scpp_sym_lovasz` 根本没开 Lovász**（见 §5） | 该结论作废 |

---

## §5 顺带查出的两个假名字

### 5.1 ScanNet++ 的 Lovász 实验从未开启 Lovász

```
scpp_sym_lovasz/config.py:
    criteria=[dict(type='CrossEntropyLoss', ...)]
    ← 没有 use_lovasz = True

scannet200/sym_lovasz_s42/config.py:
    use_lovasz = True          ← 这个才是真开了
```

**Sc++ 的 102 个 config 里 `use_lovasz=True` 的数量是 0。**

所以 `scpp_sym_lovasz`（0.2986）、`scpp_base_lovasz`（0.2804）、`scpp_base_lovasz_k5`（0.3150）
**名字带 lovasz，实际是纯交叉熵**。

这推翻了两条流传的结论：
1. 「Sc++ 上加洛瓦兹掉 3.00」——实际是 **seed 25559278** vs **seed 42** 的纯 CE 对比
2. 「Sc++ 的洛瓦兹单点 0.2986 比 s42 的 0.2619 高 3.67」——这是 seed 方差，不是损失效果

**副作用（更重要）**：这两条说明 **Sc++ 1-shot 的 seed 方差可能高达 3–4 分**，
远大于 5-shot 的 1.42（`base_k5` 三 seed 极差）。

### 5.2 `ai_thresh` 两个数据集不同

| 数据集 | `ai_thresh` | 数量 |
|---|---|---|
| ScanNet++ | **0.95** | 101 个（另 1 个是 2.0 的异常值） |
| ScanNet200 | **0.9** | 113 个 |

`NAMING.md` 记的 0.95 是 Sc++ 的值。跨数据集比较时这不是同一个 AI 补点强度。

---

## §6 变量盘点（12 个维度按处置优先级三层归类）

### 层 1：官方固定或无影响 → 直接忽略（5 个）

| 变量 | 取值 | 说明 |
|---|---|---|
| `ps_thresh` | 0.6 | 两库一致 |
| `epoch` | 20 | 两库一致 |
| `batch_size` | 12（全局） | 两库一致 |
| 评估方式 | crop=None + 2 cm 体素 | 两库一致 |
| `num_worker` | 2–24 共 9 种 | **只影响读取速度，不影响结果**；以后统一 ≥16 提速 |

### 层 2：按数据集区分的官方预设 → 只需同格内一致（3 个）

| 变量 | ScanNet++ | ScanNet200 | 处置 |
|---|---|---|---|
| `ai_thresh` | **0.95** | **0.90** | 官方各自的默认值，不跨库强行统一 |
| 学习率（head） | **0.007** | **0.001** | 同上（backbone 均为 ×0.1） |
| 注册折数 | regis1 单折 | regis1 单折 | 内部对比统一单折；最终投稿主表再跑 regis1–regis5 |

**判据**：只要同一个「数据集 × shot」单元格里的 base / sym / M5 用同一套预设，就不构成污染。

### 层 3：必须严格锁死的核心控制变量（4 个）

| 变量 | 规定 | 当前违规 |
|---|---|---|
| **伪标签模式** | 同格内不得混比 | Sc200 1-shot：base 在线、sym/M5 离线 |
| **训练点数** | Sc++ 全 102400；Sc200 1-shot 65536、5-shot 102400 | Sc200 混用（65536×103 / 102400×14） |
| **随机种子** | 只认 s42 / s7 / s13 三 seed 配对 | 有 7 个随机 seed 的实验 |
| **损失函数** | 主表一律纯交叉熵 | Sc++ 4 个假 Lovász 已作废；Sc200 26 个真 Lovász 移出主表 |

**影响量级排序**：seed（**3–4 分**）> 伪标签模式（1.4–2.9）> 折数（1–2）> 点数（0.35）> num_worker（0）。

**seed 排第一**是因为 Sc++ 1-shot 实测出现过 3.7 分的 seed 差（见 §5.1）。这决定了「三 seed 配对」不是可选项。

---

## §6b 必须明确废弃 / 移出主表的 5 类实验

| # | 类别 | 具体范围 | 处理 | 连带作废的旧结论 |
|---|---|---|---|---|
| 1 | **ScanNet++ 假 Lovász** | `scpp_base_lovasz` 0.2804<br>`scpp_base_lovasz_k5` 0.3150<br>`scpp_sym_lovasz` 0.2986<br>`scpp_sym_lovasz_k5` 0.3103 | **作废** | 「Sc++ 加洛瓦兹掉 3.00」、「早期 −0.47」 |
| 2 | **跨折 / VLM Bug** | `sc200_k1_offline` 0.4038（regis2）<br>`support_ontr_s42_BUG_vlm`<br>`support_xl_s42_BUG_vlm` | **作废** | 无（从未进主表） |
| 3 | **训练中断的残缺目录** | `scpp_k5_m5_s7`（断 Ep3）<br>`scpp_k5_m5_s13`（断 Ep4） | **重跑**（2026-11-27 已启动） | 无 |
| 4 | **跨口径混比结论** | Sc200 1-shot sym「**+3.61**」= 在线 0.3861 vs 离线 0.4222 | **改写**为离线内部对比（实际约 +1.1 ~ +1.8，等新实验定值） | 该涨幅表述 |
| 5 | **ScanNet200 真 Lovász** | 26 个 `use_lovasz=True` 的实验（含 0.4414 / 0.4553） | **不作废，移出主表** | 不得与纯 CE 的 GFS-VL / HOP3D / UGA 直接比高下 |

**注**：ScanNet200 1-shot 上近期跑的 M5 伪标签变体（`entq10` / `gce07` / `elr` / `cleancon` / `m1vote` / `d2r10` / `fullent*`）
经日志核实**全部是在线**，它们内部对标在线基线 0.3861 是干净合法的，**不需要废弃**。

---

## §7 重跑方案的两个版本（不要再混着说）

| 方案 | 规则 | 局数 | 成本 |
|---|---|---|---|
| **A. 同格内一致（采用）** | 每格内部锁死伪标签模式，格与格之间不横比 | **8 局** | 8 卡约 1 小时 |
| **B. 全表纯在线** | 四格全部在线，彻底消除 Sc200 5-shot 沿用 1-shot 缓存的瑕疵 | **17 局** | 8 卡约 10–13 小时 |

**方案 A 的 8 局**（2026-11-27 12:05 已全部启动）：

| # | 实验 | 模式 |
|---|---|---|
| 1–2 | `scpp_k5_m5` s7 / s13 | 在线 |
| 3–4 | `sc200_k1_m5` s7 / s13 | 离线 |
| 5 | `sc200_k1_base` s42（补离线基线第三点） | 离线 |
| 6–8 | `sc200_k5_m5` s42 / s7 / s13 | 离线 |

**方案 A 下 Sc200 1-shot 的对比口径**：base（离线 `BASEu_s7` 0.4046 / `base_s13` 0.4116 / 新 `base_s42`）
vs sym（离线 0.4222 三 seed 均值）vs M5（离线三 seed）。**在线基线 0.3861 不得用于这组对比。**

**方案 B 如果以后要做**：Sc200 1-shot sym 3 + M5 3（在线）、Sc200 5-shot base/sym/M5 各 3（在线）共 15 局，
加 Sc++ 2 局 = 17 局。留到投稿前跑五折主表时一并处理。

---

## §7 当前唯一干净可下的结论

> **ScanNet++ 1-shot，102400 点，在线伪标签，s42/s7/s13 三 seed：**
> M5 比 base 高 **+0.29**（0.2909 vs 0.2880）；sym 比 base 低 **−0.71**（0.2809 vs 0.2880）。

其余跨单元格对比，至少混了 1 个未受控变量。

---

## §8 复现命令

```bash
# 10 秒定性任意一个 run
grep -c 'sparseunet32_636.pth to GPU' exp/<dataset>/<run>/train.log
# 1 = 在线，0 = 离线

# 全量扫描
cd exp && for ds in scannetpp scannet200; do
  for d in $ds/*/; do
    [ -f "$d/train.log" ] || continue
    grep -q 'sparseunet32_636.pth to GPU' "$d/train.log" \
      && echo "ONLINE  $d" || echo "OFFLINE $d"
  done
done
```
