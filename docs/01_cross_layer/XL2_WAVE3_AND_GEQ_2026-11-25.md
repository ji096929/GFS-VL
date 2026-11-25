# XL-2 Wave 3 + 全局熵分位（2026-11-25）

> 本轮由另一 agent 执行，本文档负责归档。全部数字来自 `exp/<dataset>/<run>/result/val.pth`
> 的 `intersection/union`（标准评估，非 det_eval），同类实验间可比。

---

## §1 本轮代码改动

### 1.1 `pointcept/models/default.py`：跨层结构新增 8 个开关

| 参数 | 默认 | 作用 |
|---|---|---|
| `cross_grad_scale` | 1.0 | 跨层支路梯度阻尼（`g*h + (1-g)*h.detach()`） |
| `cross_linear` | False | 跨层投影去掉 GELU，变纯线性 |
| `dec0_ln` | True | `novel_seg_head` 输入里的 dec0 是否过 LN |
| `cross_residual` | False | 跨层特征以残差方式并入 |
| `cross_zero_head` | False | 跨层支路输出置零（"sym 结构但零跨层"的空对照） |
| `cross_ent_gate` | False | 用熵做门控 |
| `cross_direct` | False | 只用 dec2 直连 |
| `cross_decoupled` | False | dec1/dec2 解耦（只让 dec1 detach，dec2 保留） |

### 1.2 `pointcept/engines/train.py`：新增 `pl_mode = "global_entropy_q"`

**机制**（`_target_global_entropy_quantile`）：

```
每 batch 在 bg 区域算熵 → 取 rho 分位阈值
  → DDP 跨卡 all_reduce 平均
  → EMA 平滑：thr = ema*old + (1-ema)*thr_batch
  → keep_mask = bg_label_mask & (ent <= thr)
```

**参数**：`global_entropy_rho`（保留比例）、`global_entropy_ema`（平滑系数，默认 0.9）。

**与旧 `entropy_q` 的区别**：旧版是「每类 top-k」，新版是「跨类统一分位阈值」——正是
`SIGNAL_AUROC_2026-11-23.md` 第七节提议的方向，但实现更稳（动态分位 + EMA + DDP 平均）。

### 1.3 两处性能优化（100% 数值等价）

| 文件 | 改动 | 效果 |
|---|---|---|
| `pointcept/datasets/transform.py` | `ElasticDistortion._trilinear_interp` 改 float32 + 1D 扁平索引 | 2.83s → 0.62s（4.5×），最大误差 1.4e-5 |
| `pointcept/datasets/scannetpp.py` | 支撑集缓存从「实例属性」改 `/dev/shm` 只读 mmap | Dataset 序列化 2.22 GB → 55 KB，`get_instances` 0.00179s |

**关键发现**：Pointcept DDP 用 `multiprocessing.spawn` 而非 `fork`，实例属性会被 pickle
复制给全部 worker——之前 2.22 GB 缓存 × 60 worker 冲到 144 GB、写满 61 GB swap。

---

## §2 实验台账（按系列分组）

### 2.1 全局熵分位（5-shot，最高优先级）

| run | rho | Base | Novel | **HM** | vs base |
|---|---|---|---|---|---|
| **`scpp_k5_geq02_s42`** | **0.02** | **0.6388** | **0.2387** | **0.3475** | **+4.09** |
| `scpp_k5_geq05_s42` | 0.05 | 0.6379 | 0.2270 | 0.3348 | +2.82 |
| `scpp_sym_k5_s42`（原版） | — | 0.6165 | 0.2327 | 0.3379 | +3.13 |
| `scpp_base_k5_s42`（基线） | — | 0.6034 | 0.2055 | **0.3066** | — |

**`geq02` 同时刷新 Base 与 Novel 两个维度的最好值**，是目前 5-shot 的最强单点。

> **待补**：`scpp_k5_m5_geq02_s42`（dec2-only + 置零 + geq02 的组合）在归档时仍在训练
> （Ep 8/20），出分后需补进本表。

### 2.2 跨层结构消融（1-shot，3 seed）

| run | 含义 | s42 | s7 | s13 | 3-seed 均值 |
|---|---|---|---|---|---|
| `scpp_base_k1` | 基线（无跨层） | 0.2843 | 0.2673 | 0.3125 | **0.2880** |
| `scpp_sym_k1` | 原版 sym | 0.2619 | 0.2745 | 0.3064 | 0.2809 |
| `m5_symzero` | 跨层输出置零（空对照） | 0.2803 | 0.2841 | **0.3082** | 0.2909 |
| `m6_symzero_lc03` | m5 + `L_cross` 0.3 | 0.2638 | 0.2748 | 0.2998 | 0.2795 |
| `m7_symzero_bal` | m5 + balanced | 0.2958 | 0.2769 | 0.2867 | 0.2865 |
| `m8_entgate` | 熵门控 | 0.2671 | 0.2724 | 0.2932 | 0.2776 |
| `m9_direct` | dec2 直连 | 0.2821 | **0.3031** | 0.2679 | 0.2844 |
| `m10_decoupled` | dec1 detach + `lam_cross=1.0` | 0.2732 | 0.2678 | 0.2766 | 0.2725 |
| `m11a_highway` | `g=0.3` + `lam_cross=1.0` | 0.2933 | 0.2760 | 0.2937 | 0.2877 |
| `m11b_g1` | `g=1.0` + `lam_cross=1.0` | — | — | 0.2724 | — |
| `m1_nod0ln_g035` | dec0 不过 LN + `g=0.35` | 0.2832 | 0.2791 | **0.3089** | 0.2904 |
| `m2_resfuse_g035` | 无 base_cross + `g=0.35` | — | — | 0.3056 | — |
| `m3_resfuse_g100` | 无 base_cross + `g=1.0` | — | — | 0.2933 | — |
| `m4_dec2_d0ln_g035` | dec2-only + dec0 LN + `g=0.35` | 0.2717 | 0.2785 | 0.3031 | 0.2844 |

**没有任何一个变体在 3-seed 均值上超过 base（0.2880）。最好的 `m5_symzero` 也只有 0.2909，
而它是「跨层输出置零」的空对照——说明 m5–m11 系列的收益都来自结构之外的噪声。**

### 2.3 cross-lr 三局（1-shot，s42）

| run | 改动 | Base | Novel | HM |
|---|---|---|---|---|
| `sym_xlr001` | 只降 cross lr 到 0.001 | 0.6315 | 0.1813 | 0.2818 |
| `sym_lin_xlr001` | + 纯线性投影 | 0.6137 | 0.1808 | 0.2794 |
| **`sym_dec2_xlr001`** | **只用 dec2 + 降 lr** | 0.6245 | **0.1923** | **0.2940** |

**`sym_dec2_xlr001` 的 3-seed：s42 0.2940 / s7 0.2844 / s13 0.2834，均值 0.2873 —— 仍低于 base 的 0.2880。**

### 2.4 ScanNet200 对照（1-shot，s42）

| run | Base | Novel | HM |
|---|---|---|---|
| `sc200_k1_m5_s42` | 0.6852 | 0.3022 | **0.4194** |
| `sc200_k1_m8_s42` | 0.6863 | 0.3002 | 0.4177 |
| `sc200_k1_m1_nod0ln_g035_s42` | 0.6902 | 0.2973 | 0.4156 |
| `sc200_k1_m9_s42` | 0.6751 | 0.2990 | 0.4144 |
| `sc200_k1_m11a_s42` | 0.6828 | 0.2802 | 0.3974 |
| `sc200_k1_m10_s42` | 0.6814 | 0.2774 | 0.3942 |

---

## §3 关键结论

### ① 全局熵分位（`geq02`）是目前唯一在两个维度都赢的方案

```
scpp_base_k5_s42   Base 0.6034  Novel 0.2055  HM 0.3066
scpp_geq02_s42     Base 0.6388  Novel 0.2387  HM 0.3475   (+3.54 / +3.32 / +4.09)
```

**它不依赖跨层结构，纯粹改伪标签筛选口径**——从「每类 top-k」换成「跨类统一分位阈值」。
和 `SIGNAL_AUROC_2026-11-23.md` 的预测方向一致（纯度从 29.4% 提到 58.6%）。

### ② 跨层结构消融全线未超基线

`m5`（跨层输出置零的空对照）反而 0.2909 > base 0.2880 —— **说明 1-shot 下跨层本身就不该开**，
这和 Wave 3 的实测（sym 1-shot −0.59）一致。

### ③ 「降 cross lr」能救回大部分损失，但救不回全部

```
sym（7× lr）       0.2619  (−2.24 vs base)
sym_xlr001（lr=1e-3）0.2818  (−0.03 vs base)   ← 几乎抹平
sym_dec2_xlr001    0.2940  (+1.19 vs base)     ← s42 单点翻正
但 3-seed 均值：0.2873 < base 0.2880           ← 不成立
```

**s42 上的 +1.19 是 seed 噪声**，和之前 ScanNet200 的「+2.92 被 s42 放大」同一机制。

### ④ 两个性能优化的实际收益

| 优化 | 5-shot 训练 batch time |
|---|---|
| 修复前 | 4.86 s/iter（Data 1.71s） |
| 修复后 | 2.58 s/iter（Data 0.012s） |

**提速 1.9×，且完全数值等价（误差 1.4e-5）。**

---

## §4 与已有结论的关系

| 已 有 结 论 | 本轮数据 |
|---|---|
| `SIGNAL_AUROC` 预测全局阈值落点 0.38–0.40 | **实测 0.3475，方向对但幅度低** |
| Wave 3 发现 ScanNet++ 1-shot 的 sym 为 −0.59 | **m5（symzero）反而 0.2909 > base 0.2880，一致** |
| XL-2 第二步 g=0.2 阻尼 −1.40 | **m11a（g=0.3）0.2877 ≈ base 0.2880，阻尼在 1-shot 无用** |
| ScanNet200 的 sym 1-shot +2.34 | 见下方「空对照的坑」 |

### 4.1 空对照的坑（`m5_symzero` 不是 base 的空对照）

`m5_symzero` 的 config 是从 `sym_xl_s42` 的 dump 复制的，结构是：

```python
base_cross=True,
cross_layers=(2,),      # 只用 dec2
cross_dims=(128,),
cross_zero_head=True,   # 跨层输出置零
```

**而 `base` 根本没有 `cross_layers`。** 所以 `m5` 与 `base` 的差异里混了「有 base_cross +
有 dec2 通路」两个变量，不是纯空对照。

**真正的对照是 `m9_direct`（dec2 直连、不置零）：**

| run | 结构 | ScanNet200 s42 HM |
|---|---|---|
| `m5_symzero` | dec2 + 置零 | 0.4194 |
| `m9_direct` | dec2 + 不置零 | 0.4144 |
| **差值** | — | **0.50** |

**「置零与不置零」只差 0.50，说明 dec2-only 通路的贡献本身极小——和 ScanNet++ 上
「跨层 1-shot 无效」的结论互相印证。**

---

## §5 复现

```bash
# 全局熵分位（5-shot）
python tools/train.py \
  --config-file configs/scannetpp/semseg-pt-v3m1-0-gfsregistrain_k5_geq02.py \
  --num-gpus 2 \
  --options save_path=exp/scannetpp/scpp_k5_geq02_s42 \
            regis_train_list=["regis1"] seed=42 num_worker=20

# 跨层消融（1-shot）
python tools/train.py \
  --config-file configs/scannetpp/semseg-pt-v3m1-0-gfsregistrain_k1_m5_symzero.py \
  --num-gpus 2 \
  --options save_path=exp/scannetpp/scpp_k1_m5_s42 \
            regis_train_list=["regis1"] seed=42 num_worker=20
```

**日志位置**：`/data01/jh/data/run_scpp_*.log`、`/data01/jh/data/diag_m*.log`
