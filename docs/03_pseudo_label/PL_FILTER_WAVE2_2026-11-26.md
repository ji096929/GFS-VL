# 伪标签筛选第二轮：六种方案 × 两库两 shot（2026-11-26）

> 续 `XL2_WAVE3_AND_GEQ_2026-11-25.md`。上一轮结束时，ScanNet++ 5-shot 最强单点是
> `m5_geq02_s42` 的 0.3600。
> 本轮把「伪标签怎么筛」扩成六种方案，在两库 × 两 shot 上做横向对照。
> **全部数字来自 `exp/<ds>/<run>/train.log` 的 `Eval result` 行。**

---

## §1 代码改动

### 1.1 `pointcept/engines/train.py`（+563 行，四个新 pl_mode + 两个新 loss 项）

| 新增 | 类型 | 作用 |
|---|---|---|
| `dual_eye_and` | pl_mode | 两视角各取 top-q 后逻辑与 |
| `dual_eye_h1` | pl_mode | 训练前校准 Otsu(全局) + 每类 top-q，再与 `m_nv>0` |
| `dual_eye_m1_vote` | pl_mode | M1 全空间 argmax + dec2 边际投票 |
| `psai_dec2_rank` | pl_mode | PS+AI 全保留，dec2 在 N_novel 内部排名 ≤ k 才留 |
| `pl_gce_q` | loss 项 | Generalized Cross Entropy `(1-p_y^q)/q` |
| `elr_lam` / `elr_ema_decay` / `elr_warmup_epochs` | loss 项 | ELR 正则 `log(1 - <p,t>)` |

### 1.2 `pointcept/engines/hooks/evaluator.py`

`mid_eval_interval`：Ep5/10/15 各评一次，输出 `[MidEval ep=N] mIoU=… Base=… Novel=… HM=…`。

**实测已生效**：`psai_d2r10` 两局各 3 条，共 6 条。

### 1.3 新增 17 个 config + 1 个诊断脚本

`scripts/diag_dec2_eye_auroc.py`（15.7 KB）。

---

## §2 六种筛选方案的定义

| 方案 | 判据 | 关键参数 | 依赖 support 数 |
|---|---|---|---|
| **PS+AI（基线）** | 原型余弦 PS≥0.6 → AI≥0.95 补点 | — | 是 |
| **entq10** | 跳过 PS/AI，VLM 熵按类 top-10% | `rho=0.1` | 否 |
| **geq02** | 跳过 PS/AI，VLM 熵全局 top-2% | `rho=0.02` + EMA | 否 |
| **dualeye85/90** | `s1`(VLM 负熵) 与 `s2`(dec2 原型余弦) 各 top-q 后 AND | `q=0.15 / 0.10` | **是** |
| **dualeye90_mnv** | 同上，但 `s2` 换成 dec2 新类边际 `m_nv` | `q=0.10` | **是** |
| **h1** | 训练前校准 Otsu(全局) + 每类 top-q + `m_nv>0` | `q=0.10`, `cal_batches=30` | 是 |
| **m1vote** | M1 全空间 argmax + `m_nv>0` | 零阈值 | 是 |
| **psai_dec2_rank** | **PS+AI 全保留** → dec2 内部排名 ≤ `ceil(0.1·N_novel)` | `k=2(Sc++)/5(Sc200)` | 是 |
| **gce07** | PS/AI + GCE loss（不改筛选） | `q=0.7` | — |
| **elr** | PS/AI + ELR 正则（不改筛选） | `lam=3.0` | — |

**注**：`gce07` 和 `elr` 不改伪标签，只改 loss——所以严格说不是「筛选方案」，是同轮并跑的另外两条腿。

---

## §3 ScanNet++ 5-shot 全表

**基线（同 seed）：** s42 0.3060 / s7 0.3189 / s13 0.3047，3-seed 均值 **0.3099**

| run | Base | Novel | **HM** | Δ vs 同 seed base |
|---|---|---|---|---|
| `base_k5_s42` | 0.6036 | 0.2050 | 0.3060 | — |
| `base_k5_s7` | 0.6193 | 0.2148 | 0.3189 | — |
| `base_k5_s13` | 0.6206 | 0.2020 | 0.3047 | — |
| **`m5_psai_d2r10_s7`** | 0.6258 | **0.2574** | **0.3647** | **+4.58** |
| `m5_dualeye90_mnv_s7` | 0.6282 | 0.2530 | 0.3607 | +4.18 |
| `m5_geq02_s42` | 0.6186 | 0.2539 | 0.3600 | +5.40 |
| `m5_m1vote_s7` | 0.5949 | 0.2534 | 0.3554 | +3.65 |
| `m5_dualeye90_mnv_s13` | 0.6061 | 0.2513 | 0.3553 | +5.06 |
| `m5_h1_s7` | 0.5970 | 0.2504 | 0.3528 | +3.39 |
| `m5_dualeye85_s42` | 0.6076 | 0.2490 | 0.3532 | +4.72 |
| `m5_gce07_s7` | 0.6118 | 0.2288 | 0.3331 | +1.42 |
| `m5_geq02_s7` | 0.6222 | 0.2154 | 0.3200 | +0.11 |
| `m5_dualeye90_mnv_s42` | 0.5353 | 0.2189 | 0.3107 | +0.47 |
| `m5_elr_s7` | — | — | **崩（见 §6）** | — |
| `m5_entq10_s7` | 0.6015 | 0.2200 | **0.3222** | **+0.33** |

### 3.1 单 seed 方差比任何方法差异都大

| 配方 | s42 | s7 | s13 | 极差 |
|---|---|---|---|---|
| `base_k5` | 0.3060 | 0.3189 | 0.3047 | **1.42** |
| `m5_geq02` | 0.3600 | 0.3200 | — | **4.00** |
| `m5_dualeye90_mnv` | 0.3107 | 0.3607 | 0.3553 | **5.00** |

**在这片方差里，0.3647 / 0.3607 / 0.3528 / 0.3554 之间的差别都不具决定性。**
**唯一有三个 seed 的是 `dualeye90_mnv`：均值 0.3422，3/3 全胜，但配对差标准差 2.02（t≈2.75, df=2, p≈0.11）。**

---

## §4 ScanNet++ 1-shot 全表

**基线（同 seed）：** s42 0.2843 / s7 0.2673 / s13 0.3125，3-seed 均值 **0.2880**

| run | Base | Novel | **HM** | Δ vs base_k1_s7 |
|---|---|---|---|---|
| `m5_m1vote_s7` | 0.6265 | 0.1868 | **0.2877** | **+2.04** |
| `m5_h1_s7` | 0.6272 | 0.1697 | 0.2672 | −0.01 |
| `m5_dualeye90_mnv_s7` | 0.6008 | **0.1358** | **0.2216** | **−4.57** |

**1-shot 上三个新方案只有 `m1vote` 站住（+2.04），另外两个持平或崩盘。**

**崩盘的直接原因：`dualeye*` 和 `h1` 都依赖 dec2 原型，而 1-shot 时原型只来自 1 个 support 场景。**
`m1vote` 的 `m_nv>0` 是相对判据（只要排序对），对原型噪声更宽容——所以它是三者中唯一没崩的。

---

## §5 ScanNet200 全表

### 5.1 1-shot

**基线参考：** 3-seed 均值 0.4246（`Week1 §X5`）；同 seed 单点未单独存 log，故 Δ 仅作参考。

| run | Base | Novel | **HM** |
|---|---|---|---|
| **`m5_gce07_s7`** | 0.6799 | **0.2618** | **0.3780** |
| `m5_psai_d2r10_s7` | 0.6680 | 0.2584 | 0.3727 |
| `m5_entq10_s7` | 0.6728 | 0.2297 | 0.3425 |
| `m5_m1vote_s7` | 0.6684 | 0.1885 | 0.2941 |
| `m5_elr_s7` | — | — | **崩（见 §6）** |

**基线口径必须分开看（这一条容易把结论看反）：**

| 基线 | 1-shot HM |
|---|---|
| 纯交叉熵 `ONLINE_s7` | **0.3904** |
| 纯交叉熵 `ONLINE_s42` | 0.3855 |
| 加洛瓦兹损失 `base_lovasz` 三 seed 均值 | 0.4246 |

**所以「掉多少」完全取决于拿谁当基线：**

| run | vs 纯交叉熵 0.3904 | vs 洛瓦兹 0.4246 |
|---|---|---|
| `m5_psai_d2r10_s7` | −1.77 | −5.19 |
| `m5_gce07_s7` | −1.24 | −4.66 |
| `m5_entq10_s7` | −4.79 | −8.21 |
| `m5_m1vote_s7` | −9.63 | −13.05 |

**注意 `m1vote` 在 ScanNet200 1-shot 上崩到 0.2941（Novel 0.1885）——和它在 ScanNet++ 1-shot 的 +2.04 正好相反。**

> **⚠️ 口径补充（2026-11-27）**：上表那批 `m5_*` 变体经 VLM 加载日志核实**全部是在线**模式，
> 所以「vs 纯交叉熵 0.3904」这一列是**在线对在线**，合法。
>
> 但**纯 M5 的 `sc200_k1_m5_s42`（0.4194）是离线**，它是**离线的 1-shot 缓存**口径，
> 不能套用这张表去比。离线侧的正确基线是 `BASEu_s7` 0.4046 / `base_s13` 0.4116 / `sc200_k1_base_s42`（2026-11-27 补跑）。
> 详见 `docs/00_ledger/DEPRECATED_2026-11-27.md` §4。

### 5.2 5-shot

| run | Base | Novel | HM |
|---|---|---|---|
| `base_k5_102400_s42` | 0.6744 | 0.3067 | **0.4216** |
| `m5_m1vote_s7` | 0.6716 | 0.2948 | 0.4098 |

**`m1vote` 在 ScanNet200 5-shot 上比 base 低 1.18。**

---

## §6 ELR 事故：loss 变负发散

**现象**（`scpp_k5_m5_elr_s7.log` / `sc200_k1_m5_elr_s7.log`，19:52 后停止写入）：

```
scpp Ep13 iter37  loss: -1.8234
scpp Ep13 iter38  loss: -4.4727
sc200 Ep15 iter11 loss: -2.2367
sc200 Ep15 iter12 loss: -3.7505
```

**两局都在 Ep13–15 之间退出（日志末尾只有 `resource_tracker: leaked semaphore` 警告，无 Traceback）。**

**根因（`train.py:808-812`）：**

```python
_dot = (_p_prob * _t_prob).sum(dim=-1).clamp(min=1e-4, max=1.0 - 1e-4)
_elr_loss = torch.log(1.0 - _dot).sum() / (_valid.sum().float() + 1e-6)
loss = loss + _elr_lam * _elr_loss          # elr_lam = 3.0
```

**`log(1-dot)` 在 `dot ∈ (0,1)` 上恒为负**，且 `clamp` 上限 `1-1e-4` 让它最低可到 `log(1e-4) ≈ -9.21`。

**ELR 原论文里 `dot` 是 soft 的（真实标签 one-hot 经 EMA 平滑），所以 `log(1-dot)` 在 `-0.7` 量级。**
**但我们的 target 是伪标签——模型几轮就把它记死，`p ≈ t ≈ one-hot` → `dot → 1` → 正则项冲向 `-9`。**

**`CE（≈0.2） + 3.0 × (-9) ≈ -27`**，训练目标被彻底改写，梯度方向失去意义。

**结论：ELR 不适用于「伪标签 + 强过拟合」的设定，除非把 target 真正做软（label smoothing / mixup）或者对 `dot` 加上界约束。这一条不值得重修，归档留证。**

---

## §7 三套离线诊断缓存

| 目录 | 时间 | 内容 |
|---|---|---|
| `data/dec2_eye_audit/` | 11-25 17:50 | 4 shard + protos（含 support 泄漏版） |
| `data/dec2_eye_audit_noleak/` | 11-25 18:20 | **403 纯未见房间**，18 novel 类，10 列分数 |
| `data/dec2_1shot_audit/` | 11-26 10:22 | **1-shot 双库零泄露**（Sc++ 纯度 83.3% / Sc200 94.6%） |

**缓存字段：** `vals(N,10)` / `cls` / `pol` / `n_pos_true` / `n_neg_true` / `n_chunks`。

**已验证：`vals[:,0]` 是 18 类 softmax 负熵，值域只有 0.0044（`-2.8902 ~ -2.8858`），但 AUC 0.811。**
**任何绝对阈值在这个窄带上都不可靠，只有排序/分位可用——这是 `h1` 的 Otsu 与 `geq02` 分位表现差异的直接原因。**

---

## §8 与本轮之外的两条事实

### 8.1 UGA 的 baseline 与方法是不同训练预算

从开源仓库（`anonymous.4open.science/r/UGA`）实测：

```python
# semseg-pt-v3m1-0-gfsregistrain_k5.py  （它的 baseline）
batch_size = 12;  epoch = 20
# phase2_uga_5shot.py                   （它自己）
batch_size = 64;  epoch = 40
```

**这不是单变量对比。** 它的 UB-Filter 是 `PS → 熵过滤 → AI`（叠加），`ratio=0.1, level=class_wise`；
我们的 `entq10` 完全等价于它的 UB-Filter，`geq02` 是全局变体，`psai_dec2_rank` 是唯一「PS+AI 全保留」的版本。

### 8.2 跨层 × 3D-OCSB 的失败记录已补档

见 `OCSB_BRIDGE_DECLINE_2026-11-26.md`：AUC 0.94 但 thr=0.9 时 Precision 仅 3.1–3.3%。

---

## §9 待补与缺口

| 项 | 状态 |
|---|---|
| `m5_entq10` 两局（Sc++ 5-shot / Sc200 1-shot） | **已出分**：Sc++ 0.3222 / Sc200 0.3425，两库都接近零增益或负增益 |
| `psai_d2r10` 的 s42 / s13 | 只有 s7 单点 |
| `gce07` 的 s42 / s13 | 只有 s7 单点 |
| `dualeye90_mnv` 的 1-shot 另两 seed | 只有 s7 |
| ScanNet200 base 1-shot 的单 seed 点 | 只有 3-seed 均值 0.4246（且那是洛瓦兹口径，纯交叉熵是 0.3904 / 0.3855 / 0.3823） |
| **BA-AI 的离线表**（500 Sc200 + 373 Sc++ 房间） | **服务器上无落盘，只存在于对话记录** |
| ~~**ScanNet++ 从未跑过「洛瓦兹 + 十万两千四百点 + 跨层」**~~ | **本条作废，见 §9.2（2026-11-27 更正）** |

---

### 9.1 ScanNet++ 从未跑过洛瓦兹——原名带 lovasz 的实验是假名字

**2026-11-27 复核结论：本节原表与结论全部作废。**

| 原记载 | 复核事实 | 证据 |
|---|---|---|
| `scpp_sym_lovasz_k5` 是「洛瓦兹 + 65536」 | 它是**纯交叉熵 + 102400** | config 无 `use_lovasz=True`；`point_max=102400` |
| 「加洛瓦兹掉 3.00」 | 0.3103（seed **25559278**）vs 0.3403（seed **42**），差的是 seed | `scpp_sym_lovasz_k5/config.py` 的 `seed = 25559278` |
| 「Sc++ 从未跑过 102400 + 跨层」 | Sc++ **全部 102 个实验都是 102400** | 全量 grep `point_max=102400` |

**`scpp_base_lovasz` / `scpp_base_lovasz_k5` / `scpp_sym_lovasz` / `scpp_sym_lovasz_k5` 四局全部没开 Lovász**，
名字里的 `lovasz` 是误标。Sc++ 的 102 个 config 里 `use_lovasz=True` 的数量是 **0**。

**连带作废的结论：**
1. 「ScanNet++ 上加洛瓦兹反而比不加低 3.00」——不成立，那 3.00 是 seed 方差
2. 「Sc++ 早期 Lovász −0.47」——同上

**唯一仍然成立的是 ScanNet200 那一侧**：`sym_lovasz_s42`（0.4414）与 `sym_xl_s42`（0.4257）是真开了 Lovász 的，
但两者 seed 相同、点数相同（均 65536），所以那组对比有效。

**顺带得到一个更重要的数字：Sc++ 1-shot 的 seed 方差可达 3.7 分**（0.2986 随机 seed vs 0.2619 s42），
远大于 5-shot `base_k5` 的 1.42。这条直接说明为什么必须锁死三 seed。

详细审计见 `docs/00_ledger/PL_MODE_AUDIT_2026-11-27.md`。

---

## §10 复现

```bash
cd /data01/jh/code/GFS-VL
# 例：psai_dec2_rank（当前两库最强）
python tools/train.py \
  --config-file configs/scannetpp/semseg-pt-v3m1-0-gfsregistrain_k5_m5_psai_d2r10.py \
  --num-gpus 2 \
  --options save_path=exp/scannetpp/scpp_k5_m5_psai_d2r10_s7 \
            regis_train_list=["regis1"] seed=7 num_worker=20

# 中途评估已在 evaluator.py 落地，config 里写 mid_eval_interval = 5
```
