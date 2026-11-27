# 废弃与移出主表清单（2026-11-27 统一梳理）

> **单一信息源。** 任何报告、PPT、脚本要引用历史数字前，先查这张表。
> 配套文档：`PL_MODE_AUDIT_2026-11-27.md`（伪标签模式审计）、`NAMING.md`（命名辞典）。
> 本文档取代此前散落在各报告里的「作废」「无效」「不可用」标注。

---

## §0 一句话纪律

> **同一张表里：离线只和离线比，在线只和在线比。主表一律纯交叉熵、三 seed、regis1 单折。**

---

## §1 作废（结果不可用，不得引用）

| 类别 | 具体实验 | 数值 | 作废原因 |
|---|---|---|---|
| **Sc++ 假 Lovász** | `scpp_base_lovasz` | 0.2804 | config **无** `use_lovasz=True`，实际是纯 CE + 随机 seed 20363522 |
| | `scpp_base_lovasz_k5` | 0.3150 | 同上，随机 seed 20269292 |
| | `scpp_sym_lovasz` | 0.2986 | 同上，随机 seed 31853493 |
| | `scpp_sym_lovasz_k5` | 0.3103 | 同上，随机 seed 25559278 |
| **跨折违规** | `sc200_k1_offline` | 0.4038 | config 写 5 折，实际跑完 regis1 + regis2，最终值是 **regis2** 的 |
| **VLM Bug** | `support_ontr_s42_BUG_vlm` | — | 已知 VLM 加载 bug |
| | `support_xl_s42_BUG_vlm` | — | 同上 |
| **实现 bug** | `support_base_s42` / `support_rep3_s42` / `support_sym_s42` / `support_ontr_linear_s42` / `support_xl_linear_s42` | 0.10–0.15 | support 集微调实现缺陷，PL 只在 45 个 support 场景读取 + `gfs==-1` 产生 100% 假阳性 |
| **崩溃** | `sc200_k1_XLu_s7` | — | 训练崩溃，无结果 |
| **ELR 发散** | `m5_elr_s7`（两库） | loss 变负 | `log(1-dot)` 在伪标签上冲向 −9，训练目标被改写（见 `PL_FILTER_WAVE2 §6`） |

### 1.1 因「假 Lovász」连带作废的结论

| 原结论 | 状态 |
|---|---|
| 「ScanNet++ 上加洛瓦兹反而比不加低 3.00」 | **作废**。0.3103 vs 0.3403 是 seed 25559278 vs seed 42 的纯 CE 对比 |
| 「Sc++ 早期 Lovász −0.47」 | **作废**，同因 |
| 「Sc++ 从未跑过洛瓦兹 + 102400 + 跨层」 | **作废**。Sc++ 全部 102 个实验都是 102400，且从未开过 Lovász |

---

## §2 移出主表（数值有效，但不得与纯 CE 并列比较）

| 类别 | 范围 | 处置 |
|---|---|---|
| **ScanNet200 真 Lovász** | 26 个 `use_lovasz=True` 的实验，含 `sym_lovasz_s42` 0.4414 / `sym_lovasz_k5_102400_s42` **0.4553** | 留作附录/历史对照。**不得**与纯 CE 的 GFS-VL / HOP3D / UGA 直接比高下 |
| **Sc200 假 Lovász 的孪生** | `scpp_sym_lovasz*` 之外，Sc++ 无其他假 Lovász | — |
| **五折实验** | `sym_lovasz_5split_s42` | 名字已标注，只作跨折参考 |
| **随机 seed 实验** | `scpp_sym_lovasz`(31853493) / `scpp_sym_lovasz_k5`(25559278) / `scpp_base_lovasz`(20363522) / `scpp_base_lovasz_k5`(20269292) / `sonata_s_s2_encfrozen`(40045561) / `sonata_s_s2_support`(54639613) / Sc200 的 3 个随机 seed | **严禁参与任何三 seed 配对统计** |

---

## §3 残缺待补（正在重跑）

| 实验 | 中断位置 | 状态 |
|---|---|---|
| `scpp_k5_m5_s7` | Ep3/20，0 条 Eval | 2026-11-27 12:05 重跑中 |
| `scpp_k5_m5_s13` | Ep4/20，0 条 Eval | 同上 |

**注意**：这两个目录里有 554 MB 的 `regis1_model_last.pth`，但那是 Ep3/Ep4 的快照，**不能用来补评估**。

---

## §4 口径改写（结论保留，数字要换）

| 原表述 | 错在哪 | 正确写法 |
|---|---|---|
| 「ScanNet200 1-shot sym **+3.61**」 | 用**在线**基线 0.3861 去比**离线**方法 0.4222 | 离线内部：0.4222 vs 离线基线 ≈ 0.405 → **约 +1.7**（等 `sc200_k1_base_s42` 出分后定值） |
| 「ScanNet200 1-shot M5 **+3.33**」 | 同上（0.4194 vs 0.3861） | 离线内部约 **+1.4** |

**在线基线 0.3861 的合法用途**：只用于和**其他在线实验**比，例如
`sc200_k1_m5_entq10_s7`（0.3425）/ `gce07_s7`（0.3780）/ `psai_d2r10_s7`（0.3727）/ `m1vote_s7`（0.2941）
/ `cleancon_s7` / `fullent10_s7` / `fullent30_pp_s7` 这一批。

---

## §5 保留（不必废弃）

| 类别 | 说明 |
|---|---|
| **Sc200 1-shot 的 M5 伪标签变体** | `entq10` / `gce07` / `m1vote` / `psai_d2r10` / `cleancon*` / `fullent10` / `fullent30_pp` **全部是在线**（已用 VLM 加载日志核实），对标在线基线 0.3861 干净合法 |
| **Sc200 1-shot 的离线 sym / M5 基座** | `sym_xl_s42/s7/s13`（0.4257/0.4214/0.4194）、`sc200_k1_m5_s42`（0.4194）**本身无问题**，只要和离线基线比 |
| **ScanNet++ 全部 97 个实验** | 全在线（数据集不支持离线），无需任何重跑 |
| **`sc200_k1_BASEu_s7` / `sc200_k1_XLu_s42`** | config 写了 5 折，但都在 regis1 结束就中断，最终 Eval 是 regis1 的，**结果干净** |

---

## §6 复核方法

```bash
# 1. 判在线/离线
grep -c 'sparseunet32_636.pth to GPU' exp/<ds>/<run>/train.log   # 1=在线 0=离线

# 2. 判 Lovász 真假
grep -c 'use_lovasz = True' exp/<ds>/<run>/config.py             # 0=假名字

# 3. 判折数违规（config 写 5 折 不等于结果用了 5 折，要看最终 Eval 落在哪个 regis）
grep -oE 'regis[0-9]_model_last' exp/<ds>/<run>/train.log | sort -u
grep -n 'Eval result' exp/<ds>/<run>/train.log
```

---

## §7 变更记录

| 日期 | 变更 |
|---|---|
| 2026-11-27 | 建立本文档；作废 4 个假 Lovász + 1 个跨折 + 7 个 bug/crash；改写 2 处跨口径涨幅；把 26 个真 Lovász 移出主表 |
