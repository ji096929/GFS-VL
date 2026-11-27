# 结构与配方全表（2026-11-27 自动生成）

> 由每个 run 的 `config.py` 逐字段提取。**结构代号按参数指纹判定，不看目录名。**
> **注意**：`config.py` 曾被覆盖过（34 个 run 的 config 比 train.log 新，已从日志恢复）。
> **要绝对可靠的结构判定，用 `tools/gen_cells.py`——它直接读权重 zip 的 key，0.02 秒/文件。**
> 重新生成：`python tools/gen_structure_map.py > docs/00_ledger/STRUCTURE_MAP.md`

## 结构代号定义

| 代号 | 判据 | 含义 |
|---|---|---|
| `base` | 无 `cross_layers` | 只吃 dec0 |
| `xl` | 有跨层 + `base_cross=False` | 只有新类头接跨层 |
| `sym` | `cross_layers=(1,2)` + `base_cross=True` | 新类拼接 192 + 基类残差 128（新类 dec0 过 LN） |
| `M5` | `(2,)` + `base_cross=True` + `zero_head=True` | 两路零初始化残差，都用原始 dec0 |
| `m9` | 同 M5 但 `cross_direct=True` | dec2 直连，不经过投影层 |
| `dec2only` | `(2,)` + `base_cross=True` + 无 zero_head | dec2 拼进新类头 |
| `rep3` | `rep_dec0=True` | dec0 复制两份冒充跨层（对照） |
| `gated` | `depth_gate=True` | 逐点门控 |
| `mlp_dec0` | `mlp_hidden>0` | dec0-only 两层 MLP |
| `lnonly` | `ln_only=True` | 只加 LayerNorm |

## 命名与结构不符的 run

| 目录 | 真实结构 | 问题 |
|---|---|---|
| `scpp_sym_k1_dec2_xlr001_s13` | **dec2only** | 名字带 xl，结构是 dec2only |
| `scpp_sym_k1_dec2_xlr001_s42` | **dec2only** | 名字带 xl，结构是 dec2only |
| `scpp_sym_k1_dec2_xlr001_s7` | **dec2only** | 名字带 xl，结构是 dec2only |
| `support_xl_linear_s42` | **dec2only** | 名字带 xl，结构是 dec2only |

## base（71 个）

| 目录 | 数据集 | shot | PL模式 | PL方案 | Lovász | seed | HM |
|---|---|---|---|---|---|---|---|
| `1A0_det` | Sc++ | 5 | - | ps_ai | 否 | 45863570 | - |
| `_cm_1ca` | Sc++ | 5 | - | 'entropy_q' | 否 | 24780262 | - |
| `_cm_1ce` | Sc++ | 5 | - | 'entropy_q' | 否 | 24578095 | - |
| `_cm_l1h` | Sc++ | 5 | - | 'student_consensus' | 否 | 31682593 | - |
| `_l1_smoke` | Sc++ | 5 | 在线 | 'student_consensus' | 否 | 42 | 0.0703 |
| `_l1_smoke3` | Sc++ | 5 | 在线 | 'student_consensus' | 否 | 42 | 0.1369 |
| `scpp_base_k1_s13` | Sc++ | 1 | 在线 | ps_ai | 否 | 13 | 0.3125 |
| `scpp_base_k1_s42` | Sc++ | 1 | 在线 | ps_ai | 否 | 42 | 0.2843 |
| `scpp_base_k1_s7` | Sc++ | 1 | 在线 | ps_ai | 否 | 7 | 0.2673 |
| `scpp_base_k5_s13` | Sc++ | 5 | 在线 | ps_ai | 否 | 13 | 0.3047 |
| `scpp_base_k5_s42` | Sc++ | 5 | 在线 | ps_ai | 否 | 42 | 0.3060 |
| `scpp_base_k5_s7` | Sc++ | 5 | 在线 | ps_ai | 否 | 7 | 0.3189 |
| `scpp_base_lovasz` | Sc++ | 1 | 在线 | ps_ai | 否 | 20363522 | 0.2804 |
| `scpp_base_lovasz_k5` | Sc++ | 5 | 在线 | ps_ai | 否 | 20269292 | 0.3150 |
| `scpp_base_lovasz_k5_entq10` | Sc++ | 5 | 在线 | 'entropy_q' | 否 | 42 | 0.3505 |
| `scpp_base_lovasz_k5_entq10_fix` | Sc++ | 5 | 在线 | 'entropy_q' | 否 | 42 | 0.3420 |
| `scpp_base_lovasz_k5_entq50` | Sc++ | 5 | 在线 | 'entropy_q' | 否 | 42 | 0.2642 |
| `scpp_base_lovasz_k5_lr1e3` | Sc++ | 5 | 在线 | ps_ai | 否 | 42 | 0.2618 |
| `scpp_base_lovasz_k5_noai` | Sc++ | 5 | 在线 | ps_ai | 否 | 42 | 0.2731 |
| `scpp_base_lovasz_k5_psaithenq` | Sc++ | 5 | 在线 | 'ps_ai_then_entropy' | 否 | 42 | 0.3452 |
| `scpp_entq10_veto` | Sc++ | 5 | 在线 | 'entropy_q' | 否 | 42 | 0.3414 |
| `scpp_entq_abs2` | Sc++ | 5 | 在线 | 'entropy_q' | 否 | 22598486 | - |
| `scpp_fix1` | Sc++ | 5 | 在线 | 'entropy_q' | 否 | 17308501 | 0.3323 |
| `scpp_fix2` | Sc++ | 5 | 在线 | 'entropy_q' | 否 | 51256953 | 0.3308 |
| `scpp_k5_geq02_s42` | Sc++ | 5 | 在线 | 'global_entropy_q' | 否 | 42 | 0.3475 |
| `scpp_k5_geq05_s42` | Sc++ | 5 | 在线 | 'global_entropy_q' | 否 | 42 | 0.3348 |
| `scpp_l1hybrid` | Sc++ | 5 | 在线 | 'student_consensus' | 否 | 42 | 0.3281 |
| `scpp_l1pure` | Sc++ | 5 | 在线 | 'student_consensus' | 否 | 42 | 0.3250 |
| `scpp_m1a` | Sc++ | 5 | 在线 | 'm1_argmax' | 否 | 42 | 0.3135 |
| `scpp_m1b` | Sc++ | 5 | 在线 | 'm1_cos' | 否 | 42 | 0.3356 |
| `scpp_oracle_filter_m0` | Sc++ | 5 | 在线 | 'oracle_m0' | 否 | 42 | 0.4120 |
| `scpp_oracle_filter_m1` | Sc++ | 5 | 在线 | 'oracle_m1' | 否 | 42 | 0.3613 |
| `scpp_oracle_fullsup20` | Sc++ | 5 | 在线 | ps_ai | 否 | 42 | 0.4905 |
| `base_k5_102400_s13` | Sc200 | 5 | 离线 | ps_ai | 否 | 13 | 0.4275 |
| `base_k5_102400_s42` | Sc200 | 5 | 离线 | ps_ai | 否 | 42 | 0.4216 |
| `base_k5_102400_s7` | Sc200 | 5 | 离线 | ps_ai | 否 | 7 | 0.4346 |
| `base_k5_s42` | Sc200 | 5 | 离线 | ps_ai | 否 | 42 | 0.4181 |
| `base_k5_true_s42` | Sc200 | 5 | 在线 | ps_ai | 否 | 42 | 0.4105 |
| `base_k5_truemulti_s42` | Sc200 | 5 | 在线 | ps_ai | 否 | 42 | 0.4108 |
| `base_la_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.3885 |
| `base_lovasz_k5_102400_s42` | Sc200 | 5 | 离线 | ps_ai | 是 | 42 | 0.4501 |
| `base_lovasz_s13` | Sc200 | 1 | 离线 | ps_ai | 是 | 13 | 0.4308 |
| `base_lovasz_s42` | Sc200 | 1 | 离线 | ps_ai | 是 | 42 | 0.4122 |
| `base_lovasz_s7` | Sc200 | 1 | 离线 | ps_ai | 是 | 7 | 0.4303 |
| `base_s13` | Sc200 | 1 | 离线 | ps_ai | 否 | 13 | 0.4116 |
| `cos_base_k5_s42` | Sc200 | 5 | 离线 | ps_ai | 否 | 42 | 0.3835 |
| `frozen_base_lovasz_s42` | Sc200 | 1 | 离线 | ps_ai | 是 | 42 | 0.1370 |
| `frozen_ontr_lovasz_s42` | Sc200 | 1 | 离线 | ps_ai | 是 | 42 | 0.2445 |
| `ontr_base_lovasz_s13` | Sc200 | 1 | 离线 | ps_ai | 是 | 13 | 0.4181 |
| `ontr_base_lovasz_s42` | Sc200 | 1 | 离线 | ps_ai | 是 | 42 | 0.4236 |
| `ontr_base_lovasz_s7` | Sc200 | 1 | 离线 | ps_ai | 是 | 7 | 0.4150 |
| `ontr_eval_bk` | Sc200 | None | - | ps_ai | 否 | 7675605 | - |
| `ontr_eval_ontr` | Sc200 | None | - | ps_ai | 否 | 52267582 | - |
| `sc200_k1_A2_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.3861 |
| `sc200_k1_BASEu_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | - |
| `sc200_k1_BASEu_s7` | Sc200 | 1 | 离线 | ps_ai | 否 | 7 | 0.4046 |
| `sc200_k1_ONLINE_s13` | Sc200 | 1 | 在线 | ps_ai | 否 | 13 | 0.3823 |
| `sc200_k1_ONLINE_s42` | Sc200 | 1 | 在线 | ps_ai | 否 | 42 | 0.3855 |
| `sc200_k1_ONLINE_s7` | Sc200 | 1 | 在线 | ps_ai | 否 | 7 | 0.3904 |
| `sc200_k1_SW2_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | - |
| `sc200_k1_SW3_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.3817 |
| `sc200_k1_SW3_s7` | Sc200 | 1 | 离线 | ps_ai | 否 | 7 | 0.4006 |
| `sc200_k1_base_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.3853 |
| `sc200_k1_offline` | Sc200 | 1 | 离线 | ps_ai | 否 | 3808952 | 0.4038 |
| `sc200_k1_offline_A` | Sc200 | 1 | 离线 | ps_ai | 否 | 2580662 | 0.3890 |
| `sc200_k1_offline_B` | Sc200 | 1 | 离线 | ps_ai | 否 | 55252509 | 0.3815 |
| `sc200_k1_regis1` | Sc200 | 1 | - | ps_ai | 否 | 12900599 | - |
| `sonata_s_s2_unfrozen_lovasz` | Sc200 | 1 | 离线 | ps_ai | 是 | 42 | 0.4206 |
| `sonata_small_stage1a` | Sc200 | None | 离线 | ps_ai | 否 | 45983941 | - |
| `spunet_bk` | Sc200 | None | 离线 | ps_ai | 否 | 42 | - |
| `support_base_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | - |

## xl（6 个）

| 目录 | 数据集 | shot | PL模式 | PL方案 | Lovász | seed | HM |
|---|---|---|---|---|---|---|---|
| `sc200_k1_XLu_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.4265 |
| `sc200_k1_XLu_s7` | Sc200 | 1 | 离线 | ps_ai | 否 | 7 | - |
| `sc200_k1_XLu_s7_eval` | Sc200 | 1 | - | ps_ai | 否 | 60683218 | - |
| `support_ontr_s42_BUG_vlm` | Sc200 | 1 | 在线 | ps_ai | 否 | 42 | 0.3866 |
| `support_xl_s42_BUG_vlm` | Sc200 | 1 | 在线 | ps_ai | 否 | 42 | - |
| `xl_s13` | Sc200 | 1 | 离线 | ps_ai | 否 | 13 | 0.4191 |

## sym（50 个）

| 目录 | 数据集 | shot | PL模式 | PL方案 | Lovász | seed | HM |
|---|---|---|---|---|---|---|---|
| `_sym_nobc` | Sc++ | 5 | - | ps_ai | 否 | 25559278 | - |
| `scpp_k5_sym_gtproj_s7` | Sc++ | 5 | 在线 | ps_ai | 否 | 7 | 0.0001 |
| `scpp_sym_k1_lin_xlr001_s42` | Sc++ | 1 | 在线 | ps_ai | 否 | 42 | 0.2794 |
| `scpp_sym_k1_s13` | Sc++ | 1 | 在线 | ps_ai | 否 | 13 | 0.3064 |
| `scpp_sym_k1_s42` | Sc++ | 1 | 在线 | ps_ai | 否 | 42 | 0.2619 |
| `scpp_sym_k1_s7` | Sc++ | 1 | 在线 | ps_ai | 否 | 7 | 0.2745 |
| `scpp_sym_k1_xlr001_s42` | Sc++ | 1 | 在线 | ps_ai | 否 | 42 | 0.2818 |
| `scpp_sym_k5_g02_s42` | Sc++ | 5 | 在线 | ps_ai | 否 | 42 | 0.3245 |
| `scpp_sym_k5_s13` | Sc++ | 5 | 在线 | ps_ai | 否 | 13 | 0.3222 |
| `scpp_sym_k5_s42` | Sc++ | 5 | 在线 | ps_ai | 否 | 42 | 0.3403 |
| `scpp_sym_k5_s7` | Sc++ | 5 | 在线 | ps_ai | 否 | 7 | 0.3309 |
| `scpp_sym_lovasz` | Sc++ | 1 | 在线 | ps_ai | 否 | 31853493 | 0.2986 |
| `scpp_sym_lovasz_k5` | Sc++ | 5 | 在线 | ps_ai | 否 | 25559278 | 0.3103 |
| `cos_sym_k1_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.4201 |
| `cos_sym_k5_ep12` | Sc200 | 5 | - | ps_ai | 否 | 35284487 | - |
| `cos_sym_k5_ep16` | Sc200 | 5 | - | ps_ai | 否 | 36632934 | - |
| `cos_sym_k5_ep4` | Sc200 | 5 | - | ps_ai | 否 | 6512452 | - |
| `cos_sym_k5_ep8` | Sc200 | 5 | - | ps_ai | 否 | 36206359 | - |
| `cos_sym_k5_s42` | Sc200 | 5 | 离线 | ps_ai | 否 | 42 | 0.4326 |
| `cos_sym_k5_true_s42` | Sc200 | 5 | 在线 | ps_ai | 否 | 42 | 0.4290 |
| `detach_sym_k5_s42` | Sc200 | 5 | 离线 | ps_ai | 否 | 42 | 0.4066 |
| `detach_sym_lovasz_s7` | Sc200 | 1 | 离线 | ps_ai | 是 | 7 | 0.4089 |
| `lcross_cos_k5_s42` | Sc200 | 5 | 离线 | ps_ai | 否 | 42 | 0.4320 |
| `lcross_sym_k1_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.4230 |
| `lcross_sym_k5_s42` | Sc200 | 5 | 离线 | ps_ai | 否 | 42 | 0.4304 |
| `ontr_sym_lovasz_s42` | Sc200 | 1 | 离线 | ps_ai | 是 | 42 | 0.4404 |
| `ortho_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.4228 |
| `ortho_s7` | Sc200 | 1 | 离线 | ps_ai | 否 | 7 | 0.4222 |
| `sc200_k1_sym_gtproj_s7` | Sc200 | 1 | 在线 | ps_ai | 否 | 10 | 0.3575 |
| `sc200_k1_sym_s13` | Sc200 | 1 | 离线 | ps_ai | 否 | 13 | 0.4194 |
| `sc200_k1_sym_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.4257 |
| `sc200_k1_sym_s7` | Sc200 | 1 | 离线 | ps_ai | 否 | 7 | 0.4214 |
| `sc200_k5_sym_65536_s42` | Sc200 | 5 | 离线 | ps_ai | 否 | 42 | - |
| `sonata_s_s2_encfrozen_lovasz` | Sc200 | 1 | 离线 | ps_ai | 是 | 40045561 | 0.4027 |
| `sonata_s_s2_support_lovasz` | Sc200 | 1 | 离线 | ps_ai | 是 | 54639613 | 0.1511 |
| `support_sym_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.1492 |
| `sym_k1_a090` | Sc200 | 1 | - | ps_ai | 否 | 34509238 | - |
| `sym_k5_102400_s13` | Sc200 | 5 | 离线 | ps_ai | 否 | 13 | 0.4373 |
| `sym_k5_102400_s42` | Sc200 | 5 | 离线 | ps_ai | 否 | 42 | 0.4410 |
| `sym_k5_102400_s7` | Sc200 | 5 | 离线 | ps_ai | 否 | 7 | 0.4388 |
| `sym_k5_a080` | Sc200 | 5 | - | ps_ai | 否 | 22515937 | - |
| `sym_k5_a090` | Sc200 | 5 | - | ps_ai | 否 | 21872749 | - |
| `sym_la_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.4117 |
| `sym_lcross01_k1_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.4234 |
| `sym_lcross01_k5_102400_s42` | Sc200 | 5 | 离线 | ps_ai | 否 | 42 | 0.4406 |
| `sym_lovasz_5split_s42` | Sc200 | 1 | 在线 | ps_ai | 是 | 42 | - |
| `sym_lovasz_k5_102400_s42` | Sc200 | 5 | 离线 | ps_ai | 是 | 42 | 0.4553 |
| `sym_lovasz_s13` | Sc200 | 1 | 离线 | ps_ai | 是 | 13 | 0.4379 |
| `sym_lovasz_s42` | Sc200 | 1 | 离线 | ps_ai | 是 | 42 | 0.4414 |
| `sym_lovasz_s7` | Sc200 | 1 | 离线 | ps_ai | 是 | 7 | 0.4438 |

## M5（62 个）

| 目录 | 数据集 | shot | PL模式 | PL方案 | Lovász | seed | HM |
|---|---|---|---|---|---|---|---|
| `scpp_k1_m10_s13` | Sc++ | 1 | 在线 | ps_ai | 否 | 13 | 0.2766 |
| `scpp_k1_m10_s42` | Sc++ | 1 | 在线 | ps_ai | 否 | 42 | 0.2732 |
| `scpp_k1_m10_s7` | Sc++ | 1 | 在线 | ps_ai | 否 | 7 | 0.2678 |
| `scpp_k1_m11a_s13` | Sc++ | 1 | 在线 | ps_ai | 否 | 13 | 0.2937 |
| `scpp_k1_m11a_s42` | Sc++ | 1 | 在线 | ps_ai | 否 | 42 | 0.2933 |
| `scpp_k1_m11a_s7` | Sc++ | 1 | 在线 | ps_ai | 否 | 7 | 0.2760 |
| `scpp_k1_m11b_s13` | Sc++ | 1 | 在线 | ps_ai | 否 | 13 | 0.2724 |
| `scpp_k1_m5_dualeye90_mnv_s7` | Sc++ | 1 | 在线 | 'dual_eye_and' | 否 | 7 | 0.2216 |
| `scpp_k1_m5_h1_s7` | Sc++ | 1 | 在线 | 'dual_eye_h1' | 否 | 7 | 0.2672 |
| `scpp_k1_m5_m1vote_s7` | Sc++ | 1 | 在线 | 'dual_eye_m1_vote' | 否 | 7 | 0.2877 |
| `scpp_k1_m5_s13` | Sc++ | 1 | 在线 | ps_ai | 否 | 13 | 0.3082 |
| `scpp_k1_m5_s42` | Sc++ | 1 | 在线 | ps_ai | 否 | 42 | 0.2803 |
| `scpp_k1_m5_s7` | Sc++ | 1 | 在线 | ps_ai | 否 | 7 | 0.2841 |
| `scpp_k1_m6_s13` | Sc++ | 1 | 在线 | ps_ai | 否 | 13 | 0.2998 |
| `scpp_k1_m6_s42` | Sc++ | 1 | 在线 | ps_ai | 否 | 42 | 0.2638 |
| `scpp_k1_m6_s7` | Sc++ | 1 | 在线 | ps_ai | 否 | 7 | 0.2748 |
| `scpp_k1_m7_s13` | Sc++ | 1 | 在线 | ps_ai | 否 | 13 | 0.2867 |
| `scpp_k1_m7_s42` | Sc++ | 1 | 在线 | ps_ai | 否 | 42 | 0.2958 |
| `scpp_k1_m7_s7` | Sc++ | 1 | 在线 | ps_ai | 否 | 7 | 0.2769 |
| `scpp_k1_m8_s13` | Sc++ | 1 | 在线 | ps_ai | 否 | 13 | 0.2932 |
| `scpp_k1_m8_s42` | Sc++ | 1 | 在线 | ps_ai | 否 | 42 | 0.2671 |
| `scpp_k1_m8_s7` | Sc++ | 1 | 在线 | ps_ai | 否 | 7 | 0.2724 |
| `scpp_k5_m5_cleancon_s7` | Sc++ | 5 | 在线 | ps_ai | 否 | 7 | 0.3421 |
| `scpp_k5_m5_cleancon_v2_s7` | Sc++ | 5 | 在线 | ps_ai | 否 | 7 | 0.3343 |
| `scpp_k5_m5_cleancon_v2_softent_s7` | Sc++ | 5 | 在线 | ps_ai | 否 | 7 | 0.3531 |
| `scpp_k5_m5_dualeye85_s42` | Sc++ | 5 | 在线 | 'dual_eye_and' | 否 | 42 | 0.3532 |
| `scpp_k5_m5_dualeye90_mnv_s13` | Sc++ | 5 | 在线 | 'dual_eye_and' | 否 | 13 | 0.3553 |
| `scpp_k5_m5_dualeye90_mnv_s42` | Sc++ | 5 | 在线 | 'dual_eye_and' | 否 | 42 | 0.3107 |
| `scpp_k5_m5_dualeye90_mnv_s7` | Sc++ | 5 | 在线 | 'dual_eye_and' | 否 | 7 | 0.3607 |
| `scpp_k5_m5_elr_s7` | Sc++ | 5 | 在线 | 'ps_ai' | 否 | 7 | - |
| `scpp_k5_m5_entq10_s7` | Sc++ | 5 | 在线 | 'entropy_q' | 否 | 7 | 0.3222 |
| `scpp_k5_m5_gce07_s7` | Sc++ | 5 | 在线 | 'ps_ai' | 否 | 7 | 0.3331 |
| `scpp_k5_m5_geq02_s42` | Sc++ | 5 | 在线 | 'global_entropy_q' | 否 | 42 | 0.3600 |
| `scpp_k5_m5_geq02_s7` | Sc++ | 5 | 在线 | 'global_entropy_q' | 否 | 7 | 0.3200 |
| `scpp_k5_m5_h1_s7` | Sc++ | 5 | 在线 | 'dual_eye_h1' | 否 | 7 | 0.3528 |
| `scpp_k5_m5_m1vote_s7` | Sc++ | 5 | 在线 | 'dual_eye_m1_vote' | 否 | 7 | 0.3554 |
| `scpp_k5_m5_psai_d2r10_s7` | Sc++ | 5 | 在线 | 'psai_dec2_rank' | 否 | 7 | 0.3647 |
| `scpp_k5_m5_psai_fullent10_s7` | Sc++ | 5 | 在线 | 'ps_ai_then_entropy' | 否 | 7 | 0.3860 |
| `scpp_k5_m5_psai_fullent30_pp_s7` | Sc++ | 5 | 在线 | 'ps_ai_then_entropy' | 否 | 7 | 0.3614 |
| `scpp_k5_m5_s13` | Sc++ | 5 | 在线 | ps_ai | 否 | 13 | 0.2907 |
| `scpp_k5_m5_s42` | Sc++ | 5 | 在线 | ps_ai | 否 | 42 | 0.3309 |
| `scpp_k5_m5_s7` | Sc++ | 5 | 在线 | ps_ai | 否 | 7 | 0.3278 |
| `sc200_k1_m10_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.3942 |
| `sc200_k1_m11a_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.3974 |
| `sc200_k1_m5_cleancon_s7` | Sc200 | 1 | 在线 | ps_ai | 否 | 7 | 0.4067 |
| `sc200_k1_m5_cleancon_v2_s7` | Sc200 | 1 | 在线 | ps_ai | 否 | 7 | 0.4037 |
| `sc200_k1_m5_cleancon_v2_softent_s7` | Sc200 | 1 | 在线 | ps_ai | 否 | 7 | 0.4025 |
| `sc200_k1_m5_elr_s7` | Sc200 | 1 | 在线 | 'ps_ai' | 否 | 7 | - |
| `sc200_k1_m5_entq10_s7` | Sc200 | 1 | 在线 | 'entropy_q' | 否 | 7 | 0.3425 |
| `sc200_k1_m5_gce07_s7` | Sc200 | 1 | 在线 | 'ps_ai' | 否 | 7 | 0.3780 |
| `sc200_k1_m5_m1vote_s7` | Sc200 | 1 | 在线 | 'dual_eye_m1_vote' | 否 | 7 | 0.2941 |
| `sc200_k1_m5_psai_d2r10_s7` | Sc200 | 1 | 在线 | 'psai_dec2_rank' | 否 | 7 | 0.3727 |
| `sc200_k1_m5_psai_fullent10_s7` | Sc200 | 1 | 在线 | 'ps_ai_then_entropy' | 否 | 7 | 0.3364 |
| `sc200_k1_m5_psai_fullent30_pp_s7` | Sc200 | 1 | 在线 | 'ps_ai_then_entropy' | 否 | 7 | 0.4088 |
| `sc200_k1_m5_s13` | Sc200 | 1 | 离线 | ps_ai | 否 | 13 | 0.4163 |
| `sc200_k1_m5_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.4194 |
| `sc200_k1_m5_s7` | Sc200 | 1 | 离线 | ps_ai | 否 | 7 | 0.4014 |
| `sc200_k1_m8_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.4177 |
| `sc200_k5_m5_m1vote_s7` | Sc200 | 5 | 在线 | 'dual_eye_m1_vote' | 否 | 7 | 0.4098 |
| `sc200_k5_m5_s13` | Sc200 | 5 | 离线 | ps_ai | 否 | 13 | 0.4309 |
| `sc200_k5_m5_s42` | Sc200 | 5 | 离线 | ps_ai | 否 | 42 | 0.4262 |
| `sc200_k5_m5_s7` | Sc200 | 5 | 离线 | ps_ai | 否 | 7 | 0.4388 |

## m9（4 个）

| 目录 | 数据集 | shot | PL模式 | PL方案 | Lovász | seed | HM |
|---|---|---|---|---|---|---|---|
| `scpp_k1_m9_s13` | Sc++ | 1 | 在线 | ps_ai | 否 | 13 | 0.2679 |
| `scpp_k1_m9_s42` | Sc++ | 1 | 在线 | ps_ai | 否 | 42 | 0.2821 |
| `scpp_k1_m9_s7` | Sc++ | 1 | 在线 | ps_ai | 否 | 7 | 0.3031 |
| `sc200_k1_m9_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.4144 |

## dec2only（15 个）

| 目录 | 数据集 | shot | PL模式 | PL方案 | Lovász | seed | HM |
|---|---|---|---|---|---|---|---|
| `scpp_k1_m1_nod0ln_g035_s13` | Sc++ | 1 | 在线 | ps_ai | 否 | 13 | 0.3089 |
| `scpp_k1_m1_nod0ln_g035_s42` | Sc++ | 1 | 在线 | ps_ai | 否 | 42 | 0.2832 |
| `scpp_k1_m1_nod0ln_g035_s7` | Sc++ | 1 | 在线 | ps_ai | 否 | 7 | 0.2791 |
| `scpp_k1_m2_resfuse_g035_s13` | Sc++ | 1 | 在线 | ps_ai | 否 | 13 | 0.3056 |
| `scpp_k1_m3_resfuse_g100_s13` | Sc++ | 1 | 在线 | ps_ai | 否 | 13 | 0.2933 |
| `scpp_k1_m4_s13` | Sc++ | 1 | 在线 | ps_ai | 否 | 13 | 0.3031 |
| `scpp_k1_m4_s42` | Sc++ | 1 | 在线 | ps_ai | 否 | 42 | 0.2717 |
| `scpp_k1_m4_s7` | Sc++ | 1 | 在线 | ps_ai | 否 | 7 | 0.2785 |
| `scpp_sym_k1_dec2_xlr001_s13` | Sc++ | 1 | 在线 | ps_ai | 否 | 13 | 0.2834 |
| `scpp_sym_k1_dec2_xlr001_s42` | Sc++ | 1 | 在线 | ps_ai | 否 | 42 | 0.2940 |
| `scpp_sym_k1_dec2_xlr001_s7` | Sc++ | 1 | 在线 | ps_ai | 否 | 7 | 0.2844 |
| `sc200_k1_m1_nod0ln_g035_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.4156 |
| `support_ontr_linear_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.1359 |
| `support_xl_linear_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.1022 |
| `sym_k1_dec2_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.4188 |

## rep3（9 个）

| 目录 | 数据集 | shot | PL模式 | PL方案 | Lovász | seed | HM |
|---|---|---|---|---|---|---|---|
| `frozen_ontr_rep3_lovasz_s42` | Sc200 | 1 | 离线 | ps_ai | 是 | 42 | 0.2955 |
| `frozen_rep3_lovasz_s42` | Sc200 | 1 | 离线 | ps_ai | 是 | 42 | 0.2291 |
| `rep3_lovasz_s13` | Sc200 | 1 | 离线 | ps_ai | 是 | 13 | 0.4202 |
| `rep3_lovasz_s42` | Sc200 | 1 | 离线 | ps_ai | 是 | 42 | 0.4379 |
| `rep3_lovasz_s7` | Sc200 | 1 | 离线 | ps_ai | 是 | 7 | 0.4242 |
| `rep3_s13` | Sc200 | 1 | 离线 | ps_ai | 否 | 13 | 0.4191 |
| `rep3_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.4204 |
| `rep3_s7` | Sc200 | 1 | 离线 | ps_ai | 否 | 7 | 0.4047 |
| `support_rep3_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.1304 |

## gated（2 个）

| 目录 | 数据集 | shot | PL模式 | PL方案 | Lovász | seed | HM |
|---|---|---|---|---|---|---|---|
| `gated_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.4260 |
| `gated_s7` | Sc200 | 1 | 离线 | ps_ai | 否 | 7 | - |

## mlp_dec0（3 个）

| 目录 | 数据集 | shot | PL模式 | PL方案 | Lovász | seed | HM |
|---|---|---|---|---|---|---|---|
| `mlp_dec0_s13` | Sc200 | 1 | 离线 | ps_ai | 否 | 13 | 0.3991 |
| `mlp_dec0_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.4078 |
| `mlp_dec0_s7` | Sc200 | 1 | 离线 | ps_ai | 否 | 7 | 0.4125 |

## lnonly（3 个）

| 目录 | 数据集 | shot | PL模式 | PL方案 | Lovász | seed | HM |
|---|---|---|---|---|---|---|---|
| `frozen_ln_lovasz_s42` | Sc200 | 1 | 离线 | ps_ai | 是 | 42 | 0.1580 |
| `lnonly_s42` | Sc200 | 1 | 离线 | ps_ai | 否 | 42 | 0.3982 |
| `ontr_ln_lovasz_s42` | Sc200 | 1 | 离线 | ps_ai | 是 | 42 | 0.4234 |


