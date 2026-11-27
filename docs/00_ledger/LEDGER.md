# GFS-VL 实验台账

> 自动生成，勿手工编辑。重新生成：`python tools/gen_ledger.py --md docs/00_ledger/LEDGER.md`
>
> **口径**：`抖动` = train.log 最后一次 val（`shuffle_orders` 随机，同权重两次可差 0.5–0.8）；
> `确定性` = `det_eval/` 重评（已修 `shuffle_orders`，两次评估 0.0000 误差）。**跨实验比较只用确定性口径。**

实验目录 **228** 个：done 205 / unevaluated 8 / incomplete 2 / failed 1 / invalid 7 / infra 5。

确定性命中 6 个（`det_eval/`）。

- **invalid**（实现有 bug，结果不可用，7）：`support_base_s42`、`support_ontr_linear_s42`、`support_ontr_s42_BUG_vlm`、`support_rep3_s42`、`support_sym_s42`、`support_xl_linear_s42`、`support_xl_s42_BUG_vlm`
- **failed**（训练/评估崩溃，1）：`sc200_k1_XLu_s7`
- **unevaluated**（有权重但从未评估，8）：`gated_s7`、`sc200_k1_BASEu_s42`、`sc200_k1_m5_elr_s7`、`sc200_k5_sym_65536_s42`、`sonata_small_stage1a`、`sym_lovasz_5split_s42`、`ptv3_bk`、`scpp_k5_m5_elr_s7`
- **incomplete**（空壳或未跑完，2）：`sc200_k1_SW2_s42`、`scpp_entq_abs2`
- **infra**（基础设施/备份，非实验，5）：`_guard_test`、`ontr_eval_bk`、`ontr_eval_ontr`、`ptv3_bk`、`spunet_bk`

## 00 基础设施与备份（5）

| 目录 | 数据集 | 结构/损失 | 伪标签 | 抖动 Base/Novel/HM | 确定性 Base/Novel/HM | 种子 | 口径备注 | 权重 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| `_guard_test` | scannet200 | — / ce | — | — | — | — | — | — | infra |
| `ontr_eval_bk` | scannet200 | — / ce | — | — | — | 7675605 | OneCycleLR,AdamW,续训:ptv3_bk | — | infra |
| `ontr_eval_ontr` | scannet200 | — / ce | — | — | — | 52267582 | OneCycleLR,AdamW,续训:exp | — | infra |
| `ptv3_bk` | scannet200 | — / ce | — | — | — | — | — | 1个/0.5G | infra |
| `spunet_bk` | scannet200 | — / ce | — | — | — | 42 | OneCycleLR,AdamW | 2个/0.9G | infra |

## 01 跨层读出主干与结构变体（24）

| 目录 | 数据集 | 结构/损失 | 伪标签 | 抖动 Base/Novel/HM | 确定性 Base/Novel/HM | 种子 | 口径备注 | 权重 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| `base_lovasz_s13` | scannet200 | base / lovasz | 离线 | 0.6680/0.3179/0.4308 | 0.6619/0.3124/0.4245 | 13 | 主干:model_best.pth | 7个/3.6G | done |
| `base_lovasz_s42` | scannet200 | base / lovasz | 离线 | 0.6862/0.2946/0.4122 | 0.6784/0.2889/0.4052 | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `base_lovasz_s7` | scannet200 | base / lovasz | 离线 | 0.6754/0.3157/0.4303 | 0.6659/0.3111/0.4241 | 7 | 主干:model_best.pth | 7个/3.6G | done |
| `base_s13` | scannet200 | base / ce | 离线 | 0.6685/0.2974/0.4116 | — | 13 | 主干:model_best.pth | 7个/3.6G | done |
| `detach_sym_lovasz_s7` | scannet200 | sym/detach / lovasz | 离线 | 0.6547/0.2972/0.4089 | — | 7 | 主干:model_best.pth | 7个/3.6G | done |
| `gated_s42` | scannet200 | gated / ce | 离线 | 0.6843/0.3093/0.4260 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `gated_s7` | scannet200 | gated / ce | — | — | — | 7 | 主干:model_best.pth | 3个/1.6G | unevaluated |
| `lnonly_s42` | scannet200 | lnonly / ce | 离线 | 0.6874/0.2803/0.3982 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `mlp_dec0_s13` | scannet200 | mlp_dec0 / ce | 离线 | 0.6898/0.2808/0.3991 | — | 13 | 主干:model_best.pth | 7个/3.6G | done |
| `mlp_dec0_s42` | scannet200 | mlp_dec0 / ce | 离线 | 0.6788/0.2914/0.4078 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `mlp_dec0_s7` | scannet200 | mlp_dec0 / ce | 离线 | 0.6953/0.2932/0.4125 | — | 7 | 主干:model_best.pth | 7个/3.6G | done |
| `ortho_s42` | scannet200 | ortho / ce | 离线 | 0.6813/0.3065/0.4228 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `ortho_s7` | scannet200 | ortho / ce | 离线 | 0.6771/0.3067/0.4222 | — | 7 | 主干:model_best.pth | 7个/3.6G | done |
| `rep3_lovasz_s13` | scannet200 | rep3 / lovasz | 离线 | 0.6900/0.3021/0.4202 | — | 13 | 主干:model_best.pth | 7个/3.2G | done |
| `rep3_lovasz_s42` | scannet200 | rep3 / lovasz | 离线 | 0.6758/0.3239/0.4379 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `rep3_lovasz_s7` | scannet200 | rep3 / lovasz | 离线 | 0.6821/0.3078/0.4242 | — | 7 | 主干:model_best.pth | 7个/3.6G | done |
| `rep3_s13` | scannet200 | rep3 / ce | 离线 | 0.6868/0.3016/0.4191 | — | 13 | 主干:model_best.pth | 7个/3.6G | done |
| `rep3_s42` | scannet200 | rep3 / ce | 离线 | 0.6728/0.3058/0.4204 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `rep3_s7` | scannet200 | rep3 / ce | 离线 | 0.6907/0.2862/0.4047 | — | 7 | 主干:model_best.pth | 7个/3.6G | done |
| `sym_lovasz_5split_s42` | scannet200 | sym / lovasz | 在线 | — | — | 42 | 主干:model_best.pth | 4个/2.1G | unevaluated |
| `sym_lovasz_s13` | scannet200 | sym / lovasz | 离线 | 0.6785/0.3233/0.4379 | 0.6752/0.3191/0.4334 | 13 | 主干:model_best.pth | 7个/3.6G | done |
| `sym_lovasz_s42` | scannet200 | sym / lovasz | 离线 | 0.6867/0.3253/0.4414 | 0.6805/0.3207/0.4359 | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `sym_lovasz_s7` | scannet200 | sym / lovasz | 离线 | 0.6812/0.3291/0.4438 | 0.6775/0.3237/0.4381 | 7 | 主干:model_best.pth | 7个/3.6G | done |
| `xl_s13` | scannet200 | xl / ce | 离线 | 0.6725/0.3044/0.4191 | — | 13 | 主干:model_best.pth | 7个/3.6G | done |

## 02 损失函数（2）

| 目录 | 数据集 | 结构/损失 | 伪标签 | 抖动 Base/Novel/HM | 确定性 Base/Novel/HM | 种子 | 口径备注 | 权重 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| `base_la_s42` | scannet200 | base / la | 离线 | 0.6763/0.2725/0.3885 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `sym_la_s42` | scannet200 | sym / la | 离线 | 0.6657/0.2980/0.4117 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |

## 03 伪标签来源与 1-shot 口径（39）

| 目录 | 数据集 | 结构/损失 | 伪标签 | 抖动 Base/Novel/HM | 确定性 Base/Novel/HM | 种子 | 口径备注 | 权重 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| `sc200_k1_A2_s42` | scannet200 | — / ce | a2/离线 | 0.6871/0.2685/0.3861 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_BASEu_s42` | scannet200 | — / ce | baseu | — | — | 42 | 主干:model_best.pth | 2个/1.0G | unevaluated |
| `sc200_k1_BASEu_s7` | scannet200 | — / ce | baseu/离线 | 0.6804/0.2879/0.4046 | — | 7 | 主干:model_best.pth | 8个/4.1G | done |
| `sc200_k1_ONLINE_s13` | scannet200 | — / ce | online/在线 | 0.6736/0.2669/0.3823 | — | 13 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_ONLINE_s42` | scannet200 | — / ce | online/在线 | 0.6871/0.2679/0.3855 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_ONLINE_s7` | scannet200 | — / ce | online/在线 | 0.6642/0.2765/0.3904 | — | 7 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_SW2_s42` | scannet200 | — / ce | sw2 | — | — | 42 | 主干:model_best.pth | — | incomplete |
| `sc200_k1_SW3_s42` | scannet200 | — / ce | sw3/离线 | 0.6863/0.2644/0.3817 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_SW3_s7` | scannet200 | — / ce | sw3/离线 | 0.6845/0.2832/0.4006 | — | 7 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_XLu_s42` | scannet200 | — / ce | xlu/离线 | 0.6831/0.3100/0.4265 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_XLu_s7` | scannet200 | — / ce | xlu | — | — | 7 | 主干:model_best.pth | 7个/3.6G | failed |
| `sc200_k1_XLu_s7_eval` | scannet200 | — / ce | xlu/离线 | 0.6717/0.3003/0.4150 | — | 60683218 | 续训:sc200_k1_XLu_s7,主干:model_best.pth | — | done |
| `sc200_k1_base_s42` | scannet200 | — / ce | 离线 | 0.6738/0.2698/0.3853 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_m10_s42` | scannet200 | — / ce | 离线 | 0.6814/0.2774/0.3942 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_m11a_s42` | scannet200 | — / ce | 离线 | 0.6828/0.2802/0.3974 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_m1_nod0ln_g035_s42` | scannet200 | — / ce | 离线 | 0.6902/0.2973/0.4156 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_m5_cleancon_s7` | scannet200 | — / ce | 在线 | 0.6805/0.2900/0.4067 | — | 7 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_m5_cleancon_v2_s7` | scannet200 | — / ce | 在线 | 0.6760/0.2878/0.4037 | — | 7 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_m5_cleancon_v2_softent_s7` | scannet200 | — / ce | 在线 | 0.6779/0.2862/0.4025 | — | 7 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_m5_elr_s7` | scannet200 | — / ce | 在线 | — | — | 7 | 主干:model_best.pth | 5个/2.6G | unevaluated |
| `sc200_k1_m5_entq10_s7` | scannet200 | — / ce | 在线 | 0.6728/0.2297/0.3425 | — | 7 | PL:entropy_q,主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_m5_gce07_s7` | scannet200 | — / ce | 在线 | 0.6799/0.2618/0.3780 | — | 7 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_m5_m1vote_s7` | scannet200 | — / ce | 在线 | 0.6684/0.1885/0.2941 | — | 7 | PL:dual_eye_m1_vote,主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_m5_psai_d2r10_s7` | scannet200 | — / ce | 在线 | 0.6680/0.2584/0.3727 | — | 7 | PL:psai_dec2_rank,主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_m5_psai_fullent10_s7` | scannet200 | — / ce | 在线 | 0.6764/0.2239/0.3364 | — | 7 | PL:ps_ai_then_entropy,主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_m5_psai_fullent30_pp_s7` | scannet200 | — / ce | 在线 | 0.6694/0.2942/0.4088 | — | 7 | PL:ps_ai_then_entropy,主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_m5_s13` | scannet200 | — / ce | 离线 | 0.6639/0.3032/0.4163 | — | 13 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_m5_s42` | scannet200 | — / ce | 离线 | 0.6852/0.3022/0.4194 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_m5_s7` | scannet200 | — / ce | 离线 | 0.6775/0.2852/0.4014 | — | 7 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_m8_s42` | scannet200 | — / ce | 离线 | 0.6863/0.3002/0.4177 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_m9_s42` | scannet200 | — / ce | 离线 | 0.6751/0.2990/0.4144 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_offline` | scannet200 | — / ce | offline/离线 | 0.6805/0.2500/0.3656 | — | 3808952 | 主干:model_best.pth | 8个/4.1G | done |
| `sc200_k1_offline_A` | scannet200 | — / ce | offline/离线 | 0.6872/0.2711/0.3889 | — | 2580662 | 续训:sc200_k1_offline_A,主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_offline_B` | scannet200 | — / ce | offline/离线 | 0.6917/0.2520/0.3695 | — | 55252509 | 续训:sc200_k1_offline_B,主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_regis1` | scannet200 | — / ce | official/离线 | 0.6816/0.2880/0.4050 | — | 12900599 | 续训:sc200_k1_regis1,主干:model_best.pth | 1个/0.5G | done |
| `sc200_k1_sym_gtproj_s7` | scannet200 | sym / ce | 在线 | 0.6212/0.2510/0.3575 | — | 7 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_sym_s13` | scannet200 | sym / ce | 离线 | 0.6844/0.3024/0.4194 | — | 13 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_sym_s42` | scannet200 | sym / ce | 离线 | 0.6877/0.3083/0.4257 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `sc200_k1_sym_s7` | scannet200 | sym / ce | 离线 | 0.6775/0.3058/0.4214 | — | 7 | 主干:model_best.pth | 7个/3.6G | done |

## 04 ONTR 与冻结骨干（17）

| 目录 | 数据集 | 结构/损失 | 伪标签 | 抖动 Base/Novel/HM | 确定性 Base/Novel/HM | 种子 | 口径备注 | 权重 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| `frozen_base_lovasz_s42` | scannet200 | — / lovasz | 离线 | 0.6570/0.0765/0.1370 | — | 42 | 主干:model_best.pth | 7个/1.2G | done |
| `frozen_ln_lovasz_s42` | scannet200 | — / lovasz | 离线 | 0.6419/0.0901/0.1580 | — | 42 | 主干:model_best.pth | 7个/1.2G | done |
| `frozen_ontr_lovasz_s42` | scannet200 | — / lovasz | 离线 | 0.6783/0.1491/0.2445 | — | 42 | 主干:ptv3_bk_ontr.pth | 7个/1.2G | done |
| `frozen_ontr_rep3_lovasz_s42` | scannet200 | rep3 / lovasz | 离线 | 0.6697/0.1896/0.2955 | — | 42 | 主干:ptv3_bk_ontr.pth | 7个/1.2G | done |
| `frozen_rep3_lovasz_s42` | scannet200 | rep3 / lovasz | 离线 | 0.6521/0.1390/0.2291 | — | 42 | 主干:model_best.pth | 7个/1.2G | done |
| `ontr_base_lovasz_s13` | scannet200 | — / lovasz | 离线 | 0.6615/0.3057/0.4181 | — | 13 | 主干:ptv3_bk_ontr.pth | 7个/3.6G | done |
| `ontr_base_lovasz_s42` | scannet200 | — / lovasz | 离线 | 0.6854/0.3065/0.4236 | — | 42 | 主干:ptv3_bk_ontr.pth | 7个/3.6G | done |
| `ontr_base_lovasz_s7` | scannet200 | — / lovasz | 离线 | 0.6732/0.3000/0.4150 | — | 7 | 主干:ptv3_bk_ontr.pth | 7个/3.6G | done |
| `ontr_ln_lovasz_s42` | scannet200 | — / lovasz | 离线 | 0.6799/0.3074/0.4234 | — | 42 | 主干:ptv3_bk_ontr.pth | 7个/3.6G | done |
| `ontr_sym_lovasz_s42` | scannet200 | sym / lovasz | 离线 | 0.6885/0.3238/0.4404 | — | 42 | 主干:ptv3_bk_ontr.pth | 7个/3.6G | done |
| `support_base_s42` | scannet200 | — / ce | — | — | — | 42 | 主干:model_best.pth | 7个/3.6G | invalid |
| `support_ontr_linear_s42` | scannet200 | — / ce | 离线 | 0.6359/0.0761/0.1359 | — | 42 | 主干:ptv3_bk_ontr.pth | 7个/3.6G | invalid |
| `support_ontr_s42_BUG_vlm` | scannet200 | — / ce | 在线 | 0.6705/0.2716/0.3866 | — | 42 | 主干:ptv3_bk_ontr.pth | 7个/3.6G | invalid |
| `support_rep3_s42` | scannet200 | rep3 / ce | 离线 | 0.6401/0.0726/0.1304 | — | 42 | 主干:model_best.pth | 7个/3.6G | invalid |
| `support_sym_s42` | scannet200 | sym / ce | 离线 | 0.6370/0.0845/0.1492 | — | 42 | 主干:model_best.pth | 7个/3.6G | invalid |
| `support_xl_linear_s42` | scannet200 | — / ce | 离线 | 0.6385/0.0556/0.1022 | — | 42 | 主干:model_best.pth | 7个/3.6G | invalid |
| `support_xl_s42_BUG_vlm` | scannet200 | — / ce | 在线 | — | — | 42 | 主干:model_best.pth | — | invalid |

## 05 Sonata 基模替换（4）

| 目录 | 数据集 | 结构/损失 | 伪标签 | 抖动 Base/Novel/HM | 确定性 Base/Novel/HM | 种子 | 口径备注 | 权重 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| `sonata_s_s2_encfrozen_lovasz` | scannet200 | — / lovasz | 离线 | 0.6426/0.2932/0.4027 | — | 40045561 | 主干:model_best.pth | 7个/1.6G | done |
| `sonata_s_s2_support_lovasz` | scannet200 | — / lovasz | 离线 | 0.5919/0.0866/0.1511 | — | 54639613 | 主干:model_best.pth | 7个/1.6G | done |
| `sonata_s_s2_unfrozen_lovasz` | scannet200 | — / lovasz | 离线 | 0.6448/0.3121/0.4206 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `sonata_small_stage1a` | scannet200 | — / ce | — | — | — | 45983941 | OneCycleLR,AdamW,续训:jh | 2个/0.5G | unevaluated |

## 07 shot 数与点数上限（34）

| 目录 | 数据集 | 结构/损失 | 伪标签 | 抖动 Base/Novel/HM | 确定性 Base/Novel/HM | 种子 | 口径备注 | 权重 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| `base_k5_102400_s13` | scannet200 | base / ce | 离线 | 0.6740/0.3130/0.4275 | — | 13 | 主干:model_best.pth | 1个/0.5G | done |
| `base_k5_102400_s42` | scannet200 | base / ce | 离线 | 0.6744/0.3067/0.4216 | — | 42 | 主干:model_best.pth | 1个/0.5G | done |
| `base_k5_102400_s7` | scannet200 | base / ce | 离线 | 0.6706/0.3215/0.4346 | — | 7 | 主干:model_best.pth | 1个/0.5G | done |
| `base_k5_s42` | scannet200 | base / ce | 离线 | 0.6740/0.3030/0.4181 | — | 42 | 主干:model_best.pth | 1个/0.5G | done |
| `base_k5_true_s42` | scannet200 | base / ce | 在线 | 0.6815/0.2937/0.4105 | — | 42 | 主干:model_best.pth | 1个/0.5G | done |
| `base_k5_truemulti_s42` | scannet200 | base / ce | 在线 | 0.6824/0.2939/0.4108 | — | 42 | 主干:model_best.pth | 1个/0.5G | done |
| `base_lovasz_k5_102400_s42` | scannet200 | base / lovasz | 离线 | 0.6715/0.3385/0.4501 | — | 42 | 主干:model_best.pth | 1个/0.5G | done |
| `cos_base_k5_s42` | scannet200 | cos / ce | 离线 | 0.6886/0.2657/0.3835 | — | 42 | 余弦退火,AdamW,主干:model_best.pth | 6个/3.1G | done |
| `cos_sym_k1_s42` | scannet200 | sym/cos / ce | 离线 | 0.6839/0.3031/0.4201 | — | 42 | 余弦退火,AdamW,主干:model_best.pth | 7个/3.6G | done |
| `cos_sym_k5_ep12` | scannet200 | sym/cos / ce | 离线 | 0.6807/0.2994/0.4158 | — | 35284487 | 余弦退火,AdamW,续训:cos_sym_k5_s42,主干:model_best.pth | — | done |
| `cos_sym_k5_ep16` | scannet200 | sym/cos / ce | 离线 | 0.6804/0.3174/0.4329 | — | 36632934 | 余弦退火,AdamW,续训:cos_sym_k5_s42,主干:model_best.pth | — | done |
| `cos_sym_k5_ep4` | scannet200 | sym/cos / ce | 离线 | 0.6751/0.2613/0.3768 | — | 6512452 | 余弦退火,AdamW,续训:cos_sym_k5_s42,主干:model_best.pth | — | done |
| `cos_sym_k5_ep8` | scannet200 | sym/cos / ce | 离线 | 0.6572/0.2845/0.3971 | — | 36206359 | 余弦退火,AdamW,续训:cos_sym_k5_s42,主干:model_best.pth | — | done |
| `cos_sym_k5_s42` | scannet200 | sym/cos / ce | 离线 | 0.6835/0.3165/0.4326 | — | 42 | 余弦退火,AdamW,主干:model_best.pth | 6个/3.1G | done |
| `cos_sym_k5_true_s42` | scannet200 | sym/cos / ce | 在线 | 0.6788/0.3136/0.4290 | — | 42 | 余弦退火,AdamW,主干:model_best.pth | 6个/3.1G | done |
| `detach_sym_k5_s42` | scannet200 | sym/detach / ce | 离线 | 0.6838/0.2894/0.4066 | — | 42 | 主干:model_best.pth | 1个/0.5G | done |
| `lcross_cos_k5_s42` | scannet200 | lcross+cos/cos / ce | 离线 | 0.6817/0.3162/0.4320 | — | 42 | 余弦退火,AdamW,主干:model_best.pth | 6个/3.1G | done |
| `lcross_sym_k1_s42` | scannet200 | lcross / ce | 离线 | 0.6843/0.3061/0.4230 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `lcross_sym_k5_s42` | scannet200 | lcross / ce | 离线 | 0.6779/0.3152/0.4304 | — | 42 | 主干:model_best.pth | 1个/0.5G | done |
| `sc200_k5_m5_m1vote_s7` | scannet200 | — / ce | 在线 | 0.6716/0.2948/0.4098 | — | 7 | PL:dual_eye_m1_vote,主干:model_best.pth | 1个/0.5G | done |
| `sc200_k5_m5_s13` | scannet200 | — / ce | 离线 | 0.6806/0.3152/0.4309 | — | 13 | 主干:model_best.pth | 1个/0.5G | done |
| `sc200_k5_m5_s42` | scannet200 | — / ce | 离线 | 0.6711/0.3122/0.4262 | — | 42 | 主干:model_best.pth | 1个/0.5G | done |
| `sc200_k5_m5_s7` | scannet200 | — / ce | 离线 | 0.6876/0.3222/0.4388 | — | 7 | 主干:model_best.pth | 1个/0.5G | done |
| `sc200_k5_sym_65536_s42` | scannet200 | sym / ce | — | — | — | 42 | 主干:model_best.pth | 1个/0.5G | unevaluated |
| `sym_k1_a090` | scannet200 | sym / ce | 离线 | 0.6972/0.2976/0.4171 | — | 34509238 | α=0.90,续训:sc200_k1_sym_s42,主干:model_best.pth | — | done |
| `sym_k1_dec2_s42` | scannet200 | sym / ce | 离线 | 0.6856/0.3015/0.4188 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `sym_k5_102400_s13` | scannet200 | sym / ce | 离线 | 0.6793/0.3224/0.4373 | — | 13 | 主干:model_best.pth | 1个/0.5G | done |
| `sym_k5_102400_s42` | scannet200 | sym / ce | 离线 | 0.6843/0.3253/0.4410 | — | 42 | 主干:model_best.pth | 1个/0.5G | done |
| `sym_k5_102400_s7` | scannet200 | sym / ce | 离线 | 0.6716/0.3258/0.4388 | — | 7 | 主干:model_best.pth | 1个/0.5G | done |
| `sym_k5_a080` | scannet200 | sym / ce | 离线 | 0.6927/0.3032/0.4218 | — | 22515937 | α=0.80,续训:sc200_k5_sym_65536_s42,主干:model_best.pth | — | done |
| `sym_k5_a090` | scannet200 | sym / ce | 离线 | 0.6834/0.3154/0.4316 | — | 21872749 | α=0.90,续训:sc200_k5_sym_65536_s42,主干:model_best.pth | — | done |
| `sym_lcross01_k1_s42` | scannet200 | sym+lcross01 / ce | 离线 | 0.6848/0.3064/0.4234 | — | 42 | 主干:model_best.pth | 7个/3.6G | done |
| `sym_lcross01_k5_102400_s42` | scannet200 | sym+lcross01 / ce | 离线 | 0.6853/0.3246/0.4406 | — | 42 | 主干:model_best.pth | 1个/0.5G | done |
| `sym_lovasz_k5_102400_s42` | scannet200 | sym / lovasz | 离线 | 0.6835/0.3413/0.4553 | — | 42 | 主干:model_best.pth | 1个/0.5G | done |

## 08 ScanNet++（跨数据集验证）（103）

| 目录 | 数据集 | 结构/损失 | 伪标签 | 抖动 Base/Novel/HM | 确定性 Base/Novel/HM | 种子 | 口径备注 | 权重 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| `1A0_det` | scannetpp | — / ce | 离线 | 0.6126/0.2111/0.3140 | — | 45863570 | PS0.60/AI0.95,续训:scpp_base_lovasz_k5,主干:model_best.pth | — | done |
| `_cm_1ca` | scannetpp | — / ce | 离线 | 0.6028/0.2476/0.3511 | — | 24780262 | PL:entropy_q,续训:scpp_base_lovasz_k5_entq10,主干:model_best.pth | — | done |
| `_cm_1ce` | scannetpp | — / ce | 离线 | 0.5920/0.2413/0.3428 | — | 24578095 | PL:entropy_q,续训:scpp_base_lovasz_k5_entq10_fix,主干:model_best.pth | — | done |
| `_cm_l1h` | scannetpp | — / ce | 离线 | 0.6219/0.2230/0.3283 | — | 31682593 | PL:student_consensus,warm4,tau0.40,续训:scpp_l1hybrid,主干:model_best.pth | — | done |
| `_l1_smoke` | scannetpp | — / ce | 在线 | 0.5981/0.0374/0.0703 | — | 42 | PL:student_consensus,warm0,tau0.40,主干:model_best.pth | 1个/0.5G | done |
| `_l1_smoke3` | scannetpp | — / ce | 在线 | 0.5805/0.0776/0.1369 | — | 42 | PL:student_consensus,warm0,tau0.40,主干:model_best.pth | 1个/0.5G | done |
| `_sym_nobc` | scannetpp | sym / ce | 离线 | 0.6019/0.2021/0.3026 | — | 25559278 | PS0.60/AI0.95,续训:jh,主干:model_best.pth | — | done |
| `ptv3_bk` | scannetpp | — / ce | — | — | — | — | — | 1个/0.5G | unevaluated |
| `scpp_base_k1_s13` | scannetpp | — / ce | 在线 | 0.6321/0.2075/0.3125 | — | 13 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_base_k1_s42` | scannetpp | — / ce | 在线 | 0.6273/0.1838/0.2843 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_base_k1_s7` | scannetpp | — / ce | 在线 | 0.6144/0.1708/0.2673 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_base_k5_s13` | scannetpp | — / ce | 在线 | 0.6206/0.2020/0.3047 | — | 13 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_base_k5_s42` | scannetpp | — / ce | 在线 | 0.6036/0.2050/0.3060 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_base_k5_s7` | scannetpp | — / ce | 在线 | 0.6193/0.2148/0.3189 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_base_lovasz` | scannetpp | — / lovasz | 在线 | 0.6120/0.1819/0.2804 | — | 20363522 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_base_lovasz_k5` | scannetpp | — / lovasz | 在线 | 0.6155/0.2117/0.3150 | — | 20269292 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_base_lovasz_k5_entq10` | scannetpp | — / lovasz | 在线 | 0.6040/0.2468/0.3505 | — | 42 | PL:entropy_q,主干:model_best.pth | 1个/0.5G | done |
| `scpp_base_lovasz_k5_entq10_fix` | scannetpp | — / lovasz | 在线 | 0.5924/0.2404/0.3420 | — | 42 | PL:entropy_q,主干:model_best.pth | 1个/0.5G | done |
| `scpp_base_lovasz_k5_entq50` | scannetpp | — / lovasz | 在线 | 0.5238/0.1767/0.2642 | — | 42 | PL:entropy_q,主干:model_best.pth | 1个/0.5G | done |
| `scpp_base_lovasz_k5_lr1e3` | scannetpp | — / lovasz | 在线 | 0.6224/0.1658/0.2618 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_base_lovasz_k5_noai` | scannetpp | — / lovasz | 在线 | 0.6189/0.1752/0.2731 | — | 42 | PS0.60/AI2.00,主干:model_best.pth | 1个/0.5G | done |
| `scpp_base_lovasz_k5_psaithenq` | scannetpp | — / lovasz | 在线 | 0.6231/0.2387/0.3452 | — | 42 | PL:ps_ai_then_entropy,主干:model_best.pth | 1个/0.5G | done |
| `scpp_entq10_veto` | scannetpp | — / ce | 在线 | 0.6302/0.2342/0.3414 | — | 42 | PL:entropy_q,主干:model_best.pth | 1个/0.5G | done |
| `scpp_entq_abs2` | scannetpp | — / ce | 在线 | — | — | 22598486 | PL:entropy_q,主干:model_best.pth | — | incomplete |
| `scpp_fix1` | scannetpp | — / ce | 在线 | 0.5924/0.2309/0.3323 | — | 17308501 | PL:entropy_q,主干:model_best.pth | 1个/0.5G | done |
| `scpp_fix2` | scannetpp | — / ce | 在线 | 0.5878/0.2302/0.3308 | — | 51256953 | PL:entropy_q,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m10_s13` | scannetpp | — / ce | 在线 | 0.6093/0.1790/0.2766 | — | 13 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m10_s42` | scannetpp | — / ce | 在线 | 0.5983/0.1770/0.2732 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m10_s7` | scannetpp | — / ce | 在线 | 0.5839/0.1737/0.2678 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m11a_s13` | scannetpp | — / ce | 在线 | 0.6191/0.1925/0.2937 | — | 13 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m11a_s42` | scannetpp | — / ce | 在线 | 0.6147/0.1926/0.2933 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m11a_s7` | scannetpp | — / ce | 在线 | 0.6013/0.1791/0.2760 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m11b_s13` | scannetpp | — / ce | 在线 | 0.6010/0.1761/0.2724 | — | 13 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m1_nod0ln_g035_s13` | scannetpp | — / ce | 在线 | 0.6320/0.2044/0.3089 | — | 13 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m1_nod0ln_g035_s42` | scannetpp | — / ce | 在线 | 0.6169/0.1838/0.2832 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m1_nod0ln_g035_s7` | scannetpp | — / ce | 在线 | 0.6204/0.1800/0.2791 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m2_resfuse_g035_s13` | scannetpp | — / ce | 在线 | 0.6374/0.2010/0.3056 | — | 13 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m3_resfuse_g100_s13` | scannetpp | — / ce | 在线 | 0.6415/0.1901/0.2933 | — | 13 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m4_s13` | scannetpp | — / ce | 在线 | 0.6395/0.1987/0.3031 | — | 13 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m4_s42` | scannetpp | — / ce | 在线 | 0.6105/0.1747/0.2717 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m4_s7` | scannetpp | — / ce | 在线 | 0.6143/0.1800/0.2785 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m5_dualeye90_mnv_s7` | scannetpp | — / ce | 在线 | 0.6008/0.1358/0.2216 | — | 7 | PL:dual_eye_and,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m5_h1_s7` | scannetpp | — / ce | 在线 | 0.6272/0.1697/0.2672 | — | 7 | PL:dual_eye_h1,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m5_m1vote_s7` | scannetpp | — / ce | 在线 | 0.6265/0.1868/0.2877 | — | 7 | PL:dual_eye_m1_vote,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m5_s13` | scannetpp | — / ce | 在线 | 0.6320/0.2038/0.3082 | — | 13 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m5_s42` | scannetpp | — / ce | 在线 | 0.6137/0.1816/0.2803 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m5_s7` | scannetpp | — / ce | 在线 | 0.6244/0.1839/0.2841 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m6_s13` | scannetpp | — / ce | 在线 | 0.6285/0.1968/0.2998 | — | 13 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m6_s42` | scannetpp | — / ce | 在线 | 0.6154/0.1679/0.2638 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m6_s7` | scannetpp | — / ce | 在线 | 0.5999/0.1783/0.2748 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m7_s13` | scannetpp | — / ce | 在线 | 0.6353/0.1851/0.2867 | — | 13 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m7_s42` | scannetpp | — / ce | 在线 | 0.6143/0.1948/0.2958 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m7_s7` | scannetpp | — / ce | 在线 | 0.6050/0.1796/0.2769 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m8_s13` | scannetpp | — / ce | 在线 | 0.6102/0.1930/0.2932 | — | 13 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m8_s42` | scannetpp | — / ce | 在线 | 0.6151/0.1706/0.2671 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m8_s7` | scannetpp | — / ce | 在线 | 0.5994/0.1762/0.2724 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m9_s13` | scannetpp | — / ce | 在线 | 0.6206/0.1708/0.2679 | — | 13 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m9_s42` | scannetpp | — / ce | 在线 | 0.6294/0.1817/0.2821 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k1_m9_s7` | scannetpp | — / ce | 在线 | 0.6202/0.2005/0.3031 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_geq02_s42` | scannetpp | — / ce | 在线 | 0.6388/0.2387/0.3475 | — | 42 | PL:global_entropy_q,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_geq05_s42` | scannetpp | — / ce | 在线 | 0.6379/0.2270/0.3348 | — | 42 | PL:global_entropy_q,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_m5_cleancon_s7` | scannetpp | — / ce | 在线 | 0.6065/0.2383/0.3421 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_m5_cleancon_v2_s7` | scannetpp | — / ce | 在线 | 0.6006/0.2316/0.3343 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_m5_cleancon_v2_softent_s7` | scannetpp | — / ce | 在线 | 0.5954/0.2509/0.3531 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_m5_dualeye85_s42` | scannetpp | — / ce | 在线 | 0.6076/0.2490/0.3532 | — | 42 | PL:dual_eye_and,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_m5_dualeye90_mnv_s13` | scannetpp | — / ce | 在线 | 0.6061/0.2513/0.3553 | — | 13 | PL:dual_eye_and,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_m5_dualeye90_mnv_s42` | scannetpp | — / ce | 在线 | 0.5353/0.2189/0.3107 | — | 42 | PL:dual_eye_and,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_m5_dualeye90_mnv_s7` | scannetpp | — / ce | 在线 | 0.6282/0.2530/0.3607 | — | 7 | PL:dual_eye_and,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_m5_elr_s7` | scannetpp | — / ce | 在线 | — | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | unevaluated |
| `scpp_k5_m5_entq10_s7` | scannetpp | — / ce | 在线 | 0.6015/0.2200/0.3222 | — | 7 | PL:entropy_q,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_m5_gce07_s7` | scannetpp | — / ce | 在线 | 0.6118/0.2288/0.3331 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_m5_geq02_s42` | scannetpp | — / ce | 在线 | 0.6186/0.2539/0.3600 | — | 42 | PL:global_entropy_q,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_m5_geq02_s7` | scannetpp | — / ce | 在线 | 0.6222/0.2154/0.3200 | — | 7 | PL:global_entropy_q,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_m5_h1_s7` | scannetpp | — / ce | 在线 | 0.5970/0.2504/0.3528 | — | 7 | PL:dual_eye_h1,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_m5_m1vote_s7` | scannetpp | — / ce | 在线 | 0.5949/0.2534/0.3554 | — | 7 | PL:dual_eye_m1_vote,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_m5_psai_d2r10_s7` | scannetpp | — / ce | 在线 | 0.6258/0.2574/0.3647 | — | 7 | PL:psai_dec2_rank,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_m5_psai_fullent10_s7` | scannetpp | — / ce | 在线 | 0.6441/0.2755/0.3860 | — | 7 | PL:ps_ai_then_entropy,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_m5_psai_fullent30_pp_s7` | scannetpp | — / ce | 在线 | 0.6011/0.2584/0.3614 | — | 7 | PL:ps_ai_then_entropy,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_m5_s13` | scannetpp | — / ce | 在线 | 0.6017/0.1916/0.2907 | — | 13 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_m5_s42` | scannetpp | — / ce | 在线 | 0.6236/0.2252/0.3309 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_m5_s7` | scannetpp | — / ce | 在线 | 0.6258/0.2221/0.3278 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_k5_sym_gtproj_s7` | scannetpp | sym / ce | 在线 | 0.3455/0.0000/0.0001 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_l1hybrid` | scannetpp | — / ce | 在线 | 0.6226/0.2227/0.3281 | — | 42 | PL:student_consensus,warm4,tau0.40,主干:model_best.pth | 1个/0.5G | done |
| `scpp_l1pure` | scannetpp | — / ce | 在线 | 0.6176/0.2205/0.3250 | — | 42 | PL:student_consensus,warm4,tau0.40,主干:model_best.pth | 1个/0.5G | done |
| `scpp_m1a` | scannetpp | — / ce | 在线 | 0.6197/0.2098/0.3135 | — | 42 | PL:m1_argmax,主干:model_best.pth | 1个/0.5G | done |
| `scpp_m1b` | scannetpp | — / ce | 在线 | 0.6218/0.2298/0.3356 | — | 42 | PL:m1_cos,主干:model_best.pth | 1个/0.5G | done |
| `scpp_oracle_filter_m0` | scannetpp | — / ce | 在线 | 0.6393/0.3039/0.4120 | — | 42 | PL:oracle_m0,主干:model_best.pth | 1个/0.5G | done |
| `scpp_oracle_filter_m1` | scannetpp | — / ce | 在线 | 0.6205/0.2548/0.3613 | — | 42 | PL:oracle_m1,主干:model_best.pth | 1个/0.5G | done |
| `scpp_oracle_fullsup20` | scannetpp | — / ce | 在线 | 0.6497/0.3940/0.4905 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_sym_k1_dec2_xlr001_s13` | scannetpp | sym / ce | 在线 | 0.5477/0.1912/0.2834 | — | 13 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_sym_k1_dec2_xlr001_s42` | scannetpp | sym / ce | 在线 | 0.6245/0.1923/0.2940 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_sym_k1_dec2_xlr001_s7` | scannetpp | sym / ce | 在线 | 0.6142/0.1850/0.2844 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_sym_k1_lin_xlr001_s42` | scannetpp | sym / ce | 在线 | 0.6137/0.1808/0.2794 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_sym_k1_s13` | scannetpp | sym / ce | 在线 | 0.6097/0.2046/0.3064 | — | 13 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_sym_k1_s42` | scannetpp | sym / ce | 在线 | 0.5916/0.1681/0.2619 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_sym_k1_s7` | scannetpp | sym / ce | 在线 | 0.6206/0.1762/0.2745 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_sym_k1_xlr001_s42` | scannetpp | sym / ce | 在线 | 0.6315/0.1813/0.2818 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_sym_k5_g02_s42` | scannetpp | sym / ce | 在线 | 0.6328/0.2182/0.3245 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_sym_k5_s13` | scannetpp | sym / ce | 在线 | 0.6304/0.2164/0.3222 | — | 13 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_sym_k5_s42` | scannetpp | sym / ce | 在线 | 0.6178/0.2349/0.3403 | — | 42 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_sym_k5_s7` | scannetpp | sym / ce | 在线 | 0.6244/0.2251/0.3309 | — | 7 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_sym_lovasz` | scannetpp | sym / lovasz | 在线 | 0.6177/0.1969/0.2986 | — | 31853493 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |
| `scpp_sym_lovasz_k5` | scannetpp | sym / lovasz | 在线 | 0.6243/0.2064/0.3103 | — | 25559278 | PS0.60/AI0.95,主干:model_best.pth | 1个/0.5G | done |

