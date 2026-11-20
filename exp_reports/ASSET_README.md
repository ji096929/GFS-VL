# CUT2 实验资产清单（2026-11-20 冻结）

## 一、三层保险

| 层 | 位置 | 内容 |
|---|---|---|
| 结果固化 | results_all.json | 17 个 run 的汇总指标 + 57 类 IoU，不依赖日志存活 |
| 日志快照 | logs_snapshot_20261120.tar.gz (899 MB) | 全部 48 份日志，含失败重试记录 |
| 权重清单 | ckpt_manifest.txt (175 条) | 所有 .pth 的路径与大小 |

checkpoint 本体：/data01/jh/code/GFS-VL/exp/scannet200/<dir>/model/（约 91 GB）

## 二、run 名 → 路径映射（早期 run 目录名不同，最容易找错）

| 报告名 | 日志文件 | checkpoint 目录 |
|---|---|---|
| base_s42 | base_s42.log（软链→sc200_k1_A2_s42/train.log） | sc200_k1_A2_s42 |
| base_s7 | sc200_k1_BASEu_s7.log | sc200_k1_BASEu_s7 |
| base_s13 | base_s13.log | base_s13 |
| xl_s42 | sc200_k1_XLu_s42_resume.log | sc200_k1_XLu_s42 |
| xl_s7 | xl_s7_eval.log | sc200_k1_XLu_s7 |
| mlp_dec0_s42 / _s7 | mlp_dec0_s42.log / _s7.log | mlp_dec0_s42 / _s7 |
| sym_xl_s42 / _s7 / _s13 | sym_xl_s42.log / _s7.log / _s13.log | 同名 |
| gated_sym_xl_s42 | gated_sym_xl_s42.log | gated_sym_xl_s42 |
| ortho_sym_xl_s42 / _s7 | 同名 .log | 同名 |
| base_k5_s42 | base_k5_s42.log | base_k5_s42 |
| sym_xl_k5_s42 | sym_xl_k5_s42_eval.log（训练日志 sym_xl_k5_s42.log） | sym_xl_k5_s42 |
| support_base_s42 | support_base_s42_eval.log（训练 support_base_s42.log） | support_base_s42 |
| support_sym_s42 | support_sym_s42.log | support_sym_s42 |

## 三、checkpoint 完整度

- 7 个 ckpt：base_s13, mlp_dec0_s42/s7, sym_xl_s42/s7/s13, gated_sym_xl_s42,
  ortho_sym_xl_s42/s7, support_base_s42, support_sym_s42, sc200_k1_A2_s42,
  sc200_k1_XLu_s42/s7, sc200_k1_BASEu_s7, sc200_k1_ONLINE_s42/s7/s13,
  sc200_k1_SW3_s42/s7, sc200_k1_offline, sc200_k1_offline_A/B
- 仅 last.pth：base_k5_s42, sym_xl_k5_s42, ptv3_bk, sc200_k1_regis1
- gated_sym_xl_s7：3 个（实验被主动终止）

## 四、以后新增 run 的规范

1. 日志一律 >> 追加，绝不 > 覆盖
2. 启动命令固定 save_path=./exp/scannet200/<run_name>，与日志名一致
3. 训练结束立刻跑 extract_results.py，把结果并入 results_all.json
4. 需要逐 epoch 诊断的实验显式设 save_freq=4 或更小
5. 每轮实验后重打一次日志快照（带日期后缀）
