# GFS-VL 资产索引（服务器 /data01/jh）

> 目的：上下文压缩后据此恢复「哪些产物还在、能不能删、重建要多久」。
> 生成：2026-11-22。**数字都是 `du -sh` 实测，不是估计。**
>
> 依赖等级：★★ = 当前结论直接依赖，删了要重跑实验；★ = 还可复用；无标 = 已完成的阶段产物。

---

## 〇、总量

| 位置 | 大小 | 说明 |
|---|---|---|
| `/data01/jh/data` | **486 G** | 数据、PL、特征缓存、诊断产物 |
| `/data01/jh/code/GFS-VL/exp` | **231 G** | 100 个实验的 checkpoint（每实验约 7 个 epoch × 554 M） |
| `/data01/jh/data/*.log` | 18 M | 散在 data 下的启动日志 |
| **合计** | **717 G** | |

> 旧文档 `3D_OCSB/docs/SERVER_CACHE_INDEX.md` 写的是「data 152 G / exp 35 G」，那是 **2026-11-19** 的数。
> 之后新增了 `cut2_feat` 270 G 和 60+ 个 checkpoint，**以本文件为准**。

---

## 一、数据集（★★ 不可删）

| 路径 | 大小 | 内容 |
|---|---|---|
| `data/scannet200_gfs` ★★ | 12 G | GFS 数据集：train 1201 / val 312 场景，`coord/color/normal/segment200/segment20/instance.npy` |
| `data/ScanNetpp` ★★ | 36 G | ScanNet++ 数据 |
| `data/ScanNetpp.tar.gz` | 22 G | 上述数据集的压缩包，**已解压** |
| `data/scannet_raw` | 9.9 G | ScanNet v2 原始 scans（无图像/位姿/内参） |
| `data/scannetpp_splits` | 小 | 划分 json |

---

## 二、伪标签缓存（★★ 口径命脉）

**这是最不能丢的一类。** `GFS_OFFLINE_PL` 指向哪个目录，直接决定实验口径。

| 路径 | 大小 | 内容 | 对应实验 |
|---|---|---|---|
| `data/online_pl` ★★ | 37 M | 官方在线 PS 与 PS+AI 的点级伪标签（1201 场景） | ONLINE 系列 |
| `data/online_pl_w` ★★ | 1003 M | 逐点软权重（PS `0.25+0.75σ(2z)`，AI `0.05+0.55σ(2z)`） | SW3 |
| `data/online_pl_w_uniform` ★★ | 336 M | 均匀权重版 | A2 / XLu / BASEu |
| `data/pl_k1_s10/20/30/40/50` ★★ | 各 336 M | 5 折 1-shot 的 PL（s10 = 当前所有主实验用的） | 全部 k1 实验 |
| `data/offline_pl` | 4.3 G | 旧版离线 PL（`sc200_regis_1_10{,_A,_B,_SW}` + 20..50_A） | offline / offline_A / offline_B |
| `data/online_pl_w_s20/30/40/50` | 各 1003 M | 其他 seed 的软权重 | 未用 |

> **坑**：`pl_k1_s10` 被软链到 `/dev/shm/jh_pl_k1_s10` 供训练加速。
> 重启后 `/dev/shm` 清空，必须重新 `cp -r`，否则所有 k1 实验读不到 PL。

---

## 三、特征缓存（诊断用，占用最大）

| 路径 | 大小 | 内容 | 重建 |
|---|---|---|---|
| `data/cut2_feat` | **270 G** | CUT2 阶段逐点特征 + 混淆矩阵 + 探针。子目录 `S1`（主）、`S1_ontr`、`S2`、`S2_40ep` | 无现成一键脚本，单卡 VLM 前向约数小时 |
| `data/scene_feats_cache` | **87 G** | 训练集 1201 场景逐点 VLM 特征（`feats (N,512) fp16` + `gfs`） | 单卡约 20 min/场景 |
| `data/val_sonata_cos` | 4.2 G | 312 val 逐点 45 类 Sonata 余弦（`float16[N,45]`） | `dump_val_sonata_cos.py`，2 卡 6 min |
| `data/sonata_feat_cache` | 763 M | Sonata 特征缓存 | — |
| `data/val_logits` | 492 M | 312 val 逐点 top5（ONLINE_s42） | `val_dump_logits2.py`，7 卡 2 min |
| `data/val_logits_s7` | 110 M | s7，**只跑了 73/312 场景** | 同上 |
| `data/val_logits_s13` | 32 M | s13，**只跑了 21/312 场景** | 同上 |

**这两个大头（cut2_feat 270 G + scene_feats_cache 87 G）合计 357 G，占全部资产的一半。**
它们服务的诊断（拆解 2/3、4 桶、教师vs学生）**已全部完成并写进 `DIAGNOSIS_REPORT_2026-11-22.md`**。
只有要重跑诊断才需要它们。

---

## 四、诊断产物（★ 结论直接引用）

| 路径 | 大小 | 内容 |
|---|---|---|
| `data/cm_analysis/pl_tristage_audit_s0.npz` ★★ | — | 1201 场景三阶段 PL 体检（拆解 1 的原始数据） |
| `data/cm_analysis/sym_k1/result/val_cm.npy` ★★ | — | 4 桶归因的 57×57 混淆矩阵 |
| `data/cm_analysis/teacher_train_cm.npy` ★ | — | 教师 vs 学生的训练集混淆矩阵 |
| `data/cm_analysis/det_eval/{base,sym}_{s42,s7,s13}/` ★★ | — | **确定性重评的 6 个 run.log**，跨层增益 +1.79±1.14 的唯一出处 |
| `data/cm_analysis/audit_sym/` | — | 诊断 A 的三组扫描结果（`head_audit.npz`） |
| `data/cm_analysis/det_r1` `det_r2` | — | 确定性验证（同权重两次评估 0.0000） |
| `data/ledger.json` ★ | — | 100 个实验的结构化台账 |

---

## 五、模型权重（★ 231 G）

每个实验目录 3.7 G，构成：
- `regis1_model_last.pth` — 必须留
- `epoch_3/6/9/12/15/18.pth` — **中间检查点，各 554 M，占 3.3 G**
- 少数目录有 `best.pth`

**只保留 last/best 可省约 150–170 G。**

不可删的主干权重：

| 路径 | 大小 | 用途 |
|---|---|---|
| `exp/scannet200/ptv3_bk/model/model_best.pth` ★★ | 529 M | Stage-1 PTv3 主干（frozen 系列、Sonata 迁移的起点） |
| `exp/scannet200/ptv3_bk_ontr.pth` ★★ | 529 M | ONTR 重参数化后的主干 |
| `exp/scannet200/sc200_k1_regis1/model/epoch_20.pth` ★★ | 530 M | 官方 Model Zoo 权重 |

当前主方法的 3 个 seed（`sym_lovasz_s42/s7/s13`、`base_lovasz_s42/s7/s13`）和 ScanNet++ 四局
（`scpp_*`）的 last 权重**绝对不能删**，它们是「+1.79」和「+1.82」的唯一物理载体。

---

## 六、清理候选（按风险从低到高）

| 档 | 目标 | 可回收 | 风险 |
|---|---|---|---|
| **A 安全** | 100 个实验的 `epoch_3..18.pth`（保留 last/best） | **约 150–170 G** | 只影响「逐 epoch 曲线」复查，不影响任何结论 |
| **A 安全** | `data/ScanNetpp.tar.gz`（已解压） | 22 G | 删前先校验 `ScanNetpp` 完整 |
| **B 需确认** | `data/scannet_train_v2` + `data/ScanNet`（OCSB block 缓存） | 24.5 G | GFS-VL 完全不用；OCSB 若还要跑需保留 |
| **B 需确认** | `data/ocsb_train2k` + `data/ocsb_cache` | 6.5 G | 属 OCSB 主线，不属 GFS-VL |
| **C 高风险** | `data/cut2_feat` | 270 G | 诊断已收口；重跑诊断才需要 |
| **C 高风险** | `data/scene_feats_cache` | 87 G | 同上 |

**建议顺序**：先做 A（零风险，回收近 200 G）；B/C 等论文定稿、确认不再重跑诊断后再动。

---

## 七、日志与快照

| 路径 | 内容 |
|---|---|
| `exp/<ds>/<run>/train.log` | 每个实验的完整训练日志（含全量 config dump、OOM/Traceback） |
| `exp/<ds>/<run>/test.log` | 离线重评日志（若有） |
| `/data01/jh/data/*.log` | 启动脚本的重定向输出（命名与 run 不一定一致，见 `exp_reports/ASSET_README.md` 的映射表） |
| `exp_reports/results_all.json` | 17 个 CUT2 run 的汇总指标 + 57 类 IoU |
| `exp_reports/ckpt_manifest.txt` | 175 条权重路径与大小 |
| `exp_reports/logs_snapshot_20261120.tar.gz` | 48 份日志快照（899 M，在 OCSB 仓库/网盘） |

---

## 八、跨层诊断文档索引

| 文档 | 内容 | 关键结论 |
|---|---|---|
| `docs/01_cross_layer/CUT2_EXT_REPORT.md` | cat192 有效秩、base_cross_head 逐基类范数 | 对称残差没有提升读出端有效秩 |
| `docs/01_cross_layer/OVERNIGHT_CUT2_REPORT.md` | CUT2 扩展实验汇总 | — |
| `docs/01_cross_layer/WEEK1_ZERO_TRAIN_2026-11-24.md` | 第一周零训练诊断（X1/X3/X5/X8） | `dec0` 有效秩在 9 个实验里都锁在 10–11.5；`sym` 的 dec0 跨 seed 余弦 0.98（base 0.81） |
| `docs/01_cross_layer/XL1_GRADIENT_HIGHWAY_2026-11-24.md` | **XL-1 结案：跨层增益来自反向梯度捷径** | `detach` HM 0.4089 < base 0.4241 < sym 0.4381；ep20 时 dec2 的梯度 81.4% 来自跨层支路；detach 后三层探针全最低 |
| `docs/01_cross_layer/XL2_WAVE3_AND_GEQ_2026-11-25.md` | **Wave 3 + 全局熵分位归档**：`global_entropy_q` 机制、14 个跨层消融、cross-lr 三局、两处性能优化 | **`geq02` HM 0.3475（5-shot 最强，+4.09 vs base）**；跨层消融全线未超基线；`m5_symzero` 不是 base 的空对照（结构来自 sym_xl）；transform 优化 4.5× |
| `docs/01_cross_layer/OCSB_BRIDGE_DECLINE_2026-11-26.md` | **跨层 × 3D-OCSB 桥梁判据失败补档**：B/C/Neg 定义、四层 AUC 表、阈值扫描、口径备注（dec2 维度 64/128 之争、Base 原型 placeholder） | **AUC 0.94（dec1+Base 拒斥）但 thr=0.9 时 Precision 仅 3.1–3.3%**；唯一一次跨层在 OCSB 判据上赢基线，兑不出来；XL-1 证明那 0.92 来自梯度倒灌而非前向读出 |
