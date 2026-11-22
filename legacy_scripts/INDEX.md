# 散落脚本归档索引

> 归档日期 **2026-11-22**。源目录三个，**都不是 git 仓库**（真正的仓库是它们的兄弟目录
> `code/GFS-VL/`，所以这些脚本从第一天起就在版本控制之外）：
>
> | 源目录 | 脚本数 | 归属 |
> |---|---|---|
> | `/data01/jh/code/ps_diag/` | 117 | **全部 GFS-VL**（拆解 1/2/3、4 桶、教师vs学生、ONTR、Sonata 接入） |
> | `/data01/jh/afford_diag/` | 54 | **全部 GFS-VL**（配置生成器 + 报告补丁） |
> | `/data01/jh/code/`（顶层） | 158 | GFS-VL 与 OCSB 混杂，逐个判定 |
>
> **本目录是只读副本，原处文件保留不动**（脚本内硬编码绝对路径，挪动即失效）。
> 重新归档：`python legacy_scripts/archive_legacy.py`

## 归档统计

共扫描 **366** 个脚本：GFS-VL **203** 个（已复制），OCSB **163** 个（只在清单里，物理副本在 `3D_OCSB/legacy_scripts/`）。

- `01_config_gen` 配置生成器（生成 100 个实验配置的源头）：**56** 个
- `02_upper_bound` 上限估计与 Oracle：**4** 个
- `05_sonata` Sonata 接入与探针：**19** 个
- `03_signal_gate` 门控与信号探测：**3** 个
- `04_pseudo_label` 伪标签生成与筛选：**18** 个
- `06_diagnosis` 诊断与归因：**68** 个
- `07_reproduce` 校准与复现：**6** 个
- `08_train_eval` 训练与评估入口：**9** 个
- `09_launchers` 启动与运维脚本（含 PL 口径证据）：**20** 个
- `90_other` 未归类：**0** 个

日志索引：`303` 个（`code/` 104 + `data/` 199），**只建索引不复制**。

---

## 01_config_gen 配置生成器（生成 100 个实验配置的源头）（56）

对应报告：docs/00_ledger/LEDGER.md

| 脚本 | 修改日 | 说明 |
|---|---|---|
| `gen_offline_pl.py` | 2026-11-19 | 生成点级离线伪标签（最优方案：Sonata z>=0 提纯 + G3z 补 C 池）。 输出 /data01/jh/data/offline_pl/{split}/{scene}.npy int16[N]，值为 GFS  |
| `gen_sw.py` | 2026-11-19 | 生成软权重伪标签：A 组全保留 + 每点权重 w=0.2+0.8*sigmoid(z)，小类保底 1.0。 |
| `group_experiment.py` | 2026-11-18 | 三组伪标签对照：保留 PS 已接受的结果，只考察 PS 拒绝的区域。 固定条件 -------- - 候选池 C：同一批 PS 拒绝的区域（VLM 预测为 novel 类 c，但场景级 pooled cos(u_c, p |
| `make_scpp_sym_cfg.py` | 2026-11-22 | 为 ScanNet++ 派生 sym_lovasz（跨层残差双头）配置。 从 k1/k5 的 base_lovasz 版派生，只加 4 个 RegisTrainSegmentor 参数： base_cross=True, |
| `make_sonata_s2_linear_cfg.py` | 2026-11-22 | 从实测跑出 0.4122 的 base_lovasz_s42 存档配置派生 sonata_small 版。 改动（与 Stage-1a 严格对齐）： 1. in_channels 6 -> 9 2. backbone 加 |
| `make_sonata_stage1_cfg.py` | 2026-11-21 | 从 scratch 版 Stage-1 config 派生 sonata_small 版。 改动（6 处）： 1. in_channels 6 -> 9 2. 加 embedding_type="linear"（对齐 s |
| `make_sonata_stage2_cfg.py` | 2026-11-21 | 派生 Stage-2 的 sonata_small 配置（A: enc 冻结 / B: enc 解冻）。 相对 semseg-pt-v3m1-0-gfsregistrain_k1_sym_lovasz.py： 1. in |
| `patch_det_eval.py` | 2026-11-22 | 让 shuffle_orders 只在训练时生效，eval 完全确定。 训练：保持随机（数据增强不变，与历史 bit-for-bit 一致） 评估：序列化顺序固定 → 同一 checkpoint 两次评估必须 0 误差 |
| `patch_dsv2_freeze_enc.py` | 2026-11-21 | 把 freeze_enc 补到 DefaultSegmentorV2（Stage-1 用的是这个类）。 |
| `patch_freeze_enc.py` | 2026-11-21 | 给 DefaultSegmentorV2 加 freeze_enc：只冻结 encoder，decoder 与读出头照常训练。 Stage-1a 用：sonata_small 的预训练 encoder 冻结，只训新初始化 |
| `patch_head_audit.py` | 2026-11-22 | 给 one_regis_test 加分类头体检（GFS_HEAD_AUDIT=1）： 1. 静态范数：seg_head / novel_seg_head 的每类 \|\|W_c\|\| 与 bias 2. 加性偏置扫描：logi |
| `patch_head_audit2.py` | 2026-11-22 | 诊断 A（修正版）：验证集实测尺度 + 三组互不混淆的扫描。 修正了两个数学错误： 1. base logits 含 base_cross_head（128d），不能只比 seg_head 2. \|\|W\|\|/sqrt(d |
| `patch_pl_seed.py` | 2026-11-21 | np.savez('/data01/jh/data/online_pl_shard%s%d.npz' % (args.tag, args.shard), **agg) with open('/data01/jh/data |
| `patch_save_cm.py` | 2026-11-22 | 给 GFSSemSegTester.one_regis_test 加可选的真实混淆矩阵保存（GFS_SAVE_CM=1）。 用途：把 Novel mIoU 的丢分拆成 4 个互斥桶 A: Novel -> Base（含  |
| `patch_vlm_gpu.py` | 2026-11-22 | import io P = "/data01/jh/code/ps_diag/vlm_common.py" src = io.open(P, encoding="utf-8").read() if "return_gpu |
| `add_detach.py` | 2026-11-20 | 1) 构造参数（据注释） |
| `add_lcross.py` | 2026-11-20 | 1) 构造参数（据注释） |
| `add_lnonly.py` | 2026-11-20 | (无说明) |
| `add_lovasz_la.py` | 2026-11-21 | 1) __init__ 里读配置（找到 GFS_VL_Trainer 的初始化）（据注释） |
| `add_multi_proto.py` | 2026-11-20 | 1) 初始化多原型容器（据注释） |
| `add_repdec0.py` | 2026-11-20 | 1) 构造参数（据注释） |
| `add_resume.py` | 2026-11-20 | Execute the training loop for the current registration dataset（据注释） |
| `append_cut2.py` | 2026-11-20 | 任务 5：5-shot 诊断与最终配方（2026-11-20 下午）（据注释） |
| `decide_s13.py` | 2026-11-20 | (无说明) |
| `extract_results.py` | 2026-11-20 | 从所有训练日志提取最终结果（汇总 + 57 类），固化到 results_all.json 即使日志后来被删/覆盖，结果仍可重建。 |
| `fix_attrib.py` | 2026-11-20 | (无说明) |
| `fix_chunk_cos.py` | 2026-11-20 | (M,45,512) 广播会撑爆显存。分块不改变结果。（据注释） |
| `fix_genpl.py` | 2026-11-20 | 等价于 default_setup 的派生字段（脚本不走 launch）（据注释） |
| `fix_genpl2.py` | 2026-11-20 | 在 import 区补 point_collate_fn（据注释） |
| `fix_genpl3.py` | 2026-11-20 | (无说明) |
| `fix_review.py` | 2026-11-20 | ① L_cross 与主 loss 共用加权口径（据注释） |
| `gen_ab_configs.py` | 2026-11-20 | ---------- 实验 A: detach + 旧头降 LR ----------（据注释） |
| `gen_batch2.py` | 2026-11-21 | (无说明) |
| `gen_k1_lcross.py` | 2026-11-20 | (无说明) |
| `gen_k1_lcross_cos.py` | 2026-11-20 | (无说明) |
| `gen_k1cos.py` | 2026-11-20 | (无说明) |
| `gen_k5_102400.py` | 2026-11-21 | (无说明) |
| `gen_k5_lovasz.py` | 2026-11-21 | (无说明) |
| `gen_lcross01.py` | 2026-11-21 | (无说明) |
| `gen_lcross_cfg.py` | 2026-11-20 | 模型里启用辅助损失（据注释） |
| `gen_lcross_cos.py` | 2026-11-20 | (无说明) |
| `gen_lnonly.py` | 2026-11-20 | (无说明) |
| `gen_rep3.py` | 2026-11-20 | (无说明) |
| `gen_rep3_support.py` | 2026-11-20 | (无说明) |
| `mem_budget.py` | 2026-11-20 | (无说明) |
| `patch_erank.py` | 2026-11-21 | (无说明) |
| `patch_k5.py` | 2026-11-20 | 配对 Δ（sym - base）（据注释） |
| `patch_logit_scale.py` | 2026-11-20 | 推理期 logit 校准：可对 novel 段整体缩放，（据注释） |
| `patch_record_save.py` | 2026-11-20 | 校验块边界（torch.save 到 ) 共 8 行）（据注释） |
| `patch_report.py` | 2026-11-20 | 结果（312 val，单卡）（据注释） |
| `patch_report_7cls.py` | 2026-11-21 | (无说明) |
| `patch_report_final.py` | 2026-11-21 | (无说明) |
| `patch_report_lcross.py` | 2026-11-21 | 3.5 四种"修 Base"机制的统一性对照（据注释） |
| `patch_support.py` | 2026-11-20 | 任务 4：纯 3D Support-Only（无 VLM 伪标签）（据注释） |
| `patch_support2.py` | 2026-11-20 | 模块的独立价值：三档监督强度（据注释） |
| `update_report.py` | 2026-11-20 | (无说明) |

## 02_upper_bound 上限估计与 Oracle（4）

对应报告：docs/06_diagnostics/DIAGNOSIS_REPORT_2026-11-22.md §3

| 脚本 | 修改日 | 说明 |
|---|---|---|
| `conflict_oracle.py` | 2026-11-19 | Step 2：局部 3D 冲突区「完美仲裁」的 Oracle 上限。 对 312 val： 1. 用 top-2 概率边际定义冲突点（margin < tau） 2. 分成两类冲突：Base–Novel（top1/top |
| `instance_bound.py` | 2026-11-18 | 细节 2：几何实例边界约束 AI 补标范围的上限对照。 两种容器边界： A. 真实实例边界  instance.npy 的每个 instance id B. 无监督边界    region_cache 里「预测 nove |
| `oracle_cand.py` | 2026-11-19 | 候选集受限的 Oracle：仲裁器只能在 top-k 候选里选时，上限是多少。 对比三种： oracle_all   任意类（此前的 +4.5pp 就是这个） oracle_top2  仅 top-1/top-2 出现过 |
| `oracle_cand2.py` | 2026-11-19 | 候选集受限的 Oracle：仲裁器只能在 top-k 候选里选时，上限是多少。 对比三种： oracle_all   任意类（此前的 +4.5pp 就是这个） oracle_top2  仅 top-1/top-2 出现过 |

## 05_sonata Sonata 接入与探针（19）

对应报告：docs/05_sonata/SONATA_SMALL_INTEGRATION.md

| 脚本 | 修改日 | 说明 |
|---|---|---|
| `build_sonata_base.py` | 2026-11-19 | 用 Sonata 池化 base 预测区的区域特征（AI 新增点的主要来源）。 与 build_sonata_region.py 完全相同的特征口径，只是输入换成 base_index （VLM 判成 12 个基类之一的 |
| `build_sonata_region.py` | 2026-11-18 | 细节 4 第二步：用 Sonata 纯几何特征池化同一批区域（sonata 环境运行）。 输入  /data01/jh/data/region_index/regions_index_{split}_s{N}.pt 输出 |
| `build_support_protos_sonata.py` | 2026-11-18 | Sonata 视角的 support 原型（sonata 环境运行）。 与 VLM 版 (build_support_protos.py) 完全相同的 split 与 GT 选点逻辑，只是把 特征换成 Sonata 的  |
| `convert_sonata_ckpt.py` | 2026-11-21 | 把 sonata_small.pth 转成 GFS-VL 的 checkpoint 格式。 sonata 的 key:  embedding.* / enc.* 我们 DefaultSegmentorV2 期望: bac |
| `dump_val_sonata_cos.py` | 2026-11-19 | 为 312 val 场景预计算逐点 Sonata 对 45 个新类原型的余弦（仲裁用）。 输出 /data01/jh/data/val_sonata_cos/{scene}_cos.npy  float16[N,45]  |
| `sonata_1shot.py` | 2026-11-21 | 格 1 真 1-shot：冻结 Sonata 1232d 特征 + 57 类线性头。 口径（与 GFS-VL 1-shot 严格一致）： Base 12 类  : 1201 个训练场景的全部 Base GT（每场景每类最 |
| `sonata_gate.py` | 2026-11-21 | Sonata-Small -> PTv3 权重注入 + 1 场景特征一致性验收。 gate: 同一份 point dict 喂两个模型，enc4 输出 max_err < 1e-4 才允许开训。 为什么要这个 gate： |
| `sonata_gate2.py` | 2026-11-21 | 逐层对比 sonata 官方实现与我们的 PTv3，定位第一处数值偏离。 |
| `sonata_gate3.py` | 2026-11-21 | 逐层对比 sonata 官方实现与我们的 PTv3，定位第一处数值偏离。 |
| `sonata_gate4.py` | 2026-11-21 | 验证 hypothesis：enc1+ 的误差只是体素行排列不同（Z-order vs 字典序）， 用 pooling_inverse 展开回原始点顺序后两边应该逐位一致。 |
| `sonata_lp.py` | 2026-11-21 | 格 1：Sonata 冻结特征的 57 类线性探针。 与 cut2_variants_lp.py 同口径： - 按类均匀采样，每类上限 CAP=2000 点 - 线性头（无隐藏层），AdamW(lr=1e-3, wd=5 |
| `sonata_lp2.py` | 2026-11-21 | 逐场景前向 -> 1232 维全分辨率特征 -> 每类最多 per_scene 点。 |
| `sonata_lp3.py` | 2026-11-21 | 返回 (N, 1232) 的 GPU 张量，全程不落 CPU。 沿 pooling_parent 链逐层 cat，最后用 base_inv 映回原始点数。 |
| `sonata_probe.py` | 2026-11-21 | 每层 GridPooling 存 pooling_inverse，用于逐层反池化。 |
| `sonata_proto.py` | 2026-11-21 | 纯 3D 1-shot 机理诊断：验证死结 1（1232 维无约束线性拟合过拟合）。 对照 sonata_1shot.py 的 0.1594（40 轮 AdamW + wcls 加权 + 无约束线性头）： 本脚本不改数据 |
| `sonata_proto2.py` | 2026-11-21 | 纯 3D 1-shot 机理诊断 v2：一次前向，三个分类器，带特征缓存。 v1（sonata_proto.py）的 bug： base 线性头与 novel 余弦分支只对齐了 std，没对齐 mean。 min(12) |
| `sonata_proto3.py` | 2026-11-21 | 可学习 L2 归一化余弦分类头。返回 (mu, sd, Wn)，Wn 已按列归一化 (D, 12)。 |
| `sonata_rand.py` | 2026-11-21 | 排查：eval 模式下我们的模型两次前向是否完全一致（drop_path / shuffle_orders / traceable）。 |
| `run_sonata_s2_batch.sh` | 2026-11-22 | !/bin/bash（据注释） |

## 03_signal_gate 门控与信号探测（3）

对应报告：docs/04_ontr_frozen/DAYTIME_ONTR_REPORT*.md

| 脚本 | 修改日 | 说明 |
|---|---|---|
| `analyze_gate.py` | 2026-11-20 | (无说明) |
| `retry_gated_s7.sh` | 2026-11-20 | !/bin/bash（据注释） |
| `retry_gated_s7_v2.sh` | 2026-11-20 | !/bin/bash（据注释） |

## 04_pseudo_label 伪标签生成与筛选（18）

对应报告：docs/03_pseudo_label/margin_filter_experiment.md

| 脚本 | 修改日 | 说明 |
|---|---|---|
| `b_pool_purity.py` | 2026-11-18 | 诊断：用基类负例增量 + Sonata 跨视角，对脏 B 池（PS keep 区域）排序提纯。 对每个 novel 类 c，B 池 = PS keep 的区域（pred == c）。这批区域的 GT 精度 已经知道很低（ |
| `build_base_cache.py` | 2026-11-18 | 缓存「VLM 预测为基类」的几何区域（pred < 12）。 为什么不是 pred == -1：GFS-VL 的 VLM 输出是 57 类 argmax，**从不输出 -1**。 -1 是 PS 后处理把 base 类和 |
| `build_miss_cache.py` | 2026-11-18 | 缓存 VLM 漏检区（pred == -1）的几何区域特征。 region_cache 只收「预测为 novel 类」的区域，覆盖不到 PS 拒绝池之外的空白区。 这里补上 pred == -1 的部分：这些点 VLM  |
| `build_pl_weights.py` | 2026-11-21 | 生成最终点级伪标签 + Sonata 软权重。 输入 /data01/jh/data/online_pl/{scene}.npz      ps / psai（官方在线 PS、PS+AI） /data01/jh/data |
| `build_pp_score.py` | 2026-11-19 | 逐点 Sonata 打分：score_i = cos(f_sonata[i], proto[psai_i])，供软权重使用。 输出 /data01/jh/data/pp_score/{scene}_s.npy  floa |
| `build_region_cache.py` | 2026-11-18 | 缓存训练场景的 VLM 预测区域特征（三组伪标签对照的底座）。 对每个训练场景： 1. vlm_forward -> 每点 512 维特征 + 预测类（GFS 空间 0-56） 2. 对每个 novel 预测类（>=12 |
| `build_w_pp.py` | 2026-11-19 | 用【逐点】Sonata 分数生成软权重，并分 PS / AI 两组做五等分精度校验。 权重区间（按建议） PS keep : w = 0.25 + 0.75 * sigmoid(2z)      ∈ [0.25, 1.0 |
| `build_w_simple.py` | 2026-11-19 | 把 online_pl_w 的软权重改写成「来源两级权重」：PS keep=1.0，AI/补标=0.45。 依据（实测，同一 Y_b==-1 口径）： PS keep 41.4M 点 精度 0.404 AI 新增 16. |
| `build_w_uniform.py` | 2026-11-21 | 建「均匀权重」目录：_pl.npy 软链接自 online_pl_w，_w.npy 全 1.0。 用途：A2 = 同一份 PL（PS+AI+补标）但权重全 1，用来把 SW3 的效果拆成 ① ONLINE -> A2   |
| `dump_base_index.py` | 2026-11-19 | 导出 base 池（pred < 12）的区域点索引，供 AI 补标写入点级伪标签。 |
| `dump_online_pl.py` | 2026-11-21 | 逐场景 dump 官方原版在线 PS + 逐点 AI 的点级伪标签。 口径（对齐 pointcept/engines/train.py 的 target_calibrate） ---------------------- |
| `dump_online_pl_1shot_cmp.py` | 2026-11-20 | 逐场景 dump 官方原版在线 PS + 逐点 AI 的点级伪标签。 口径（对齐 pointcept/engines/train.py 的 target_calibrate） ---------------------- |
| `dump_online_pl_5shot.py` | 2026-11-20 | 逐场景 dump 官方原版在线 PS + 逐点 AI 的点级伪标签。 口径（对齐 pointcept/engines/train.py 的 target_calibrate） ---------------------- |
| `dump_region_index.py` | 2026-11-18 | 细节 4 第一步：导出每个训练场景的「点级 VLM 预测 + 区域点索引」。 跨视角对照要求 Sonata 和 VLM 用**完全相同的区域边界**。区域缓存只有区域级 pred，没有点级 pred，所以这里重跑一遍 V |
| `full_pipeline.py` | 2026-11-19 | 完整伪标签管线：统一 G_c_full 口径，一次跑完所有消融。 方案 ---- ps_full     PS keep 全部区域（原版 PS 输出） ps_ai       原版 PS + AI：对 pred==-1  |
| `genB_full.py` | 2026-11-19 | 把 AI 在 base 池的补标写入 B 组点级伪标签。 |
| `per_class_pts.py` | 2026-11-19 | 逐类点数对比：A 的离线伪标签 vs PS keep vs base 池 AI 补标。 输出 ps_full[45]    PS keep 全部区域展开的点数（按 VLM 判的类计） base_ai[45]    bas |
| `pseudo_miou.py` | 2026-11-18 | 伪标签层的 mIoU-N：原版 PS keep vs B 池提纯 vs + 补标。 支持两套打分视角 ---------------- vlm    原型/桥梁池相似度 - 基类负例（512 维 CLIP 空间） son |

## 06_diagnosis 诊断与归因（68）

对应报告：docs/06_diagnostics/DIAGNOSIS_REPORT_2026-11-22.md

| 脚本 | 修改日 | 说明 |
|---|---|---|
| `ai_auc.py` | 2026-11-19 | 测 Sonata 原型分在【AI 新增点】上的逐类宏平均 AUC（对应你问的 0.83 是否真实）。 |
| `arbiter_eval.py` | 2026-11-19 | Step 3：纯 3D 局部仲裁器（novel–novel），一次扫多组 alpha/margin。 对每个冲突连通分量（margin<tau 且 top1/top2 都是新类且不同）： 候选类 = 分量内 top-1/ |
| `arbiter_pt.py` | 2026-11-19 | 逐点仲裁：在冲突点上，用「逐点」候选分数选类，而不是分量均值。 对每个点 p（属于 novel–novel 冲突分量）： 候选 = 该点 top-k 的类 score_c(p) = alpha * prob_c(p) + |
| `arbiter_v2.py` | 2026-11-19 | 仲裁器 v2（novel–novel）：候选 top-k，多种聚合方式，一次扫完。 对每个冲突分量（margin<tau，top1/top2 同为不同新类）： 候选类 = 分量内 top-k 出现过的类 分数 = alp |
| `arbiter_v3.py` | 2026-11-19 | 仲裁器 v2（novel–novel）：候选 top-k，多种聚合方式，一次扫完。 对每个冲突分量（margin<tau，top1/top2 同为不同新类）： 候选类 = 分量内 top-k 出现过的类 分数 = alp |
| `arbiter_v4.py` | 2026-11-19 | 仲裁器 v4：跨类比较前先做【按类 z-score 校准】。 之前的 argmax 直接比原始 cos，被类级偏置主导（某些类原型天生余弦高）。 这里先用全部 val 点统计每类的 mean/std，把 cos 变成类内 |
| `arbiter_v5.py` | 2026-11-19 | 仲裁器 v4：跨类比较前先做【按类 z-score 校准】。 之前的 argmax 直接比原始 cos，被类级偏置主导（某些类原型天生余弦高）。 这里先用全部 val 点统计每类的 mean/std，把 cos 变成类内 |
| `audit_arbiter.py` | 2026-11-19 | 逐分量审计：Sonata argmax 在「可改判分量」上到底有多准。 对 τ=0.2、novel–novel 冲突： correct_now   = 分量当前多数预测 == GT 多数类 switchable    = |
| `audit_arbiter2.py` | 2026-11-19 | 逐分量审计：Sonata argmax 在「可改判分量」上到底有多准。 对 τ=0.2、novel–novel 冲突： correct_now   = 分量当前多数预测 == GT 多数类 switchable    = |
| `base_experiment.py` | 2026-11-19 | 第 2 点：在「VLM 判成基类」的漏检区上重跑三组补标对照。 候选池来自 base_cache：VLM 把这些点判成了 12 个基类之一（pred < 12）， 所以它们既不在 PS 拒绝池里，也没有 novel 标签 |
| `bd_transfer.py` | 2026-11-19 | 百度网盘分享：按场景只转存 scans/ 子目录。 接口要点（2026-11 实测） * 分享页改版，baidupcs-py 的 shared_paths() 解析不到 yunData.setData * share/l |
| `bridge_holdout.py` | 2026-11-19 | 留出集检验：提纯多实例原型 P~ 是否比 1-shot 支持原型 p 更能排序区域。 做法 用奇数场景的区域（按支持原型挑 top10%）构造 P~_c， 在偶数场景的区域上比较 AUC（正例 = GT 该类占比 >=  |
| `bridge_precheck.py` | 2026-11-19 | 零 GPU 预检：用缓存区域特征构造「提纯多实例原型」，看与 1-shot 支持原型差多少。 |
| `cm_bucket_analysis.py` | 2026-11-22 | import sys import numpy as np NC_BASE = 12 def main(): path = sys.argv[1] nr_base = int(sys.argv[2]) if len(sy |
| `cmp_classes.py` | 2026-11-19 | (无说明) |
| `cmp_online_pl.py` | 2026-11-19 | Task 1 汇总：官方在线 PS / PS+逐点 AI / 我们 A 的逐类点数与伪标签质量。 全部用同一口径：只统计写入 Y_b == -1 的点（训练时 dataset 唯一会填的位置）。 输入 /data01/j |
| `cut2_analyze.py` | 2026-11-19 | Cut 2 汇总：合并各分片 CM，打印分层指标总表。 用法: python cut2_analyze.py [tag ...] |
| `cut2_audit1.py` | 2026-11-19 | 疑点 1a：各层特征在 val 上的 L2 范数量级。 判据：若 dec3/enc4 的范数是 dec0 的 3 倍以上，multi 的裸 concat 就会被高维粗糙层主导。 |
| `cut2_audit2.py` | 2026-11-19 | 疑点 2：ptv3_bk 原生 12 类 seg_head 在 312 val 上的 mIoU-B。 用来交叉验证「S1 dec0 Linear Probe 只有 0.5136」是不是均匀采样造成的。 S1 的 nove |
| `cut2_blindA.py` | 2026-11-19 | 盲区 A：δ 反向扫描（给新类 logits 加 δ，δ<0 = 压新类/抬基类）。 用 val_logits 的 top-5：跨类型 argmax 只需比较「最高的 base」与「最高的 novel」， 这两者必在 t |
| `cut2_blindB.py` | 2026-11-19 | 盲区 B：Cut 1 父子类吞噬体检 + 全点无约束 Top-k Oracle。 用法: python cut2_blindB.py [pred_dir] pred_dir 默认 /data01/jh/data/val_ |
| `cut2_common.py` | 2026-11-21 | Cut 2 公共库：PTv3 逐层特征提取（精确 unpool 映射）+ 模型/数据构建。 层位与分辨率编码（start_level）： enc4  1/16 -> 4 dec3  1/8  -> 3 dec2  1/4 |
| `cut2_cosine.py` | 2026-11-19 | 盲区 A 补：Cosine Classifier / 权重归一化对 Base→Novel 假阳性的影响。 用法: python cut2_cosine.py <mode> [shard] [shards] mode =  |
| `cut2_eval.py` | 2026-11-19 | Cut 2 阶段三：用训练好的 Linear Probe 在 312 val 上流式累计混淆矩阵。 用法: python cut2_eval.py <S1\|S2> [shard] [shards] 输出 /data01/ |
| `cut2_extract.py` | 2026-11-21 | Cut 2 阶段一：从指定 checkpoint 提取分层特征。 用法: python cut2_extract.py <S1\|S2> train   [shard] [shards] python cut2_extra |
| `cut2_oneshot.py` | 2026-11-19 | Cut 2 阶段三补：1-shot 支持原型 Probe（免训练）。 对每层： proto[c] = L2normalize(mean(feat[gt==c]))，c 取 45 个新类 在 312 val 的新类真值点上 |
| `cut2_train.py` | 2026-11-19 | Cut 2 阶段二：训练 6 组 Linear Probe，并计算 eRank / Base 子空间能量。 用法: python cut2_train.py <S1\|S2> [epochs] 6 组 probe: enc |
| `cut2_variants.py` | 2026-11-19 | 疑点 1b/1c：多读出变体 × 采样方式，312 val 一次评估。 变体： dec0_u       dec0 + 均匀采样（每类 2000） dec0_n       dec0 + 自然分布采样 dec0_lp   |
| `cut2_variants_lp.py` | 2026-11-21 | 补测：log-prior 校准对 multi / dec012 / dec01 是否同样有效。 只跑 shard0（104 场景），与 cut2_variants.py 的 dec0_u/dec0_lp 同口径对比。 输 |
| `cut2_verify.py` | 2026-11-19 | Cut 2 前置验证：确认 5 层 hook 与 pooling_inverse 链完全正确。 判据（全部必须通过）： V1  enc4/dec3/dec2/dec1/dec0 特征长度 == 对应层级的点数 V2  用 |
| `fisher_vlm512.py` | 2026-11-22 | 拆解 2 第一把尺子：512d VLM 特征的 Fisher 判别比 + 三阶段逐类明细。 输入 pl_tristage_audit_s0.npz（含 feat_sum/feat_sq/feat_cnt 与三阶段混淆矩阵 |
| `miss_experiment.py` | 2026-11-18 | 第 2 点：在真实待填充区（pred == -1 的漏检区）上重跑三组补标对照。 与 group_experiment 的区别 -------------------------- - 候选池来自 miss_cache： |
| `obj_diag.py` | 2026-11-19 | 把伪标签误差拆成「objectness（是不是物体）」与「class identity（是不是该类）」。 对每个 PL 点，用逐点 Sonata 分数（= cos 到其被赋类原型）做两个二分类： objectness:  |
| `ontr_step_a.py` | 2026-11-21 | Track 2 Step A：预计算正交零空间基与 dec2 目标 PCA 基。 产出 /data01/jh/data/ontr_basis.pt Q_perp (64, 52)   基类分类头 W_b 的精确正交零空间 |
| `ontr_step_b.py` | 2026-11-21 | 特征的有效秩：exp(谱熵)，x 为 (N, D)。 |
| `ontr_step_b_v2.py` | 2026-11-21 | 特征的有效秩：exp(谱熵)，x 为 (N, D)。 |
| `ontr_step_c.py` | 2026-11-21 | Track 2 Step C 体检：Base mIoU + dec0 的 eRank（新类点上）。 用法 python ontr_step_c.py --weight <ckpt> --tag <名字> 口径（与训练 v |
| `perpoint_check.py` | 2026-11-19 | 逐点 Sonata 打分 + 假阳性成分分析（抽样 30 场景）。 回答两个问题： 1. PS / AI 的假阳性点里，真值分别是 base(0-11) / novel(12-56) / ignore(-1) 各占多少  |
| `pl_tristage_audit.py` | 2026-11-22 | 全场景统计：把 PL>=12 的预测与该点 GT 比较；gfs==-1 的点若被预测成 novel 就是 FP。 |
| `pl_tristage_audit_gpu.py` | 2026-11-22 | 拆解 1+2（全 GPU 版）：VLM 伪标签三阶段全场景体检 + 512d 特征 Fisher 矩。 相比 CPU 版的关键改动： - vlm_forward(return_gpu=True)，特征全程不落 CPU（避 |
| `probe_omission.py` | 2026-11-20 | Omission Recovery probe. Question: for points that RegionPLC's PS missed (ps == -1) but whose true GT is a nov |
| `probe_rlp_auc.py` | 2026-11-20 | RegionPLC baseline AUC for the omission-recovery probe. Same B/C/Neg split as probe_omission.py, but similarit |
| `proto_separability.py` | 2026-11-22 | 返回每层的原型、半径、对每对的 Fisher 比与自测混淆率。 |
| `ps_diag.py` | 2026-11-18 | PS/AI diagnosis for GFS-VL (RegionPLC 636 + ScanNet200). Reproduces GFS-VL's Pseudo-label Selection (PS) and A |
| `task2_offline.py` | 2026-11-19 | Task 2：在【真实在线 PS+AI 底盘】上验证两个改动。 (a) Sonata + D_base 对真实底盘写入点的排序区分度 - 用 PS 保留区域（kept）做评估集：纯度 = hist_n[c] / (siz |
| `task2b_auc.py` | 2026-11-19 | 补充：Sonata+D_base 的【逐类】区分度（宏平均 AUC）与 G3z 逐类精度。 |
| `task2c_scores.py` | 2026-11-19 | 对比多种候选分数在 keep 区域上的逐类 AUC（判断软权重是否可行）。 |
| `task2d_g3z_proto.py` | 2026-11-19 | 用 Sonata 原型相似度（而非桥梁池）做 G3z 选点，对比 mIoU-N。 |
| `teacher_vs_student.py` | 2026-11-22 | 遍历 1201 训练场景，用 VLM 伪标签 vs GT 累积 57x57 矩阵。 |
| `val_cls_count.py` | 2026-11-19 | 统计 312 val 场景里每个 novel 类的 GT 点数，并和逐类 IoU 对照。 目的：判断 A/B 相对基线的跌幅是集中在「点极少的小类」（IoU 不稳定/噪声） 还是「点不少的类」（真实退化）。 |
| `val_cm_probe.py` | 2026-11-19 | 零训练剖析：312 val 混淆矩阵拆分 + 基类 logit 偏置 δ 扫描。 用法: python val_cm_probe.py <exp_config.py> [weight] 输出 ① 45 个新类真值点的错误 |
| `val_cm_probe2.py` | 2026-11-19 | 零训练剖析 v2：按 tester 的【分片推理】跑 312 val，输出混淆矩阵拆分 + δ 扫描。 用法: python val_cm_probe2.py <exp_config.py> <weight.pth> |
| `val_dump_logits.py` | 2026-11-19 | Step 1：保存 312 val 的逐点 top-5 概率/索引/预测/GT，供冲突区仲裁分析。 按 tester 的分片推理；输出 /data01/jh/data/val_logits/{scene}.npz top |
| `val_dump_logits2.py` | 2026-11-19 | Step 1：保存 312 val 的逐点 top-5 概率/索引/预测/GT（支持分片）。 用法: python val_dump_logits2.py <config.py> <weight.pth> [out_di |
| `val_dump_logits3.py` | 2026-11-19 | Step 1：保存 312 val 的逐点 top-5 概率/索引/预测/GT（支持分片）。 用法: python val_dump_logits2.py <config.py> <weight.pth> [out_di |
| `vlm_common.py` | 2026-11-22 | Shared setup for the GFS-VL 3D VLM (RegionPLC, sparseunet32_636). Builds the text embedding (CLIP ViT-B/32, ls |
| `vlm_val_probe.py` | 2026-11-22 | 90 秒摸底：冻结 VLM_3D 在 312 个验证场景上的真实 Novel mIoU。 目的：区分两种情况 情况 1（val Novel >= 0.40）：信息在教师里，损耗在 PS+AI 硬标签压缩 + Stage- |
| `cmp_norms.py` | 2026-11-20 | (无说明) |
| `pr_analysis.py` | 2026-11-20 | 从 val.pth 的逐场景 inter/union/target 计算每类 Precision / Recall / IoU。 定义（与 intersection_and_union 一致）： I = TP, targ |
| `pr_compare.py` | 2026-11-20 | (无说明) |
| `quick_pr.py` | 2026-11-20 | (无说明) |
| `task1_merge.py` | 2026-11-20 | (无说明) |
| `task1_sym.py` | 2026-11-20 | (无说明) |
| `weight_norms.py` | 2026-11-20 | (无说明) |
| `xl_compare.py` | 2026-11-20 | 解析 4 个 run 的逐类 IoU，输出 Base/Novel/HM 与逐类 delta。 用法: python xl_compare.py [log_dir] |
| `xl_erank.py` | 2026-11-21 | 训练后诊断：读出支路 eRank + novel_head 三切片范数占比。 用法: python xl_erank.py <run> [shard shards] run: base_s42 / xl_s42 / ba |
| `xl_erank_merge.py` | 2026-11-20 | 合并 eRank 分片并计算有效秩。 |
| `run_ontr_stage2.sh` | 2026-11-21 | !/bin/bash（据注释） |

## 07_reproduce 校准与复现（6）

对应报告：docs/legacy/SESSION_STATE.md

| 脚本 | 修改日 | 说明 |
|---|---|---|
| `check_weights.py` | 2026-11-19 | 校验软权重：权重大小是否与伪标签真实精度单调相关。 |
| `flash_diff_test.py` | 2026-11-22 | 对比 enable_flash=True/False 的数值差异（同权重、同数据、同 patch 配置）。 关键：PTv3 在 enable_flash=False 时会把 patch_size 动态设为 min(bat |
| `gpu_watch.py` | 2026-11-19 | GPU 占用记录器：每 60s 追加一行，用于摸清 lwh 等共享用户的周期性评估峰值。 输出格式（制表符分隔） 时间 \| g0_used/g0_util ... g7_used/g7_util \| pid:user:m |
| `retry_err.py` | 2026-11-19 | 重试 transfer 超时（errno=4）的场景。 只处理 bd_transfer.log 里 ERR 行对应的场景：重新取 scans 的 fs_id， transfer 到 /scannetpp/<scene>（ |
| `verify_cross.py` | 2026-11-19 | 启动前验证：cross_layers 分支的 shape / 尺度 / 参数分组 / 梯度 / 缓存。 |
| `verify_variants.py` | 2026-11-20 | Pre-flight 校验：mlp / sym / gated 三种分支。 |

## 08_train_eval 训练与评估入口（9）

对应报告：docs/07_shot_protocol/OVERNIGHT_ICCV_REPORT.md

| 脚本 | 修改日 | 说明 |
|---|---|---|
| `build_support_protos.py` | 2026-11-18 | 计算 10 套 1-shot 划分的 support 原型（45 个 novel 类，512 维）。 带场景特征缓存，避免同一场景在多个 split 上重复前向。 |
| `run_sw3.sh` | 2026-11-19 | !/bin/bash（据注释） |
| `run_sw_s42.sh` | 2026-11-19 | !/bin/bash（据注释） |
| `train_guard.sh` | 2026-11-19 | !/bin/bash（据注释） |
| `run_5split.sh` | 2026-11-21 | !/bin/bash（据注释） |
| `run_scpp_batch.sh` | 2026-11-22 | !/bin/bash（据注释） |
| `run_support_base.sh` | 2026-11-20 | !/bin/bash（据注释） |
| `run_xl.sh` | 2026-11-19 | !/bin/bash（据注释） |
| `assemble_scannet200.py` | 2026-11-18 | 从服务器现有 ply + 本地传来的 segment200 组装 pointcept 格式的 ScanNet200。 点序已验证：本地 coord.npy 与服务器 ply 顶点逐点一致（81,369 点，首点相同）。  |

## 09_launchers 启动与运维脚本（含 PL 口径证据）（20）

对应报告：docs/legacy/SESSION_STATE.md（**含 GFS_OFFLINE_PL 口径证据**）

| 脚本 | 修改日 | 说明 |
|---|---|---|
| `dl_scannetpp.sh` | 2026-11-19 | !/bin/bash（据注释） |
| `dl_scannetpp_v2.sh` | 2026-11-19 | !/bin/bash（据注释） |
| `dl_scannetpp_gfs.sh` | 2026-11-21 | !/bin/bash（据注释） |
| `dl_scannetpp_parallel.sh` | 2026-11-21 | !/bin/bash（据注释） |
| `dl_scanpp_parallel.sh` | 2026-11-22 | !/bin/bash（据注释） |
| `guard_support_sym_eval.sh` | 2026-11-20 | !/bin/bash（据注释） |
| `guard_xl_s7.sh` | 2026-11-20 | !/bin/bash（据注释） |
| `overnight_cut2.sh` | 2026-11-20 | !/bin/bash（据注释） |
| `overnight_watchdog.sh` | 2026-11-20 | !/bin/bash（据注释） |
| `pl_pipeline_seed.sh` | 2026-11-21 | !/bin/bash（据注释） |
| `pl_pipeline_shot.sh` | 2026-11-21 | !/bin/bash（据注释） |
| `reeval_deterministic.sh` | 2026-11-22 | !/bin/bash（据注释） |
| `report_watcher.sh` | 2026-11-20 | !/bin/bash（据注释） |
| `resume_xl_s42.sh` | 2026-11-19 | !/bin/bash（据注释） |
| `retry_failed.sh` | 2026-11-20 | !/bin/bash（据注释） |
| `retry_support_sym.sh` | 2026-11-20 | !/bin/bash（据注释） |
| `retry_sym_k5.sh` | 2026-11-20 | !/bin/bash（据注释） |
| `start_xl_s7_after_base.sh` | 2026-11-19 | !/bin/bash（据注释） |
| `watch_and_start_xl_s7.sh` | 2026-11-19 | !/bin/bash（据注释） |
| `install_deps.sh` | 2026-11-16 | !/bin/bash（据注释） |

## OCSB 主线（163，已分流到 3D_OCSB 仓库）

这些脚本属于 3D 可供性 / 桥接 / 环实验主线，**不在本仓库保存副本**。
物理副本：`experiment/3D_OCSB/legacy_scripts/code_ocsb/`。

`adaptive_signal.py`、`analyze_cal.py`、`analyze_diag.py`、`analyze_full.py`、`analyze_overnight.py`、`analyze_paired.py`、`analyze_task6.py`、`analyze_toplk_oracle.py`、`auc_signals.py`、`audit_merge.py`、`audit_stage2.py`、`audit_u_infer.py`、`audit_u_merge.py`、`block_coord_probe.py`、`build_intra_cache.py`、`build_ocsb_cache.py`、`build_train2k.py`、`build_u_cache.py`、`check.py`、`check_coseg.py`、`check_pair.py`、`clip_vit.py`、`cohens_topk.py`、`collect_train_blocks.py`、`combo_signals.py`、`compare_u_mode.py`、`control_u_confound.py`、`coseg_mem_probe.py`、`coseg_su.py`、`cross_block_probe.py`、`death_test.py`、`diag_bc.py`、`diag_cache.py`、`diag_cmc.py`、`diag_signals.py`、`diag_sonata2.py`、`diag_step2.py`、`dump_for_sonata.py`、`eval_baseline.py`、`export_counter_by_block.py`、`export_counter_gt.py`、`export_ply2.py`、`export_room_raw.py`、`export_room_scene.py`、`extract_orth_feats.py`、`fake_tp.py`、`gate_probe.py`、`group_experiment.py`、`local_geometry.py`、`make_class2scans.py`、`make_manifest.py`、`ocsb_advanced.py`、`ocsb_cmc.py`、`ocsb_core.py`、`ocsb_faithful.py`、`ocsb_v3.py`、`official_compare.py`、`or_gate2.py`、`perclass_overnight.py`、`phase1_smooth.py`、`phase2_adjacent.py`、`phase2_clip.py`、`phase2_full.py`、`phase2_orth.py`、`phase2_v2.py`、`phase3_calib.py`、`phase3_erode3d.py`、`phase3_full.py`、`phase3_gate.py`、`phase3_v2.py`、`phase4_p2_perclass.py`、`phase4_report.py`、`phase4_v2.py`、`phase4_v3.py`、`phase4_veto.py`、`post_ub560.py`、`post_ub_official.py`、`post_v3.py`、`probe_agg.py`、`probe_calib.py`、`probe_fp.py`、`probe_gap.py`、`probe_gap2.py`、`probe_layers.py`、`probe_repro.py`、`probe_retrieval.py`、`probe_retrieval_merge.py`、`purity_analysis.py`、`purity_fix_probe.py`、`query_gate.py`、`rebuild_meta.py`、`recompute_filtered.py`、`repro_probe.py`、`repro_probe2.py`、`ring1_balance.py`、`ring2_check.py`、`ring2_export_ply.py`、`ring2_physical.py`、`ring2b_fixed.py`、`ring2b_unannotated.py`、`ring2c_full.py`、`ring3_purify.py`、`ring4_isolated.py`、`ring4_overlap.py`、`ring4_report.py`、`ring4_ucover.py`、`run_ablation.py`、`run_cmc.py`、`run_cmc_s2.py`、`run_official_eval.py`、`run_s2.py`、`run_s34.py`、`same_scene_distance.py`、`scan_block_classes.py`、`scene_eval.py`、`scene_merge.py`、`scene_sampler_probe.py`、`selfcheck_4_4.py`、`sonata_auc_merge.py`、`sonata_b_auc.py`、`sonata_check.py`、`sonata_p2_fuse.py`、`sonata_p2_merge.py`、`sonata_p3_merge.py`、`sonata_probe2.py`、`sonata_probe3.py`、`stability_signal.py`、`stats_blocks.py`、`step1_gate.py`、`step23_gate.py`、`summarize_full.py`、`sweep_fusion.py`、`sweep_gamma.py`、`sweep_gamma_fine.py`、`sweep_step2.py`、`test_ocsb_unit.py`、`topk_curve.py`、`train_bridge_scorer.py`、`train_episode_gate.py`、`train_omission_gate.py`、`train_stage2.py`、`train_su.py`、`train_sweep.py`、`train_sweep2k.py`、`u_block_diag.py`、`upper_bound.py`、`upper_bound_test.py`、`upper_bound_v2.py`、`v3_build_cache.py`、`v3_build_gate_cache.py`、`v3g_build_cache.py`、`v3h_build_cache.py`、`v3i_intra_query.py`、`verify_controls.py`、`verify_gain.py`、`verify_replicate.py`、`weak_support_test.py`、`run_k_sweep.sh`、`run_paired.sh`、`run_prep_rest.sh`、`run_same_scene.sh`、`run_task6.sh`、`watch_s2.sh`

---

## 日志索引（不复制内容）

| 目录 | 文件 | 大小 | 修改日 |
|---|---|---|---|
| `code` | `adv_smoke.log` | 1 K | 2026-11-17 |
| `code` | `adv_smoke2.log` | 1 K | 2026-11-17 |
| `code` | `adv_smoke3.log` | 1 K | 2026-11-17 |
| `code` | `cache_s0.log` | 1 K | 2026-11-17 |
| `code` | `cache_s1.log` | 1 K | 2026-11-17 |
| `code` | `cache_s1b.log` | 1 K | 2026-11-17 |
| `code` | `cache_s2.log` | 1 K | 2026-11-17 |
| `code` | `cache_s3.log` | 1 K | 2026-11-17 |
| `code` | `check_new.log` | 1 K | 2026-11-17 |
| `code` | `collect_shard0.log` | 1 K | 2026-11-16 |
| `code` | `collect_shard1.log` | 1 K | 2026-11-16 |
| `code` | `collect_shard2.log` | 1 K | 2026-11-16 |
| `code` | `collect_shard3.log` | 1 K | 2026-11-16 |
| `code` | `cross_block100.log` | 1 K | 2026-11-17 |
| `code` | `eg.log` | 1 K | 2026-11-17 |
| `code` | `eg_final.log` | 1 K | 2026-11-17 |
| `code` | `eg_l03.log` | 1 K | 2026-11-17 |
| `code` | `gate2_s0.log` | 1 K | 2026-11-17 |
| `code` | `gate2_s1.log` | 1 K | 2026-11-17 |
| `code` | `gate2_s2.log` | 1 K | 2026-11-17 |
| `code` | `gate2_s3.log` | 1 K | 2026-11-17 |
| `code` | `gate_cache_s0.log` | 1 K | 2026-11-17 |
| `code` | `gate_cache_s1.log` | 1 K | 2026-11-17 |
| `code` | `gate_cache_s2.log` | 1 K | 2026-11-17 |
| `code` | `gate_cache_s3.log` | 1 K | 2026-11-17 |
| `code` | `gate_signal100.log` | 1 K | 2026-11-17 |
| `code` | `install_deps.log` | 1 K | 2026-11-16 |
| `code` | `pip_torch.log` | 1 K | 2026-11-16 |
| `code` | `purity100.log` | 1 K | 2026-11-17 |
| `code` | `purity_fix100.log` | 1 K | 2026-11-17 |
| `code` | `repro2_A.log` | 1 K | 2026-11-17 |
| `code` | `repro2_B.log` | 1 K | 2026-11-17 |
| `code` | `repro_probe.log` | 1 K | 2026-11-17 |
| `code` | `ring4_full.log` | 3 K | 2026-11-16 |
| `code` | `ring4_overlap.log` | 1 K | 2026-11-16 |
| `code` | `room2blocks_train.log` | 44 K | 2026-11-16 |
| `code` | `sc_s0.log` | 1 K | 2026-11-17 |
| `code` | `sc_s1.log` | 1 K | 2026-11-17 |
| `code` | `sc_s2.log` | 1 K | 2026-11-17 |
| `code` | `sc_s3.log` | 1 K | 2026-11-17 |
| `code` | `sc_s4.log` | 1 K | 2026-11-17 |
| `code` | `sc_s5.log` | 1 K | 2026-11-17 |
| `code` | `sc_s6.log` | 1 K | 2026-11-17 |
| `code` | `sc_s7.log` | 1 K | 2026-11-17 |
| `code` | `scorer_folds.log` | 1 K | 2026-11-17 |
| `code` | `sonata_env.log` | 5 K | 2026-11-16 |
| `code` | `train_gate.log` | 1 K | 2026-11-17 |
| `code` | `train_scorer.log` | 1 K | 2026-11-17 |
| `code` | `train_scorer2.log` | 1 K | 2026-11-17 |
| `code` | `ub50_repro.log` | 1 K | 2026-11-17 |
| `code` | `ub560_s0.log` | 2 K | 2026-11-17 |
| `code` | `ub560_s1.log` | 2 K | 2026-11-17 |
| `code` | `ub560_s2.log` | 2 K | 2026-11-17 |
| `code` | `ub560_s3.log` | 2 K | 2026-11-17 |
| `code` | `ub560x_s0.log` | 2 K | 2026-11-17 |
| `code` | `ub560x_s1.log` | 2 K | 2026-11-17 |
| `code` | `ub560x_s2.log` | 2 K | 2026-11-17 |
| `code` | `ub560x_s3.log` | 2 K | 2026-11-17 |
| `code` | `ub560x_s5.log` | 1 K | 2026-11-17 |
| `code` | `ub_full.log` | 1 K | 2026-11-17 |
| `code` | `ub_s0.log` | 1 K | 2026-11-17 |
| `code` | `ub_s1.log` | 1 K | 2026-11-17 |
| `code` | `ub_s2.log` | 1 K | 2026-11-17 |
| `code` | `ub_s3.log` | 1 K | 2026-11-17 |
| `code` | `v3_s0.log` | 3 K | 2026-11-17 |
| `code` | `v3_s1.log` | 3 K | 2026-11-17 |
| `code` | `v3_s2.log` | 3 K | 2026-11-17 |
| `code` | `v3_s3.log` | 3 K | 2026-11-17 |
| `code` | `v3_smoke.log` | 1 K | 2026-11-17 |
| `code` | `v3_smoke2.log` | 2 K | 2026-11-17 |
| `code` | `v3g2_s0.log` | 1 K | 2026-11-17 |
| `code` | `v3g2_s1.log` | 1 K | 2026-11-17 |
| `code` | `v3g2_s2.log` | 1 K | 2026-11-17 |
| `code` | `v3g2_s3.log` | 1 K | 2026-11-17 |
| `code` | `v3g_s0.log` | 1 K | 2026-11-17 |
| `code` | `v3g_s1.log` | 1 K | 2026-11-17 |
| `code` | `v3g_s2.log` | 1 K | 2026-11-17 |
| `code` | `v3g_s3.log` | 1 K | 2026-11-17 |
| `code` | `v3h_s0.log` | 1 K | 2026-11-17 |
| `code` | `v3h_s1.log` | 1 K | 2026-11-17 |
| `code` | `v3h_s2.log` | 1 K | 2026-11-17 |
| `code` | `v3h_s3.log` | 1 K | 2026-11-17 |
| `code` | `v3h_s4.log` | 1 K | 2026-11-17 |
| `code` | `v3h_s5.log` | 1 K | 2026-11-17 |
| `code` | `v3h_s6.log` | 1 K | 2026-11-17 |
| `code` | `v3h_s7.log` | 1 K | 2026-11-17 |
| `code` | `v3i_s0.log` | 1 K | 2026-11-17 |
| `code` | `v3i_s1.log` | 1 K | 2026-11-17 |
| `code` | `v3i_s2.log` | 1 K | 2026-11-17 |
| `code` | `v3i_s3.log` | 1 K | 2026-11-17 |
| `code` | `v3i_s4.log` | 1 K | 2026-11-17 |
| `code` | `v3i_s5.log` | 1 K | 2026-11-17 |
| `code` | `v3i_s6.log` | 1 K | 2026-11-17 |
| `code` | `v3i_s7.log` | 1 K | 2026-11-17 |
| `code` | `v3ir_s0.log` | 1 K | 2026-11-17 |
| `code` | `v3ir_s1.log` | 1 K | 2026-11-17 |
| `code` | `v3ir_s2.log` | 1 K | 2026-11-17 |
| `code` | `v3ir_s3.log` | 1 K | 2026-11-17 |
| `code` | `v3ir_s4.log` | 1 K | 2026-11-17 |
| `code` | `v3ir_s5.log` | 1 K | 2026-11-17 |
| `code` | `v3ir_s6.log` | 1 K | 2026-11-17 |
| `code` | `v3ir_s7.log` | 1 K | 2026-11-17 |
| `code` | `verify_rep.log` | 1 K | 2026-11-17 |
| `code` | `weak_support.log` | 3 K | 2026-11-16 |
| `data` | `5split_sym_lovasz_5split_s42.log` | 213 K | 2026-11-21 |
| `data` | `ab_s.log` | 1 K | 2026-11-05 |
| `data` | `ab_x.log` | 1 K | 2026-11-05 |
| `data` | `audit_sym.log` | 139 K | 2026-11-22 |
| `data` | `base_s.log` | 2 K | 2026-11-05 |
| `data` | `base_x.log` | 2 K | 2026-11-05 |
| `data` | `cal_s.log` | 2 K | 2026-11-05 |
| `data` | `cal_x.log` | 2 K | 2026-11-05 |
| `data` | `clip_s.log` | 1 K | 2026-11-05 |
| `data` | `clip_x.log` | 1 K | 2026-11-05 |
| `data` | `cm_sym_k1.log` | 354 K | 2026-11-22 |
| `data` | `copy.log` | 548 K | 2026-11-04 |
| `data` | `cs2.log` | 2 K | 2026-11-05 |
| `data` | `cx2.log` | 2 K | 2026-11-05 |
| `data` | `det_eval_all.log` | 1 K | 2026-11-22 |
| `data` | `det_r1.log` | 130 K | 2026-11-22 |
| `data` | `det_r2.log` | 130 K | 2026-11-22 |
| `data` | `diag_100.log` | 1 K | 2026-11-05 |
| `data` | `diag_500.log` | 2 K | 2026-11-05 |
| `data` | `diag_500_v2.log` | 1 K | 2026-11-05 |
| `data` | `diag_502_s101.log` | 2 K | 2026-11-05 |
| `data` | `diag_502_s102.log` | 2 K | 2026-11-05 |
| `data` | `diag_502_s103.log` | 2 K | 2026-11-05 |
| `data` | `diag_502_s104.log` | 2 K | 2026-11-05 |
| `data` | `diag_all_test.log` | 2 K | 2026-11-05 |
| `data` | `diag_all_test2.log` | 1 K | 2026-11-05 |
| `data` | `diag_all_test3.log` | 2 K | 2026-11-05 |
| `data` | `diag_dbg.log` | 1 K | 2026-11-05 |
| `data` | `diag_dbg2.log` | 1 K | 2026-11-05 |
| `data` | `diag_dbg3.log` | 1 K | 2026-11-05 |
| `data` | `diag_dbg4.log` | 1 K | 2026-11-05 |
| `data` | `diag_full_s91.log` | 2 K | 2026-11-05 |
| `data` | `diag_full_s92.log` | 2 K | 2026-11-05 |
| `data` | `diag_gap_s81.log` | 2 K | 2026-11-05 |
| `data` | `diag_gap_s82.log` | 2 K | 2026-11-05 |
| `data` | `diag_gate_s121.log` | 2 K | 2026-11-05 |
| `data` | `diag_gate_s122.log` | 2 K | 2026-11-05 |
| `data` | `diag_gate_s123.log` | 2 K | 2026-11-05 |
| `data` | `diag_gate_s124.log` | 2 K | 2026-11-05 |
| `data` | `diag_k_s131.log` | 2 K | 2026-11-05 |
| `data` | `diag_k_s133.log` | 2 K | 2026-11-05 |
| `data` | `diag_multi100.log` | 1 K | 2026-11-05 |
| `data` | `diag_multi_v2.log` | 1 K | 2026-11-05 |
| `data` | `diag_new5.log` | 1 K | 2026-11-05 |
| `data` | `diag_on_s71.log` | 2 K | 2026-11-05 |
| `data` | `diag_on_s72.log` | 2 K | 2026-11-05 |
| `data` | `diag_on_s73.log` | 2 K | 2026-11-05 |
| `data` | `diag_on_s74.log` | 2 K | 2026-11-05 |
| `data` | `diag_or2_s41.log` | 1 K | 2026-11-05 |
| `data` | `diag_or2_s42.log` | 1 K | 2026-11-05 |
| `data` | `diag_or_s31.log` | 1 K | 2026-11-05 |
| `data` | `diag_or_s32.log` | 1 K | 2026-11-05 |
| `data` | `diag_or_s33.log` | 1 K | 2026-11-05 |
| `data` | `diag_or_s34.log` | 1 K | 2026-11-05 |
| `data` | `diag_proto_s51.log` | 1 K | 2026-11-05 |
| `data` | `diag_proto_s52.log` | 1 K | 2026-11-05 |
| `data` | `diag_proto_s53.log` | 1 K | 2026-11-05 |
| `data` | `diag_s1.log` | 1 K | 2026-11-05 |
| `data` | `diag_s2.log` | 1 K | 2026-11-05 |
| `data` | `diag_s3.log` | 1 K | 2026-11-05 |
| `data` | `diag_s4.log` | 1 K | 2026-11-05 |
| `data` | `diag_s5.log` | 1 K | 2026-11-05 |
| `data` | `diag_same_s121.log` | 2 K | 2026-11-05 |
| `data` | `diag_same_s122.log` | 1 K | 2026-11-05 |
| `data` | `diag_same_s123.log` | 2 K | 2026-11-05 |
| `data` | `diag_stab_s111.log` | 2 K | 2026-11-05 |
| `data` | `diag_stab_s112.log` | 2 K | 2026-11-05 |
| `data` | `diag_stab_s113.log` | 2 K | 2026-11-05 |
| `data` | `diag_stab_s114.log` | 2 K | 2026-11-05 |
| `data` | `diag_t6s_s131.log` | 2 K | 2026-11-05 |
| `data` | `diag_t6s_s133.log` | 2 K | 2026-11-05 |
| `data` | `diag_t6x_s121.log` | 2 K | 2026-11-05 |
| `data` | `diag_t6x_s123.log` | 1 K | 2026-11-05 |
| `data` | `diag_test.log` | 3 K | 2026-11-05 |
| `data` | `diag_up_s21.log` | 1 K | 2026-11-05 |
| `data` | `diag_up_s22.log` | 1 K | 2026-11-05 |
| `data` | `diag_up_s23.log` | 1 K | 2026-11-05 |
| `data` | `diag_up_s24.log` | 1 K | 2026-11-05 |
| `data` | `diag_up_test.log` | 1 K | 2026-11-05 |
| `data` | `diag_v3_s1.log` | 1 K | 2026-11-05 |
| `data` | `diag_v3_s2.log` | 2 K | 2026-11-05 |
| `data` | `diag_v3_s3.log` | 1 K | 2026-11-05 |
| `data` | `diag_v3_s4.log` | 1 K | 2026-11-05 |
| `data` | `diag_v3_s5.log` | 1 K | 2026-11-05 |
| `data` | `diag_v3_s6.log` | 1 K | 2026-11-05 |
| `data` | `diag_v4_s1.log` | 1 K | 2026-11-05 |
| `data` | `diag_v4_s2.log` | 2 K | 2026-11-05 |
| `data` | `diag_v4_s3.log` | 1 K | 2026-11-05 |
| `data` | `diag_v4_s4.log` | 1 K | 2026-11-05 |
| `data` | `diag_v4_s5.log` | 1 K | 2026-11-05 |
| `data` | `diag_v4_s6.log` | 1 K | 2026-11-05 |
| `data` | `diag_v5_s11.log` | 1 K | 2026-11-05 |
| `data` | `diag_v5_s12.log` | 1 K | 2026-11-05 |
| `data` | `diag_v5_s13.log` | 1 K | 2026-11-05 |
| `data` | `diag_v5_s14.log` | 1 K | 2026-11-05 |
| `data` | `dl_scannetpp_gfs.log` | 1 K | 2026-11-21 |
| `data` | `eval_baseline.log` | 1 K | 2026-11-05 |
| `data` | `eval_baseline2.log` | 1 K | 2026-11-05 |
| `data` | `extract2.log` | 1 K | 2026-11-05 |
| `data` | `extract3.log` | 1 K | 2026-11-05 |
| `data` | `extract_ontr_s0.log` | 2 K | 2026-11-21 |
| `data` | `extract_ontr_s0b.log` | 2 K | 2026-11-21 |
| `data` | `extract_ontr_s1.log` | 2 K | 2026-11-21 |
| `data` | `extract_ontr_s1b.log` | 2 K | 2026-11-21 |
| `data` | `extract_ontr_s2.log` | 2 K | 2026-11-21 |
| `data` | `extract_ontr_s2b.log` | 2 K | 2026-11-21 |
| `data` | `extract_orth.log` | 1 K | 2026-11-05 |
| `data` | `extract_p0.log` | 3 K | 2026-11-21 |
| `data` | `extract_p1.log` | 3 K | 2026-11-21 |
| `data` | `extract_p2.log` | 3 K | 2026-11-21 |
| `data` | `frozen_base.log` | 701 K | 2026-11-21 |
| `data` | `frozen_ln.log` | 701 K | 2026-11-21 |
| `data` | `frozen_ontr.log` | 701 K | 2026-11-21 |
| `data` | `frozen_ontr_rep3.log` | 702 K | 2026-11-21 |
| `data` | `frozen_rep3.log` | 702 K | 2026-11-21 |
| `data` | `gate_s.log` | 2 K | 2026-11-05 |
| `data` | `gate_x.log` | 2 K | 2026-11-05 |
| `data` | `kernels3.log` | 1 K | 2026-11-05 |
| `data` | `kernels4.log` | 1 K | 2026-11-05 |
| `data` | `kernels_build.log` | 1 K | 2026-11-05 |
| `data` | `ontr_base_s13.log` | 191 K | 2026-11-21 |
| `data` | `ontr_base_s13b.log` | 556 K | 2026-11-21 |
| `data` | `ontr_base_s7.log` | 191 K | 2026-11-21 |
| `data` | `ontr_base_s7b.log` | 556 K | 2026-11-21 |
| `data` | `ontr_c_bk.log` | 1 K | 2026-11-21 |
| `data` | `ontr_c_bk50.log` | 1 K | 2026-11-21 |
| `data` | `ontr_diag.log` | 1 K | 2026-11-21 |
| `data` | `ontr_lp.log` | 1 K | 2026-11-21 |
| `data` | `ontr_lp2.log` | 1 K | 2026-11-21 |
| `data` | `ontr_lp3.log` | 2 K | 2026-11-21 |
| `data` | `ontr_stage2_ontr_base_lovasz_s42.log` | 702 K | 2026-11-21 |
| `data` | `ontr_stage2_ontr_ln_lovasz_s42.log` | 701 K | 2026-11-21 |
| `data` | `ontr_stage2_ontr_sym_lovasz_s42.log` | 702 K | 2026-11-21 |
| `data` | `ontr_step_b.log` | 24 K | 2026-11-21 |
| `data` | `ontr_test_bk.log` | 343 K | 2026-11-21 |
| `data` | `ontr_test_ontr.log` | 343 K | 2026-11-21 |
| `data` | `orth_s.log` | 1 K | 2026-11-05 |
| `data` | `orth_x.log` | 1 K | 2026-11-05 |
| `data` | `p3s.log` | 1 K | 2026-11-05 |
| `data` | `p3x.log` | 1 K | 2026-11-05 |
| `data` | `pl_pipeline_k5_s20.log` | 1 K | 2026-11-21 |
| `data` | `pl_pipeline_k5_s30.log` | 1 K | 2026-11-21 |
| `data` | `pl_pipeline_k5_s40.log` | 1 K | 2026-11-21 |
| `data` | `pl_pipeline_k5_s50.log` | 1 K | 2026-11-21 |
| `data` | `pl_pipeline_s20.log` | 6 K | 2026-11-21 |
| `data` | `pl_pipeline_s30.log` | 6 K | 2026-11-21 |
| `data` | `pl_pipeline_s40.log` | 6 K | 2026-11-21 |
| `data` | `pl_pipeline_s50.log` | 6 K | 2026-11-21 |
| `data` | `pl_tristage_audit.log` | 1 K | 2026-11-22 |
| `data` | `pl_tristage_audit_gpu.log` | 2 K | 2026-11-22 |
| `data` | `pm_s_bal.log` | 4 K | 2026-11-05 |
| `data` | `pm_s_fps.log` | 1 K | 2026-11-05 |
| `data` | `pm_x_bal.log` | 1 K | 2026-11-05 |
| `data` | `pm_x_fps.log` | 1 K | 2026-11-05 |
| `data` | `pointops2_build.log` | 27 K | 2026-11-05 |
| `data` | `prep_rest.log` | 3 K | 2026-11-05 |
| `data` | `prep_val.log` | 1 K | 2026-11-05 |
| `data` | `rep3_lovasz_s13.log` | 128 K | 2026-11-21 |
| `data` | `rep3_lovasz_s13_v2.log` | 561 K | 2026-11-21 |
| `data` | `rep3_lovasz_s7.log` | 97 K | 2026-11-21 |
| `data` | `rep3_lovasz_s7_v2.log` | 624 K | 2026-11-21 |
| `data` | `room2blocks.log` | 11 K | 2026-11-05 |
| `data` | `s1_lp.log` | 2 K | 2026-11-21 |
| `data` | `s2_cross.log` | 1 K | 2026-11-05 |
| `data` | `s2_lr4_v100.log` | 1 K | 2026-11-05 |
| `data` | `s2_main.log` | 1 K | 2026-11-05 |
| `data` | `s2_main2.log` | 1 K | 2026-11-05 |
| `data` | `s2_watch.log` | 1 K | 2026-11-05 |
| `data` | `scpp_base_k5.log` | 254 K | 2026-11-22 |
| `data` | `scpp_base_lovasz.log` | 256 K | 2026-11-22 |
| `data` | `scpp_sym_k5.log` | 255 K | 2026-11-22 |
| `data` | `scpp_sym_lovasz.log` | 257 K | 2026-11-22 |
| `data` | `sonata_1shot.log` | 1 K | 2026-11-21 |
| `data` | `sonata_1shot_pl_fixed.log` | 2 K | 2026-11-21 |
| `data` | `sonata_1shot_vlm.log` | 1 K | 2026-11-21 |
| `data` | `sonata_ex_s0.log` | 49 K | 2026-11-21 |
| `data` | `sonata_ex_s1.log` | 59 K | 2026-11-21 |
| `data` | `sonata_ex_s2.log` | 55 K | 2026-11-21 |
| `data` | `sonata_lp.log` | 1 K | 2026-11-21 |
| `data` | `sonata_lp2.log` | 1 K | 2026-11-21 |
| `data` | `sonata_lp3.log` | 1 K | 2026-11-21 |
| `data` | `sonata_proto.log` | 1 K | 2026-11-21 |
| `data` | `sonata_proto2.log` | 1 K | 2026-11-21 |
| `data` | `sonata_proto3.log` | 1 K | 2026-11-21 |
| `data` | `sonata_s2_batch.log` | 1 K | 2026-11-22 |
| `data` | `sonata_s2_encfrozen.log` | 485 K | 2026-11-22 |
| `data` | `sonata_s2_support.log` | 485 K | 2026-11-22 |
| `data` | `sonata_s2_unfrozen.log` | 484 K | 2026-11-22 |
| `data` | `sonata_stage1a.log` | 1080 K | 2026-11-22 |
| `data` | `sonata_val.log` | 40 K | 2026-11-21 |
| `data` | `sonata_x0.log` | 51 K | 2026-11-21 |
| `data` | `sonata_x1.log` | 51 K | 2026-11-21 |
| `data` | `sonata_x2.log` | 51 K | 2026-11-21 |
| `data` | `support_ontr.log` | 702 K | 2026-11-21 |
| `data` | `support_ontr_linear.log` | 701 K | 2026-11-21 |
| `data` | `support_xl.log` | 39 K | 2026-11-21 |
| `data` | `support_xl_linear.log` | 701 K | 2026-11-21 |
| `data` | `vs.log` | 2 K | 2026-11-06 |
| `data` | `vx.log` | 2 K | 2026-11-06 |

