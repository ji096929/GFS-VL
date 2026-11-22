# 散落脚本归档索引

> 归档日期 **2026-11-22**。源目录 `/data01/jh/code/`（158 个 .py）与
> `/data01/jh/afford_diag/`（54 个 .py）**都不是 git 仓库**，真正的仓库是它们的子目录
> `code/GFS-VL/`。所以这些脚本从第一天起就在版本控制之外。
>
> **本目录是只读副本，原处文件保留不动**（脚本内硬编码绝对路径，挪动即失效）。
> 重新归档：`python legacy_scripts/archive_legacy.py`

## 归档统计

- `01_config_gen` 配置生成器（生成 100 个实验配置的源头）：**42** 个
- `02_upper_bound` 上限估计与 Oracle：**5** 个
- `03_signal_gate` 门控与信号探测：**4** 个
- `04_pseudo_label` 伪标签生成与筛选：**5** 个
- `05_sonata` Sonata 迁移探针：**10** 个
- `06_diagnosis` 诊断与归因：**22** 个
- `07_reproduce` 校准与复现：**6** 个
- `08_train_eval` 训练与评估入口：**5** 个
- `90_other` 未归类：**0** 个
- `98_ocsb` 属 OCSB 主线（非 GFS-VL）：**113** 个，应迁往 3D_OCSB 仓库

日志：`303` 个（`code/` 104 + `data/` 199），**只建索引不复制**（见文末）。

---

## 01_config_gen 配置生成器（生成 100 个实验配置的源头）（42）

对应报告：docs/00_ledger/LEDGER.md（每个实验的 config 来源）

| 脚本 | 修改日 | 说明 |
|---|---|---|
| `group_experiment.py` | 2026-11-18 | 三组伪标签对照：保留 PS 已接受，只考察 PS 拒绝的区域。 Group 1: support 原型直接匹配 C（放宽阈值） Group 2: 加入跨场景 B（PS 共识池），按 B 对 C 的增量排序 Group 3 |
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

## 02_upper_bound 上限估计与 Oracle（5）

对应报告：docs/06_diagnostics/DIAGNOSIS_REPORT_2026-11-22.md §3 4桶上界

| 脚本 | 修改日 | 说明 |
|---|---|---|
| `post_ub560.py` | 2026-11-17 | 合并全部分片的 ub560 结果：全量 λ 表 + 配对检验 + 2 折交叉验证 + 逐类/分层表。 支持两轮 λ（不同文件名前缀），会自动发现 rows 里出现的所有 λ。 |
| `post_ub_official.py` | 2026-11-18 | 合并官方口径（min_gt=0.05，435 集）的四路上限结果。 与 post_ub560.py 的区别：rows 里的 ep 已经是全局编号（upper_bound.py 的 --min-gt 分支直接存 g_ep） |
| `upper_bound.py` | 2026-11-18 | OCSB 第一步：外部池 U 的四种用法能带来多少上限。 四行对照，全部零训练，只改送进相关性网络的原型来源： Baseline      仅 support 的 1-shot 原型 Naive-B       supp |
| `upper_bound_test.py` | 2026-11-17 | 第一步：用 COSeg 原生多样本接口测 B_pos / B_pos∪C_pos 的真实上限。 四行对照（全部零训练，只改支持集）： Baseline    仅 1-shot S（k_shot=1，100 个原型） Na |
| `upper_bound_v2.py` | 2026-11-17 | 第一步（v2）：修复三个 bug 后的 B / B_pos / B_pos∪C_pos 上限测试。 修复： Bug1  每个 U 块独立 getFeatures，绝不把两个减过质心的块拼进同一个 offset Bug2  |

## 03_signal_gate 门控与信号探测（4）

对应报告：docs/04_ontr_frozen/DAYTIME_ONTR_REPORT*.md

| 脚本 | 修改日 | 说明 |
|---|---|---|
| `gate_probe.py` | 2026-11-17 | 门控信号探针：找能区分"高基线任务"（Q4/Q5）与"低基线任务"（Q1/Q2）的无监督指标。 背景 ---- `frac_q`（预测前景占比）在 Q1–Q5 上都是 0.37–0.44，完全无法区分难度—— 所有任务都 |
| `train_episode_gate.py` | 2026-11-17 | 任务级 EpisodeGate + 点级保 TP 融合。 为什么这样拆 ------------ 上一版把 8 维任务特征和 3 维点级特征混进一个 200 参数的 MLP，用逐点 CE 盲训。 一折只有 270 个任务 |
| `train_omission_gate.py` | 2026-11-17 | 训练 OmissionGate：逐点决定「这一点的预测要不要采信 U」。 输入 11 维 = 3 维点级 + 8 维任务级： 点级：p0_s(i), p0_u(i), relu(p0_u(i) − p0_s(i)) 任务 |
| `analyze_gate.py` | 2026-11-20 | (无说明) |

## 04_pseudo_label 伪标签生成与筛选（5）

对应报告：docs/03_pseudo_label/margin_filter_experiment.md

| 脚本 | 修改日 | 说明 |
|---|---|---|
| `post_v3.py` | 2026-11-17 | v3 后处理：合并分片、2 折 CV 选 (λ, gate_thr)、分层表、TP/FP/FN。 |
| `purity_analysis.py` | 2026-11-17 | 离线提纯信号分析：哪些信号能把 B 的精度从 0.475 拉上去。 只读已有缓存，不训练、不改主管道。 分析对象 -------- B = U 块中初始 p0 > fg_hi 的点（基准精度约 0.475），标签取降采样 |
| `purity_fix_probe.py` | 2026-11-17 | 验证三个根因：FPS 离群放大、反向完整管道校验、以及采样策略。 只读缓存，不改主管道。输出四组对比： 1. FPS 种子精度 vs 全体候选精度 `getMutiplePrototypes` 用 3D 坐标做 FPS  |
| `recompute_filtered.py` | 2026-11-17 | 按 query 目标占比阈值重算 539 集 baseline / 自桥接（用已有缓存，不重跑）。 |
| `weak_support_test.py` | 2026-11-16 | 弱标注支持集痛点摸底（零训练，val_560，仅 S->Q，不引入 U）。 五种支持集掩码（均由支持集真实标签自动生成）： full      逐点真值（上界，等同原生 COSeg） tight     真实前景点 AA |

## 05_sonata Sonata 迁移探针（10）

对应报告：docs/05_sonata/SONATA_SMALL_INTEGRATION.md

| 脚本 | 修改日 | 说明 |
|---|---|---|
| `diag_sonata2.py` | 2026-11-18 | (无说明) |
| `dump_for_sonata.py` | 2026-11-18 | 把 357 个官方口径任务的 support/query 点云与 COSeg 的 p0 落盘， 供 sonata 环境（torch 2.5）二次读取（pointops 与 sonata 环境不兼容，必须分两步）。 每个任 |
| `sonata_auc_merge.py` | 2026-11-18 | 合并 sonata_b_auc.py 的 8 个分片，输出全局 AUC 与头部精度。 |
| `sonata_b_auc.py` | 2026-11-18 | 用 Sonata 独立三维特征判别 B_pos / B_neg。 B 的定义与 OCSB 一致：query 上 COSeg 初始预测 p0 > thresh 的点。 B_pos = B 中真值为前景，B_neg = B  |
| `sonata_check.py` | 2026-11-18 | Sonata 环境验证（headless，不依赖 open3d 可视化）。 用本地 ScanNet block 构造点云（coord + color + 估计法向）， 跑 Sonata forward，输出逐点特征统计并 |
| `sonata_p2_fuse.py` | 2026-11-18 | 验A + 验B(CPU 部分)：全图融合与 Sonata 自选头部纯度。 验A: 融合得分 w*z(cat_top3) + z(p0)，两折 CV 定阈值，357 集 Cumul/Arith。 纯后处理，不碰 GPU。  |
| `sonata_p2_merge.py` | 2026-11-18 | 检查1(提纯) + 检查2(零训练分割水位) 的汇总。 检查1: 在 B={p0>0.7} 上对比 7 种打分的 AUC 与 Top-X% 精度 检查2: 6 种 Sonata 打分当零训练 1-shot 分割器，全 q |
| `sonata_p3_merge.py` | 2026-11-18 | 检验B 汇总：原型采样空间（feat/nfeat/coord）× k 的 CV Cumul/Arith + 头部纯度。 |
| `sonata_probe2.py` | 2026-11-18 | (无说明) |
| `sonata_probe3.py` | 2026-11-18 | 检验B(GPU)：原型采样空间与子原型数量 k 的影响。 对 357 任务，用 Sonata cat(1232 维) 特征比较： 空间 feat  : 原始特征空间 FPS（与 p2 的 cat_multi/cat_to |

## 06_diagnosis 诊断与归因（22）

对应报告：docs/06_diagnostics/DIAGNOSIS_REPORT_2026-11-22.md

| 脚本 | 修改日 | 说明 |
|---|---|---|
| `audit_merge.py` | 2026-11-18 | 审计汇总：逐 ckpt 指标 + 两折盲选轮次。 |
| `audit_stage2.py` | 2026-11-18 | Stage II 审计：ep0 基线、逐 checkpoint 逐集指标（TP/FP/FN、B 池纯度/AUC）。 |
| `audit_u_infer.py` | 2026-11-18 | Exp1-low ep1 的推理时 U 消融 + 训练前后 U 上的桥梁质量。 对 357 个任务，Exp1-low ep1 模型分别在： official  官方 forward，不传 U（= 之前 0.4153 的口 |
| `audit_u_merge.py` | 2026-11-18 | U 推理消融汇总：4 种推理模式的 Cumul/Arith + U 块上的 B 池质量。 |
| `probe_agg.py` | 2026-11-18 | (无说明) |
| `probe_calib.py` | 2026-11-17 | 只做前景比例校准能把整场景 IoU 提到多少。 对 baseline 的概率 p1 取 top-k 分位作为预测，k 分别用： s_ratio   support 的前景占比（无标签可得） oracle    query |
| `probe_fp.py` | 2026-11-18 | (无说明) |
| `probe_gap.py` | 2026-11-18 | (无说明) |
| `probe_gap2.py` | 2026-11-18 | 修正版对拍：官方 forward（fp16/fp32）vs predict_agg（fp32）。 上次 probe 的错误：把 dump 的 s_lab（0/1 二值）当成 21 类 id 传给 `(label == c |
| `probe_layers.py` | 2026-11-18 | (无说明) |
| `probe_retrieval.py` | 2026-11-18 | 免训练对照：块级检索命中率 + 配对分割收益。 每个任务固定同一批 S/Q、同一 500 块候选池、同一权重，只允许送 3 块： random3    从池里随机抽 3 块 retrieve3  用 S 的 fg/bg  |
| `probe_retrieval_merge.py` | 2026-11-18 | 检索对照汇总：命中率 + Cumul mIoU。 |
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

## 07_reproduce 校准与复现（6）

对应报告：docs/legacy/SESSION_STATE.md

| 脚本 | 修改日 | 说明 |
|---|---|---|
| `check_pair.py` | 2026-11-17 | 对拍：同一 episode 上，官方 COSeg.forward 与我们复刻的 predict_agg 是否给出相同 IoU。 |
| `repro_probe.py` | 2026-11-17 | 定位 Baseline 的随机源：同进程连跑 3 次，打印各阶段校验和。 |
| `repro_probe2.py` | 2026-11-17 | 验证 per-episode 种子锁死后，跨进程 Baseline 是否一致。 用法：起两个独立进程（不同卡），比较同一 --gps 的输出。 |
| `verify_replicate.py` | 2026-11-17 | 验证 predict_agg 是否精确复刻 COSeg 原生 forward 的后半段。 原生：model(support, query, epoch=100) -> final_pred [1, 2, N_raw] 复 |
| `verify_cross.py` | 2026-11-19 | 启动前验证：cross_layers 分支的 shape / 尺度 / 参数分组 / 梯度 / 缓存。 |
| `verify_variants.py` | 2026-11-20 | Pre-flight 校验：mlp / sym / gated 三种分支。 |

## 08_train_eval 训练与评估入口（5）

对应报告：docs/07_shot_protocol/OVERNIGHT_ICCV_REPORT.md

| 脚本 | 修改日 | 说明 |
|---|---|---|
| `assemble_scannet200.py` | 2026-11-18 | 从服务器现有 ply + 本地传来的 segment200 组装 pointcept 格式的 ScanNet200。 点序已验证：本地 coord.npy 与服务器 ply 顶点逐点一致（81,369 点，首点相同）。  |
| `build_u_cache.py` | 2026-11-18 | 预计算 500 个候选块的 COSeg 降采样特征（训练池随机采，不按标签筛）。 候选池固定 seed=0，所有任务共用；与测试的 312 个 val 场景零重叠。 |
| `official_compare.py` | 2026-11-18 | 官方 5% 口径终极对比（纯后处理，不占 GPU）。 把四个来源的逐集 TP/FP/FN 统一到全局 episode 编号，再按官方 get_class2scans 的阈值过滤，重算所有方案的 Cumul / Arith |
| `run_official_eval.py` | 2026-11-17 | 用 fake_tp 垫片跑官方 main_fs.py（官方脚本没有这个 shim，会撞 torch_points3d 的依赖链）。 |
| `train_su.py` | 2026-11-18 | Stage II 训练：Exp 0（纯 S->Q 微调）与 Exp 1（S+U 可微软原型）。 用法: CUDA_VISIBLE_DEVICES=2 python train_su.py --mode s  --epoc |

## 98_ocsb OCSB 主线（非 GFS-VL，待迁往 3D_OCSB）（113）

对应报告：—

| 脚本 | 修改日 | 说明 |
|---|---|---|
| `adaptive_signal.py` | 2026-11-05 | Task 2e: per-class adaptive signal selection. For each novel class, pick the signal with the highest per-episo |
| `analyze_cal.py` | 2026-11-05 | Task 10: does margin-space calibration / a threshold-free soft gate beat the raw end-to-end fusion? All compar |
| `analyze_diag.py` | 2026-11-05 | Summarize diag_bc.py records: overall + per-class tables. |
| `analyze_full.py` | 2026-11-05 | Task 2/3 on the full pool: old 502 (v4/v5) + new full-field run. |
| `analyze_overnight.py` | 2026-11-05 | Overnight report analysis: Task 1/2/3/4 tables. |
| `analyze_paired.py` | 2026-11-05 | Paired analysis over the fixed manifest. All runs consume the same manifest prefix, so episode i is the same ( |
| `analyze_task6.py` | 2026-11-05 | Task 6 analysis: drop-off anatomy, asymmetric fusion, RSV gating, K sweep. Usage: python analyze_task6.py <cro |
| `analyze_toplk_oracle.py` | 2026-11-05 | Top-K bridge truncation + Oracle injection analysis. |
| `auc_signals.py` | 2026-11-05 | AUC of candidate negative-control signals: B_pos vs B_neg per episode. |
| `block_coord_probe.py` | 2026-11-17 | 隔离坐标平移：同样 1m block、同样 voxel、同样原型流程，只改最后一步坐标处理。 center  coord -= (min+max)/2   ← prepare_block 现行做法，范围 [-0.5,0. |
| `build_intra_cache.py` | 2026-11-13 | 补齐 intra-query 方向需要的字段。 原缓存只有 downsampled 的 q_h/q_margin/q_coord，缺少： q_seg_full / q_rgb_full   voxel 后原始点的超点 i |
| `build_ocsb_cache.py` | 2026-11-12 | 提取 3D-OCSB 的冻结特征缓存。 因为 Stage I 的主干 F0 和初始预测器 f0 完全冻结，每个 episode 只需要前向 一次并把下列量存盘，之后所有公式（1)-(19）、五项自检、五步消融都在缓存上  |
| `build_train2k.py` | 2026-11-13 | 从 train split 的 blocks 采 2000 个 base 类任务并提取缓存。 严格场景隔离：support / query / U 三者场景互不重叠。 输出 /data01/jh/data/ocsb_tr |
| `check.py` | 2026-11-17 | OCSB 管道的三种校验，合并成一个入口。 python check.py replicate                 # 复刻的 predict_agg vs COSeg 原生 forward python c |
| `check_coseg.py` | 2026-11-05 | Minimal COSeg forward check: load weights, run one support+query episode. |
| `clip_vit.py` | 2026-11-05 | Minimal CLIP ViT-B/16 vision tower, rebuilt from a TorchScript archive. The server's CLIP checkpoints are Torc |
| `cohens_topk.py` | 2026-11-05 | Task 2 completion: Cohen's d + per-signal Top-K B precision. |
| `collect_train_blocks.py` | 2026-11-13 | 并行收集 ScanNet train split 的场景 npy（第 8 列保留 seg id）。 用法：--shard 0 --nshard 4 输出：/data01/jh/data/scannet_train_v2/ |
| `combo_signals.py` | 2026-11-05 | Task 2c: linear combination search over orthogonal signals. Per-bridge signals are combined after per-episode  |
| `compare_u_mode.py` | 2026-11-05 | Compare cross-scene vs same-scene U without relying on episode pairing. Pairing turned out to be impossible: n |
| `control_u_confound.py` | 2026-11-05 | Control for the difficulty confound in the cross/same-scene comparison. same_scene happens to have an easier e |
| `coseg_mem_probe.py` | 2026-11-18 | 实测 COSeg 训练前向+反向的显存与耗时。 对比： src=block   1m block（真实训练输入，voxel 后 2-5k 点） src=scene   整场景采样到 20480 点（输入上限） freez |
| `coseg_su.py` | 2026-11-18 | COSeg + S+U 可微软原型（Stage II Exp 1）。 forward_su 相对官方 forward 只改三处： 1. 额外对 U 做一次 getFeatures（U 无标签，gt 传全 0，不做 bas |
| `cross_block_probe.py` | 2026-11-17 | 跨块互验证：把"考官"换成没参与选出候选点的独立块。 为什么需要 ---------- `proto_B -> S` 与 `S -> B` 用的是同一组余弦矩阵，只是聚合方向不同， 信息高度重叠（实测 AUC 0.626 |
| `death_test.py` | 2026-11-13 | 生死线检验：Q_bg_pred 中按桥梁得分选 Top-K%，看 Precision 能否超过 I0。 严格阈值：p > I0 = 37.44%（不是 I0/(1+I0) = 27.24%）。 同时报告相对 27.24% |
| `diag_bc.py` | 2026-11-05 | Step-1 diagnostic for 3D-OCSB premise. Frozen COSeg, 1-way 1-shot ScanNet val episodes. No training. For each  |
| `diag_cache.py` | 2026-11-12 | 检查缓存质量与 OCSB 中间量的分布。 在已缓存的 episode 上跑一次前向，统计 B/C 规模、eta 风险、V 正负、 w_C 恢复比例，以及负对照三个来源的实际数量。用来确认「w_C 全零导致 L_seg 无 |
| `diag_cmc.py` | 2026-11-12 | 诊断 delta_corr 是否对准真实漏检点。 分组：TP（margin>0 且真前景）、FN（margin<=0 且真前景，即漏检）、 FP（margin>0 且真背景）、TN（margin<=0 且真背景）。 如果 |
| `diag_signals.py` | 2026-11-13 | 四种候选信号在 TP/FN/FP/TN 上的四分组诊断。 S1  旧公式：Σ ŵ_R relu(z_R·z_q - c0)          （复现 TN 倒置） S2  纯加权相似度：Σ ŵ_R (z_R·z_q) S |
| `diag_step2.py` | 2026-11-12 | 诊断式(15) 融合后掉点的原因。 打印 s_s（COSeg margin）、超点相似度 sim/τ、ell 的分布，以及预测正例率 与真值正例率的对比。重点看 sim 与 s_s 是否量纲一致。 |
| `eval_baseline.py` | 2026-11-05 | Baseline query mIoU for COSeg 1-way 1-shot on ScanNet val. Used as the sanity gate from the task spec: make su |
| `export_counter_by_block.py` | 2026-11-20 | 按 U 块拆分导出 counter_007 的三个房间：rgb / gt / tag 三版。 tag 复现 ring2_export_ply 的超点级判定： B_pos 绿 (0,255,0)  B_neg 红 (255 |
| `export_counter_gt.py` | 2026-11-20 | 导出 counter_007 的 U 池真值上色版：counter 品红，其余灰/暗原始色。 |
| `export_ply2.py` | 2026-11-16 | 导出改进版 PLY： 1. 每个 U block 单独一个文件（原始 RGB + 区域标注） 2. counter_007 / cabinet_009 额外导出完整房间（gamma 提亮 + 切块高亮） 输出：/data |
| `export_room_raw.py` | 2026-11-20 | 用原始 vh_clean_2.ply（164k 点）导出 scene0606_00 的 gt / tag。 原始 ply 是 binary_little_endian，顶点 x,y,z float32 + rgba uc |
| `export_room_scene.py` | 2026-11-20 | 导出 scene0606_00 整房间的 rgb / gt / tag 三版。 tag 来自该房间内 block_3 的 episode 区域判定，按坐标匹配映射回整房间： 绿 B_pos / 红 B_neg / 黄 C |
| `extract_orth_feats.py` | 2026-11-05 | Phase 1: build an orthogonal visual feature f_orth with a frozen 2D encoder. Server has no CLIP/DINOv2 and no  |
| `fake_tp.py` | 2026-11-05 | Compatibility shim: provide torch_points_kernels via pointops2. COSeg only uses tp.ball_query in getFeatures() |
| `local_geometry.py` | 2026-11-05 | Task 2g: superpoint local geometry (curvature / planarity) as a signal. Reads the block npy files directly (co |
| `make_class2scans.py` | 2026-11-18 | 按官方 get_class2scans 的逻辑生成 class2scans.pkl。 官方阈值：某类点数 > max(该文件总点数 * 5%, 100) 才把这个文件计入该类。 scene 级和 block 级都用同一段 |
| `make_manifest.py` | 2026-11-05 | Build a fixed paired episode manifest. Every experiment (baseline, zero-shot 2C, Stage II validation) runs on  |
| `ocsb_advanced.py` | 2026-11-17 | 第二步：OASB + CSR 的零训练实现（五步消融）。 对应 `3D OCSB原始idea.md` 的式 (7)(8)(9)(10)(12)(13)(15)。 五步 ---- Step 1  Baseline      |
| `ocsb_cmc.py` | 2026-11-13 | OCSB-CMC：在 ocsb_faithful 之上的三个结构性修正。 Fix 1  式(5) 的 s_c^0 从"单均值原型"换成"支持集局部多子原型的 Top-M 均值"， 避免椅背/椅腿这类部件差异被误判成 Δ> |
| `ocsb_core.py` | 2026-11-17 | OCSB 第一步的共用底座：种子、episode 构造、特征提取、原型、相关性聚合。 这里的所有函数都与 `COSeg.forward` 的后半段逐行对应： getFeatures -> 前景/背景原型 -> 余弦相关性 |
| `ocsb_faithful.py` | 2026-11-12 | 3D-OCSB 的忠实实现：导师文档 Method V1 式 (1)-(19)。 只保留三处最小修正，其余严格按文档： C1  式(1)  W_phi 单位阵初始化 + \|\|W-I\|\|_F^2 正则，保证第 0 步与冻结 |
| `ocsb_v3.py` | 2026-11-17 | 三层遗漏条件化选择性桥接（v3）。 针对 v2 的三个方法学错误做的修正： 1. **第一层：U 内核心提纯 + 同块邻接吸收** 不再把 `p0 > 0.7` 的全部点直接当原型。按 `R_s = sim(f, S_f |
| `or_gate2.py` | 2026-11-13 | or_gate v2：B_max 与 oracle 在 val_560 上的 Arith + Cumul。 |
| `perclass_overnight.py` | 2026-11-05 | Per-class breakdown of the overnight experiments (offline). |
| `phase1_smooth.py` | 2026-11-13 | Phase 1：查询集超点内部一致性平滑 + 2 折交叉验证。 ell(i) = (1-alpha) * q_margin(i) + alpha * m(R_i) - theta m(R) 支持 mean / media |
| `phase2_adjacent.py` | 2026-11-13 | Phase 2：查询集内部三维邻接约束下的自桥接 A -> B_Q -> C_Q。 B_Q   查询集内 m(R) >= tau_seed 的高置信种子 C_Q   初始背景 (m(R) < tau_seed 且 > - |
| `phase2_clip.py` | 2026-11-05 | Phase 2: does the CLIP referee break the 0.610 AUC ceiling? Two variants are recorded: clip_*_r -- raw CLIP fe |
| `phase2_full.py` | 2026-11-13 | Phase 2 完整版：三种匹配度 + CSR 净贡献验证 + 2 折 CV。 打分： S_cos  max_B z_B·z_C                       （特征余弦） S_vq   max_B [ z |
| `phase2_orth.py` | 2026-11-05 | Phase 2: does the physical-appearance referee separate B_pos from B_neg better than F0-derived signals (whose  |
| `phase2_v2.py` | 2026-11-13 | Phase 2 完整版 v2：四种匹配度 + CSR 验证 + 2 折 CV + Cumul。 打分： S_cos   max_B z_B·z_C S_vq    max_B [ z_B·z_C - max_{D in  |
| `phase3_calib.py` | 2026-11-13 | Phase 3：自适应防爆阈值 + 支持集去污染诊断。 主实验：任务级自适应阈值 theta_ep = theta_0 + lam * relu(fg_ratio - 0.15) 其中 fg_ratio 是在 theta |
| `phase3_erode3d.py` | 2026-11-13 | Phase 3 补充：基于支持集三维距离 d_to_bg 的边界腐蚀。 对每个支持前景点，计算它到最近支持背景点的欧氏距离 d_to_bg， 剔除 d_to_bg 最小的 20% / 30%，重提支持前景原型，再算查询集 |
| `phase3_full.py` | 2026-11-13 | Phase 3 完整版：β_orth 正交投影 + 边界腐蚀 + 自适应阈值。 1) β_orth ∈ {0,0.2,0.5,0.8,1.0}：把支持/查询特征沿背景原型方向正交化， 重算支持前景原型，用余弦得分做超点平 |
| `phase3_gate.py` | 2026-11-06 | Phase 3: does the CLIP consensus gate predict when bridging helps? agree_orth -- CLIP-space agreement between  |
| `phase3_v2.py` | 2026-11-13 | Phase 3 v2：β_orth / 边界腐蚀 / 自适应阈值，全部带 Cumul 与 2 折 CV。 |
| `phase4_p2_perclass.py` | 2026-11-13 | 补齐 Phase 4 后两行的逐类对比：Phase 2 CV 盲测 + 去污染（同上一行）。 |
| `phase4_report.py` | 2026-11-13 | Phase 4：完整消融表（独立行 + pSE + 逐类）。 顺序按 objective： Baseline -> + 阈值校准 -> + 查询集超点一致性 -> + 空间邻接自桥接 -> + 支持集去污染 |
| `phase4_v2.py` | 2026-11-13 | Phase 4 v2：按 2 折 CV 盲测口径生成消融表。 每个组件在训练折上选超参，在盲测折上评估，两折拼接成 560 的盲测结果。 Baseline           无超参 + 阈值校准          θ  |
| `phase4_v3.py` | 2026-11-13 | Phase 4 v3：统一口径的 2 折 CV 盲测消融表。 与 v2 的区别：第四行不再读 Phase 2 的固定 baseline，而是 1) 在训练折选出 Phase 1 的最优 (agg, alpha, thet |
| `phase4_veto.py` | 2026-11-06 | Task 14: orthogonal veto as the episode gate, after fixing the eta scale. eta was computed as logmeanexp(relu( |
| `probe_repro.py` | 2026-11-05 | Minimal reproducibility probe: same manifest episode, same GPU, two runs. If two separate processes on the SAM |
| `query_gate.py` | 2026-11-05 | Task 1.5 completion: real query-correlation gate for the bridge injection. For each episode, if the strongest  |
| `rebuild_meta.py` | 2026-11-12 | 从 manifest + 已存在的 npz 重建 val_meta（不依赖分段进程写文件）。 |
| `ring1_balance.py` | 2026-11-13 | 第一环：数据与任务划分均衡性核查（10 个 novel 类）。 |
| `ring2_check.py` | 2026-11-13 | 核查 ring2 几何统计的 4 处疑点。 |
| `ring2_export_ply.py` | 2026-11-13 | 第二环：导出 8 个典型任务的 U 点云（原始 RGB + 区域标注）。 颜色约定：B_pos 绿 (0,255,0)，B_neg 红 (255,0,0)， C_pos 黄 (255,255,0)，其余背景灰 (128, |
| `ring2_physical.py` | 2026-11-13 | 第二环：B_pos / B_neg / C_pos 的语义成分与稠密几何特征。 在原始 .npy 稠密点上按 .segs.json 超点分组，计算几何量。 |
| `ring2b_fixed.py` | 2026-11-14 | 10 类 unannotated 统计 + B_pos/B_neg 物理离地高度（按 block 分别算地面）。 z_rel 定义：超点内点的平均 z 减去【该超点所属 block 自己的】最低 z，单位米。 之前误用  |
| `ring2b_unannotated.py` | 2026-11-14 | 10 类 unannotated 统计 + B_pos/B_neg 物理离地高度分布。 z_rel 的定义：超点内所有点的平均 z 减去该重块内所有点的最小 z， 单位为米（切块高 1m，所以数值自然落在 [0, 1]， |
| `ring2c_full.py` | 2026-11-16 | 第二环补全：全 10 类 × 30 eps，按 block 自己的 z_min 算离地高度。 输出每类：B 精确率、unannotated 占比、排除 unannotated 后精确率、 B_neg Top3 混淆类别， |
| `ring3_purify.py` | 2026-11-16 | 第三环：B 提纯策略 A/B/C（全 10 类 × 30 eps）。 指标：过滤后 B 精确率、B_pos 保留率、仍有 B 的任务比例。 禁止按类别写死阈值，所有规则都是类无关的。 |
| `ring4_isolated.py` | 2026-11-16 | 第四环：把不同纯度的 B 作为前景原型送入 COSeg 相关性网络， 测它对查询集的直接贡献（不恢复 C，不训练）。 两种接口： onlyB  fg 原型 100 个全部来自 B S+B    S 前景点集与 B 点集合 |
| `ring4_overlap.py` | 2026-11-16 | 第四环对照诊断：B 超点是否与查询集点云空间重合。 data_prepare_seg 会对每个 block 独立中心化，U 与 Q 的坐标不可比， 所以这里复现其 voxelize/crop 并保留到原始 .npy 的索 |
| `ring4_report.py` | 2026-11-16 | 第四环出表：读 ring4_full.json，输出主表、纯度分组、Tier3 归因。 |
| `ring4_ucover.py` | 2026-11-16 | 第四环补充：U 中真值同类物体对 Q 前景的形态覆盖。 不依赖模型，纯几何。对每个 episode： - U 中 label==cls 的原始点（真值前景）点数 / bbox 体积 - Q 中 label==cls 的原 |
| `run_ablation.py` | 2026-11-13 | 导师文档第五部分的五步消融。 Step 1  原始 COSeg（不使用 U） Step 2  高置信超点 B 直接扩展原型（无 OASB/CSR，不恢复 C，W_phi=I） Step 3  OASB（eta=0），恢复 |
| `run_cmc.py` | 2026-11-12 | 在 val_100 上评测 CMC 注入（Fix 1 + Fix 2 + Fix 3）。 对每个 gamma 跑 4 种注入： baseline  不挂 hook true      真实 delta_corr mean |
| `run_cmc_s2.py` | 2026-11-13 | S2（纯加权相似度，中心化）的 CMC 注入评测。 与 run_cmc.py 的区别：注入信号从 delta_corr（S1）换成中心化的 S2。 |
| `run_s2.py` | 2026-11-13 | S2（纯加权相似度）的端到端评测。 S2 有正确的四分组排序（FN > TP > FP > TN），但整体为正常数，必须先中心化 才能避免全局阈值平移。这里测： centered  : ell = q_margin +  |
| `run_s34.py` | 2026-11-13 | S3 / S4（减背景得分）的 Logit 注入评测。 S3：Σ ŵ_R relu(z_R·z_q - c_bg(i)) S4：模块一/二基准也换成前景-背景裕度后，再算 S3 |
| `same_scene_distance.py` | 2026-11-05 | Does the same-scene gain decay with distance between query and U blocks? Only meaningful within a scene (block |
| `scan_block_classes.py` | 2026-11-13 | 一次性扫描所有 block 的类别，缓存为 json。 |
| `scene_eval.py` | 2026-11-17 | 整场景评估：按 COSeg 官方协议（scenes/data + data_prepare_v101）测 baseline 与 U 的 Oracle 上限。 与 block 口径的区别 ----------------- |
| `scene_merge.py` | 2026-11-17 | 合并 scene_eval.py 的多分片结果，重算 Cumul / Arith / 逐类 / U 命中率。 |
| `scene_sampler_probe.py` | 2026-11-17 | 整场景下三种 voxel_max 采样策略对 baseline IoU 的影响。 v101    官方 data_prepare_v101：全场景随机采样（缓存目录名里的 rdmsp） sphere  球状裁剪 + 中心 |
| `selfcheck_4_4.py` | 2026-11-12 | 导师文档 4.4 节的五项强制检查。 1. B/C 的划分只依赖初始预测 q_c，不使用 U 的真值 2. 负对照 N_c 在训练与测试阶段都能构建（不读 U 标签） 3. 强制 B_c 为空时，查询得分与原始 COSe |
| `stability_signal.py` | 2026-11-05 | Task 2f: cross-episode B stability as an orthogonal signal. A superpoint is identified by (block name, raw seg |
| `stats_blocks.py` | 2026-11-05 | Gate-3 check: block count and superpoints per block. |
| `step1_gate.py` | 2026-11-05 | Step 1: add the end-to-end column to the baseline-stratified table, plus the two idealised gating ceilings. Th |
| `step23_gate.py` | 2026-11-05 | Step 2/3: can an unsupervised episode-level signal predict (a) the baseline IoU and (b) whether bridging will  |
| `summarize_full.py` | 2026-11-12 | 汇总全量 560 eps 的 Step 1-4 与 Step 5 网格结果。 |
| `sweep_fusion.py` | 2026-11-12 | 扫描式(15)修正后的 fusion_mode 与 gamma。 每个 (step, mode) 只前向一次拿 delta_mem，然后在同一增量上扫 gamma。 delta_mem 已乘回 tau，与 margin  |
| `sweep_gamma.py` | 2026-11-12 | 扫 gamma：融合强度从 0 到 0.2，找不伤 IoU 的临界点。 gamma -> 0 时式(15) 严格退回 COSeg。这条曲线能证明掉点来自融合项本身， 而不是 CSR 选错了桥。 |
| `sweep_gamma_fine.py` | 2026-11-12 | Step 4 的 gamma 细化扫描（1.0-4.0），找最优点。 |
| `sweep_step2.py` | 2026-11-12 | Step 2 掉点诊断：扫描 tau 与 Top-K，确认式(15)的量纲问题。 对每个 tau 只前向一次，然后在同一 out 上换 Top-K 与融合方式，比较 Query IoU。同时测一个「margin 校准」变 |
| `test_ocsb_unit.py` | 2026-11-12 | ocsb_faithful 的纯 CPU 单元测试（不加载 COSeg，秒级）。 覆盖：前向形状、B 空退化、三项损失梯度、B/C 与 u_y/q 解耦。 |
| `topk_curve.py` | 2026-11-05 | How does B precision decay as K grows, for the scores we could use? The end-to-end runs only evaluated K in {2 |
| `train_bridge_scorer.py` | 2026-11-17 | 训练 BridgeScorer：把 10 个相关性空间统计量组合成一个原型权重。 参数量 193（10→16→1），远低于 1000，杜绝过拟合。 监督 ---- - 类别平衡 BCE：正样本按 (N_neg / N_p |
| `train_stage2.py` | 2026-11-05 | Stage II: light fine-tuning of the bridge-path HCA only. Design constraints that motivate every choice here: * |
| `train_sweep.py` | 2026-11-12 | Step 5 的 W_phi 训练网格：w_pos / w_reg / lr，逐轮评测。 复用 run_ablation 的缓存加载与评测逻辑。Epoch 0 的评测在更新前做， 用于确认它与 Step 4 的数值完全一 |
| `train_sweep2k.py` | 2026-11-13 | 在 2000 个零泄漏 train split 任务上训练 W_phi，每 500 步评测 val_560。 |
| `u_block_diag.py` | 2026-11-05 | Block/scene-level diagnostics for the bridge set B. The 21 superpoint-level signals all cap at AUC 0.615. This |
| `v3_build_cache.py` | 2026-11-17 | 缓存 BridgeScorer 的训练特征（类别无关的相关性空间统计量）。 对每个 U 块的候选点（B 与邻接模糊带），提取 10 维特征： 1 p0            初始前景概率 2 sp_mean_p0     |
| `v3_build_gate_cache.py` | 2026-11-17 | 缓存 OmissionGate 的训练样本：pred_S / pred_U / 任务级与点级特征。 预测器设计 ---------- `pred_U` 用训练好的 BridgeScorer 重新构造 U 原型集： 1.  |
| `v3g_build_cache.py` | 2026-11-17 | 生成干净的 pred_S / pred_U 缓存。 修复三处算法级问题 + 一处泄漏： A. 不再把两个 U 块的 [2K, C] 原型矩阵按行平均。 两个房间的 FPS 原型在物理部位和语义通道上没有对应关系，按行平均 |
| `v3h_build_cache.py` | 2026-11-17 | 生成 pred_S / pred_U 缓存（含连续权重与单向保护）。 相比 v3g_build_cache 的增量： 1. 保留 BridgeScorer 的连续权重：每个原型槽位的权重取簇内平均 w_u， 而不是把选中 |
| `v3i_intra_query.py` | 2026-11-17 | 查询集内自桥接（U=Q）：在查询集自己身上挑 B_Q / C_Q 构造补充原型。 与跨房间版本的唯一区别：把 `uf` 换成 `qf`。 必须警惕的风险：自匹配 -------------------- `sparse_ |
| `verify_controls.py` | 2026-11-13 | 三个对照，检验 Step 4 的增益是否来自真实空间匹配。 检验 1  常数阈值平移：ell = q_margin - theta（完全不用 U） 检验 2  空间打乱：把 delta_mem 替换为任务均值 / 随机打 |
| `verify_gain.py` | 2026-11-12 | 核查 +2.13pp 的两个乐观来源。 1. 插值控制：baseline 用原始点级 margin，Step4 用 downsampled margin 插值。 如果 interp(margin_down) 本身的 Io |

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

