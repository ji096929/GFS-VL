# 实验数据导览

> 最后更新：2026-11-22 20:30
> 这份文件是**入口**。具体数字去 `LEDGER.md`，命名含义去 `NAMING.md`，产物路径去 `ASSETS.md`。

---

## 一、台账总览：111 个实验

| 文件 | 内容 | 什么时候看 |
|---|---|---|
| `LEDGER.md` | 111 个实验的目录名、结构参数、种子、双口径指标、状态 | 想知道「某个实验跑过没有、结果多少」 |
| `NAMING.md` | 目录名里每个词的含义（`sym` / `lcross` / `rep3` / `SW3` / `XLu`…） | 想看懂一个陌生目录名 |
| `ASSETS.md` | 717 GB 各类缓存的路径、大小、重建成本、删除风险 | 想清理磁盘或重建缓存 |

**台账的分节**（按研究问题）：

| 节 | 实验数 | 内容 |
|---|---|---|
| `00_infra` | 6 | 备份与基础设施 |
| `01_cross_layer` | 33 | 跨层读出主干 + 12 个结构变体 |
| `02_loss` | 2 | Lovász / LA |
| `03_pseudo_label` | 16 | ONLINE / offline / SW3 / A2 / XLu / BASEu |
| `04_ontr_frozen` | 17 | ONTR / 冻结骨干 / support 微调 |
| `05_sonata` | 4 | Sonata-Small 基模替换 |
| `07_shot_protocol` | 33 | 1-shot / 5-shot / 102400 点 |

**今天新增的 9 个实验全部集中在 `LEDGER.md` 的 `08 ScanNet++（跨数据集验证）` 一节**（共 16 个实验）。
口径备注列会显示 `PL:entropy_q` / `PL:student_consensus,warm4,tau0.40` 等，非 `ps_ai` 模式下 PS/AI 参数不显示（它们是死参数）。

---

## 二、今天新增的 9 个实验（ScanNet++ 5-shot）

**口径**：单折 `regis1`（seed=10）、纯 CE、确定性评估（`shuffle_orders` 只在训练生效）、`point_max=102400`、lr=7e-3/7e-4（1A 除外）。

| # | 实验 | 一句话目的 | 结果（Base/Novel/HM） | 结论 |
|---|---|---|---|---|
| 1A0 | 基线确定性重评 | 建立干净锚点 | 0.6126 / 0.2111 / **0.3140** | 锚点（抖动口径是 0.3150，差 0.10） |
| 1A | lr=1e-3/1e-4 | 剥离 UGA 的 7× lr 差异 | 0.6224 / 0.1658 / 0.2618 | **−5.22**，lr 不能裸降 |
| 1B | 关 AI（`ai_thresh=2.0`） | 测 AI 的真实贡献 | 0.6189 / 0.1752 / 0.2731 | **−4.09**，AI 是正贡献 |
| 1C-a | 屏蔽 base + 熵过滤 ρ=0.1 | 验证熵过滤方向 | 0.6040 / 0.2468 / **0.3505** | **+3.65**，静态最佳（含 bug） |
| 1C-b | 同上，ρ=0.5 | 过滤强度 | 0.5238 / 0.1767 / 0.2642 | −4.98，**过滤要狠** |
| 1C-c | PS+AI 后再叠熵过滤 | 复现 UGA 配方 | 0.6231 / 0.2387 / 0.3452 | +3.12，**不能叠加**（保 Base 伤 Novel） |
| 1C-e | 修 `bg_mask` bug 后重跑 | 量化 bug 的净效应 | 0.5924 / 0.2404 / 0.3420 | 修 bug **反而 −0.85** |
| L1-Pure | Ep0-3 纯 GT 热身 + 学生共识 | 零 UGA 路线 | 0.6176 / 0.2205 / 0.3250 | +1.10，Base 最高但 Novel 不够 |
| L1-Hybrid | Ep0-3 熵过滤热身 + 学生共识 | 组合 | 0.6226 / 0.2227 / 0.3281 | +1.41 |

**对照**：UGA-Entropy（论文）= 0.6343 / 0.2592 / **0.3678**。

### 按问题分组

**Q1：UGA 的 lr 能不能直接搬？** → 1A。答案：不能，−5.22。

**Q2：AI（Adaptive Infilling）是正贡献还是噪声？** → 1B、1C-c。
答案：+4.09，正贡献；PS+AI 保 Base（+1.91）但伤 Novel（−0.81）。

**Q3：UGA 的熵过滤在我们数据上成不成立？** → 1C-a/b/e。
答案：成立（+3.65），但 ρ 必须狠（0.1 比 0.5 好 8.6）；修 bug 后 −0.85，说明 1C-a 的增益部分来自 bug 的「意外严格」。

**Q4：能不能用学生仲裁替代熵过滤？** → L1-Pure/Hybrid、Level 0。
答案：伪标签质量能提升 2.4 倍（Level 0），但训练中的学生不如训练完的学生，HM 只有 0.325–0.328。

### 文件路径

| 类型 | 路径 |
|---|---|
| config | `configs/scannetpp/semseg-pt-v3m1-0-gfsregistrain_k5{,_lr1e3,_noai,_entq10,_entq50,_psaithenq,_l1pure,_l1hybrid,_l1smoke}.py` |
| 日志 | `/data01/jh/data/scpp_day1_1A_lr1e3.log`、`scpp_day1_1B_noai.log`、`scpp_day1_1C_entq10.log`、`scpp_day1_1C_entq50.log`、`scpp_day1_1Cc_psaithenq.log`、`scpp_day1_1Ce_fix.log`、`scpp_l1pure.log`、`scpp_l1hybrid.log` |
| 权重 | `exp/scannetpp/scpp_{base_lovasz_k5_lr1e3,base_lovasz_k5_noai,base_lovasz_k5_entq10,base_lovasz_k5_entq50,base_lovasz_k5_psaithenq,base_lovasz_k5_entq10_fix,l1pure,l1hybrid}/model/regis1_model_last.pth` |
| 基线权重 | `exp/scannetpp/scpp_base_lovasz_k5/model/regis1_model_last.pth` |

---

## 三、今天的诊断数据（不是训练实验，但结论更硬）

| 数据 | 文件 | 结论 |
|---|---|---|
| **Base 抢占摸底** | `data/probe_topk_scpp2.json` | 49.50% 的真 novel 点 Top-1 被 12 个 base 类抢走；屏蔽后 Top-1 0.4734 → 0.7943（+32.1 pp）；Top-2/3 = 0.8376 / 0.8746 |
| **Level 0 仲裁对比** | `data/level0_full.json` | 50 个未见 val 场景，零泄漏：A（熵过滤）P 0.209 / R 0.123 / mIoU 0.064；B2（学生仲裁）P 0.353 / R 0.460 / mIoU **0.152** |
| **熵-模长相关** | 见 `tools/probe_norm_entropy_corr.py` | 原始 `seg_scores` 的熵与 `||f||` 的 Spearman = **−0.928**；余弦化后降到 −0.397 |
| **熵分布** | 见 `tools/probe_entropy_dist.py` | novel 点熵 2.65–3.29（接近 30 类上限 ln30=3.40）；base 点 9% 熵≈0 |

**脚本**：`tools/probe_topk_recall.py`、`tools/probe_level0_arbitration.py`、`tools/probe_entropy_dist.py`、`tools/probe_norm_entropy_corr.py`。

---

## 四、今天的三条硬结论

1. **UGA 的 lr 不能单独搬**（−5.22）。它靠 UG-Update 的梯度保护才扛得住小 lr。
2. **AI 是正贡献 +4.09**，机制是「在 18 个 novel 原型间匹配，捞回被 base 抢走的 49.5%」。
3. **静态伪标签方案的顶点在 0.342–0.351**（屏蔽 base + 熵过滤），距 UGA 的 0.3678 差 1.73，差距在 Base（我们 0.60 vs 它 0.63）。

## 五、代码改动（今天新增，都可回滚）

| 文件 | 改动 | 开关 |
|---|---|---|
| `pointcept/engines/train.py` | 1C：`_target_entropy_quantile`（屏蔽 base + 余弦熵分位） | `pl_mode="entropy_q"` |
| 同上 | 1C-c：`_entropy_prune`（PS+AI 后叠熵过滤） | `pl_mode="ps_ai_then_entropy"` |
| 同上 | Level 1：`_student_consensus_pl`（学生 × Top-K 共识）+ `run_step` 复用 `seg_logits.detach()` | `pl_mode="student_consensus"` + `warmup_epochs` |
| 备份 | `train.py.bak_1c_entropy` / `.bak_1cc` / `.bak_l1` / `.bak_l1b` | — |
| `tools/` | `gen_ledger.py`（台账生成）、4 个诊断探针 | — |

**切换方式**：config 里设 `pl_mode`，缺省 `ps_ai` 保持原 GFS-VL 流程不变。
