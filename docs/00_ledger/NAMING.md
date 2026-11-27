# GFS-VL 目录命名辞典

> 每一条都用 `config.py` 或源码核实过，标注证据位置。**没有核实的写「未记录」，不猜。**
> 最后更新：2026-11-22

---

## 一、目录名语法

```
<结构>[_<损失>][_k<shot>][_<伪标签>][_<点数>][_<其他>][_s<种子>]
```

例子：

| 目录名 | 拆解 |
|---|---|
| `sym_lovasz_s42` | 对称跨层 + Lovász + 1-shot + seed 42 |
| `sym_lcross01_k5_102400_s42` | 对称跨层 + L_cross λ=0.1 + 5-shot + 102400 点 + seed 42 |
| `frozen_ontr_rep3_lovasz_s42` | 冻结主干（ONTR 权重）+ rep3 读出 + Lovász + seed 42 |
| `sc200_k1_XLu_s42` | ScanNet200 + 1-shot + xl 读出 + 均匀权重 PL + seed 42 |

成分可缺省。**缺省 ≠ 默认值，要逐个看 config。**

---

## 二、结构词（全部由 config 参数决定）

证据：`pointcept/models/default.py:99-230`（`RegisTrainSegmentor`）

| 词 | config 参数 | 实际含义 |
|---|---|---|
| （无）/ `base` | — | 基线。`novel_seg_head = Linear(64→45, bias=False)`，只吃 dec0 |
| `xl` | `cross_layers=(1,2)` `cross_dims=(64,128)` `cross_dim=64` | 跨层读出。novel head 吃 `[LN(dec0), proj(dec1), proj(dec2)]` = **192d** |
| `sym` | `xl` + `base_cross=True` | **对称**。Base 头额外加零初始化残差 `Linear(128→12, bias=False)`，吃 `proj1/proj2` |
| `sym_xl` | **已废弃（2026-11-27）** | 与 `sym` 定义完全相同，纯命名冗余。当天已把 8 个目录全部改名（`sc200_k1_sym_*` / `sc200_gated_*` / `sc200_ortho_*` / `sc200_k5_sym_65536_s42`），**全库已无此名**。以后统一用 `sym` |
| `lcross` | `sym` + `lam_cross=0.3` | 加 L_cross 损失，λ=0.3 |
| `lcross01` | `sym` + `lam_cross=0.1` | 加 L_cross 损失，λ=0.1 |
| `ortho` | `sym` + `decorr_weight=0.1` | 加去相关正则（orthogonality），权重 0.1 |
| `gated` | `sym` + `depth_gate=True` | 逐点门控 `Linear(192→64→2)` 缩放 `proj1/proj2`，末层零初始化 |
| `rep3` | `sym` + `rep_dec0=True` | **对照**：dec0 复制两份走独立投影。结构与 sym 相同，但特征源**只有 dec0**（证明增益来自「真跨层」而非「多两条支路」） |
| `mlp_dec0` | `mlp_hidden=192` | **对照**：dec0-only 两层 MLP（LN→192→GELU→LN→45），无跨层 |
| `lnonly` | `ln_only=True` | **对照**：novel head 前加一层 LayerNorm，其余同基线 |
| `detach` | `sym` + `detach_cross=True` | 跨层分支梯度隔离（`proj1/proj2` 不回传到 decoder） |

**四个容易搞错的点：**

1. **`sym` 本身已含 `xl`**。`sym_xl_s42` 和 `sym_lovasz_s42` 的差别只是损失（CE vs Lovász），不是结构。
2. **`rep3` 是假跨层**。它和 `sym` 层数、参数完全一样，但输入的 dec1/dec2 被替换成 dec0 的复制。它存在的唯一目的就是做这个消融。
3. **`mlp_dec0` 参数量比 `sym` 还多**，但没有跨层信息。它排除了「增益只是来自更多参数」的解释。
4. **`base` 不是「base 类专用」**，它是「没有任何跨层增强」的基线。

---

## 三、损失词

| 词 | config 参数 | 含义 |
|---|---|---|
| （缺省） | `criteria=[dict(type='CrossEntropyLoss', ignore_index=-1)]` | 纯 CE |
| `lovasz` | `use_lovasz = True` | CE + Lovász-Softmax（`lovasz_w` 默认 1.0） |
| `la` | `use_la = True` `la_tau = 0.5` | CE + **训练期 Logit Adjustment**（Menon et al., ICLR 2021）。`logits += 0.5·log(π)`，π 来自 `pl_class_prior_full.npy` |

证据：`pointcept/engines/train.py:721-736`

**注意**：`la` 是**训练期**偏置，和诊断 A 的推理期 δ 扫描是两回事，不要混。

---

## 四、口径词

| 词 | 含义 | 证据 |
|---|---|---|
| `k1` / `k5` | 1-shot / 5-shot support 集 | `regis_train_list` + support 采样 |
| `102400` | `SphereCrop(point_max=102400)`，默认 65536 | config `transform` 段 |
| `true` / `truemulti` | 真 5-shot（多实例多场景），对比早期近似实现 | config |
| `s42` / `s7` / `s13` | 随机种子（**已写进 config 的 `seed` 字段**） | 台账「种子」列 |
| `ep4/8/12/16` | 余弦退火调度下只训 N 个 epoch（学习率退火周期消融） | `epoch = N` |
| `a080` / `a090` | `novel_logit_scale = 0.8 / 0.9`，推理前对 novel logits 乘性缩放 | config 顶层 |
| `XLu` / `BASEu` / `A2` | 均匀权重 PL（`online_pl_w_uniform`），u = uniform | `RUNBOOK.md` |
| `offline_A` / `offline_B` | 旧版离线 PL 的 A/B 两套（`gen_ab.py`） | `SERVER_CACHE_INDEX.md` |
| `regis1` | 官方 Model Zoo 权重（`epoch 20`, `optimizer.step 1999`） | `docs/_archive/ASSET_README.md` |

**`a080/a090` 的额外事实**：这两个实验的 config 里 `weight` 指向 `sym_xl_k5_s42/model/regis1_model_last.pth`，`seed=22515937`，即**从 sym_xl_k5 权重续训 20 epoch**，不是从头训。

---

## 五、伪标签来源词

| 词 | 环境变量 / 目录 | 含义 |
|---|---|---|
| `ONLINE` | 不设 `GFS_OFFLINE_PL` | 官方在线 PS（VLM 前向 + 原型提取），最慢但最贴近原论文 |
| `offline` | `GFS_OFFLINE_PL=/data01/jh/data/offline_pl` | 旧版离线 PL（4.3 G） |
| `SW3` | `GFS_OFFLINE_PL=/data01/jh/data/online_pl_w` | 逐点软权重 v3（`0.25+0.75σ(2z)` / `0.05+0.55σ(2z)`） |
| `SW2` | 同上早期版 | **废弃**，启动时 `Collect` 漏字段，已 kill |
| `A2` | `GFS_OFFLINE_PL=/data01/jh/data/online_pl_w_uniform` | 均匀权重消融 |
| `XLu` / `BASEu` | 同上 | 均匀权重 + 对应结构 |

**口径缺口（必须知道）**：`GFS_OFFLINE_PL` **只存在于启动命令**，`config.py` 和 `train.log` 都不记录。
43 个离线 PL 实验的实际 PL 版本目前只能靠目录名 + `exp_reports/RUNBOOK.md` 推断。

**2026-11-27 补：判据已经找到，不再靠猜。**

```bash
# 10 秒定性任意一个 run
grep -c 'sparseunet32_636.pth to GPU' exp/<dataset>/<run>/train.log
# 1 = 在线（VLM 被加载），0 = 离线（VLM_3D = None）
```

原理：`train.py:373` 的 `if not os.environ.get("GFS_OFFLINE_PL", "")` 是唯一分叉点，
离线时整个 3D VLM 都不构建，日志里不可能出现 `network_template.py` 的输出。

**实测分布（2026-11-27 全量扫描）：**

| 数据集 | 在线 | 离线 | 说明 |
|---|---|---|---|
| **ScanNet++** | **97** | **0** | `scannetpp.py` 没有 `GFS_OFFLINE_PL` 分支，物理上不支持 |
| **ScanNet200** | **21** | **85** | 混用严重 |

**离线缓存与 shot 数绑定**：`online_pl_w_uniform` 是用 **1-shot 原型**生成的。
1-shot 训练读它匹配；5-shot 训练读它等于白扔 4 个 support（`CUT2_EXT_REPORT.md:224` 已记录）。

完整审计见 `AUDIT_2026-11-27.md`。

证据：`pointcept/engines/train.py:373,577-584,880`；`pointcept/datasets/scannet.py:423`

---

## 五之二、伪标签筛选词（2026-11-25 ~ 26 新增，全部为在线 VLM，不读 `GFS_OFFLINE_PL`）

> 这一批命名是实验过程中随手拼的，**没有统一规则**：有的是英文取首段字母 + 参数，
> 有的是消融编号。下表逐条对应到 config 里的真名。
> **最容易踩的坑：`dualeye85` 里的 85 是分位数，保留比例是 0.15；而 `entq10` / `d2r10`
> 里的 10 是保留百分比。同一个数字在两套命名里方向相反。**

| 缩写 | config 真名 | 中文含义 |
|---|---|---|
| `ps_ai` / `psai` | `pl_mode='ps_ai'` | 官方原型筛选（余弦 ≥ 0.6）+ 自适应填充（余弦 ≥ 0.95） |
| `entq10` | `pl_mode='entropy_q'`，`entropy_keep_rho=0.1` | **每类**保留熵最低的 10% |
| `geq02` / `geq05` | `pl_mode='global_entropy_q'`，`global_entropy_rho=0.02/0.05` | **全局**保留熵最低的 2% / 5%，动态分位 + EMA 0.9 |
| `dualeye85` | `pl_mode='dual_eye_and'`，`dual_eye_q1=q2=0.15` | 两视角各取前 15%，取交集 |
| `dualeye90` | 同上，`dual_eye_q1=q2=0.10` | 两视角各取前 10%，取交集 |
| `mnv` | `dual_eye_score='margin_nv'` | 第二层特征里，对被判新类的相似度要高于其他任何新类 |
| `h1` | `pl_mode='dual_eye_h1'` | 训练前 30 batch 校准：全局大津阈值 + 每类熵前 10% + 新类边际 > 0 |
| `m1vote` | `pl_mode='dual_eye_m1_vote'` | 全类别取最大值 + 第二层边际投票 |
| `d2r10` | `pl_mode='psai_dec2_rank'`，`dual_eye_rank_alpha=0.10` | PS/AI 全保留后，第二层在 N_novel 内部排名进前 10% 才留 |
| `gce07` | `pl_mode='ps_ai'`，`pl_gce_q=0.7` | 伪标签点改用广义交叉熵 `(1-p^q)/q`，q=0.7；GT 点仍是普通 CE |
| `m5` | `cross_zero_head=True`，`cross_layers=(2,)` | 新类头只吃末层；基类头额外加第二层跨层。**只置零新类侧的跨层，基类侧是活的** |

**`m1`~`m11b` 是 Wave3 结构消融的编号**，不是方案名，见
`docs/01_cross_layer/XL2_WAVE3_AND_GEQ_2026-11-25.md` §1。

**`Sc++` = ScanNet++，`Sc200` = ScanNet200，`k1`/`k5` = 1-shot / 5-shot，`s42/s7/s13` = 随机种子。**

---

## 六、冻结 / 主干词

| 词 | config 参数 | 含义 |
|---|---|---|
| `frozen` | `freeze_backbone=True` | 主干完全冻结，只训读出头与分类头 |
| `ontr` | `backbone_weight = './exp/scannet200/ptv3_bk_ontr.pth'` | 主干换成 ONTR 重参数化后的 Stage-1 权重 |
| `support` | `freeze_backbone` + support 集微调 | **全部 5 个 `support_*` 实验作废**：PL 只在 45 个 support 场景读取 + `gfs==-1` 筛出 100% 假阳性 |
| `sonata` | `backbone.type='PT-v3m1'` + `norm_all_ln=True` + `embedding_type='linear'` | Sonata-Small 38.65M 权重迁移（全 LayerNorm、9 通道输入） |
| `spunet` | `backbone.type='SpUNet-v1m1'` | SpUNet 基线 |

**`frozen_base` vs `frozen_ontr` 的唯一差别**就是 `backbone_weight` 那一行（原始 Stage-1 权重 vs ONTR 权重）。

---

## 七、状态词（台账 `状态` 列）

| 状态 | 含义 |
|---|---|
| `done` | 有 val 结果 |
| `unevaluated` | 训完了、有权重，但从未跑过评估 |
| `incomplete` | 空壳或主动终止 |
| `failed` | 训练/评估崩溃（日志有 Traceback / OOM） |
| `invalid` | 实现有 bug，结果不可用 |
| `infra` | 基础设施/备份，不是实验 |

---

## 八、目录名对照速查（历史遗留，最易找错）

| 报告里的名字 | 实际目录 | 说明 |
|---|---|---|
| `base_s42` | `sc200_k1_A2_s42` | 软链 + 命名不一致，最坑 |
| `base_s7` | `sc200_k1_BASEu_s7` | 同上 |
| `xl_s42` | `sc200_k1_XLu_s42` | 早期 xl 实验 |
| `xl_s7` | `sc200_k1_XLu_s7` | 崩溃，无结果 |
| `sym_xl_s42/s7/s13` | 同名 | CE 口径的对称跨层 |
| `sym_lovasz_s42/s7/s13` | 同名 | **当前主方法的三个 seed** |

来源：`docs/_archive/ASSET_README.md`（2026-11-20 冻结）
