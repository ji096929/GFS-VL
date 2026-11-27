# 项目总览：我们做了什么，结论是什么

> **后来者先读这份。** 10 分钟看完整个项目的脉络。
> 数字全部来自 `LEDGER.md` 的 `Eval result`，结构全部经权重文件验证。
> 最后更新：2026-11-27

---

## 一、任务

**Generalized Few-Shot 3D Point Cloud Segmentation（GFS-PCS）**

给一个房间的点云，判断每个点属于哪一类。30–57 个类分成两组：

| | 标注量 | 难点 |
|---|---|---|
| **基类**（12 个） | 大量 | 不能忘 |
| **新类**（18 / 45 个） | 只有 1 个或 5 个标注场景 | 要泛化 |

基线是 **GFS-VL**（CVPR 2025）：PTv3 主干 + 两个线性分类头（基类 / 新类），
第二阶段用新类的少量样本微调 20 epoch，在线生成 VLM 伪标签。

两个数据集：

| | 场景数 | 类别 | 1-shot 基线 | 5-shot 基线 |
|---|---|---|---|---|
| **ScanNet++** | 230 train / 312 val | 12 base + 18 novel | 0.2880 | 0.3099 |
| **ScanNet200** | 1201 train / 312 val | 12 base + 45 novel | 0.4005（离线口径） | 0.4279 |

---

## 二、实验流程（时间线）

### 阶段 1（11-19 ~ 11-20）：建立基线，引入跨层

- 复现 GFS-VL 基线，确认官方 config 与论文的差异（点数 65536 vs 102400、折数单折 vs 五折）
- **Cut 2**：把解码器中间层 dec1/dec2 接到分类头上，两种接法：
  - `xl`：只有新类头接跨层
  - `sym`：新类头和基类头都接（`base_cross=True`），新类走 192 维拼接
- 结论：`sym` 在 ScanNet200 1-shot 上 **+1.66**（0.4244 → 0.4410，3/3 seed 全正）

### 阶段 2（11-21 ~ 11-22）：三条"换主干/换方向"的路全部证伪

| 尝试 | 最好结果 | 结论 |
|---|---|---|
| **冻结主干 + ONTR** | 0.2955 | 全参微调是 0.4122，**差 11.7 点** |
| **换 Sonata-Small 基模** | 0.4206 | 低于 `sym_lovasz` 的 0.4414 |
| **3D-OCSB 桥梁判据** | AUC 0.94 | 阈值化后 precision 只有 **3.1%**，类不平衡吃掉一切 |

**这三条路都已归档到 `docs/_archive/` 或 `docs/01_cross_layer/OCSB_BRIDGE_DECLINE`，不要再走。**

同时确立了一条方法论：**AUC / 精确率这类排序指标，对阈值化后的可用性没有预测力**。

### 阶段 3（11-23 ~ 11-24）：拆开跨层为什么有效

用三个诊断（有效秩 / 线性探针 / 梯度占比）回答「dec1 和 dec2 各自贡献多少」：

| 检查 | dec1 | dec2 |
|---|---|---|
| 拼接后归一化秩 | cat192 = 16.9%，比 dec0 的 16.0% 只高 0.9 点 | 同 |
| base 权重下的探针 | **0.4174（比 dec0 的 0.4422 还低）** | 0.4317 |
| 跨层梯度占比 | 42.1% | **81.4%** |

**结论：dec1 既没额外信息、又没被训练改写；dec2 是唯一被真正重塑的层。**

但 `detach` 对照发现：**dec2 的判别力高度依赖梯度通路**——掐断后从 0.4884 掉到 0.3945，比没训练过的 base（0.4317）还低。

**这引出了两个后续：**
1. 只留 dec2（`M5` 结构）
2. 「dec2 的特征能不能在推理时兑现」成了一个真问题（后面 `psai_d2r10` 试了，在 ScanNet200 1-shot 上失败）

### 阶段 4（11-25）：结构消融 + 全局熵分位

- 跑了 14 个跨层结构变体（m1–m11b），**全部没超过 base**
- 意外收获：**全局熵分位（`geq02`）在 ScanNet++ 5-shot 上 +4.09**，靠改伪标签筛选口径而非结构

### 阶段 5（11-26）：伪标签筛选六方案

在 `PS+AI` 之外试了：`entq10` / `geq02` / `dualeye85/90` / `h1` / `m1vote` / `psai_dec2_rank` / `gce07` / `elr` / `cleancon` / `fullent`。

| 结论 | 数据 |
|---|---|
| Sc++ 5-shot 最强 | `psai_d2r10` **0.3647**（+4.58） |
| Sc++ 1-shot 最强 | `m1vote` **0.2877**（+2.04） |
| ScanNet200 1-shot 最强 | `fullent30_pp` **0.4088** |
| ELR | **崩了**（loss 变负发散），不适用伪标签设定 |
| cleancon | 本质是基类原型锚定，方向不对，均值正但 s13 是 −1.40 |

### 阶段 6（11-27）：口径审计，定稿 12 格

查出三个问题并修掉：
1. `config.py` 被覆盖过（34 个 run），结构判定改用**权重 zip key**
2. `sym_xl ≡ sym` 的冗余命名，8 个目录改名
3. 在线/离线、65536/102400、CE/Lovász 三个口径混用

---

## 三、当前定稿的 12 格

**三 seed（s42 / s7 / s13），regis1 单折，纯 CE。**

| 数据集 | shot | base | sym | M5 | sym Δ | M5 Δ |
|---|---|---|---|---|---|---|
| ScanNet++ | 1 | 0.2880 | 0.2809 | **0.2909** | **−0.71** | **+0.29** |
| ScanNet++ | 5 | 0.3099 | **0.3311** | 0.3165 | **+2.12** | +0.66 |
| ScanNet200 | 1 | 0.4005 | **0.4222** | 0.4124 | **+2.17** | +1.19 |
| ScanNet200 | 5 | 0.4279 | **0.4390** | 0.4320 | **+1.11** | +0.41 |

### 两个结构是什么

**`sym`（主方法）**：
```
新类 logits = Linear_192([LN(dec0), proj(dec1), proj(dec2)])
基类 logits = Linear_64(dec0) + Linear_128_zero([proj(dec1), proj(dec2)])
```

**`M5`（1-shot 稳健版）**：
```
新类 logits = Linear_64(dec0) + Linear_64_zero(proj(dec2))
基类 logits = Linear_64(dec0) + Linear_64_zero(proj(dec2))
```

**关键差别**：sym 的新类走 192 维拼接（从第 0 步就带跨层噪声）；M5 两个跨层头**零初始化**，训练初期完全等价基线。

### 怎么读这张表

- **sym 只在 ScanNet++ 1-shot 为负（−0.71）**，其余三格 +1.1 ~ +2.2
- **M5 四格全正（+0.29 ~ +1.19），但幅度小**
- **M5 唯一赢 sym 的格子，就是 sym 唯一输的那格**
- 所以 M5 不是「更好的 sym」，是「support 极少时不会反噬的保险版本」

---

## 四、关键数字速查

| 项 | 值 | 出处 |
|---|---|---|
| 系统理论上界（Sc++ 全监督） | 0.4905 | `scpp_oracle_fullsup20` |
| 环节完美上界（Sc++ 完美伪标签） | 0.4120 | `scpp_oracle_filter_m0` |
| 剩余空间 | 18.06 点 | 0.4905 − 0.3099 |
| seed 方差（Sc++ 1-shot） | **可达 3.7 点** | 随机 seed vs s42 |
| 加 Lovász（ScanNet200 1-shot） | +2.85 | 0.4216 → 0.4501 |
| 冻结主干（ScanNet200 1-shot） | −11.7 点 | 0.4122 → 0.2955 |

---

## 五、文档地图

| 想知道什么 | 看哪份 |
|---|---|
| **项目全貌 / 时间线** | **本文档** |
| 某个实验跑过没有、多少分 | `LEDGER.md`（228 个，自动生成） |
| 哪些数字作废、口径怎么锁 | `AUDIT_2026-11-27.md` |
| 目录名里每个词什么意思 | `NAMING.md` |
| 每个 run 的真实结构（base/xl/sym/M5） | `STRUCTURE_MAP.md` |
| 跨层方法的完整证据链 | `01_cross_layer/` 六份 |
| 伪标签筛选的全部方案 | `03_pseudo_label/PL_FILTER_WAVE2_2026-11-26.md` |
| 失败实验的机理分析 | `06_diagnostics/` 三份 |
| 已放弃的方向 | `_archive/README.md` |
| 服务器上有什么、能删什么 | `ASSETS.md` |

---

## 六、复现一个实验

```bash
cd /data01/jh/code/GFS-VL

export PYTHONPATH=/data01/jh/code/GFS-VL:/data01/jh/code/GFS-VL/pointcept/models/PLA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GFS_RESERVE_GB=30                     # 必带，防别人抢显存
export GFS_OFFLINE_PL=/data01/jh/data/online_pl_w_uniform   # 仅离线口径需要；ScanNet++ 不能设

# 例：ScanNet++ 1-shot sym
python tools/train.py \
  --config-file configs/scannetpp/semseg-pt-v3m1-0-gfsregistrain_k1.py \
  --num-gpus 1 \
  --options save_path=exp/scannetpp/scpp_sym_k1_s42 seed=42 \
            regis_train_list=["regis1"]

# 日志在 exp/<ds>/<run>/train.log，结果看 "Eval result" 行
```

**三个必守的坑**：
1. `regis_train_list` 默认是 5 个 split，必须显式写 `["regis1"]`
2. `GFS_OFFLINE_PL` 只存在于启动命令，config 不记录——**这就是为什么后来需要按权重判结构**
3. ScanNet++ 的 dataset 代码没有离线分支，设了也没用

---

## 七、工具

| 脚本 | 用途 |
|---|---|
| `tools/gen_ledger.py` | 重新生成 `LEDGER.md`（含在线/离线列） |
| `tools/gen_cells.py` | 列出所有三 seed 齐全的格子，**结构从权重判定** |
| `tools/gen_structure_map.py` | 生成全库「结构 vs 配方」表 |
| `tools/restore_config.py` | 从 `train.log` 恢复被覆盖的 `config.py` |
