# 跨层 × 3D-OCSB 桥梁判据：失败归档（2026-11-26 补档）

> **原始记录**：`docs/_archive/SESSION_STATE.md` §8.3（2026-11-21 00:25 快照）
> **探针脚本**：`tools/probe_omission.py`（253 行；副本在 `legacy_scripts/ps_diag/probe_omission.py`）
> **状态**：**已砍**。本文档把当时只有一张表 + 一句「已砍」的结论补成可复核的完整档案。

---

## §1 当时的实验设计

用 3D-OCSB 的桥梁结构去问一个问题：**能否用 PTv3 的高秩层（dec1/dec2）区分「PS 漏检的前景」和「真的背景」？**

脚本头部的定义（`probe_omission.py:8-13`，逐字）：

```
B   = {ps == c}                     # PS 已标，用来算该类原型
C   = {gt == c} & {ps == -1}        # 漏检正样本（要恢复的）
Neg = {ps == -1} & {gt != c}        # 负样本
Prototype = mean feature of B. Similarity = cosine. Metric = AUC(C vs Neg).
Layers compared: dec0 (collapsed), dec1 / dec2 (high-rank), and a Base-reject
```

- 场景数：**50**
- 相似度：点特征与「B 池均值原型」的余弦
- Base 拒斥版：`s = cos(z, proto_c) − max_b cos(z, proto_b)`，b 取 12 个基类
- 阈值扫描：0.5 / 0.6 / 0.7 / 0.8 / 0.85 / 0.9 / 0.95

---

## §2 原始数字（§8.3 原表，未改）

| 特征空间 | AUC | AUC + Base 拒斥 |
|---|---|---|
| **RegionPLC（基线）** | **0.9323** | 0.9162 |
| PTv3 dec0 | 0.8444 | 0.9262 |
| PTv3 dec1 | 0.9133 | **0.9478** |
| PTv3 dec2 | 0.9222 | **0.9433** |

**阈值化后 Precision：3.1–3.3%（多类 argmax，thr=0.9）。AUC 高但 Precision 不可用。**

---

## §3 失败的确切形态

### 3.1 单看 AUC，跨层全线不如基线

```
RegionPLC  0.9323   ← 基线最高
dec2       0.9222   (−1.01)
dec1       0.9133   (−1.90)
dec0       0.8444   (−8.79)
```

**唯一的例外是「加 Base 拒斥后 dec1 反超」（+3.16）和 dec2（+2.71）——这也是整个项目里唯一一次跨层在 OCSB 判据上赢过基线。**

### 3.2 但这个「赢」兑不出来

**阈值 0.9 时 precision 只有 3.1–3.3%。** 被判为「遗漏前景」的点里 97% 是假的。

原因不是判据错，而是**任务本身的类不平衡**：`Neg`（背景）点数远多于 `C`（漏检前景），AUC 是排序指标，对绝对阈值下的 precision 几乎没有预测力。

**这条判据的 AUC 0.94 与 Exp B 的「精确率 0.968」、SIGNAL_AUROC 的「S1 AUROC 0.754」是同一个陷阱——见 §4。**

---

## §4 与另外两次「AUC 高、precision 崩」的对照

| 记录 | 排序指标 | 阈值化后的真实可用性 |
|---|---|---|
| **§8.3 跨层 OCSB 桥梁** | AUC **0.94**（dec1 + Base 拒斥） | Precision **3.1–3.3%** |
| `EXP_B_VETO_2026-11-23.md` | 否决精确率 **0.968** | 只清掉 23% 假点；误杀 15.7% 真 novel（集中在贴附类） |
| `SIGNAL_AUROC_2026-11-23.md`（v2） | S1 AUROC **0.709**（中位 0.746） | 每类 top-10% 纯度 0.283；smoke detector 的 P@10% 仅 **0.003** |

**三次的共性：在 45 类长尾 + 背景占比 90%+ 的设定里，AUC / 精确率这类排序指标对「阈值化后留下多少真点」没有预测力。任何新判据在进训练前，必须先看阈值扫描表，不能只看 AUC。**

---

## §5 与 XL-1 的机理闭环

§8.3 用的是训练好的 `sym` 权重，所以 dec2 的判别力看起来不错。XL-1（`XL1_GRADIENT_HIGHWAY_2026-11-24.md`）后来补上了机理层：

| 权重 | dec0 探针 | dec1 | dec2 |
|---|---|---|---|
| base_s7 | **0.4422** | 0.4174 | 0.4317 |
| sym_s7 | 0.3835 | 0.4234 | **0.4884** |
| **detach_sym_s7** | **0.3432** | **0.3349** | **0.3945** |

**截断跨层梯度后，dec2 的探针从 0.4884 掉到 0.3945（三层全最低）。**

**结论：§8.3 里 dec2 那 0.92 的判别力，是靠「新类损失经跨层支路倒灌进 dec2」（ep20 时 dec2 的梯度 81.4% 来自跨层）灌出来的，不是 dec2 自己「看见」的。**

**而 OCSB 的桥梁池只需要「眼睛」，不需要「梯度旁路」。** 所以即使换成 high-rank 的 dec2，也拿不到它真正需要的那个能力。

---

## §6 口径备注（复核时必须知道）

1. **dec2 的维度在两个文档里不同**：§8.1 记的是 64 维（eRank 25.32 / 39.6%），`probe_omission.py` 的 `DIM` 表写 `dec2: 128`，XL-1 的 eRank 也是 128 维口径（40.51）。**比较时不要混。**
2. **Base 原型有 placeholder 坑**：`probe_omission.py:102` 写的是
   ```python
   base_protos[b] = unit(feat['dec1'][m].mean(0))  # placeholder dim
   ```
   **Base 拒斥用的是 dec1 特征，不是被评估那一层的特征。** 所以「AUC + Base 拒斥」这一列不是严格的同层对照。
3. **Precision 3.1–3.3% 的口径是「多类 argmax + thr 0.9」**，不是逐类独立阈值。

---

## §7 复现

```bash
cd /data01/jh/code/GFS-VL
/data01/jh/conda_envs/gfs_vl/bin/python tools/probe_omission.py
# 输出三张表：AUC 汇总 / 阈值扫描 / Base 拒斥版阈值扫描
```

依赖：`data/val_logits`（312 val 的逐点 top5）与 `data/cut2_feat`（逐层特征，270 G）。
**注意 `cut2_feat` 已在清理候选 C 档（`ASSETS.md` 第六节），删掉后本探针需重建特征才能复跑。**

---

## §8 结论：这条路不要再走

1. **跨层在 OCSB 判据上唯一一次「赢」（dec1 + Base 拒斥 +3.16 AUC）兑不出 precision。**
2. **即使兑得出，XL-1 也说明那部分判别力来自梯度倒灌，而 OCSB 用的是前向读出，用不到它。**
3. **真正的瓶颈在类不平衡，不在特征空间** —— 换任何高秩空间都不能改变 `Neg/C` 的体量比。

**如果将来还要碰「用跨层特征做判据」的题，先做 §3.2 那种阈值扫描表，AUC 只能当入场券。**
