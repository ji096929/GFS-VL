p = '/data01/jh/code/GFS-VL/OVERNIGHT_ICCV_REPORT.md'
s = open(p, encoding='utf-8').read()

old = '''### 3.4 L_cross λ=0.1 统一性验证（进行中）

- `sym_lcross01_k1_s42`（1-shot, 65536, λ=0.1）— 检验 1-shot 是否零掉点
- `sym_lcross01_k5_102400_s42`（5-shot, 102400, λ=0.1）— 检验 5-shot 是否护住 Base

**状态：05:40 刚启动，约 06:50 出分。** 本报告发布时暂无数据。'''

new = '''### 3.4 L_cross λ=0.1 统一性验证

**1-shot 结果（已完成）**

| 配置 | Base | Novel | HM | ΔHM vs sym_xl |
|---|---|---|---|---|
| sym_xl（CE） | 0.6877 | 0.3083 | 0.4257 | — |
| sym + L_cross λ=0.3 | 0.6843 | 0.3061 | 0.4230 | −0.27 |
| **sym + L_cross λ=0.1** | 0.6848 | 0.3064 | **0.4234** | **−0.23** |

**结论：L_cross 不是 1/5-shot 统一配方。** 降低 λ 只把 1-shot 的损失从 −0.27 降到 −0.23，仍在噪声内。原因与 α、余弦退火一致：1-shot 的 Base 本来就没被压制（+0.06），任何"修 Base"的机制都是纯干扰。

**5-shot 结果**：`sym_lcross01_k5_102400_s42` 进行中（约 07:55 出分）。

### 3.5 四种"修 Base"机制的统一性对照

| 机制 | 1-shot ΔHM | 5-shot ΔHM | 是否统一 |
|---|---|---|---|
| **Lovász（损失端）** | **+1.57** | **+1.94** | ✅ **是** |
| L_cross λ=0.1（损失端，跨层子空间） | −0.23 | 待出 | ❌ |
| 推理期 α=0.9 | −0.86 | +0.53 | ❌ |
| 余弦退火 + AdamW | −0.56 | +0.63 | ❌ |

**唯一同时满足 1-shot 与 5-shot 双正的机制是 Lovász。** 它与其他三者不冲突的原因：Lovász 直接优化 IoU（对少数类的边界误差加权），跨层改善特征可分性——一个在损失端、一个在表示端，物理上正交。'''

assert s.count(old) == 1, 'anchor=%d' % s.count(old)
s = s.replace(old, new)
open(p, 'w', encoding='utf-8').write(s)
print('report patched')
