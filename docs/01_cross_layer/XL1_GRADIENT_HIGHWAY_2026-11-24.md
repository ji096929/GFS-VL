# XL-1 结案：跨层增益来自「反向梯度捷径」而非「前向读出」（2026-11-24）

## §1 核心结论

**一句话：`sym` 的 +1.40 HM 增益由「跨层反向梯度捷径」主导。**

把 `dec1/dec2` 接到分类头上时，跨层连线同时做了两件事：

- **机制 A（前向读出）**：推理时把 dec1/dec2 的高秩几何特征送给分类头
- **机制 B（反向梯度捷径）**：训练时让新类损失的梯度绕过坍塌的 `dec0`，直接灌进 dec1/dec2

**实测判决：B 主导，A 不仅无效而且有害。**

截断跨层梯度（`detach_cross=True`）后：

```
base_lovasz_s7         HM 0.4241
sym_lovasz_s7          HM 0.4381   (+1.40)
detach_sym_lovasz_s7   HM 0.4089   (−2.92 vs sym，−1.52 vs base)
```

**不仅丢掉全部增益，还比不接跨层的 `base` 低 1.52。**

---

## §2 关键实验总表

### 表 1：实验 2 确定性 HM（det_eval，`enable_flash=False`）

| run | Base | Novel | HM |
|---|---|---|---|
| base_lovasz_s7 | 0.6659 | 0.3111 | **0.4241** |
| sym_lovasz_s7 | 0.6775 | 0.3237 | **0.4381** |
| detach_sym_lovasz_s7 | 0.6547 | 0.2973 | **0.4089** |

`detach` 与标准评估完全一致（两边都是 0.4089），没有 base/sym 那种 −0.6 的口径偏移。

### 表 2：三层相对 Stage-1 的漂移（1a，52 场景）

Stage-1 锚点：`exp/scannet200/ptv3_bk/model/model_best.pth`（epoch=81）。

| case | dec0 均值余弦 / 点级余弦 | dec1 | dec2 |
|---|---|---|---|
| base_s7 | 0.8828 / 0.7344 | 0.9094 / 0.7598 | 0.9214 / 0.8312 |
| **sym_s7** | 0.9177 / 0.8172 | **0.7989 / 0.7709** | **0.7403 / 0.8044** |
| rep3_s7 | 0.9347 / 0.8221 | 0.8352 / 0.7861 | 0.9234 / **0.8570** |
| lnonly_s42 | 0.8880 / 0.7459 | 0.8832 / 0.7591 | 0.9235 / 0.8349 |

对应的 eRank（协方差特征值熵口径）：

| case | dec0 (64d) | dec1 (64d) | dec2 (128d) |
|---|---|---|---|
| Stage1_ep81 | 9.49 | 17.95 | 40.51 |
| base_s7 | 10.07 | 18.58 | 39.19 |
| sym_s7 | 10.32 | 18.14 | **41.98** |
| rep3_s7 | 11.00 | 18.10 | 40.52 |
| lnonly_s42 | 11.23 | 17.69 | 40.65 |

**读法：只有 `sym`（真接 dec1/dec2）的 dec2 发生剧烈语义重塑（0.7403）；`rep3` 只复制 dec0，dec2 几乎不动（0.9234）。**

### 表 3：三层单层线性探针（1b，固定 seed，45 类 mIoU / macro Acc）

| case | dec0 | dec1 | dec2 |
|---|---|---|---|
| base_s7 | **0.4422 / 0.6995** | 0.4174 / 0.6523 | 0.4317 / 0.6561 |
| sym_s7 | 0.3835 / 0.6125 | 0.4234 / 0.6793 | **0.4884 / 0.7148** |
| **detach_sym_s7** | **0.3432 / 0.5991** | **0.3349 / 0.5892** | **0.3945 / 0.6383** |

**三层全部是三者最低，不是「停留在 base 水平」，是每一层都退化了。**

### 表 4：静态梯度与跨层贡献占比（8 场景平均，L2）

| 状态 | detach_cross | dec0 | dec1 | dec2 | other |
|---|---|---|---|---|---|
| ep0（Stage-1 + 随机 sym 头） | False | 1.843 | 1.117 | 0.795 | 2.315 |
| | True | 1.843 | 0.874 | 0.329 | 2.314 |
| **跨层贡献** | | **0%** | **21.8%** | **58.6%** | — |
| ep20（sym_s7 收敛态） | False | 4.904 | 3.324 | 3.211 | 3.739 |
| | True | 4.904 | 1.926 | 0.597 | 3.621 |
| **跨层贡献** | | **0%** | **42.1%** | **81.4%** | — |

**dec0 的梯度在两种模式下逐比特相同（1.843/1.843、4.904/4.904），证明截断只作用于跨层支路。**

---

## §3 机理闭环

```
base:      分类头只看 dec0（64 维）
           → 100% 的新类梯度死磕 dec0 → dec0 探针 0.4422（三者最高）
           → 但 dec0 只有 ~10 维有效秩，承载力有限 → HM 0.4241

sym:       分类头看 [LN(dec0), proj1, proj2]（256 维）
           → dec0 被「抢食」变懒 → 0.3835（比 base 低 5.9）
           → 但 dec2 的梯度 81.4% 来自跨层捷径，被重塑 → 0.4884（三者最高）
           → dec2 接管主力 → HM 0.4381

detach:    分类头同样看三层 → dec0 同样被抢食变懒（0.3432）
           → 但 dec2 的 81.4% 梯度被 detach 截断，没被重塑（0.3945 < base 的 0.4317）
           → 而且 dec0 变懒后，原本经 dec0→dec1→dec2 上渗的约 18% 梯度也枯竭
           → 两头踏空，三层全最低 → HM 0.4089（比 base 还低 1.52）
```

**「梯度抢食」的量化：** `sym` 的 dec0 探针比 `base` 低 5.9 个点，说明跨层支路确实把训练压力从 dec0 吸走了。

**「梯度捷径」的量化：** ep20 时 dec2 的 81.4%、dec1 的 42.1% 梯度来自跨层支路；dec0 为 0%。

**「两头踏空」的量化：** detach 的 dec0（0.3432）低于 base（0.4422），dec2（0.3945）也低于 base（0.4317），三层全最低。

---

## §4 口径核验与局限

### 4.1 伪标签来源一致

三份权重（base_lovasz_s7 / sym_lovasz_s7 / detach_sym_lovasz_s7）**均使用同一离线伪标签**：

```
GFS_OFFLINE_PL=/data01/jh/data/online_pl_w_uniform
```

这条曾在首轮判读中被误判为「基线用在线 VLM、detach 用离线 PL」——错误来源是只翻了 `config.py`，而环境变量不写进 config dump（见 `docs/00_ledger/NAMING.md` 记录过的坑）。

### 4.2 工程差异

| 项 | base/sym 基线 | detach 本局 |
|---|---|---|
| 卡数 / 每卡 batch | 单卡 bs=12 | 3 卡 DDP，每卡 bs=4 |
| `use_lovasz` | True | True |
| `enable_flash` | False | False |

Lovász 损失按单卡 batch 计算，可能带来零点几个点的波动，但**无法解释 −2.92**。

**独立佐证**：5-shot 单卡纯 CE 的 `detach_sym_k5_s42` 同样低于 `base_k5_s42`（HM 0.4066 < 0.4181，−1.15）。两套完全不同的设定（1-shot Lovász 3 卡 / 5-shot CE 单卡）复现了同一现象。

### 4.3 测量口径

- **eRank**：协方差矩阵特征值的熵指数 `exp(−Σ p_i log p_i)`，在 ScanNet200 val 集上每 6 个场景取 1 个（52 场景）累积
- **均值余弦 / 点级余弦**：每场景固定 512 个采样点（`rng(20261124)`），点级余弦是同一场景同一位置的逐点余弦
- **线性探针**：104 场景，固定 seed 打乱后 70/30 切分，每层按训练集标准化，带 bias 的线性头 + 类别逆频次加权 CE，Adam 150 步，每层独立 `torch.manual_seed(42)`

### 4.4 未解决的问题

1. **`detach` 为什么比 `base` 还低 1.52** —— 目前只有「梯度抢食 + 两头踏空」的机制解释和三层探针的实证支持，但没有直接测量「dec0→dec1→dec2 上渗梯度」在 detach 下的衰减量
2. **标准评估与 det_eval 的 −0.6 偏移** —— 在 base/sym 上稳定出现，在 detach 上没有出现；机制未查清（怀疑与复用 `trainer.model` 对象的 DDP buffer 状态有关）

---

## §5 资产与复现

### 权重

| 名称 | 路径 |
|---|---|
| Stage-1 锚点 | `exp/scannet200/ptv3_bk/model/model_best.pth`（ep81） |
| base | `exp/scannet200/base_lovasz_s7/model/regis1_model_last.pth` |
| sym | `exp/scannet200/sym_lovasz_s7/model/regis1_model_last.pth` |
| rep3 | `exp/scannet200/rep3_lovasz_s7/model/regis1_model_last.pth` |
| lnonly | `exp/scannet200/lnonly_s42/model/regis1_model_last.pth` |
| detach | `exp/scannet200/detach_sym_lovasz_s7/model/regis1_model_last.pth` |

### 脚本

| 脚本 | 作用 |
|---|---|
| `tools/probe_layer_drift_v2.py` | 三层 eRank + 均值余弦 + 点级余弦 |
| `tools/probe_linear_v2.py` | 三层 45 类线性探针（固定 seed） |
| `tools/probe_grad_static_v2.py` | ep0/ep20 两态 × detach 两模式的梯度对比 |
| `tools/probe_erank.py` | 9 个 ScanNet200 权重的逐层 eRank |

### 代码改动

| 文件 | 改动 |
|---|---|
| `pointcept/models/default.py:359-372` | `forward` 的 `finally` 里，当 `GFS_CAPTURE_LAYERS` 环境变量存在时，把 `dec0` 与 `dec1/dec2`（`_to_full` 上采样）存入 `self._cr_last`；默认关闭，不影响正常训练/评估 |
| `pointcept/engines/train.py:1484-1529` | `_gn_probe`：每 20 iter 统计一次，分 dec0/dec1/dec2/other 四组记录 L2 与 RMS；AMP 下用 `scaler.get_scale()` 还原真实量级；GPU 上累加标量，循环外只调一次 `.item()` |
| `configs/scannet200/…k1_detach_sym_lovasz_s7.py` | 复制 `sym_lovasz_s7` 的完整配置，仅加 `detach_cross=True` 与 `log_grad_norm=True` |

### 复现命令

```bash
# 实验 2 训练
cd /data01/jh/code/GFS-VL
export PYTHONPATH=/data01/jh/code/GFS-VL:/data01/jh/code/GFS-VL/pointcept/models/PLA
export GFS_OFFLINE_PL=/data01/jh/data/online_pl_w_uniform
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=<uuid0>,<uuid1>,<uuid2> \
  /data01/jh/tools/numactl/usr/bin/numactl --cpunodebind=0 --membind=0 \
  python tools/train.py \
  --config-file configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_detach_sym_lovasz_s7.py \
  --num-gpus 3 \
  --options save_path=exp/scannet200/detach_sym_lovasz_s7 num_worker=18

# 确定性评估
export GFS_OFFLINE_PL=/dev/shm/jh_pl_k1_s10
CUDA_VISIBLE_DEVICES=<uuid0>,<uuid1>,<uuid2> \
  /data01/jh/tools/numactl/usr/bin/numactl --cpunodebind=0 --membind=0 \
  python tools/test.py \
  --config-file configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_detach_sym_lovasz_s7.py \
  --num-gpus 3 \
  --options weight=exp/scannet200/detach_sym_lovasz_s7/model/regis1_model_last.pth \
            save_path=/data01/jh/data/det_eval_detach_s7 \
            model.backbone.enable_flash=False \
            "regis_train_list=['regis1']"

# 三层漂移
GFS_CAPTURE_LAYERS=1 python tools/probe_layer_drift_v2.py
# 三层探针
GFS_CAPTURE_LAYERS=1 python tools/probe_linear_v2.py
# 静态梯度
python tools/probe_grad_static_v2.py
```

**注意**：`tools/test.py` 的 `--options` 必须用 `key=value` 格式（空格分隔会报 `ValueError: not enough values to unpack`）。
