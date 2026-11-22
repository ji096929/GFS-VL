# Sonata-Small 接入 GFS-VL 正式管线（2026-11-21 夜）

> 目的：用**同参数量**的 3D 基础模型（sonata_small 38.65M enc + 7.50M dec = 46.14M）
> 替换 scratch PTv3（46.17M），在**正式 GFS-VL 管线**里验证
> 「换基模能否打破 44–45 HM 天花板」以及「微调是否洗坏预训练特征」。
> 所有探针已封存，只认正式管线结果。

---

## 一、为什么这条路特别干净

| 项 | 我们的 PTv3 | `sonata_small.pth` |
|---|---|---|
| `enc_channels` | (32,64,128,256,512) | (32,64,128,256,512) **完全相同** |
| `enc_depths` | (2,2,2,6,2) | (2,2,2,6,2) **完全相同** |
| `enc_num_head` | (2,4,8,16,32) | 同 |
| `enc_patch_size` | 1024 ×5 | 同 |
| encoder 参数量 | **38.65M** | **38.65M** |
| decoder | 7.50M（4 层） | 无（`enc_mode=True`） |
| 全模型 | **46.17M** | 38.65M |
| **接入后总参数** | — | **46.14M（差 −0.03M）** |

**结论：这是一次精确 1:1 同参数量替换，不是“以大欺小”。**

---

## 二、BN/LN 不兼容（致命坑，必须修）

`sonata_small` 的 `state_dict` 里 **`running_mean` 键数量 = 0**（全套 LayerNorm）。

我们原版 PTv3（`point_transformer_v3m1_base.py:596-611`，`pdnorm_bn=False, pdnorm_ln=False`）：

| 位置 | 我们 | sonata | 风险 |
|---|---|---|---|
| `embedding.stem.norm` | BatchNorm1d | LayerNorm | **雷** |
| `enc.enc1..4.down.norm.0` | BatchNorm1d | LayerNorm | **雷** |
| `Block.norm1/norm2` | LayerNorm | LayerNorm | 安全 |
| `dec.*.up.proj.*` | BatchNorm1d | 无 decoder | 无关 |

**为什么危险**：LN 与 BN 的 `weight`/`bias` 形状完全相同 `(C,)`，`load_state_dict(strict=False)`
**不会报任何 warning**，但前向会用 `running_mean=0 / var=1` 归一化，预训练特征直接失效。

---

## 三、三处代码改动

### 1. `pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py`

- `Embedding.__init__` 加 `embedding_type="conv"`，`"linear"` 时用
  `nn.Linear(in_channels, embed_channels)`（**bias=True**）+ `mask_token = nn.Parameter(zeros(1,C))`
  → 构造出的 key 是 `stem.linear.weight/bias`、`mask_token`，与官方 ckpt 逐字一致
- `PointTransformerV3.__init__` 加 `norm_all_ln=False`、`embedding_type="conv"`
- `norm_all_ln=True` 时 `bn_layer = ln_layer = nn.LayerNorm`

备份：`point_transformer_v3m1_base.py.bak_presonata`

### 2. `pointcept/models/default.py`

`DefaultSegmentorV2` 加 `freeze_enc`（**注意不是 RegisTrainSegmentor**）：
```python
self.freeze_enc = bool(freeze_enc)
if self.freeze_enc:
    for _name, _p in self.backbone.named_parameters():
        if _name.startswith(("embedding.", "enc.")):
            _p.requires_grad = False
```
备份：`default.py.bak_dsv2enc`

### 3. 权重转格式 `ps_diag/convert_sonata_ckpt.py`

`embedding.* / enc.*` → `backbone.embedding.* / backbone.enc.*`
产出 `/data01/jh/weights/sonata_small_ptv3.pth`（273 键 / 38.65M）

### 4. 新 config `configs/scannet200/semseg-pt-v3m1-0-gfspretrain_sonata_small.py`

生成脚本：`ps_diag/make_sonata_stage1_cfg.py`。相对 scratch 版改 6 处：

```
in_channels 6 -> 9
+ embedding_type="linear"
+ norm_all_ln=True
+ freeze_enc=True        (DefaultSegmentorV2 的参数，不是 backbone 的)
feat_keys ("color","normal") -> ("coord","color","normal")   × 3 处
epoch 800 -> 40；num_worker 24 -> 6
```

### 5. `pointcept/engines/train.py:169` 的既有 bug

```python
self.max_epoch = cfg.eval_epoch        # 原始
self.max_epoch = cfg.epoch             # 修正
```

**`cfg.epoch` 在原代码里根本没被读取**，总轮数由 `eval_epoch` 决定。
历史所有实验都是 `epoch = eval_epoch = X`，所以这个修正**不改变任何历史结果的可比性**。
若不改，想跑 40 轮就必须 `eval_epoch=40`，代价是全程只有最后一次评估。

备份：`train.py.bak_maxepoch`


---

## 四、逐层特征验收（1 场景，`shuffle_orders=False`）

| layer | max_err | 判定 |
|---|---|---|
| **embedding** | **0.000e+00** | ✅ 逐位一致 |
| **enc.enc0** | **3.492e-07** | ✅ fp32 噪声级 |
| enc.enc1 | 4.87e-01（原始） / **2.09e-06（按 pooling_inverse 展开）** | ✅ 同一批体素，行顺序不同 |
| enc.enc2 | 2.09 | ❌ |
| enc.enc3 | 5.21 | ❌ |
| enc.enc4 | 4.58 | ❌ |

**enc2 以上为何对不齐**：两边 `SerializedPooling` 是两套实现
- pointcept：用 `serialized_code >> pooling_depth*3` 聚类，输出按 **Z-order** 排列
- sonata：用 `grid_coord // stride` 聚类，输出按 **字典序** 排列

enc1 展开后 2e-06 证明**聚的是同一批体素、聚合结果相同**；但 enc1 的体素**行顺序不同**，
enc2 的输入 `serialized_code` 排序因此不同 → attention 窗口划在不同点上 → 特征分叉。
**这是排序语义差异，不是权重错误，也不是 bug。**

**Gate 判定：PASS（权重层面）**。三重证据：embedding 0 + enc0 3.5e-07 + enc1 展开 2e-06。

---

## 五、side finding：`shuffle_orders=True` 在 eval 模式下也随机

`pointcept/pointcept/models/utils/structure.py:94-98` 与
`point_transformer_v3m1_base.py:432-436` 的 `if shuffle_orders: perm = torch.randperm(...)`
**都没有 `self.training` 判断**。

实测：同一模型、同一输入、`eval()` 模式，两次前向 enc4 差 **4.89**。

**影响**：过去所有 GFS 评估都带这个抖动，单 seed 间差异里有一部分不是种子差异。
**处置**：正式评估时强制 `shuffle_orders=False`（训练保持 True，与历史同口径）。

---

## 六、2 场景干跑结果

```
freeze_enc: 冻结 38.65M，可训练 7.49M
Loading weight at: /data01/jh/weights/sonata_small_ptv3.pth
Train: [1/1][1/100]   Data 2.970  Batch 10.409  loss: 3.7687
Train: [1/1][10/100]  Data 0.006  Batch 0.489   loss: 1.9786
```

- **1 epoch = 100 iter × 0.5 s ≈ 50 秒**（1201 场景 / bs 12 = 100 iter）
- **Data 只要 6 ms** —— dataloader 49× 优化在这里全额兑付
- 40 epoch ≈ **33 分钟**

---

## 七、Stage-1a 正式训练

```bash
cd /data01/jh/code/GFS-VL
PYTHONPATH=/data01/jh/code/GFS-VL:/data01/jh/code/GFS-VL/pointcept/models/PLA \
CUDA_VISIBLE_DEVICES=<空卡 uuid> GFS_RESERVE_GB=20 OMP_NUM_THREADS=1 \
python tools/train.py \
  --config-file configs/scannet200/semseg-pt-v3m1-0-gfspretrain_sonata_small.py \
  --num-gpus 1 \
  --options epoch=40 eval_epoch=10 \
            weight=/data01/jh/weights/sonata_small_ptv3.pth \
            save_path=<exp>/sonata_small_stage1a
```

日志：`/data01/jh/data/sonata_stage1a.log`

---

## 八、待验证的三个问题

1. **同参数量下 3D 基础模型能打多高**：Stage-1a 的 Base mIoU 能否接近/超过 scratch PTv3 的 **0.6870**？
2. **解冻是否洗坏特征**：Stage-1b（enc 解冻 lr=1e-4）对比 Stage-1a，Base 涨多少、Novel 掉多少
3. **正式 GFS 管线水位**：Stage-2 1-shot 对比 `base_lovasz 0.4122` / `sym_lovasz 0.4414`，
   以及对手 `GFS-VL 40.92` / `UGA 43.83`

---

## 九、Stage-1a 最终结果

```
epoch 15  val  mIoU 0.7715 / allAcc 0.9306
最终 test mIoU 0.7634 / mAcc 0.8498 / allAcc 0.9229   （3 卡 gather，312 场景全量）
```

对标 scratch PTv3 的 12 类真值 **0.8057**（`/data01/jh/data/ontr_test_bk.log`）：
**0.7634 = 94.7%**，而训练量只有 **45 遍 vs 800 遍**（1/17.8），可训练参数只有 7.49M（16%）。

**Stage-1 这一层是成功的**：预训练 encoder 确实把知识带进来了。

---

## 十、Stage-2 第 1 局：换基模在正式管线上是净负（2026-11-22 00:44）

配置：裸线性头（无跨层）、全解冻、backbone lr=1e-4、Adam、CE+Lovász、VLM PL、1-shot regis1、3 卡。

| 方法 | 参数量 | Stage-1 遍数 | Base | Novel | HM |
|---|---|---|---|---|---|
| `base_lovasz` 3-seed 均值 | 46.17M | 800 | **0.6765** | 0.3094 | **0.4244** |
| **`sonata_s_s2_unfrozen`** | **46.14M** | **45** | **0.6448** | **0.3121** | **0.4206** |
| `sym_lovasz` 3-seed 均值 | 46.17M | 800 | 0.6867 | 0.3253 | 0.4410 |

**delta（vs scratch 3-seed 均值）：Base −3.17、Novel +0.27（噪声内）、HM −0.38。**

**结论：换成 Sonata-Small 预训练 encoder，在正式 GFS 管线上没有兑现成增益，Base 明显下降。**
全解冻让 GFS 的 45 类梯度稀释了 Base 判别边界，而 Novel 并没有因此多拿。

### 外部旁证（与本结果一致）

- **Sonata 原论文 Table 5**：纯 3D 的 Sonata 在 ScanNet200 上（33.5）输给从零训的 PTv3（35.3）。
- **GeoGuide (CVPR'26) Table 5**：换冻结 Sonata + 逐点 VLM 监督 + 简单头，53.9 < 54.2（低于原基线）。

**三条独立证据指向同一件事：在有 VLM 的 ScanNet200 上，纯 3D 预训练初始化不构成优势；
真正的瓶颈在逐点 VLM 伪标签噪声、低秩末层读出、以及丢弃文本先验。**

---

## 十一、三局全部完成 —— 换基模支线结案（2026-11-22 01:48）

| 局 | 配置 | PL | Base | Novel | **HM** |
|---|---|---|---|---|---|
| 1 | 全解冻 | VLM | 0.6448 | **0.3121** | **0.4206** |
| 2 | enc 冻结 | VLM | 0.6426 | 0.2932 | **0.4027** |
| 3 | enc 冻结 | 无（纯 3D） | 0.5919 | **0.0866** | **0.1511** |

### scratch 基线对照

| 方法 | 口径 | Base | Novel | HM |
|---|---|---|---|---|
| `base_lovasz` 3-seed 均值 | Stage-2 完整 | **0.6765** | 0.3094 | **0.4244** |
| `sym_lovasz` 3-seed 均值 | Stage-2 完整 | 0.6867 | **0.3253** | **0.4410** |
| `support_xl_linear` | 探针（冻结骨干） | 0.6385 | 0.0556 | 0.1022 |
| `support_ontr_linear` | 探针（冻结骨干） | 0.6359 | 0.0761 | 0.1359 |

### 结论

**1. 三个配置全部没有正收益**
- 全解冻：HM 0.4206 vs 0.4244（**−0.38**，在单 seed 噪声内）
- enc 冻结：HM 0.4027（**−2.17**，比全解冻还差）
- 纯 3D：Base 0.5919 明显低于所有对照（−4.4 ~ −4.7）

**2. “冻结预训练 encoder”只在同任务有效**
- Stage-1（12 类，与预训练同构）：冻结 enc + 训 dec → **0.7634**，达 scratch 的 94.7% ✅
- Stage-2（57 类 + GFS 双头 + 伪标签）：冻结 enc → **0.4027**，比全解冻低 1.79 ❌

原因：GFS 第二阶段需要 encoder 适配 45 个新类的判别-度量结构，decoder 单靠 7.49M 参数、
20 轮补不回来。

**3. 唯一的正面结论在 Stage-1**
45 遍（vs scratch 800 遍）达成 94.7%，证明预训练知识真实存在；
但它在 Stage-2 的 GFS 目标下无法转化为分数。

**4. 支线封存**：不再在 sonata_small 上排新实验。
若要继续基础模型路线，唯一有物理依据的是 **Concerto**（Sonata 论文实测 ScanNet200 +8.1 mIoU、
ScanNet++ +6.7 mIoU），但必须加同骨干基线与未见域防护（Concerto 预训练见过 val split）。
