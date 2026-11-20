# 实验启动 Runbook（CUT2）

## 通用环境（所有训练 run 必须带）

```bash
cd /data01/jh/code/GFS-VL
export PYTHONPATH=/data01/jh/code/GFS-VL:/data01/jh/code/GFS-VL/pointcept/models/PLA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GFS_RESERVE_GB=30            # 圈地，防被挤
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4   # 必须，否则 88 线程超订
PY=/data01/jh/conda_envs/gfs_vl/bin/python
```

伪标签开关：
- **1-shot 离线 PL**：`export GFS_OFFLINE_PL=/data01/jh/data/online_pl_w_uniform`
- **纯 3D**：`export GFS_OFFLINE_PL=/data01/jh/data/empty_pl`
- **真 5-shot**：**不要设** `GFS_OFFLINE_PL`（走在线原型提取，多 5-8 分钟）

## 训练模板

```bash
CUDA_VISIBLE_DEVICES=<GPU-UUID> $PY tools/train.py \
  --config-file configs/scannet200/<config>.py \
  --num-gpus 1 \
  --options save_path=./exp/scannet200/<run_name> seed=<seed> num_worker=4 \
  >> /data01/jh/afford_diag/xl_runs/<run_name>.log 2>&1
```

## 评估模板（不训练，只打分）

```bash
CUDA_VISIBLE_DEVICES=<GPU-UUID> $PY tools/test.py \
  --config-file configs/scannet200/<config>.py \
  --num-gpus 1 \
  --options weight=./exp/scannet200/<run_name>/model/regis1_model_last.pth num_worker=2 \
  >> /data01/jh/afford_diag/xl_runs/<run_name>_eval.log 2>&1
```

`num_worker=2` 是过载机器上的关键：配置默认 15，在本机 load>300 时会卡死在 futex。

## 关键配置对照

| 名称 | 文件 | 结构 |
|---|---|---|
| base | semseg-pt-v3m1-0-gfsregistrain_k1.py | seg_head(dec0) |
| xl | ..._k1_xl.py | +proj(dec1,dec2) 拼接 |
| mlp | ..._k1_mlp.py | dec0 + 两层 MLP |
| sym | ..._k1_sym.py | +base_cross_head 对称残差 |
| gated | ..._k1_gated.py | +逐点深度门控（已证伪） |
| ortho | ..._k1_ortho.py | +正交补去相关（已证伪） |
| k5 | ..._k5.py | 5-shot 基线 |
| k5_sym | ..._k5_sym.py | 5-shot 对称 |
| k5_detach | ..._k5_detach.py | detach_cross + seg_head lr=1e-4（实验 A） |
| k5_cos_sym/base | ..._k5_cos_*.py | AdamW+Cosine+save_freq=4（实验 B） |

## 五条红线

1. 日志一律 `>>`
2. `save_path` 与日志名一致
3. 显存阈值 `free >= 34000 MiB` 才启动，`GFS_RESERVE_GB=30`
4. 每 run 结束跑 `extract_results.py` 并入 `results_all.json`
5. 需要 epoch 曲线的显式 `save_freq=4`
