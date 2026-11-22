#!/bin/bash
# 单个 support 种子的完整离线伪标签链：dump -> 软权重 -> 均匀权重
# 用法: bash pl_pipeline_seed.sh <seed> <gpu_uuid>
set -u
SEED="$1"
KSHOT="${3:-1}"
GPU="$2"
PY=/data01/jh/conda_envs/gfs_vl/bin/python
LOG=/data01/jh/data/pl_pipeline_k${KSHOT}_s${SEED}.log

cd /data01/jh/code/ps_diag || exit 1
export PYTHONPATH="/data01/jh/code/GFS-VL:/data01/jh/code/GFS-VL/pointcept/models/PLA${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="$GPU"

PROTO_VLM=/data01/jh/data/support_protos/sc200_regis_${KSHOT}_${SEED}.npy
PROTO_SON=/data01/jh/data/support_protos/sc200_regis_${KSHOT}_${SEED}_sonata.npy
ONLINE=/data01/jh/data/online_pl_k${KSHOT}_s${SEED}
WEIGHTED=/data01/jh/data/online_pl_w_k${KSHOT}_s${SEED}
FINAL=/data01/jh/data/pl_k${KSHOT}_s${SEED}
REJ=/data01/jh/data/online_pl_rej_k${KSHOT}_s${SEED}_s0.json

echo "[$(date '+%F %T')] === seed=$SEED gpu=$GPU 开始 ===" >> "$LOG"

# 1) 逐场景 PS + AI 伪标签
$PY dump_online_pl.py --proto "$PROTO_VLM" --out "$ONLINE" \
    --tag "_k${KSHOT}_s${SEED}" --shard 0 --shards 1 >> "$LOG" 2>&1
rc1=$?
echo "[$(date '+%F %T')] step1 dump rc=$rc1 场景数=$(ls "$ONLINE" 2>/dev/null | wc -l)" >> "$LOG"
[ $rc1 -ne 0 ] && exit $rc1

# 2) Sonata 标准化软权重 + G3z 拒绝池补标
$PY build_pl_weights.py --proto "$PROTO_SON" --online "$ONLINE" \
    --out "$WEIGHTED" --rej "$REJ" >> "$LOG" 2>&1
rc2=$?
echo "[$(date '+%F %T')] step2 weights rc=$rc2 场景数=$(ls "$WEIGHTED" 2>/dev/null | wc -l)" >> "$LOG"
[ $rc2 -ne 0 ] && exit $rc2

# 3) 均匀权重目录（_pl 软链 + _w 全 1），训练直接读这个
$PY build_w_uniform.py --src "$WEIGHTED" --dst "$FINAL" >> "$LOG" 2>&1
rc3=$?
echo "[$(date '+%F %T')] step3 uniform rc=$rc3 文件数=$(ls "$FINAL" 2>/dev/null | wc -l)" >> "$LOG"

echo "[$(date '+%F %T')] === seed=$SEED 完成 rc=$rc3 ===" >> "$LOG"
exit $rc3
