#!/bin/bash
# Track 2 Step D：ONTR 源头权重 + Stage-2 实战。
#
# 用法:
#   bash run_ontr_stage2.sh <gpu_uuid> <配置路径> <save_path>
#
# 两个配置只差末端读出方式，backbone_weight 都指向 ptv3_bk_ontr.pth：
#   ontr_base_lovasz : 原版裸线性 dec0 + Lovász
#   ontr_sym_lovasz  : 跨层残差双头 + Lovász
set -u
GPU="${1:?需要 gpu uuid}"
CFG="${2:?需要配置文件}"
SAVE="${3:?需要 save_path}"
TAG="$(basename "$SAVE")"

cd /data01/jh/code/GFS-VL || exit 1
export PYTHONPATH="/data01/jh/code/GFS-VL:/data01/jh/code/GFS-VL/pointcept/models/PLA${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export GFS_RESERVE_GB=24
export GFS_OFFLINE_PL=/data01/jh/data/online_pl_w_uniform
export CUDA_VISIBLE_DEVICES="$GPU"

LOG=/data01/jh/data/ontr_stage2_${TAG}.log
echo "[$(date '+%F %T')] 启动 $TAG gpu=$GPU" >> "$LOG"
echo "  cfg=$CFG" >> "$LOG"
if [ ! -f ./exp/scannet200/ptv3_bk_ontr.pth ]; then
    echo "  !! ptv3_bk_ontr.pth 不存在，终止" >> "$LOG"
    exit 1
fi
mkdir -p "$SAVE"

/data01/jh/conda_envs/gfs_vl/bin/python tools/train.py \
    --config-file "$CFG" --num-gpus 1 \
    --options save_path="$SAVE" seed=42 num_worker=4 >> "$LOG" 2>&1
rc=$?
echo "[$(date '+%F %T')] $TAG 结束 rc=$rc" >> "$LOG"
exit $rc
