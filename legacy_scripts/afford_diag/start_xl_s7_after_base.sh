#!/bin/bash
cd /data01/jh/code/GFS-VL
PY=/data01/jh/conda_envs/gfs_vl/bin/python
export GFS_OFFLINE_PL=/data01/jh/data/online_pl_w_uniform
export PYTHONPATH=/data01/jh/code/GFS-VL:/data01/jh/code/GFS-VL/pointcept/models/PLA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/data01/jh/afford_diag/xl_runs/sc200_k1_XLu_s7_v2.log
while kill -0 2069921 2>/dev/null; do sleep 20; done
echo "=== base_s7 exited at $(date), start xl_s7 ==="
free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " "\$1==7{print \$2}")
echo "idx7 free=${free}MiB"
CUDA_VISIBLE_DEVICES=7 $PY tools/train.py \
  --config-file configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_xl.py \
  --num-gpus 1 \
  --options save_path=./exp/scannet200/sc200_k1_XLu_s7 seed=7 \
  > "$LOG" 2>&1
echo "=== xl_s7 DONE rc=$? $(date) ==="
