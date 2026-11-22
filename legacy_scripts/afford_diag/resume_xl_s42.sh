#!/bin/bash
cd /data01/jh/code/GFS-VL
PY=/data01/jh/conda_envs/gfs_vl/bin/python
export GFS_OFFLINE_PL=/data01/jh/data/online_pl_w_uniform
export PYTHONPATH=/data01/jh/code/GFS-VL:/data01/jh/code/GFS-VL/pointcept/models/PLA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/data01/jh/afford_diag/xl_runs/sc200_k1_XLu_s42_resume.log
CK=./exp/scannet200/sc200_k1_XLu_s42/model/epoch_6.pth
while true; do
  free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " "\$1==5{print \$2}")
  if [ -n "$free" ] && [ "$free" -ge 26000 ]; then
    echo "=== RESUME at $(date) free=${free}MiB ==="
    CUDA_VISIBLE_DEVICES=6 $PY tools/train.py \
      --config-file configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_xl.py \
      --num-gpus 1 \
      --options save_path=./exp/scannet200/sc200_k1_XLu_s42 weight=$CK resume=True seed=42 \
      >> "$LOG" 2>&1
    echo "=== RESUME DONE rc=$? $(date) ==="
    exit 0
  fi
  sleep 20
done
