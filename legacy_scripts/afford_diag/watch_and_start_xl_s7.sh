#!/bin/bash
cd /data01/jh/code/GFS-VL
PY=/data01/jh/conda_envs/gfs_vl/bin/python
export GFS_OFFLINE_PL=/data01/jh/data/online_pl_w_uniform
export PYTHONPATH=/data01/jh/code/GFS-VL:/data01/jh/code/GFS-VL/pointcept/models/PLA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/data01/jh/afford_diag/xl_runs/sc200_k1_XLu_s7_v2.log
while true; do
  line=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " "\$2>=25000{print \$1\",\"\$2; exit}")
  if [ -n "$line" ]; then
    idx=${line%,*}
    free=${line#*,}
    case $idx in
      0) cvd=2;; 1) cvd=3;; 2) cvd=4;; 3) cvd=0;;
      4) cvd=5;; 5) cvd=6;; 6) cvd=1;; 7) cvd=7;;
    esac
    echo "=== idx${idx} free=${free}MiB -> CVD ${cvd} start xl_s7 $(date) ==="
    CUDA_VISIBLE_DEVICES=$cvd $PY tools/train.py \
      --config-file configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_xl.py \
      --num-gpus 1 \
      --options save_path=./exp/scannet200/sc200_k1_XLu_s7 seed=7 \
      > "$LOG" 2>&1
    echo "=== xl_s7 DONE rc=$? $(date) ==="
    exit 0
  fi
  sleep 15
done
