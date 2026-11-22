#!/bin/bash
# 纯 3D Support-Only 基线（无 VLM 伪标签）重跑
# 规则：free >= 34GB 才启动，带 GFS_RESERVE_GB=30 圈地，失败自动重试
cd /data01/jh/code/GFS-VL || exit 1
PY=/data01/jh/conda_envs/gfs_vl/bin/python
export GFS_OFFLINE_PL=/data01/jh/data/empty_pl
export PYTHONPATH=/data01/jh/code/GFS-VL:/data01/jh/code/GFS-VL/pointcept/models/PLA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GFS_RESERVE_GB=30
LOG=/data01/jh/afford_diag/xl_runs
for attempt in 1 2 3 4 5; do
  if [ -f "$LOG/support_base_s42.log" ] && [ "$(grep -c 'Class_.*Final Result' $LOG/support_base_s42.log 2>/dev/null)" -ge 57 ]; then
    echo "already done, exit"; exit 0
  fi
  uuid=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>=34000{print $1; exit}')
  if [ -z "$uuid" ]; then
    echo "attempt $attempt: no card with 34GB free, wait"
    sleep 120
    continue
  fi
  echo "=== attempt $attempt on $uuid $(date) ==="
  CUDA_VISIBLE_DEVICES=$uuid $PY tools/train.py     --config-file configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1.py     --num-gpus 1 --options save_path=./exp/scannet200/support_base_s42 seed=42     >> "$LOG/support_base_s42.log" 2>&1
  echo "=== attempt $attempt rc=$? $(date) ==="
done
