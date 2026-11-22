#!/bin/bash
cd /data01/jh/code/GFS-VL
PY=/data01/jh/conda_envs/gfs_vl/bin/python
export GFS_OFFLINE_PL=/data01/jh/data/empty_pl
export PYTHONPATH=/data01/jh/code/GFS-VL:/data01/jh/code/GFS-VL/pointcept/models/PLA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GFS_RESERVE_GB=30
LOG=/data01/jh/afford_diag/xl_runs
while true; do
  if [ -f "$LOG/support_sym_s42.log" ] && [ "$(grep -c "Class_.*Final Result" "$LOG/support_sym_s42.log" 2>/dev/null)" -ge 57 ]; then
    echo "already done"; exit 0
  fi
  ready=0
  for r in ortho_sym_xl_s42 ortho_sym_xl_s7; do
    if [ -f "$LOG/$r.log" ] && [ "$(grep -c "Class_.*Final Result" "$LOG/$r.log" 2>/dev/null)" -ge 57 ]; then
      ready=1
    fi
  done
  if [ "$ready" = "1" ]; then
    uuid=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits | awk -F", " "\$2>=34000{print \$1; exit}")
    if [ -n "$uuid" ]; then
      echo "=== start support_sym on $uuid $(date) ==="
      CUDA_VISIBLE_DEVICES=$uuid $PY tools/train.py \
        --config-file configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_sym.py \
        --num-gpus 1 --options save_path=./exp/scannet200/support_sym_s42 seed=42 \
        >> "$LOG/support_sym_s42.log" 2>&1
      echo "=== support_sym done rc=$? $(date) ==="
      exit 0
    fi
  fi
  sleep 120
done
