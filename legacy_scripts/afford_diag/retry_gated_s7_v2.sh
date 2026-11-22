#!/bin/bash
cd /data01/jh/code/GFS-VL
PY=/data01/jh/conda_envs/gfs_vl/bin/python
export GFS_OFFLINE_PL=/data01/jh/data/online_pl_w_uniform
export PYTHONPATH=/data01/jh/code/GFS-VL:/data01/jh/code/GFS-VL/pointcept/models/PLA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/data01/jh/afford_diag/xl_runs
while true; do
  if [ -f "$LOG/gated_sym_xl_s7.log" ] && [ "$(grep -c "Class_.*Final Result" "$LOG/gated_sym_xl_s7.log" 2>/dev/null)" -ge 57 ]; then
    echo "gated_s7 already done"; exit 0
  fi
  freed=0
  for r in base_s13 sym_xl_s13 gated_sym_xl_s42; do
    if [ -f "$LOG/$r.log" ] && [ "$(grep -c "Class_.*Final Result" "$LOG/$r.log" 2>/dev/null)" -ge 57 ]; then
      freed=1
    fi
  done
  if [ "$freed" = "1" ]; then
    uuid=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits | awk -F", " "\$2>=30000{print \$1; exit}")
    if [ -n "$uuid" ]; then
      echo "=== start gated_s7 on $uuid $(date) ==="
      CUDA_VISIBLE_DEVICES=$uuid $PY tools/train.py \
        --config-file configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_gated.py \
        --num-gpus 1 --options save_path=./exp/scannet200/gated_sym_xl_s7 seed=7 \
        >> "$LOG/gated_sym_xl_s7.log" 2>&1
      echo "=== gated_s7 done rc=$? $(date) ==="
      $PY /data01/jh/afford_diag/update_report.py
      exit 0
    fi
  fi
  sleep 120
done
