#!/bin/bash
cd /data01/jh/code/GFS-VL
PY=/data01/jh/conda_envs/gfs_vl/bin/python
export GFS_OFFLINE_PL=/data01/jh/data/online_pl_w_uniform
export PYTHONPATH=/data01/jh/code/GFS-VL:/data01/jh/code/GFS-VL/pointcept/models/PLA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/data01/jh/afford_diag/xl_runs
C=configs/scannet200
RUNS="mlp_dec0_s42 mlp_dec0_s7 sym_xl_s42 sym_xl_s7 base_s13 sym_xl_s13 xl_s13 gated_sym_xl_s42 gated_sym_xl_s7"
mkdir -p $LOG

cfg_of () {
  case $1 in
    mlp_*) echo "$C/semseg-pt-v3m1-0-gfsregistrain_k1_mlp.py";;
    gated_*) echo "$C/semseg-pt-v3m1-0-gfsregistrain_k1_gated.py";;
    sym_*) echo "$C/semseg-pt-v3m1-0-gfsregistrain_k1_sym.py";;
    xl_*) echo "$C/semseg-pt-v3m1-0-gfsregistrain_k1_xl.py";;
    base_*) echo "$C/semseg-pt-v3m1-0-gfsregistrain_k1.py";;
  esac
}

for r in $RUNS; do
  if [ -f "$LOG/$r.log" ] && [ "$(grep -c "Class_.*Final Result" "$LOG/$r.log" 2>/dev/null)" -ge 57 ]; then
    echo "[$(date)] $r already done"
    continue
  fi
  # 选一张 free >= 26GB 的卡，最多等 40 分钟
  n=0
  while [ $n -lt 40 ]; do
    uuid=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits | awk -F", " "\$2>=26000{print \$1; exit}")
    [ -n "$uuid" ] && break
    sleep 60
    n=$((n+1))
  done
  if [ -z "$uuid" ]; then
    echo "[$(date)] $r skipped: no free gpu"
    continue
  fi
  cfg=$(cfg_of $r)
  seed=${r##*_s}
  echo "[$(date)] retry $r cfg=$cfg seed=$seed gpu=$uuid"
  CUDA_VISIBLE_DEVICES=$uuid $PY tools/train.py --config-file "$cfg" --num-gpus 1 \
    --options save_path="./exp/scannet200/$r" seed=$seed \
    >> "$LOG/$r.retry.log" 2>&1
  echo "[$(date)] retry $r rc=$?"
  $PY /data01/jh/afford_diag/update_report.py
  uuid=""
done
echo "[$(date)] RETRY ALL DONE"
