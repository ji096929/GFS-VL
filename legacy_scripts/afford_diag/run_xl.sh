#!/bin/bash
cd /data01/jh/code/GFS-VL
PY=/data01/jh/conda_envs/gfs_vl/bin/python
export GFS_OFFLINE_PL=/data01/jh/data/online_pl_w_uniform
export PYTHONPATH=/data01/jh/code/GFS-VL:/data01/jh/code/GFS-VL/pointcept/models/PLA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG_BASE=configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1.py
CFG_XL=configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_xl.py
LOG=/data01/jh/afford_diag/xl_runs
mkdir -p $LOG

run () {
  local gpu=$1 cfg=$2 name=$3 seed=$4
  echo "=== START $name gpu=$gpu seed=$seed $(date) ==="
  CUDA_VISIBLE_DEVICES=$gpu $PY tools/train.py --config-file "$cfg" --num-gpus 1 \
    --options save_path="./exp/scannet200/$name" seed=$seed \
    > "$LOG/$name.log" 2>&1
  echo "=== END $name rc=$? $(date) ==="
}

( run 6 "$CFG_BASE" sc200_k1_BASEu_s42 42; run 6 "$CFG_XL" sc200_k1_XLu_s42 42 ) &
P1=$!
( run 7 "$CFG_BASE" sc200_k1_BASEu_s7 7; run 7 "$CFG_XL" sc200_k1_XLu_s7 7 ) &
P2=$!
wait $P1
wait $P2
echo "=== ALL DONE $(date) ==="
