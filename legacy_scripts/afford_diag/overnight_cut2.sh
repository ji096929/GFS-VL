#!/bin/bash
cd /data01/jh/code/GFS-VL
PY=/data01/jh/conda_envs/gfs_vl/bin/python
export GFS_OFFLINE_PL=/data01/jh/data/online_pl_w_uniform
export PYTHONPATH=/data01/jh/code/GFS-VL:/data01/jh/code/GFS-VL/pointcept/models/PLA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/data01/jh/afford_diag/xl_runs
C=configs/scannet200
GA=GPU-39317709-7096-1d39-3d4e-31f84f9776ba
GB=GPU-829a5bd9-3902-f607-5ea7-7c667b233971
mkdir -p $LOG

run_one () {
  local gpu=$1 cfg=$2 name=$3 seed=$4
  echo "=== START $name gpu=$gpu seed=$seed $(date) ==="
  CUDA_VISIBLE_DEVICES=$gpu $PY tools/train.py --config-file "$cfg" --num-gpus 1 \
    --options save_path="./exp/scannet200/$name" seed=$seed \
    > "$LOG/$name.log" 2>&1
  echo "=== END $name rc=$? $(date) ==="
}

run_batch () {
  local n1=$1 c1=$2 s1=$3 n2=$4 c2=$5 s2=$6
  ( run_one $GA "$c1" "$n1" "$s1" ) &
  local A=$!
  ( run_one $GB "$c2" "$n2" "$s2" ) &
  local B=$!
  wait $A
  wait $B
  $PY /data01/jh/afford_diag/update_report.py
}

# ---- batch 1: mlp_dec0 ----
run_batch mlp_dec0_s42 "$C/semseg-pt-v3m1-0-gfsregistrain_k1_mlp.py" 42 \
          mlp_dec0_s7  "$C/semseg-pt-v3m1-0-gfsregistrain_k1_mlp.py" 7

# ---- batch 2: sym_xl ----
run_batch sym_xl_s42 "$C/semseg-pt-v3m1-0-gfsregistrain_k1_sym.py" 42 \
          sym_xl_s7  "$C/semseg-pt-v3m1-0-gfsregistrain_k1_sym.py" 7

# ---- batch 3: base_s13 + auto-choice ----
CHOICE=$($PY /data01/jh/afford_diag/decide_s13.py 2>>"$LOG/batch3_choice.log")
if [ "$CHOICE" = "sym" ]; then
  CFG2="$C/semseg-pt-v3m1-0-gfsregistrain_k1_sym.py"; NAME2=sym_xl_s13
else
  CFG2="$C/semseg-pt-v3m1-0-gfsregistrain_k1_xl.py"; NAME2=xl_s13
fi
echo "batch3 choice=$CHOICE -> $NAME2"
run_batch base_s13 "$C/semseg-pt-v3m1-0-gfsregistrain_k1.py" 13 \
          $NAME2  "$CFG2" 13

# ---- batch 4: gated_sym_xl ----
run_batch gated_sym_xl_s42 "$C/semseg-pt-v3m1-0-gfsregistrain_k1_gated.py" 42 \
          gated_sym_xl_s7  "$C/semseg-pt-v3m1-0-gfsregistrain_k1_gated.py" 7

echo "=== ALL BATCHES DONE $(date) ==="
$PY /data01/jh/afford_diag/update_report.py
