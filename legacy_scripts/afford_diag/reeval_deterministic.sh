#!/bin/bash
# 确定性重评：6 个历史权重（base/sym × s42/s7/s13 的 1-shot）
# 用修好的 shuffle_orders（eval 确定），只评估不重训。
# 分两组并行：组 A 跑 base 三个 seed，组 B 跑 sym 三个 seed。
set -u
GFSDIR=/data01/jh/code/GFS-VL
PY=/data01/jh/conda_envs/gfs_vl/bin/python
CFG_BASE=$GFSDIR/configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1.py
CFG_SYM=$GFSDIR/configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_sym_lovasz.py
PL=/dev/shm/jh_pl_k1_s10
OUT=/data01/jh/data/cm_analysis/det_eval
LOG=/data01/jh/data/det_eval_all.log
GPUS_A=GPU-8ea65afc-1652-0af2-8dbd-dac776e9cf08,GPU-db4b5085-e4a2-0c20-1c46-20273faf4592,GPU-8784a584-6940-a13f-53fe-4e1a6cd15977
GPUS_B=GPU-0f2a7e5a-432f-9c3f-8012-ae0870e15499,GPU-880bb2b3-1a85-c20b-56be-3589d20f3064,GPU-39317709-7096-1d39-3d4e-31f84f9776ba

cd "$GFSDIR" || exit 1
export PYTHONPATH="$GFSDIR:$GFSDIR/pointcept/models/PLA"
export GFS_OFFLINE_PL="$PL"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_one() {
  tag=$1; cfg=$2; seed=$3; gpus=$4
  w="$GFSDIR/exp/scannet200/${tag}_${seed}/model/regis1_model_last.pth"
  if [ ! -f "$w" ]; then
    echo "[$(date '+%F %T')] SKIP $tag $seed (无权重 $w)" >> "$LOG"
    return 1
  fi
  sp="$OUT/${tag}_${seed}"
  mkdir -p "$sp"
  echo "[$(date '+%F %T')] START $tag $seed" >> "$LOG"
  CUDA_VISIBLE_DEVICES="$gpus" "$PY" tools/test.py \
    --config-file "$cfg" --num-gpus 3 \
    --options weight="$w" save_path="$sp" \
              model.backbone.enable_flash=False \
              "regis_train_list=['regis1']" \
    > "$sp/run.log" 2>&1
  rc=$?
  r=$(grep -E 'Val result: mIoU/Base' "$sp/run.log" | tail -1 | sed 's/.*allAcc //')
  echo "[$(date '+%F %T')] DONE $tag $seed rc=$rc  $r" >> "$LOG"
}

group() {
  tag=$1; cfg=$2; gpus=$3
  for s in s42 s7 s13; do
    run_one "$tag" "$cfg" "$s" "$gpus"
  done
}

group base_lovasz "$CFG_BASE" "$GPUS_A" &
p1=$!
group sym_lovasz "$CFG_SYM" "$GPUS_B" &
p2=$!
wait $p1; r1=$?
wait $p2; r2=$?
echo "[$(date '+%F %T')] ALL DONE rc=$r1/$r2" >> "$LOG"
