#!/bin/bash
# ScanNet++ 剩余三局串行（首局 scpp_base_lovasz k1 由主控单独起）。
#   2. scpp_sym_lovasz  k1  : 跨层双头 1-shot
#   3. scpp_base_lovasz k5  : 裸线性 5-shot
#   4. scpp_sym_lovasz  k5  : 跨层双头 5-shot
# 在线 VLM：env -u GFS_OFFLINE_PL（train.py:373 触发 build_zero_shot_model）
set -u
GFSDIR=/data01/jh/code/GFS-VL
PY=/data01/jh/conda_envs/gfs_vl/bin/python
GPUS=GPU-8ea65afc-1652-0af2-8dbd-dac776e9cf08,GPU-db4b5085-e4a2-0c20-1c46-20273faf4592,GPU-8784a584-6940-a13f-53fe-4e1a6cd15977
BATCHLOG=/data01/jh/data/scpp_batch.log

cd "$GFSDIR" || exit 1
export PYTHONPATH="$GFSDIR:$GFSDIR/pointcept/models/PLA"
export CUDA_VISIBLE_DEVICES="$GPUS"
export GFS_RESERVE_GB=30
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
  tag=$1; cfg=$2
  save="$GFSDIR/exp/scannetpp/$tag"
  log="/data01/jh/data/$tag.log"
  rm -rf "$save"
  echo "[$(date '+%F %T')] === $tag START cfg=$(basename $cfg) ===" >> "$BATCHLOG"
  env -u GFS_OFFLINE_PL "$PY" tools/train.py \
    --config-file "$cfg" --num-gpus 3 \
    --options num_worker=12 \
              "regis_train_list=['regis1']" \
              find_unused_parameters=True \
              model.backbone.enable_flash=False \
              save_path="$save" \
    > "$log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] === $tag DONE rc=$rc ===" >> "$BATCHLOG"
  grep -E 'Val result: mIoU/Base' "$log" | tail -1 >> "$BATCHLOG"
  return $rc
}

run scpp_sym_lovasz      "$GFSDIR/configs/scannetpp/semseg-pt-v3m1-0-gfsregistrain_k1_sym.py"
run scpp_base_lovasz_k5  "$GFSDIR/configs/scannetpp/semseg-pt-v3m1-0-gfsregistrain_k5.py"
run scpp_sym_lovasz_k5   "$GFSDIR/configs/scannetpp/semseg-pt-v3m1-0-gfsregistrain_k5_sym.py"

echo "[$(date '+%F %T')] === ALL DONE ===" >> "$BATCHLOG"
