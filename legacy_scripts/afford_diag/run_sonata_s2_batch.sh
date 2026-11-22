#!/bin/bash
# Sonata-Small Stage-2 第 2、3 局串行（第 1 局由主控单独起）。
#
#   2 局 encfrozen : enc 冻结 + VLM PL   -> 对比第 1 局，回答「护住预训练 enc 是否更高」
#   3 局 support   : enc 冻结 + 无 VLM   -> 正式管线纯 3D，对标 support_sym 0.1492 / GW 5-shot 0.1455
#
# 用法: bash run_sonata_s2_batch.sh
set -u
GFSDIR=/data01/jh/code/GFS-VL
PY=/data01/jh/conda_envs/gfs_vl/bin/python
CFG_A=$GFSDIR/configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_sym_lovasz_sonata_small_a.py
BW=$GFSDIR/exp/scannet200/sonata_small_stage1a/model/model_best.pth
PL=/dev/shm/jh_pl_k1_s10
EPL=/data01/jh/data/empty_pl
GPUS=GPU-8ea65afc-1652-0af2-8dbd-dac776e9cf08,GPU-db4b5085-e4a2-0c20-1c46-20273faf4592,GPU-8784a584-6940-a13f-53fe-4e1a6cd15977
BATCHLOG=/data01/jh/data/sonata_s2_batch.log

cd "$GFSDIR" || exit 1
export PYTHONPATH="$GFSDIR:$GFSDIR/pointcept/models/PLA"
export CUDA_VISIBLE_DEVICES="$GPUS"
export GFS_RESERVE_GB=30
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f "$BW" ]; then
  echo "[$(date '+%F %T')] 缺 Stage-1a 权重 $BW，退出" >> "$BATCHLOG"
  exit 1
fi

run() {
  tag=$1; pl=$2
  save="$GFSDIR/exp/scannet200/sonata_s_s2_${tag}_lovasz"
  log="/data01/jh/data/sonata_s2_${tag}.log"
  rm -rf "$save"
  echo "[$(date '+%F %T')] === $tag START  pl=$pl ===" >> "$BATCHLOG"
  GFS_OFFLINE_PL="$pl" "$PY" tools/train.py \
    --config-file "$CFG_A" --num-gpus 3 \
    --options num_worker=12 \
              find_unused_parameters=True \
              backbone_weight="$BW" \
              save_path="$save" \
    > "$log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] === $tag DONE rc=$rc ===" >> "$BATCHLOG"
  grep -E 'Val result' "$log" | tail -1 >> "$BATCHLOG"
  return $rc
}

run encfrozen "$PL"
run support   "$EPL"

echo "[$(date '+%F %T')] === ALL DONE ===" >> "$BATCHLOG"
