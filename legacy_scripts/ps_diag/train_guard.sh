#!/bin/bash
# 训练守护：等显存 -> 启动 -> 异常退出 -> 等显存 -> 从最新 checkpoint 续训
#
# 用法:
#   bash train_guard.sh <CVD> <min_free_MiB> <save_path> <seed> <config> [额外 options...]
# 例:
#   bash train_guard.sh 7 30000 ./exp/scannet200/sc200_k1_A_s42 42 \
#        configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_r1.py batch_size=12
#
# 说明:
#   * _model_last.pth 每个 epoch 都会原子写入，所以续训最多丢 1 个 epoch，无需改 save_freq
#   * CVD -> smi index 按实测映射表换算，不能直接用 CVD 查 nvidia-smi
#   * 训练命令里的 GFS_RESERVE_GB 会透传（预占显存）
set -u

CVD=$1; MINFREE=$2; SAVE=$3; SEED=$4; CFG=$5
shift 5
EXTRA="$*"
LOG="/data01/jh/afford_diag/gfs_build/guard_$(basename "$SAVE").log"
GFS_RESERVE_GB=${GFS_RESERVE_GB:-0}

case "$CVD" in
  0) SMI=3;; 1) SMI=6;; 2) SMI=0;; 3) SMI=1;;
  4) SMI=2;; 5) SMI=4;; 6) SMI=5;; 7) SMI=7;;
  *) echo "bad CVD=$CVD"; exit 1;;
esac

CKPT="$SAVE/model/regis1_model_last.pth"
mkdir -p "$SAVE/model"

echo "[$(date +%H:%M:%S)] guard start CVD=$CVD smi=$SMI minfree=${MINFREE}MiB save=$SAVE" >> "$LOG"

for attempt in $(seq 1 40); do
  while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$((SMI + 1))p")
    if [ "${FREE:-0}" -ge "$MINFREE" ]; then break; fi
    echo "[$(date +%H:%M:%S)] smi$SMI free ${FREE}MiB < ${MINFREE}MiB, 等待" >> "$LOG"
    sleep 30
  done

  OPT="save_path=$SAVE seed=$SEED"
  [ -n "$EXTRA" ] && OPT="$OPT $EXTRA"
  if [ -f "$CKPT" ]; then
    OPT="$OPT weight=$CKPT resume=True"
    echo "[$(date +%H:%M:%S)] 发现 checkpoint，续训: $CKPT" >> "$LOG"
  fi

  echo "[$(date +%H:%M:%S)] attempt $attempt 启动 (smi$SMI free ${FREE}MiB)" >> "$LOG"
  cd /data01/jh/code/GFS-VL || exit 1

  if [ "${GUARD_DRY:-0}" = "1" ]; then
    echo "[$(date +%H:%M:%S)] DRY: 不真正训练" >> "$LOG"
    exit 0
  fi

  PYTHONPATH=.:pointcept/models/PLA \
  CUDA_VISIBLE_DEVICES="$CVD" \
  GFS_RESERVE_GB="$GFS_RESERVE_GB" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /data01/jh/conda_envs/gfs_vl/bin/python -u tools/train.py \
      --config-file "$CFG" --options $OPT >> "$LOG" 2>&1
  RC=$?
  echo "[$(date +%H:%M:%S)] attempt $attempt 退出码 $RC" >> "$LOG"
  [ "$RC" -eq 0 ] && exit 0
done

echo "[$(date +%H:%M:%S)] 重试次数用尽，放弃" >> "$LOG"
exit 1
