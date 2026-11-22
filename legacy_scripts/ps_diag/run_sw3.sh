#!/bin/bash
# SW3：真实在线 PS+AI 底盘 + 43.6 万补标 + 逐点 Sonata 软权重
set -u
cd /data01/jh/code/ps_diag || exit 1
export GFS_OFFLINE_PL=/data01/jh/data/online_pl_w
export GFS_RESERVE_GB=${GFS_RESERVE_GB:-6}
export GFS_DEBUG_W=1
bash train_guard.sh 7 20000 \
  /data01/jh/code/GFS-VL/exp/scannet200/sc200_k1_SW3_s42 42 \
  configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_softw.py
