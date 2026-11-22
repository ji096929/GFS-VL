#!/bin/bash
# 等 support_sym 训练结束进入评估；若评估 3 分钟内推进不足 8 个场景，
# 杀掉训练进程并用 num_worker=2 重跑评估。
LOG=/data01/jh/afford_diag/xl_runs/support_sym_s42.log
PY=/data01/jh/conda_envs/gfs_vl/bin/python
cd /data01/jh/code/GFS-VL || exit 1

# 1. 等评估开始
for i in $(seq 1 60); do
  if grep -q 'Test: 1/312' "$LOG" 2>/dev/null; then break; fi
  sleep 10
done

# 2. 观察 180 秒
n1=$(grep -oE 'Test: [0-9]+/312' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1)
sleep 180
n2=$(grep -oE 'Test: [0-9]+/312' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1)
n1=${n1:-0}; n2=${n2:-0}
echo "progress after 180s: $n1 -> $n2"

if [ "$n2" -lt $((n1 + 8)) ]; then
  echo "too slow, switching to num_worker=2"
  ps -eo pid,args | grep 'gfsregistrain_k1_sym.py' | grep -v grep | awk '{print $1}' | xargs -r kill -9
  sleep 3
  export GFS_OFFLINE_PL=/data01/jh/data/empty_pl
  export PYTHONPATH=/data01/jh/code/GFS-VL:/data01/jh/code/GFS-VL/pointcept/models/PLA:$PYTHONPATH
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  CUDA_VISIBLE_DEVICES=GPU-8ea65afc-1652-0af2-8dbd-dac776e9cf08 $PY tools/test.py \
    --config-file configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_sym.py \
    --num-gpus 1 \
    --options weight=exp/scannet200/support_sym_s42/model/regis1_model_last.pth num_worker=2 \
    >> /data01/jh/afford_diag/xl_runs/support_sym_s42_eval.log 2>&1
  echo "eval done rc=$?"
else
  echo "progress ok, let it run"
fi
