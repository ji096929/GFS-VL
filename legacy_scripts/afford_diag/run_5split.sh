#!/bin/bash
# 5-split（5 个 support 抽样种子 10/20/30/40/50）的 1-shot 训练。
# trainer 会依次跑 regis1..regis5，每个 split 结束后用 PreciseEvaluator 评估，
# 最后对 5 个 split 的逐类 IoU 做平均（train.py 的 Final Eval 逻辑）。
#
# 每个 split 的离线伪标签目录由 GFS_OFFLINE_PL_TMPL 按 {seed} 自动切换。
# 前提：/data01/jh/data/pl_k1_s{10,20,30,40,50} 已就绪。
#
# 用法: bash run_5split.sh <gpu_uuid> [配置名] [save_path]
set -u
GPU="${1:?需要 gpu uuid}"
CFG="${2:-configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1_sym_lovasz_5split.py}"
SAVE="${3:-./exp/scannet200/sym_lovasz_5split_s42}"

cd /data01/jh/code/GFS-VL || exit 1
export PYTHONPATH="/data01/jh/code/GFS-VL:/data01/jh/code/GFS-VL/pointcept/models/PLA${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export GFS_RESERVE_GB=22
export CUDA_VISIBLE_DEVICES="$GPU"
# 按 support 种子切换离线伪标签目录
export GFS_OFFLINE_PL_TMPL='/data01/jh/data/pl_k1_s{seed}'

LOG=/data01/jh/data/5split_$(basename "$SAVE").log
echo "[$(date '+%F %T')] 启动 5-split gpu=$GPU cfg=$CFG save=$SAVE" >> "$LOG"
echo "[$(date '+%F %T')] 模板 GFS_OFFLINE_PL_TMPL=$GFS_OFFLINE_PL_TMPL" >> "$LOG"
for s in 10 20 30 40 50; do
    n=$(ls "/data01/jh/data/pl_k1_s$s" 2>/dev/null | wc -l)
    echo "  伪标签目录 pl_k1_s$s: $n 文件" >> "$LOG"
    if [ "$n" -lt 2400 ]; then
        echo "  !! pl_k1_s$s 未就绪，终止" >> "$LOG"
        exit 1
    fi
done

mkdir -p "$SAVE"
/data01/jh/conda_envs/gfs_vl/bin/python tools/train.py \
    --config-file "$CFG" --num-gpus 1 \
    --options save_path="$SAVE" seed=42 num_worker=4 >> "$LOG" 2>&1
rc=$?
echo "[$(date '+%F %T')] 5-split 结束 rc=$rc" >> "$LOG"
exit $rc
