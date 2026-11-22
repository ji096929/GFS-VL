#!/bin/bash
LOG=/data01/jh/afford_diag/xl_runs/sc200_k1_XLu_s7_v2.log
while kill -0 2328599 2>/dev/null; do
  n=$(grep -c "Start Training" "$LOG" 2>/dev/null)
  if [ "$n" -ge 2 ]; then
    echo "regis2 started at $(date), killing 2328599"
    kill 2328599
    break
  fi
  sleep 15
done
echo "guard exit $(date)"
