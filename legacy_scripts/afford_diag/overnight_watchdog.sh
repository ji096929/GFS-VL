#!/bin/bash
LOG=/data01/jh/afford_diag/xl_runs
MON=$LOG/watchdog.log
RUNS="mlp_dec0_s42 mlp_dec0_s7 sym_xl_s42 sym_xl_s7 base_s13 sym_xl_s13 xl_s13 gated_sym_xl_s42 gated_sym_xl_s7"
mkdir -p $LOG
while true; do
  ts=$(date "+%F %T")
  if pgrep -f "overnight_cut2.sh" >/dev/null; then
    n=$(ps -eo args | grep "tools/train.py" | grep -v grep | grep -v "207[0-9]\{4\}" | wc -l)
    echo "[$ts] main=alive train_procs=$n" >> "$MON"
    for r in $RUNS; do
      f=$LOG/$r.log
      [ -f "$f" ] || continue
      age=$(( $(date +%s) - $(stat -c %Y "$f") ))
      err=$(grep -cE "OutOfMemory|Traceback" "$f" 2>/dev/null)
      cls=$(grep -c "Class_.*Final Result" "$f" 2>/dev/null)
      echo "[$ts]   $r age=${age}s err=$err cls=$cls" >> "$MON"
      if [ "$age" -gt 2400 ] && [ "$cls" -lt 57 ]; then
        pids=$(pgrep -f "save_path=./exp/scannet200/$r")
        if [ -n "$pids" ]; then
          echo "[$ts] STUCK $r age=${age}s cls=$cls pids=$pids -> kill" >> "$MON"
          kill $pids
        fi
      fi
    done
  else
    echo "[$ts] MAIN DEAD -> retry_failed.sh" >> "$MON"
    bash /data01/jh/afford_diag/retry_failed.sh >> "$MON" 2>&1
    echo "[$ts] retry_failed finished" >> "$MON"
    break
  fi
  sleep 600
done
