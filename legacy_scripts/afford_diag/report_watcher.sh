#!/bin/bash
while true; do
  /data01/jh/conda_envs/gfs_vl/bin/python /data01/jh/afford_diag/update_report.py >> /data01/jh/afford_diag/xl_runs/report_watcher.log 2>&1
  sleep 180
done
