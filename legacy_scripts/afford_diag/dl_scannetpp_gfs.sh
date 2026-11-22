#!/bin/bash
# GFS-PCS 官方 ScanNetpp.tar.gz 续传下载（走本机反向隧道代理 127.0.0.1:17890）
# 目标 23,142,919,323 字节；已完成部分用 curl -C - 续传
URL='https://huggingface.co/datasets/ZhaochongAn/GFS_PCS_Datasets/resolve/main/ScanNetpp.tar.gz'
OUT='/data01/jh/data/ScanNetpp.tar.gz'
WANT=23142919323
LOG='/data01/jh/data/dl_scannetpp_gfs.log'
export https_proxy=http://127.0.0.1:17890
export http_proxy=http://127.0.0.1:17890

echo "[$(date '+%F %T')] === 开始，当前 $(stat -c%s "$OUT" 2>/dev/null || echo 0) / $WANT ===" >> "$LOG"
for round in $(seq 1 300); do
  cur=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
  if [ "$cur" -ge "$WANT" ]; then
    echo "[$(date '+%F %T')] DONE size=$cur" >> "$LOG"
    exit 0
  fi
  curl -sSL -C - --retry 5 --retry-delay 10 --connect-timeout 30 \
       --speed-time 120 --speed-limit 2048 --max-time 3600 \
       -o "$OUT" "$URL" >> "$LOG" 2>&1
  new=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
  echo "[$(date '+%F %T')] round $round: $cur -> $new (+$((new-cur)))" >> "$LOG"
done
echo "[$(date '+%F %T')] GAVEUP size=$(stat -c%s "$OUT" 2>/dev/null || echo 0)" >> "$LOG"
exit 1
