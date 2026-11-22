#!/bin/bash
# 从 hf-mirror 顺序下载 Pointcept 的 ScanNet++ 预处理包（8 个分片，约 325GB）
# 支持断点续传：curl -C - 对已存在的部分文件续传；整片下完才进入下一片。
set -u
BASE='https://hf-mirror.com/datasets/Pointcept/concerto_scannetpp_compressed/resolve/main'
DEST='/data01/jh/data/scannetpp_parts'
LOG='/data01/jh/afford_diag/gfs_build/hf_download.log'
mkdir -p "$DEST"
cd "$DEST" || exit 1

for i in 0 1 2 3 4 5 6 7; do
  f=$(printf 'scannetpp.tar.gz.part%02d' "$i")
  want=42949700000
  [ "$i" = "7" ] && want=23933900000
  if [ -f "$f" ] && [ "$(stat -c%s "$f")" -ge "$want" ]; then
    echo "[$(date +%H:%M:%S)] $f 已完整，跳过" >> "$LOG"
    continue
  fi
  # 单文件重试到成功为止（-C - 续传，--retry 处理瞬时错误）
  for attempt in $(seq 1 200); do
    echo "[$(date +%H:%M:%S)] $f attempt $attempt 已有 $(stat -c%s "$f" 2>/dev/null || echo 0) B" >> "$LOG"
    curl -sL -C - --retry 50 --retry-delay 10 --retry-all-errors \
         --connect-timeout 30 --max-time 0 \
         -o "$f" "$BASE/$f" >> "$LOG" 2>&1
    sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -ge "$want" ]; then
      echo "[$(date +%H:%M:%S)] $f DONE $sz" >> "$LOG"
      break
    fi
    echo "[$(date +%H:%M:%S)] $f 不完整 $sz/$want，重试" >> "$LOG"
    sleep 15
  done
done
echo "[$(date +%H:%M:%S)] ALL DONE" >> "$LOG"
