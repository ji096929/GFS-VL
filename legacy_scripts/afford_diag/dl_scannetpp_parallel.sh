#!/bin/bash
# ScanNet++ 并行分片下载：part01..part07 各自独立 curl，独立日志，可单独观测
# part00 已完成（42949700000 字节），跳过
BASE='https://hf-mirror.com/datasets/Pointcept/concerto_scannetpp_compressed/resolve/main'
DEST='/data01/jh/data/scannetpp_parts'
mkdir -p "$DEST" || exit 1
cd "$DEST" || exit 1

dl_one() {
  i=$1
  f=$(printf 'scannetpp.tar.gz.part%02d' "$i")
  if [ "$i" = "7" ]; then want=23933900000; else want=42949700000; fi
  log="$DEST/dl_p${i}.log"
  echo "[$(date '+%F %T')] === start $f want=$want ===" >> "$log"
  for round in $(seq 1 200); do
    cur=$(stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$cur" -ge "$want" ]; then
      echo "[$(date '+%F %T')] DONE $f size=$cur" >> "$log"
      return 0
    fi
    # -C - 断点续传；速度低于 1KB/s 持续 120s 则放弃，由外层重试
    curl -sSL -C - \
         --retry 20 --retry-delay 15 --retry-all-errors \
         --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
         --max-time 7200 \
         -o "$f" "$BASE/$f" >> "$log" 2>&1
    new=$(stat -c%s "$f" 2>/dev/null || echo 0)
    echo "[$(date '+%F %T')] round $round: $cur -> $new (+$((new-cur)))" >> "$log"
  done
  echo "[$(date '+%F %T')] GAVEUP $f size=$(stat -c%s "$f" 2>/dev/null || echo 0)" >> "$log"
  return 1
}

# 低优先级：3 路并行，每组顺序下多个分片（原 7 路改成 3 路，配合 ionice -c3）
pids=""
for grp in "1 4 7" "2 5" "3 6"; do
  (
    for i in $grp; do dl_one "$i"; done
  ) &
  pids="$pids $!"
done

rc=0
for p in $pids; do
  wait "$p" || rc=1
done
echo "ALL_DONE rc=$rc $(date '+%F %T')" >> "$DEST/dl_all_status.log"
