#!/bin/bash
# 把 ScanNetpp.tar.gz 剩余部分切成 N 段并行下载（走本机 17890 反向隧道）。
# 每段独立断点续传（记录已下字节，手动追加），全部校验大小后才合并。
URL='https://huggingface.co/datasets/ZhaochongAn/GFS_PCS_Datasets/resolve/main/ScanNetpp.tar.gz'
OUT='/data01/jh/data/ScanNetpp.tar.gz'
DIR='/data01/jh/data/scanpp_parts'
WANT=23142919323
N=${1:-5}
export https_proxy=http://127.0.0.1:17890
export http_proxy=http://127.0.0.1:17890
mkdir -p "$DIR"

S0=$(stat -c%s "$OUT")
REM=$((WANT - S0))
CHUNK=$(( (REM + N - 1) / N ))
echo "[$(date '+%F %T')] === 起点 $S0 / $WANT  剩余 $REM  分 $N 段 每段 $CHUNK ===" >> "$DIR/dl.log"

dl_seg() {
  i=$1; start=$2; end=$3
  f="$DIR/seg$i"
  want=$((end - start + 1))
  for round in $(seq 1 200); do
    have=$(stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$have" -ge "$want" ]; then
      echo "[$(date '+%F %T')] seg$i DONE $have/$want" >> "$DIR/dl.log"
      return 0
    fi
    # 不要 --retry / --retry-all-errors / --max-time：
    #   curl 内部重试会以固定的 -r 起点覆盖 -o 文件 => 每次从 0 重下；
    #   且 shell 循环永远等不到 curl 退出，cat .part 的续传逻辑形同虚设。
    #   （2026-11-22 实测：每小时白烧 786MB x 5 的代理流量，一晚上下不完。）
    # 让 curl 单次跑到失败/极慢就退出，由本循环按已落盘字节续传。
    curl -sSL -r $((start + have))-${end} \
         --connect-timeout 30 --speed-time 180 --speed-limit 1024 \
         -o "$f.part" "$URL" >> "$DIR/seg$i.log" 2>&1
    if [ -s "$f.part" ]; then cat "$f.part" >> "$f"; fi
    rm -f "$f.part"
    new=$(stat -c%s "$f" 2>/dev/null || echo 0)
    echo "[$(date '+%F %T')] seg$i round $round: $have -> $new / $want" >> "$DIR/dl.log"
  done
  return 1
}

pids=""
for i in $(seq 0 $((N-1))); do
  start=$((S0 + i * CHUNK))
  end=$(( start + CHUNK - 1 ))
  [ $end -ge $WANT ] && end=$((WANT - 1))
  dl_seg "$i" "$start" "$end" &
  pids="$pids $!"
done
rc=0
for p in $pids; do wait "$p" || rc=1; done

ok=1
for i in $(seq 0 $((N-1))); do
  start=$((S0 + i * CHUNK)); end=$(( start + CHUNK - 1 ))
  [ $end -ge $WANT ] && end=$((WANT - 1))
  want=$((end - start + 1))
  have=$(stat -c%s "$DIR/seg$i" 2>/dev/null || echo 0)
  if [ "$have" -ne "$want" ]; then
    echo "[$(date '+%F %T')] seg$i 大小不符 $have != $want" >> "$DIR/dl.log"; ok=0
  fi
done

if [ $ok -eq 1 ] && [ $rc -eq 0 ]; then
  for i in $(seq 0 $((N-1))); do cat "$DIR/seg$i" >> "$OUT"; done
  tot=$(stat -c%s "$OUT")
  echo "[$(date '+%F %T')] === 合并完成 总大小 $tot / $WANT ===" >> "$DIR/dl.log"
  [ "$tot" -eq "$WANT" ] && echo "SIZE_OK" >> "$DIR/dl.log" || echo "SIZE_MISMATCH" >> "$DIR/dl.log"
  exit 0
else
  echo "[$(date '+%F %T')] 有分片不完整，未合并（分段文件保留，可重跑续传）" >> "$DIR/dl.log"
  exit 1
fi
