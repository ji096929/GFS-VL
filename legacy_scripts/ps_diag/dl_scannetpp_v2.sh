#!/bin/bash
# 从 hf-mirror 顺序下载 Pointcept 的 ScanNet++ 预处理包（8 个分片，约 325GB）
#
# 设计要点（修复 2026-11-19 的截断事故）
#   * 每次尝试都是【新的 curl 进程】，不用 --retry：curl 进程内重试会把 -C - 的偏移重置为 0，
#     把已下载的内容从头部覆盖掉（实测同一 inode 写了 17.3GB、文件只剩 5MB）
#   * 每次尝试前先做 Range 健康检查：必须返回 206 才真正下载；
#     返回 200（不支持 Range）或 5xx 时只等待，不碰文件
#   * 每轮比较文件大小：连续 5 轮无增长才放弃；服务器不可用最多等 20 轮
#   * 断点续传完全依赖磁盘上的文件大小，脚本重启后继续
set -u

BASE='https://hf-mirror.com/datasets/Pointcept/concerto_scannetpp_compressed/resolve/main'
DEST='/data01/jh/data/scannetpp_parts'
LOG='/data01/jh/afford_diag/gfs_build/hf_download.log'
mkdir -p "$DEST"
cd "$DEST" || exit 1

# 分片大小（last part 不同）；若服务器给 Content-Length 会以此为准
want_for() {
  case "$1" in
    *part07*) echo 23933900000 ;;
    *)        echo 42949700000 ;;
  esac
}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

log "===== 启动下载（pid $$）====="

for i in 0 1 2 3 4 5 6 7; do
  f=$(printf 'scannetpp.tar.gz.part%02d' "$i")
  url="$BASE/$f"
  want=$(want_for "$f")

  cur=$(stat -c%s "$f" 2>/dev/null || echo 0)
  if [ "$cur" -ge "$want" ]; then
    log "$f 已完成（$cur B），跳过"
    continue
  fi
  log "$f 开始，已有 $cur / $want B"

  stall=0
  bad_srv=0
  attempt=0
  while :; do
    attempt=$((attempt + 1))
    if [ "$attempt" -gt 800 ]; then log "$f 尝试次数用尽，退出"; exit 4; fi

    cur=$(stat -c%s "$f" 2>/dev/null || echo 0)
    [ "$cur" -ge "$want" ] && break

    # --- Range 健康检查：只接受 206 ---
    code=$(curl -sL -r 0-99 -o /dev/null -w '%{http_code}' --max-time 30 "$url" 2>/dev/null || echo 000)
    if [ "$code" != "206" ]; then
      bad_srv=$((bad_srv + 1))
      log "$f 健康检查 http=$code（需要 206），第 $bad_srv 次，等 60s"
      [ "$bad_srv" -ge 20 ] && { log "$f 服务器持续不可用，退出"; exit 2; }
      sleep 60
      continue
    fi
    bad_srv=0

    # --- 单次下载：新 curl 进程，-C - 从当前文件大小续传，不带 --retry ---
    t0=$(date +%s)
    curl -sL -C - --connect-timeout 30 --max-time 0 -o "$f" "$url" >> "$LOG" 2>&1
    rc=$?
    t1=$(date +%s)
    new=$(stat -c%s "$f" 2>/dev/null || echo 0)
    dt=$((t1 - t0)); [ "$dt" -lt 1 ] && dt=1
    rate=$(( (new - cur) / dt / 1024 ))
    log "$f attempt $attempt rc=$rc  $cur -> $new B（${rate} KB/s）"

    if [ "$new" -lt "$cur" ]; then
      log "$f !! 文件变小（$cur -> $new B），疑似被截断，立即停止"
      exit 5
    fi
    if [ "$new" -le "$cur" ]; then
      stall=$((stall + 1))
      [ "$stall" -ge 5 ] && { log "$f 连续 5 次无增长，退出"; exit 3; }
      sleep 15
    else
      stall=0
      sleep 3
    fi
  done
  log "$f 完成（$(stat -c%s "$f") B）"
done

log "===== 全部分片完成 ====="
