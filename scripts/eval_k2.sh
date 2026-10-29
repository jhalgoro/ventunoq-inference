#!/bin/bash
# K2-Horizon cycle (IFM / K2 Lab), added on request after the Qwen3.5 suite.
# Waits for results/eval_done (Qwen suite), then runs the same harness on
# K2-Horizon-4B (fits RAM) and K2-Horizon-7B (swap-dominated, reduced scope).
# Skipped: MoVA-36B-A4B Q4_K_M = 22.4 GB (breaks the 50% disk policy),
#          0.9B (below lineup range), -Uno (diffusion LoRA adapter, not AR-runnable).
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=$PWD/llama.cpp
B=./llama.cpp/llama-bench
LOG=results/DISK_LOG.md

disklog() {
  echo "" >> $LOG
  echo "### $(date -u +'%F %R UTC') — $1: \`$2\`" >> $LOG
  [ -f "$2" ] && echo '```' >> $LOG && sha256sum "$2" >> $LOG && echo '```' >> $LOG
  echo '```' >> $LOG; df -h / | tail -1 >> $LOG; echo '```' >> $LOG
}
delmodel() { disklog "DELETE (tests finished)" "$1"; rm -f "$1"; disklog "AFTER delete" "$1"; }
fits() {
  local need_kb=$(( $1 / 1024 + 2*1024*1024 ))
  local avail_kb=$(df -k / | awk 'NR==2{print $4}')
  [ "$avail_kb" -ge "$need_kb" ]
}

echo "waiting for Qwen suite (results/eval_done)..."
while [ ! -f results/eval_done ]; do sleep 60; done
echo "Qwen suite done, starting K2-Horizon cycle $(date -u +%FT%TZ)"

k2_cycle() { # $1=repo-stem  $2=size-label  $3=bench args  $4=gsm8k N  $5=humaneval N  [$6=extra bench args] $7=file-stem
  local STEM=$1 SZ=$2 BENCH_ARGS=$3 GN=$4 HN=$5 EXTRA=${6:-} FSTEM=${7:-$1}
  local TAG=K2-Horizon-$SZ F=models/$FSTEM-Q4_K_M.gguf ST=results/k2h${SZ}_status
  local URL="https://huggingface.co/IFM/$STEM-GGUF/resolve/main/$FSTEM-Q4_K_M.gguf"
  [ -f "$ST" ] && grep -q done "$ST" && { echo "$SZ already done"; return; }
  local SIZE=$(curl -sSLI -m 20 "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | head -1)
  if ! fits "${SIZE:-6000000000}"; then
    echo "SKIPPED $(date -u +%FT%TZ) $SZ: need ${SIZE} bytes, would exceed 50% disk policy" >> "$ST"
    return
  fi
  echo "downloading $STEM Q4_K_M ($((${SIZE:-0}/1024/1024)) MB)..."
  curl -sSL --fail -m 3600 -o "$F" "$URL" || { echo "download failed $SZ" >> "$ST"; return; }
  disklog "DOWNLOAD" "$F"
  $B -m "$F" $BENCH_ARGS -o csv > results/bench_k2h_${SZ}_main.csv 2>&1
  [ -n "$EXTRA" ] && $B -m "$F" $EXTRA -o csv >> results/bench_k2h_${SZ}_main.csv 2>&1
  [ "$GN" -eq 0 ] && { delmodel "$F"; echo "bench-only done" >> "$ST"; return; }
  bash scripts/serve.sh "$F" || { echo "serve failed $SZ" >> "$ST"; delmodel "$F"; return; }
  python3 scripts/gsm8k_eval.py 8181 $GN K2-Horizon-$SZ-Q4_K_M > results/gsm8k_k2h_${SZ}.log 2>&1
  python3 scripts/humaneval_eval.py 8181 $HN K2-Horizon-$SZ-Q4_K_M > results/humaneval_k2h_${SZ}.log 2>&1
  pkill -f llama-server; sleep 3
  delmodel "$F"
  echo "done" >> "$ST"
}

# 4B (3.16 GB): full cycle, same scope as Qwen 2B/4B. NOTE repo=3.7B, file=4B
k2_cycle K2-Horizon-3.7B 4B "-p 512,2048,8192 -n 32,128 -r 2" 50 20 "" K2-Horizon-4B
# 7B (5.59 GB): fits RAM only via swap like Qwen 9B -> reduced scope
k2_cycle K2-Horizon-7B 7B "-p 512,2048 -n 32,128 -r 1" 20 10 "-p 8192 -n 0 -r 1"

echo "# Final disk state (K2 cycle)" >> $LOG
df -h / | tail -1 >> $LOG
echo "K2-HORIZON CYCLE DONE" > results/k2_done
