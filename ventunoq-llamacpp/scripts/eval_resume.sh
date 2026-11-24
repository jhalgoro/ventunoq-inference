#!/bin/bash
# Resume eval_suite.sh after two bugs were fixed ($5 unbound; entry_point key).
# Skips already-finished work: LFM3 GSM8K (30/50) + perplexity (both quants) + phase-1 benches.
# Still owed: LFM3 HumanEval, Qwen3.5 2B/4B/9B full cycles.
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=$PWD/llama.cpp
B=./llama.cpp/llama-bench
LOG=results/DISK_LOG.md
Q4KM=models/LFM3-1.7B-Q4_K_M.gguf

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

# ---- A'. LFM3-1.7B HumanEval only (GSM8K + PPL already in results/) ----
if [ ! -f results/humaneval_LFM3-1.7B-Q4_K_M.json ]; then
  curl -sSL --fail -m 3600 -o "$Q4KM" "https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/main/LFM3-1.7B-Q4_K_M.gguf" || { echo "LFM3 download failed"; exit 1; }
  disklog "DOWNLOAD (resume: LFM3 humaneval)" "$Q4KM"
  bash scripts/serve.sh "$Q4KM" || exit 1
  python3 scripts/humaneval_eval.py 8181 20 LFM3-1.7B-Q4_K_M > results/humaneval_run.log 2>&1
  pkill -f llama-server; sleep 3
  delmodel "$Q4KM"
fi

# ---- C. Qwen3.5 comparison cycle (unsloth GGUFs, Q4_K_M) ----
qwen_cycle() { # $1=size  $2=bench args  $3=gsm8k N  $4=humaneval N  [$5=extra bench args]
  local S=$1 BENCH_ARGS=$2 GN=$3 HN=$4 EXTRA=${5:-} TAG=Qwen3.5-$1-GGUF F=models/Qwen3.5-$1-Q4_K_M.gguf
  [ -f "results/qwen${S}_status" ] && grep -q done "results/qwen${S}_status" && { echo "$S already done"; return; }
  local SIZE=$(curl -sSLI -m 20 "https://huggingface.co/unsloth/$TAG/resolve/main/Qwen3.5-$1-Q4_K_M.gguf" | tr -d '\r' | awk 'tolower($1)=="x-linked-size:"{print $2}')
  if ! fits "${SIZE:-6000000000}"; then
    echo "SKIPPED $(date -u +%FT%TZ) $S: need ${SIZE} bytes, would exceed 50% disk policy" >> results/qwen${S}_status
    return
  fi
  echo "downloading $TAG Q4_K_M ($((${SIZE:-0}/1024/1024)) MB)..."
  curl -sSL --fail -m 3600 -o "$F" "https://huggingface.co/unsloth/$TAG/resolve/main/Qwen3.5-$1-Q4_K_M.gguf" || { echo "download failed $S" >> results/qwen${S}_status; return; }
  disklog "DOWNLOAD" "$F"
  $B -m "$F" $BENCH_ARGS -o csv > results/bench_qwen35_${S}_main.csv 2>&1
  [ -n "$EXTRA" ] && $B -m "$F" $EXTRA -o csv >> results/bench_qwen35_${S}_main.csv 2>&1
  [ "$GN" -eq 0 ] && { delmodel "$F"; echo "bench-only done" >> results/qwen${S}_status; return; }
  bash scripts/serve.sh "$F" || { echo "serve failed $S" >> results/qwen${S}_status; delmodel "$F"; return; }
  python3 scripts/gsm8k_eval.py 8181 $GN Qwen3.5-$1-Q4_K_M > results/gsm8k_qwen35_$S.log 2>&1
  python3 scripts/humaneval_eval.py 8181 $HN Qwen3.5-$1-Q4_K_M > results/humaneval_qwen35_$S.log 2>&1
  pkill -f llama-server; sleep 3
  delmodel "$F"
  echo "done" >> results/qwen${S}_status
}
qwen_cycle 2B "-p 512,2048,8192 -n 32,128 -r 2" 50 20
qwen_cycle 4B "-p 512,2048,8192 -n 32,128 -r 2" 50 20
# 9B: fits (5.7 GB + KV < 14 GB RAM) but decodes slowly -> reduced scope
qwen_cycle 9B "-p 512,2048 -n 32,128 -r 1" 20 10 "-p 8192 -n 0 -r 1"

echo "# Final disk state" >> $LOG
df -h / | tail -1 >> $LOG
echo "EVAL SUITE DONE (resumed)" > results/eval_done
