#!/bin/bash
# Phase 2: LFM3 quality evals + perplexity, then Qwen3.5-2B/4B comparison cycle.
# Disk policy: delete each model right after its tests; log everything (DISK_LOG.md).
# Waits for phase-1 bench suite (results/suite_done) before starting.
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=$PWD/llama.cpp
B=./llama.cpp/llama-bench
LOG=results/DISK_LOG.md
Q4KM=models/LFM3-1.7B-Q4_K_M.gguf
QAD=models/LFM3-1.7B-QAD-Q4_0.gguf
LFM25=models/LFM2.5-2.6B-Q4_K_M.gguf

disklog() { # $1=action $2=file
  echo "" >> $LOG
  echo "### $(date -u +'%F %R UTC') — $1: \`$2\`" >> $LOG
  [ -f "$2" ] && echo '```' >> $LOG && sha256sum "$2" >> $LOG && echo '```' >> $LOG
  echo '```' >> $LOG; df -h / | tail -1 >> $LOG; echo '```' >> $LOG
}
delmodel() { disklog "DELETE (tests finished)" "$1"; rm -f "$1"; disklog "AFTER delete" "$1"; }

[ -f $LOG ] || { echo "# Disk usage log — delete-each-model-after-testing (target: / < 50%)" > $LOG; df -h / | tail -1 >> $LOG; }

# ---- wait for phase-1 suite ----
while [ ! -f results/suite_done ]; do sleep 30; done

# ---- A. LFM3-1.7B quality evals (server) ----
bash scripts/serve.sh $Q4KM || exit 1
python3 scripts/smoke_chat.py 8181 > results/smoke_lfm3_1.7b.txt 2>&1
python3 scripts/gsm8k_eval.py 8181 50 LFM3-1.7B-Q4_K_M > results/gsm8k_run.log 2>&1
python3 scripts/humaneval_eval.py 8181 20 LFM3-1.7B-Q4_K_M > results/humaneval_run.log 2>&1
pkill -f llama-server; sleep 3

# ---- B. perplexity (Gutenberg corpus, both quants) ----
for M in $Q4KM $QAD; do
  ./llama.cpp/llama-perplexity -m "$M" -f bench/gatsby.txt -c 2048 -b 2048 -ub 512 \
    > "results/perplexity_$(basename $M .gguf).txt" 2>&1 \
    || echo "perplexity failed for $M (see file)" >> results/perplexity_$(basename $M .gguf).txt
done
delmodel $QAD
delmodel $Q4KM
delmodel $LFM25

# ---- C. Qwen3.5 comparison cycle (unsloth GGUFs only, Q4_K_M) ----
fits() { # $1 = bytes needed; keep / below 50% with 2GB working margin
  local need_kb=$(( $1 / 1024 + 2*1024*1024 ))
  local avail_kb=$(df -k / | awk 'NR==2{print $4}')
  [ "$avail_kb" -ge "$need_kb" ]
}
qwen_cycle() { # $1=size  $2=bench args  $3=gsm8k N  $4=humaneval N  [$5=extra bench args]
  local S=$1 BENCH_ARGS=$2 GN=$3 HN=$4 EXTRA=${5:-} TAG=Qwen3.5-$1-GGUF F=models/Qwen3.5-$1-Q4_K_M.gguf
  local SIZE=$(curl -sSLI -m 20 "https://huggingface.co/unsloth/$TAG/resolve/main/Qwen3.5-$1-Q4_K_M.gguf" | tr -d '\r' | awk 'tolower($1)=="x-linked-size:"{print $2}')
  if ! fits "${SIZE:-6000000000}"; then
    echo "SKIPPED $(date -u +%FT%TZ) $S: need ${SIZE} bytes, would exceed 50% disk policy" >> results/qwen${S}_status
    disklog "SKIP download $S (disk policy)" /dev/null
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
echo "EVAL SUITE DONE" > results/eval_done
