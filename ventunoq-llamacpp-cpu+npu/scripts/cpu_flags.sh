#!/usr/bin/env bash
# L6 — CPU micro-tuning: one-flag-one-run on LFM3 + Qwen3.5-2B, pinned.
# Engine = $HEX2 runner at -ngl 0 (scope: CPU+NPU arch only — no v0.8.0 CPU-only stack).
# Baseline row = board default ($CPU_PIN, llama-bench defaults b=2048/ub=512, fa auto).
# Service tuning row = what the Bricks runner ships (-b 1024 -ub 512 -fa on).
# Usage: scripts/cpu_flags.sh <gguf> <tag>   (model must already be on disk)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export LD_LIBRARY_PATH=/tmp/opencode/libfix:$HEX2/lib
GGUF=${1:?}; TAG=${2:?}
A="-p 512,8192 -n 128 -r 1"   # r=1: the sweep's effect sizes are big; noise is in the CSV
run() { # row-name + flags
  timeout 7200 $HEX2/bin/llama-bench -m "$GGUF" -ngl 0 $CPU_PIN $A ${2:-} -o csv \
    > "results/bench_${TAG}-$1.csv" 2>&1 || true
  echo "$(date -u +%T) l6 $TAG $1" >> results/matrix_status.log
}
run base      ""
run svc       "-b 1024 -ub 512 -fa on"
run b4096     "-b 4096"
run ub1024    "-ub 1024"
run faon      "-fa on"
run nommap    "--no-mmap"
run poll0     "--poll 0"
