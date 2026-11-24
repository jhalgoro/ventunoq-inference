#!/usr/bin/env bash
# Standard MNN bench: big-core pinned, fp16, llama-bench-identical -pg 512,128.
# Usage: scripts/bench_mnn.sh models/Qwen3.5-0.8B-MNN-text <tag> [-extra llm_bench flags]
# Writes results/bench_<tag>.txt (+ _done marker). Compare with llama.cpp:
#   llama-bench -m x.gguf -ngl 0 -t 4 --cpu-mask 0xf --cpu-strict 1 -p 512 -n 128 -r 3
set -euo pipefail
cd "$(dirname "$0")/.."

MNN_SRC=${MNN_SRC:-$PWD/MNN}
CFG=${1:?usage: bench_mnn.sh <model-dir> <tag>}; TAG=${2:?need tag}; shift 2
CFG=$(realpath "$CFG")
# llm_bench -m reads the config file directly — a model dir must resolve to config.json
[ -d "$CFG" ] && CFG="$CFG/config.json"
[ -e "$CFG" ] || { echo "no config at $CFG (F9: run scripts/text_only.py?)"; exit 1; }

OUT=results/bench_$TAG.txt
taskset -c 0-3 "$MNN_SRC/build/llm_bench" -m "$CFG" -a cpu -t 4 -c 2 -pg 512,128 -rep 3 "$@" \
  > "$OUT" 2>&1 || { echo "FAILED — see $OUT"; exit 1; }
grep -E "pp512|tg128|device supports" "$OUT" | head -6
touch results/bench_${TAG}_done
echo "→ $OUT"
