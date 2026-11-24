#!/bin/bash
# KV-cache memory + long-context prefill scaling test for LFM3-1.7B.
# Measures llama.cpp's reported KV buffer sizes and peak RSS at 32K/128K ctx,
# and prefill speed at 16K/32K/64K prompt lengths (SWA efficiency claim).
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=$PWD/llama.cpp
MODEL=${1:-models/LFM3-1.7B-Q4_K_M.gguf}
TAG=$(basename "$MODEL" .gguf)

echo "== KV memory @ 32K ctx =="
/usr/bin/time -v ./llama.cpp/llama-cli -m "$MODEL" -c 32768 -p "hi" -n 1 --temp 0 2>&1 \
  | grep -E "KV buffer|compute buffer|model buffer|context size|Maximum resident" | sed 's/^ *//'

echo; echo "== KV memory @ 128K ctx =="
/usr/bin/time -v ./llama.cpp/llama-cli -m "$MODEL" -c 131072 -p "hi" -n 1 --temp 0 2>&1 \
  | grep -E "KV buffer|compute buffer|model buffer|context size|Maximum resident" | sed 's/^ *//'
