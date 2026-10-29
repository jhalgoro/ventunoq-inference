#!/bin/bash
# Start llama-server for evals; call with model path. Logs to results/server_<tag>.log
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=$PWD/llama.cpp
MODEL=${1:-models/LFM3-1.7B-Q4_K_M.gguf}
TAG=$(basename "$MODEL" .gguf)
pkill -f "llama-server.*$TAG" 2>/dev/null; sleep 1
nohup ./llama.cpp/llama-server -m "$MODEL" -c 8192 --port 8181 --temp 0.1 --top-k 50 \
  --repeat-penalty 1.1 --jinja > "results/server_${TAG}.log" 2>&1 &
for i in $(seq 1 60); do
  sleep 2
  curl -s -m 2 http://127.0.0.1:8181/health 2>/dev/null | grep -q ok && { echo "server up: $TAG"; exit 0; }
done
echo "server failed to start; see results/server_${TAG}.log"; tail -20 "results/server_${TAG}.log"; exit 1
