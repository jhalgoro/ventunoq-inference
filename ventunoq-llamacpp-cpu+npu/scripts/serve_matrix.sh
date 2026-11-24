#!/usr/bin/env bash
# L7 — serving & concurrency: llama-server -np {1,2,4} on both engines.
# Usage: scripts/serve_matrix.sh <gguf> <cpu|npu2> <tag>
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
GGUF=${1:?}; ENGINE=${2:?cpu|npu2}; TAG=${3:?}

if [ "$ENGINE" = cpu ]; then
  BIN=$HEX2/bin/llama-server; EXTRA="-ngl 0 $CPU_PIN"
  export LD_LIBRARY_PATH=/tmp/opencode/libfix:$HEX2/lib
else
  BIN=$HEX2/bin/llama-server; EXTRA="--device HTP0 -ngl 100 -t 4 --cpu-mask 0x0f -b 1024 -ub 512 -fa on"
  export LD_LIBRARY_PATH=/tmp/opencode/libfix:$HEX2/lib ADSP_LIBRARY_PATH=$HEX2/lib
  export GGML_HEXAGON_OPBATCH=2048 GGML_HEXAGON_MBUF=256,256
fi

for NP in 1 2 4; do
  PORT=$(( 8190 + NP ))
  $BIN -m "$GGUF" $EXTRA -c 8192 --parallel $NP --port $PORT -nga 1 \
      > results/serve_${TAG}_np${NP}_server.log 2>&1 &
  SPID=$!
  for _ in $(seq 1 120); do
    curl -s -o /dev/null "http://127.0.0.1:$PORT/health" && break; sleep 2
  done
  python3 scripts/serve_load.py $PORT "${TAG}_np${NP}" $NP 3
  kill $SPID; wait $SPID 2>/dev/null || true
  sleep 5
done
