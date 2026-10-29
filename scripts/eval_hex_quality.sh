#!/bin/bash
# Cycle 3 — NPU vs CPU OUTPUT-QUALITY EQUIVALENCE (see research/bricks-faster-inference.md).
# Same binary, same model (LFM2.5-2.6B-Q4_K_M), same sampler: GSM8K N=20 through an NPU
# llama-server (--device HTP0 -ngl 100, q4_K repacked to NPU + recurrent state on DSP) vs a
# -ngl 0 CPU control. Speeds already measured in cycles 1-2; this validates numerical parity.
# MBUF=128,128 (vs service's 256,256) so ctx-8192 KV fits the ~1.75 GiB DSP VA window.
set -u
cd "$(dirname "$0")/.."
PKG=/tmp/opencode/pkg-snapdragon
S=$PKG/bin/llama-server
export LD_LIBRARY_PATH=/tmp/opencode/libfix:$PKG/lib
export ADSP_LIBRARY_PATH=$PKG/lib
export GGML_HEXAGON_OPBATCH=2048 GGML_HEXAGON_MBUF=128,128
F=models/LFM2.5-2.6B-Q4_K_M.gguf
URL=https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/resolve/main/LFM2.5-2.6B-Q4_K_M.gguf

[ -f results/hex2_done ] || { echo "waiting for hex2_done..."; while [ ! -f results/hex2_done ]; do sleep 60; done; }
[ -f "$F" ] || curl -sSL --fail -m 3600 -o "$F" "$URL" || { echo "download failed" > results/hexq_status; exit 1; }

serve() { # $1=extra llama-server args  $2=tag
  pkill -f "llama-server.*LFM2.5" 2>/dev/null; sleep 3
  local CTX=${3:-8192}
  nohup $S -m "$F" -c "$CTX" --port 8181 --temp 0.1 --top-k 50 --repeat-penalty 1.1 $1 \
    > "results/server_hexq_$2.log" 2>&1 &
  for i in $(seq 1 60); do
    sleep 2
    curl -s -m 2 http://127.0.0.1:8181/health 2>/dev/null | grep -q ok && { echo "server up: $2 (ctx $CTX)"; return 0; }
  done
  echo "server failed ($2, ctx $CTX)"; tail -5 "results/server_hexq_$2.log"
  [ "$CTX" = 8192 ] && { echo "retrying ctx 4096"; return 1; }
  return 2
}

run_side() { # $1=args $2=tag
  serve "$1" "$2" || serve "$1" "$2" 4096 || return 1
  python3 scripts/gsm8k_eval.py 8181 20 "LFM2.5-2.6B-Q4_K_M-$2" > "results/gsm8k_hexq_${2}.log" 2>&1
}

ok=1
run_side "--device HTP0 -ngl 100" npu || ok=0
run_side "-ngl 0" cpu || ok=0
pkill -f "llama-server.*LFM2.5" 2>/dev/null
rm -f "$F"
echo "# Final disk state (hex quality)" >> results/DISK_LOG.md
df -h / | tail -1 >> results/DISK_LOG.md
[ $ok = 1 ] && echo done > results/hexq_status || echo "partial - see server logs" > results/hexq_status
echo "HEX QUALITY CHECK DONE" > results/hexq_done
