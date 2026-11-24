#!/bin/bash
# Hexagon NPU (HTP) bench via Arduino's prebuilt llama.cpp (release llamacpp/20261020).
# Context: research/bricks-faster-inference.md — the upstream Hexagon backend uses
# dspqueue over classic fastrpc (works on this board; NOT the QNN/smmu_proxy path of npu/NOTES.md).
# HTP0 enumerated as user 2026-10-29. Waits for results/k2_4b_done so the CPU is idle.
# NOTE: this build (ggml 0.27.0) has qwen35/k2_horizon/lfm2 but NOT lfm3 — LFM3 is untestable here.
set -u
cd "$(dirname "$0")/.."
PKG=/tmp/opencode/pkg-snapdragon
B=$PKG/bin/llama-bench
LOG=results/DISK_LOG.md
mkdir -p /tmp/opencode/libfix
ln -sf /usr/lib/aarch64-linux-gnu/libcdsprpc.so.1 /tmp/opencode/libfix/libcdsprpc.so
export LD_LIBRARY_PATH=/tmp/opencode/libfix:$PKG/lib
export ADSP_LIBRARY_PATH=$PKG/lib
# mirror src/arduino/app_services/llamacpp/service_compose.yaml (ventunoq)
export GGML_HEXAGON_OPBATCH=2048 GGML_HEXAGON_MBUF=256,256
SVC="-t 4 --cpu-mask 0x0f -b 1024 -ub 512 -fa on"
ARGS="-p 512,8192 -n 32,128 -r 2"

disklog() {
  echo "" >> $LOG
  echo "### $(date -u +'%F %R UTC') — $1: \`$2\`" >> $LOG
  [ -f "$2" ] && echo '```' >> $LOG && sha256sum "$2" >> $LOG && echo '```' >> $LOG
  echo '```' >> $LOG; df -h / | tail -1 >> $LOG; echo '```' >> $LOG
}
delmodel() { disklog "DELETE (tests finished)" "$1"; rm -f "$1"; disklog "AFTER delete" "$1"; }
fits() { local need_kb=$(( $1 / 1024 + 2*1024*1024 )); local avail_kb=$(df -k / | awk 'NR==2{print $4}'); [ "$avail_kb" -ge "$need_kb" ]; }

echo "waiting for k2_4b_done (CPU must be idle for fair numbers)..."
while [ ! -f results/k2_4b_done ]; do sleep 60; done
echo "hexagon cycle start $(date -u +%FT%TZ)"

hex_cycle() { # $1=url  $2=tag  [$3=ndev default 1]
  local URL=$1 TAG=$2 NDEV=${3:-1}
  local F=models/$(basename "$URL") ST=results/hex_${TAG}_status
  local DEV=""; for ((i=0;i<NDEV;i++)); do DEV="${DEV:+$DEV,}HTP$i"; done
  local SIZE=$(curl -sSLI -m 30 "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | head -1)
  fits "${SIZE:-2000000000}" || { echo "SKIPPED disk" > "$ST"; return; }
  curl -sSL --fail -m 3600 -o "$F" "$URL" || { echo "download failed" > "$ST"; return; }
  disklog "DOWNLOAD (hex $TAG)" "$F"
  export GGML_HEXAGON_DEVICES=$NDEV
  # NPU run: all layers to HTP, exactly like run-model-router.sh (-ngl 100)
  timeout 3600 $B -m "$F" --device "$DEV" -ngl 100 $SVC $ARGS -o csv > "results/bench_hex_${TAG}_npu.csv" 2>&1
  # CPU control in the SAME binary/build: isolates the NPU effect from the build effect
  timeout 3600 $B -m "$F" --device CPU -ngl 0 $SVC $ARGS -o csv > "results/bench_hex_${TAG}_cpu.csv" 2>&1
  delmodel "$F"
  echo "done" > "$ST"
}

hex_cycle https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_0.gguf Qwen35-08B-Q4_0
hex_cycle https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_0.gguf       Qwen35-2B-Q4_0
hex_cycle https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/resolve/main/LFM2.5-2.6B-Q4_K_M.gguf  LFM25-26B-Q4KM
hex_cycle https://huggingface.co/IFM/K2-Horizon-3.7B-GGUF/resolve/main/K2-Horizon-4B-Q4_K_M.gguf K2H-4B-Q4KM 2

echo "# Final disk state (hexagon cycle)" >> $LOG
df -h / | tail -1 >> $LOG
echo "HEXAGON CYCLE DONE" > results/hex_done
