#!/bin/bash
# Hexagon follow-up cycle (see research/bricks-faster-inference.md + npu/cumap_probe.c):
#  1) CPU controls that FAILED in cycle 1 ("invalid device: CPU" — this build does not
#     enumerate CPU as a device; correct control = no --device flag + -ngl 0).
#  2) Partial-offload ladders for models whose NPU-side weights exceed the measured
#     ~1.75 GiB 32-bit DSP VA ceiling (cumap_probe: 7x256MiB OK, then ENOMEM;
#     both HTP0/HTP1 share domain cdsp:3, so ndev=2 adds no VA).
#     Ladder tries -ngl 100 (fast-fail) then descending layer counts; first success wins.
#     Partials use GGML_HEXAGON_MBUF=128,128 (service uses 256,256) to save 256 MiB of
#     DSP VA for weights — deviation noted in the report.
set -u
cd "$(dirname "$0")/.."
PKG=/tmp/opencode/pkg-snapdragon
B=$PKG/bin/llama-bench
LOG=results/DISK_LOG.md
export LD_LIBRARY_PATH=/tmp/opencode/libfix:$PKG/lib
export ADSP_LIBRARY_PATH=$PKG/lib
export GGML_HEXAGON_OPBATCH=2048
SVC="-t 4 --cpu-mask 0x0f -b 1024 -ub 512 -fa on"
ARGS="-p 512,8192 -n 32,128 -r 2"
OKPAT='"707616feb"'   # data rows start with the build_commit field

disklog() {
  echo "" >> $LOG
  echo "### $(date -u +'%F %R UTC') — $1: \`$2\`" >> $LOG
  [ -f "$2" ] && echo '```' >> $LOG && sha256sum "$2" >> $LOG && echo '```' >> $LOG
  echo '```' >> $LOG; df -h / | tail -1 >> $LOG; echo '```' >> $LOG
}
delmodel() { disklog "DELETE (tests finished)" "$1"; rm -f "$1"; disklog "AFTER delete" "$1"; }
fits() { local need_kb=$(( $1 / 1024 + 2*1024*1024 )); local avail_kb=$(df -k / | awk 'NR==2{print $4}'); [ "$avail_kb" -ge "$need_kb" ]; }
fetch() { # url -> $F
  curl -sSL --fail -m 3600 -o "$F" "$1" || { echo "download failed" > "$ST"; return 1; }
  disklog "DOWNLOAD ($TAG)" "$F"
}

cpu_ctrl() { # $1=url $2=tag — same binary, CPU-only control
  local URL=$1 TAG=$2
  local F=models/$(basename "$URL") ST=results/hex2_${TAG}_status
  unset GGML_HEXAGON_DEVICES
  fetch "$URL" || return
  timeout 5400 $B -m "$F" -ngl 0 $SVC $ARGS -o csv > "results/bench_hex2_${TAG}_cpu.csv" 2>&1
  grep -q "$OKPAT" "results/bench_hex2_${TAG}_cpu.csv" && echo done > "$ST" || echo "bench failed" > "$ST"
  delmodel "$F"
}

npu_ladder() { # $1=url $2=tag $3=ngl ladder
  local URL=$1 TAG=$2 LADDER=$3
  local F=models/$(basename "$URL") ST=results/hex2_${TAG}_status
  export GGML_HEXAGON_DEVICES=1 GGML_HEXAGON_MBUF=128,128
  fetch "$URL" || return
  local ok=""
  for NGL in $LADDER; do
    timeout 3600 $B -m "$F" --device HTP0 -ngl "$NGL" $SVC $ARGS -o csv \
      > "results/bench_hex2_${TAG}_npu_ngl${NGL}.csv" 2>&1
    if grep -q "$OKPAT" "results/bench_hex2_${TAG}_npu_ngl${NGL}.csv"; then ok=$NGL; break; fi
    rm -f "results/bench_hex2_${TAG}_npu_ngl${NGL}.csv"   # keep only the winner
  done
  [ -n "$ok" ] && echo "done ngl=$ok" > "$ST" || echo "all ngls failed" > "$ST"
  delmodel "$F"
}

# 1) same-binary CPU controls for the three models that have NPU numbers
cpu_ctrl https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_0.gguf Qwen35-08B-Q4_0
cpu_ctrl https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_0.gguf       Qwen35-2B-Q4_0
cpu_ctrl https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/resolve/main/LFM2.5-2.6B-Q4_K_M.gguf  LFM25-26B-Q4KM
# 2) bigger models: NPU partial-offload ladder (100 expected to fail >1.75 GiB DSP VA)
npu_ladder https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_0.gguf Qwen35-4B-Q4_0 "100 24 16 8"
npu_ladder https://huggingface.co/IFM/K2-Horizon-3.7B-GGUF/resolve/main/K2-Horizon-4B-Q4_K_M.gguf K2H-4B-Q4KM "100 18 12 6"
# 3) CPU control for K2-4B in the same binary (Qwen-4B control skipped: Q4_0-NPU may succeed full)
cpu_ctrl https://huggingface.co/IFM/K2-Horizon-3.7B-GGUF/resolve/main/K2-Horizon-4B-Q4_K_M.gguf K2H-4B-Q4KM

echo "# Final disk state (hexagon cycle 2)" >> $LOG
df -h / | tail -1 >> $LOG
echo "HEXAGON CYCLE 2 DONE" > results/hex2_done
