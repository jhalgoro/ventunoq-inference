#!/bin/bash
# GPU (Vulkan / Turnip Adreno) inference matrix — completes the CPU/NPU/GPU engine trio.
# Stack: llama.cpp v0.8.0 built from source with GGML_VULKAN=ON using a sudo-free
# local prefix (/tmp/opencode/vkdeps: cmake, ninja, glslc, Vulkan headers, SPIRV-Headers).
# Device under test: Vulkan0 = Turnip (Mesa 25.2.8, Vulkan 1.3.318, vendor 0x5143).
#   vulkaninfo: GPU0=turnip integrated, GPU1=llvmpipe(CPU) — ggml-vulkan skips CPU-type
#   devices, so Vulkan0 is the real GPU.
# First run pays Turnip pipeline-compile cost; -r 2 absorbs most of it.
# Waits for results/hexq_done so it never overlaps the CPU-control GSM8K eval.
set -u
cd "$(dirname "$0")/.."
VK=/tmp/opencode/vkdeps
SRC=/tmp/opencode/llama-vulkan
BUILD=$SRC/build-vk
B=$BUILD/bin/llama-bench
LOG=results/DISK_LOG.md
export PATH=$VK/usr/bin:$PATH
export LD_LIBRARY_PATH=$VK/usr/lib/aarch64-linux-gnu
SVC="-t 4 --cpu-mask 0x0f -b 1024 -ub 512 -fa on"
ARGS="-p 512,8192 -n 32,128 -r 2"
OKPAT='"707616feb"'   # data rows start with the build_commit field

say() { echo "[$(date -u +'%F %T UTC')] $*"; }
disklog() {
  echo "" >> $LOG
  echo "### $(date -u +'%F %R UTC') — $1: \`$2\`" >> $LOG
  [ -f "$2" ] && echo '```' >> $LOG && sha256sum "$2" >> $LOG && echo '```' >> $LOG
  echo '```' >> $LOG; df -h / | tail -1 >> $LOG; echo '```' >> $LOG
}
delmodel() { disklog "DELETE (tests finished)" "$1"; rm -f "$1"; disklog "AFTER delete" "$1"; }
fits() { local need_kb=$(( $1 / 1024 + 2*1024*1024 )); local avail_kb=$(df -k / | awk 'NR==2{print $4}'); [ "$avail_kb" -ge "$need_kb" ]; }
fetch() { # $1=url  sets F
  F=models/$(basename "$1")
  curl -sSL --fail -m 3600 -o "$F" "$1" || return 1
  disklog "DOWNLOAD ($TAG)" "$F"
}

# ---------- 0) wait for the CPU-control eval to release the cores ----------
while [ ! -f results/hexq_done ]; do sleep 60; done
say "hexq_done seen; starting GPU build"

# ---------- 1) build llama-server + llama-bench with Vulkan ----------
if [ ! -x "$B" ]; then
  cmake -S $SRC -B $BUILD -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DVulkan_INCLUDE_DIR=$VK/usr/include \
    -DVulkan_LIBRARY=$VK/usr/lib/aarch64-linux-gnu/libvulkan.so \
    -DVulkan_GLSLC_EXECUTABLE=$VK/usr/bin/glslc \
    -DSPIRV-Headers_DIR=$VK/usr/share/cmake/SPIRV-Headers \
    > results/build_vk_configure.log 2>&1
  if [ $? -ne 0 ]; then echo "configure failed: $(tail -5 results/build_vk_configure.log)" > results/vk_failed; exit 1; fi
  cmake --build $BUILD --target llama-server llama-bench -j 8 > results/build_vk_build.log 2>&1
  if [ ! -x "$B" ]; then echo "build failed: $(tail -5 results/build_vk_build.log)" > results/vk_failed; exit 1; fi
fi
say "build OK"

# ---------- 2) one-model smoke test (also warms Turnip shader cache) ----------
TAG=Qwen35-08B-Q4_0
URL=https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_0.gguf
fetch "$URL" || { echo "smoke download failed" > results/vk_failed; exit 1; }
SMOKE=results/bench_vk_smoke.csv
timeout 1800 $B -m "$F" --device Vulkan0 -ngl 100 -p 128 -n 32 -r 1 -o csv > $SMOKE 2>&1
if ! grep -q "$OKPAT" $SMOKE; then
  # one retry with fp16 disabled in case Turnip advertises a feature ggml mis-uses
  GGML_VK_DISABLE_F16=1 timeout 1800 $B -m "$F" --device Vulkan0 -ngl 100 -p 128 -n 32 -r 1 -o csv > $SMOKE 2>&1
  grep -q "$OKPAT" $SMOKE || { echo "smoke bench failed: $(tail -20 $SMOKE)" > results/vk_failed; delmodel "$F"; exit 1; }
  say "smoke passed with GGML_VK_DISABLE_F16=1"
fi
say "smoke OK on Turnip"

# ---------- 3) full matrix (0.8B already on disk) ----------
vk_bench() { # $1=file $2=tag ; env VK_EXTRA for retry flags
  local F=$1 TAG=$2 OUT=results/bench_vk_$2.csv
  timeout 5400 $B -m "$F" --device Vulkan0 -ngl 100 $SVC $ARGS -o csv > $OUT 2>&1
  if ! grep -q "$OKPAT" $OUT; then
    GGML_VK_DISABLE_F16=1 timeout 5400 $B -m "$F" --device Vulkan0 -ngl 100 $SVC $ARGS -o csv > $OUT 2>&1
    grep -q "$OKPAT" $OUT && echo "done (no-f16)" > results/vk_${TAG}_status && return
  fi
  grep -q "$OKPAT" $OUT && echo done > results/vk_${TAG}_status || echo "bench failed" > results/vk_${TAG}_status
}

vk_bench "$F" Qwen35-08B-Q4_0; delmodel "$F"

for URL in \
  https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_0.gguf \
  https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/resolve/main/LFM2.5-2.6B-Q4_K_M.gguf \
  https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/main/LFM3-1.7B-Q4_K_M.gguf ; do
  TAG=$(basename "$URL" .gguf)
  fits 3000000000 || { echo "disk low, skipped" > results/vk_${TAG}_status; continue; }
  fetch "$URL" || { echo "download failed" > results/vk_${TAG}_status; continue; }
  vk_bench "$F" "$TAG"
  delmodel "$F"
done

say "matrix complete"
echo done > results/vk_done
