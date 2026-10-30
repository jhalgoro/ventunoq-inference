#!/usr/bin/env bash
# Build MNN's LLM tools on-board (~4 min on QCS8300). Idempotent.
# Flags proven 2026-10-30: ARM82 fp16 ON, KleidiAI left OFF (i8mm absent — kernel-audit),
# LOW_MEMORY for the int4 path, static libs so the binaries have no .so to chase.
set -euo pipefail
cd "$(dirname "$0")/.."

MNN_SRC=${MNN_SRC:-$PWD/MNN}
[ -d "$MNN_SRC" ] || git clone --quiet --depth 1 https://github.com/alibaba/MNN.git "$MNN_SRC"
cd "$MNN_SRC"

if [ ! -x build/llm_bench ]; then
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DMNN_BUILD_LLM=ON -DMNN_ARM82=ON -DMNN_LOW_MEMORY=ON \
    -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_AVX512=OFF > /tmp/mnn_cmake.log
  ninja -C build llm_demo llm_bench > /tmp/mnn_build.log
fi
echo "OK: $MNN_SRC/build/{llm_bench,llm_demo}"
# sanity — must print "i8sdot:1, fp16:1, i8mm:0" on this board (kernel-audit §4):
taskset -c 0-3 build/llm_bench 2>&1 | grep -m1 "device supports" || true
