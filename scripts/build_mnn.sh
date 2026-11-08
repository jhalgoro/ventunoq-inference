#!/usr/bin/env bash
# Build MNN's LLM tools on-board (~4 min first time on QCS8300). Idempotent.
# Flags proven 2026-10-30: ARM82 fp16 ON, KleidiAI left OFF (i8mm absent — kernel-audit),
# LOW_MEMORY for the int4 path, static libs so the binaries have no .so to chase.
# Extras we inject into the (gitignored) clone:
#   llm_eval   — scripts/llm_eval.cpp batch driver, the quality harness' engine (F14)
#   ppl_eval   — upstream tool, M1's gatsby-perplexity leg (ROADMAP said "no ppl tool":
#                it ships, just isn't built by default)
set -euo pipefail
cd "$(dirname "$0")/.."

MNN_SRC=${MNN_SRC:-$PWD/MNN}
[ -d "$MNN_SRC" ] || git clone --quiet --depth 1 https://github.com/alibaba/MNN.git "$MNN_SRC"
cd "$MNN_SRC"

TOOLS=transformers/llm/engine/tools
cp -f ../scripts/llm_eval.cpp "$TOOLS/llm_eval.cpp"
grep -q "llm_eval" "$TOOLS/CMakeLists.txt" || cat >> "$TOOLS/CMakeLists.txt" <<'EOF'

# ventunoq-mnn batch eval driver (copied in by scripts/build_mnn.sh)
add_executable(llm_eval ${CMAKE_CURRENT_LIST_DIR}/llm_eval.cpp)
target_link_libraries(llm_eval ${LLM_DEPS})
target_include_directories(llm_eval PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
EOF

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_LLM=ON -DMNN_ARM82=ON -DMNN_LOW_MEMORY=ON \
  -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_AVX512=OFF \
  -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON > /tmp/mnn_cmake.log
# OPENCV+IMGCODECS: LLM_SUPPORT_VISION (patch-layout vision processors like qwen2_vl,
# which LFM3-VL reuses) is only defined with MNN_BUILD_OPENCV; imread needs MNN_IMGCODECS.
# Without them any is_visual model fails init: "Unsupported vision processor family".
ninja -C build llm_demo llm_bench llm_eval ppl_eval apply_template > /tmp/mnn_build.log
# apply_template: engine minja gate — apply_template <test.json> 1 renders chat templates
# (pre-check an HF chat_template.jinja is parseable before paying for a full export).
echo "OK: $MNN_SRC/build/{llm_bench,llm_demo,llm_eval,ppl_eval}"
# sanity — must print "i8sdot:1, fp16:1, i8mm:0" on this board (kernel-audit §4):
taskset -c 0-3 build/llm_bench 2>&1 | grep -m1 "device supports" || true
