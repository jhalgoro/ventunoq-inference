#!/usr/bin/env bash
# The stack this repo evaluates — the llama.cpp **CPU+NPU** hexagon runner, nothing else.
# Scope rule: v0.8.0 CPU-only prebuilt and MNN (sibling repos) are NOT evaluated here;
# CPU controls always run the SAME runner binary with `-ngl 0`.
#   $HEX2    hexagon runner llamacpp/20261029 bin (ggml 0.28.0, lfm3 ✓)  — current
#   $HEX     hexagon runner llamacpp/20261020 bin (ggml 0.27.0, no lfm3) — old control
#   $LLAMA   parent's v0.8.0 prebuilt path — exported for parent-doc cross-refs only,
#            never on LD_LIBRARY_PATH, never benchmarked from this repo
# Fetch recipes (sha256-pinned): ../ventunoq-llamacpp README "Reproducing…" +
# research/bricks-faster-inference.md.  Source, don't execute:  source scripts/env.sh
# HEX2 fetch (no gh/token needed — plain release asset):
#   curl -sL -o /tmp/opencode/llamacpp-hexagon-20261029.tar.gz \
#     "https://github.com/arduino/app-bricks-py/releases/download/llamacpp/20261029/llamacpp-hexagon-20261029.tar.gz"
#   # sha256 fb587ac4c9b7ae838292f8350459b07beec8291b13f0baaf8ff67c457a7ffb32
#   mkdir -p /tmp/opencode/pkg-snapdragon-20261029
#   tar xzf /tmp/opencode/llamacpp-hexagon-20261029.tar.gz -C /tmp/opencode/pkg-snapdragon-20261029 --strip-components=1

export VQ_PARENT=${VQ_PARENT:-$HOME/Code/ventunoq-llamacpp}
export LLAMA=${LLAMA:-$VQ_PARENT/llama.cpp}        # out of scope — cross-ref only
export HEX=${HEX:-/tmp/opencode/pkg-snapdragon}    # contains bin/ and lib/
export HEX2=${HEX2:-/tmp/opencode/pkg-snapdragon-20261029}  # current runner: lfm3-capable

# NPU runtime fixups (parent F6): unversioned libcdsprpc.so symlink + service tuning env.
if [ -d "$HEX2/lib" ] || [ -d "$HEX/lib" ]; then
  mkdir -p /tmp/opencode/libfix
  ln -sf /usr/lib/aarch64-linux-gnu/libcdsprpc.so.1 /tmp/opencode/libfix/libcdsprpc.so
  ACTIVE_HEX=$HEX; [ -d "$HEX2/lib" ] && ACTIVE_HEX=$HEX2   # newest runner wins
  export LD_LIBRARY_PATH=/tmp/opencode/libfix:$ACTIVE_HEX/lib
  export ADSP_LIBRARY_PATH=$ACTIVE_HEX/lib
  export GGML_HEXAGON_OPBATCH=2048 GGML_HEXAGON_MBUF=256,256   # service_compose.yaml mirror
else
  echo "env.sh: WARN — hexagon runner missing at $HEX / $HEX2 (ephemeral /tmp!):" >&2
  echo "  see HEX2 fetch recipe in this file's header" >&2
fi

# The board default (both engines): 4 threads, big cores only, strict.
export CPU_PIN="-t 4 --cpu-mask 0xf --cpu-strict 1"

alias lbench='$HEX2/bin/llama-bench'
echo "HEX2=$HEX2  HEX=$HEX  (runner2/lfm3: $([ -x "$HEX2/bin/llama-bench" ] && echo ok || echo MISSING), runner1: $([ -x "$HEX/bin/llama-bench" ] && echo ok || echo MISSING))"
