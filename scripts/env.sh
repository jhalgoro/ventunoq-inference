#!/usr/bin/env bash
# Put both llama.cpp stacks on the path — this repo deliberately vendors nothing.
#   $LLAMA  v0.8.0 b11972 Ubuntu-arm64 prebuilt (CPU)          — parent's vendored copy
#   $HEX    hexagon runner llamacpp/20261020 bin (NPU)         — parent's tarball, extracted
# Fetch recipes (sha256-pinned): ../ventunoq-llamacpp README "Reproducing…" +
# research/bricks-faster-inference.md.  Source, don't execute:  source scripts/env.sh

export VQ_PARENT=${VQ_PARENT:-$HOME/Code/ventunoq-llamacpp}
export LLAMA=${LLAMA:-$VQ_PARENT/llama.cpp}
export HEX=${HEX:-/tmp/opencode/pkg-snapdragon}        # contains bin/ and lib/

# NPU runtime fixups (parent F6): unversioned libcdsprpc.so symlink + service tuning env.
if [ -d "$HEX/lib" ]; then
  mkdir -p /tmp/opencode/libfix
  ln -sf /usr/lib/aarch64-linux-gnu/libcdsprpc.so.1 /tmp/opencode/libfix/libcdsprpc.so
  export LD_LIBRARY_PATH=/tmp/opencode/libfix:$HEX/lib:$LLAMA
  export ADSP_LIBRARY_PATH=$HEX/lib
  export GGML_HEXAGON_OPBATCH=2048 GGML_HEXAGON_MBUF=256,256   # service_compose.yaml mirror
else
  echo "env.sh: WARN — hexagon runner missing at $HEX (ephemeral /tmp!):" >&2
  echo "  gh release download 'llamacpp/20261020' -R arduino/app-bricks-py -p 'llamacpp-hexagon-*.tar.gz' -R \$(pwd)" >&2
  echo "  tar xzf llamacpp-hexagon-20261020.tar.gz -C /tmp/opencode/pkg-snapdragon" >&2
  export LD_LIBRARY_PATH=$LLAMA
fi

# The board default (both engines): 4 threads, big cores only.
export CPU_PIN="-t 4 --cpu-mask 0xf --cpu-strict 1"

alias lbench='$LLAMA/llama-bench'
echo "LLAMA=$LLAMA  HEX=$HEX  (llama-bench: $([ -x "$LLAMA/llama-bench" ] && echo ok || echo MISSING), runner: $([ -x "$HEX/bin/llama-bench" ] && echo ok || echo MISSING))"
