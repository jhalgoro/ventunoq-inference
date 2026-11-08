#!/usr/bin/env bash
# LFM3-VL-2B image smoke on the board (ventunoq): llm_demo + <img> prompt.
#
#   scripts/lfm3vl_board_test.sh                      # seed-42 static-noise test image
#   scripts/lfm3vl_board_test.sh photo.jpg            # any image
#   scripts/lfm3vl_board_test.sh photo.jpg "Why is this funny?" 128
#
# Requires llm_demo built with vision:
#   cmake -S MNN -B MNN/build -DCMAKE_BUILD_TYPE=Release \
#     -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
#     -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON
#   cmake --build MNN/build -j"$(nproc)" --target llm_demo
# Model dir from `scripts/lfm3vl_um890.sh push` on the UM890 (override: LFM3VL_MODEL).
#
# Gotcha: llm_demo EXITS SILENTLY when the prompt file is missing — this script
# always writes it. Interactive REPL also works: run llm_demo with the config
# only, then type  <img>/path/to/img.jpg</img>question  at the > prompt.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLM_DEMO="${LLM_DEMO:-$REPO_ROOT/MNN/build/llm_demo}"
MODEL="${LFM3VL_MODEL:-$REPO_ROOT/lfm3-models/LFM3-VL-2B-MNN}"
WORK="${LFM3VL_TEST_DIR:-/tmp/lfm3vl-test}"
IMG="${1:-}"
QUESTION="${2:-What is in this image?}"
MAX_TOKENS="${3:-512}"

[ -x "$LLM_DEMO" ] || { echo "no $LLM_DEMO — build with -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON" >&2; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "no $MODEL/config.json — push from UM890 (scripts/lfm3vl_um890.sh push) or set LFM3VL_MODEL" >&2; exit 1; }

mkdir -p "$WORK"
if [ -z "$IMG" ]; then
  # deterministic image identical to the UM890 numerical gates (seed 42)
  IMG="$WORK/test-image.jpg"
  [ -s "$IMG" ] || python3 - "$IMG" <<'EOF'
import sys
import numpy as np
from PIL import Image
rng = np.random.default_rng(42)
# structured pattern + noise so bicubic resize has features to keep
arr = rng.integers(0, 256, size=(368, 544, 3), dtype=np.uint8)
img = Image.fromarray(arr).resize((136, 92)).resize((544, 368), Image.BICUBIC)
img.save(sys.argv[1])
EOF
fi
[ -f "$IMG" ] || { echo "no image $IMG" >&2; exit 1; }

PROMPT="$WORK/prompt-vl.txt"
printf '<img>%s</img>%s\n' "$(realpath "$IMG")" "$QUESTION" > "$PROMPT"
echo "[lfm3vl] model=$MODEL image=$IMG"
exec "$LLM_DEMO" "$MODEL/config.json" "$PROMPT" "$MAX_TOKENS"
