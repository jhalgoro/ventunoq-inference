#!/usr/bin/env bash
# lfm3vl_um890.sh — convert LiquidAI LFM3-VL-2B to MNN on an x86_64 host (Minisforum UM890 PRO).
#
# Companion of lfm3_um890.sh (same $LFM3_WORK, venv and MNN tree; the VL export
# support lives in the same patches/mnn-lfm3-export.patch).
#
#   scripts/lfm3vl_um890.sh all     # or any of: setup download convtest onnxexport onnxtest export test push
#
# What it does:
#   setup      ensure MNN @ tested rev + patch + venv (see lfm3_um890.sh; idempotent).
#   download   hf-download LiquidAI/LFM3-VL-2B (safetensors + custom modeling/processor code).
#   convtest   scripts/lfm3vl_conv_test.py — gather-conv rewrite vs HF reference + ONNX round-trip.
#   onnxexport llmexport --export onnx (keeps onnx/visual.onnx for the ORT gate).
#   onnxtest   scripts/lfm3vl_onnx_test.py — visual.onnx (ORT) vs HF get_image_features on one image.
#   export     correctness gate (--test with an image prompt, torch side) then llmexport int4 (block 64).
#   test       llm_demo image smoke generation from the exported model.
#   push       copy exported model to the QCS8300 board (LFM3VL_PUSH_TARGET/_PATH).
#
# Recommended order: setup download convtest onnxexport onnxtest export test push
# Requires: see lfm3_um890.sh. Everything is idempotent.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${LFM3_WORK:-$REPO_ROOT/lfm3-work}"
PATCH="$REPO_ROOT/patches/mnn-lfm3-export.patch"
MODEL_ID="${LFM3VL_MODEL:-LiquidAI/LFM3-VL-2B}"
MODEL_DIR="$WORK/models/$(basename "$MODEL_ID")"
OUT_DIR="$WORK/models/$(basename "$MODEL_ID")-MNN"
ONNX_DIR="$WORK/models/$(basename "$MODEL_ID")-onnx"
VENV="$WORK/venv"
PY="$VENV/bin/python"
IMG="$WORK/test-image.jpg"
log() { echo "[lfm3vl] $*"; }

do_setup() {
  # reuse the text-port setup step wholesale (deps, MNN clone+patch, venv, build)
  "$REPO_ROOT/scripts/lfm3_um890.sh" setup
}

do_download() {
  export HF_HUB_DISABLE_PROGRESS_BARS=1
  "$WORK/venv/bin/hf" download "$MODEL_ID" --local-dir "$MODEL_DIR" \
    || "$VENV/bin/huggingface-cli" download "$MODEL_ID" --local-dir "$MODEL_DIR"
  # custom code must be present (trust_remote_code export + image processor)
  for f in modeling_lfm3_vl.py configuration_lfm3_vl.py \
           image_processing_lfm3_vl.py processing_lfm3_vl.py; do
    [ -s "$MODEL_DIR/$f" ] || { log "MISSING $f in $MODEL_DIR — repo needs custom code"; exit 1; }
  done
  log "downloaded $MODEL_ID -> $MODEL_DIR ($(du -sh "$MODEL_DIR" | cut -f1))"
}

make_image() {
  [ -s "$IMG" ] && return
  log "creating test image $IMG"
  "$PY" - "$IMG" <<'EOF'
import sys
import numpy as np
from PIL import Image
rng = np.random.default_rng(42)
# structured pattern + noise so bicubic resize has features to keep
arr = rng.integers(0, 256, size=(368, 544, 3), dtype=np.uint8)
img = Image.fromarray(arr).resize((136, 92)).resize((544, 368), Image.BICUBIC)
img.save(sys.argv[1])
EOF
}

do_convtest() {
  log "gather-conv rewrite vs HF reference (torch + ONNX round-trip)"
  "$PY" "$REPO_ROOT/scripts/lfm3vl_conv_test.py" "$WORK/MNN/transformers/llm/export"
}

do_onnxexport() {
  cd "$WORK/MNN/transformers/llm/export"
  log "export visual ONNX -> $ONNX_DIR/onnx/visual.onnx"
  "$PY" llmexport.py --path "$MODEL_DIR" --export onnx --dst_path "$ONNX_DIR"
}

do_onnxtest() {
  make_image
  [ -f "$ONNX_DIR/onnx/visual.onnx" ] || do_onnxexport
  log "visual.onnx (ORT) vs HF get_image_features"
  "$PY" "$REPO_ROOT/scripts/lfm3vl_onnx_test.py" "$MODEL_DIR" "$ONNX_DIR" "$IMG"
}

do_export() {
  make_image
  cd "$WORK/MNN/transformers/llm/export"
  log "correctness gate: HF model vs exported torch graph (image prompt, greedy)"
  "$PY" llmexport.py --path "$MODEL_DIR" --test "<img>$IMG</img>What is in this image?" | tail -12
  log "export int4 (block 64, visual int8 default) -> $OUT_DIR"
  "$PY" llmexport.py --path "$MODEL_DIR" --export mnn \
    --quant_block 64 --dst_path "$OUT_DIR" \
    --mnnconvert "$WORK/MNN/build/MNNConvert"
  log "exported: $(du -sh "$OUT_DIR" | cut -f1)"
}

do_test() {
  make_image
  local dir="${1:-$OUT_DIR}"
  local cfg="$dir/config.json"
  [ -f "$cfg" ] || { log "no $cfg — run the export step first"; exit 1; }
  log "llm_demo image smoke $(basename "$dir") (greedy, max 512 tokens)"
  sed -i 's#"sampler_type".*#"sampler_type": "greedy",#' "$cfg" 2>/dev/null || true
  # prompt as FILE arg: stdin-only mode enters the REPL which never sees EOF
  printf '<img>%s</img>What is in this image?\n' "$IMG" > "$WORK/prompt-vl.txt"
  timeout 1800 "$WORK/MNN/build/llm_demo" "$cfg" "$WORK/prompt-vl.txt" 512 2>&1 | tail -60
  log "smoke done — description should mention the image content (HF reference in research/lfm3-port.md)"
}

PUSH_TARGET="${LFM3VL_PUSH_TARGET:-arduino@localhost}"
PUSH_PATH="${LFM3VL_PUSH_PATH:-/home/arduino/Code/ventunoq-mnn/lfm3-models}"
do_push() {
  local d name
  ssh "$PUSH_TARGET" "mkdir -p '$PUSH_PATH'"
  local have_rsync=no
  command -v rsync >/dev/null 2>&1 && ssh "$PUSH_TARGET" "command -v rsync" >/dev/null 2>&1 && have_rsync=yes
  d="$OUT_DIR"; name="$(basename "$d")"
  [ -f "$d/config.json" ] || { log "skip push (not exported)"; return; }
  log "pushing $name ($(du -sh "$d" | cut -f1)) -> $PUSH_TARGET:$PUSH_PATH"
  if [ "$have_rsync" = yes ]; then
    rsync -a --delete "$d/" "$PUSH_TARGET:$PUSH_PATH/$name/"
  else
    ssh "$PUSH_TARGET" "rm -rf '$PUSH_PATH/$name'"
    scp -q -r "$d" "$PUSH_TARGET:$PUSH_PATH/"
  fi
  log "push done — on the board: llm_demo $PUSH_PATH/$name/config.json"
}

# accept one or more steps, e.g. `setup download convtest`
run_step() {
  case "$1" in
    setup)      do_setup ;;
    download)   do_download ;;
    convtest)   do_convtest ;;
    onnxexport) do_onnxexport ;;
    onnxtest)   do_onnxtest ;;
    export)     do_export ;;
    test)       do_test ;;
    push)       do_push ;;
    all)        do_setup; do_download; do_convtest; do_onnxexport; do_onnxtest; do_export; do_test ;;
    *) echo "usage: $0 {setup|download|convtest|onnxexport|onnxtest|export|test|push|all} [more steps...]"; exit 2 ;;
  esac
}
for step in "${@:-all}"; do run_step "$step"; done
