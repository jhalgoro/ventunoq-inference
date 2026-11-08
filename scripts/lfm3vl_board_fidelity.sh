#!/usr/bin/env bash
# lfm3vl_board_fidelity.sh — run the LFM3-VL conversion-fidelity review
# (research/lfm3vl-fidelity.md) on the QCS8300 board itself, for when the UM890
# host is unavailable. Same measurements as scripts/lfm3vl_um890.sh {hfppl,export8,
# logitsdiff}, just with board paths and a locally-created venv.
#
#   scripts/lfm3vl_board_fidelity.sh setup       # venv: torch CPU (aarch64) + transformers>=5.12 + llmexport deps
#   scripts/lfm3vl_board_fidelity.sh download    # LiquidAI/LFM3-VL-2B (3.85 GB, public) -> lfm3-work/models/
#   scripts/lfm3vl_board_fidelity.sh hfppl       # DECISIVE: HF gatsby ppl (ppl_eval twin) -> results/fidelity_lfm3vl_hf_ppl.log
#   scripts/lfm3vl_board_fidelity.sh export8     # int8 MNN export via the board's patched MNN tree -> lfm3-work/models/...-int8
#   scripts/lfm3vl_board_fidelity.sh logitsdiff  # teacher-forced int4 (board) vs int8 -> results/fidelity_lfm3vl_logitsdiff_int4_int8.txt
#   scripts/lfm3vl_board_fidelity.sh all         # the above, in order (hfppl is the long one: ~2-4 h on 8 CPU cores)
#
# Interpretation (matrix in research/lfm3vl-fidelity.md):
#   HF ppl ~140-155  -> checkpoint property, MNN conversion faithful
#   HF ppl ~40-60    -> export/quantization bug, localize with the logits diff
#   logits-diff KL vs the LFM3-1.7B int4-vs-int8 reference: mean KL 0.206, 9/56 argmax disagreements
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${LFM3_WORK:-$REPO_ROOT/lfm3-work}"
MODEL_ID="${LFM3VL_MODEL_ID:-LiquidAI/LFM3-VL-2B}"
MODEL_DIR="$WORK/models/$(basename "$MODEL_ID")"
INT4_DIR="${LFM3VL_INT4:-$REPO_ROOT/lfm3-models/LFM3-VL-2B-MNN}"
OUT8_DIR="$WORK/models/$(basename "$MODEL_ID")-MNN-int8"
VENV="$WORK/venv"
PY="$VENV/bin/python"
RES="$REPO_ROOT/results"
log() { echo "[lfm3vl-fid] $*"; }

do_setup() {
  if [ ! -x "$PY" ]; then
    log "python venv + deps (torch CPU aarch64, transformers>=5.12, llmexport deps)"
    mkdir -p "$VENV"
    # this board has no python3-venv (ensurepip missing): fall back to --without-pip + get-pip
    python3 -m venv "$VENV" 2>/dev/null \
      || { python3 -m venv --without-pip "$VENV" \
           && curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
           && "$PY" /tmp/get-pip.py -q; }
    "$PY" -m pip install -q --upgrade pip
    "$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    "$PY" -m pip install -q 'transformers>=5.12' 'huggingface_hub[cli]' \
      sentencepiece onnx onnxslim onnxruntime tqdm numpy yaspin requests Pillow datasets
  fi
  "$PY" -c 'import torch, transformers; print("torch", torch.__version__, "| transformers", transformers.__version__)'
}

do_download() {
  export HF_HUB_DISABLE_PROGRESS_BARS=1
  "$PY" -m pip show huggingface_hub >/dev/null 2>&1 || do_setup
  "$VENV/bin/hf" download "$MODEL_ID" --local-dir "$MODEL_DIR" \
    || "$VENV/bin/huggingface-cli" download "$MODEL_ID" --local-dir "$MODEL_DIR"
  log "downloaded $MODEL_ID -> $MODEL_DIR ($(du -sh "$MODEL_DIR" | cut -f1))"
}

do_hfppl() {
  [ -f "$MODEL_DIR/config.json" ] || { log "no $MODEL_DIR — run download first"; exit 1; }
  # --fp32 is MANDATORY on this board: torch aarch64 CPU half-precision matmul is
  # ~250x slower than fp32 here (measured 2026-11-08: fp32 50.6 GF/s vs bf16 0.3 GF/s).
  local extra="${HFPPL_EXTRA:---fp32}"
  log "HF gatsby ppl (decisive test; MNN int4 = 151.77) [$extra]"
  "$PY" "$REPO_ROOT/scripts/lfm3vl_hf_ppl.py" "$MODEL_DIR" "$REPO_ROOT/bench/gatsby.txt" \
    --mnn-config "$INT4_DIR/llm_config.json" $extra | tee "$RES/fidelity_lfm3vl_hf_ppl.log"
}

do_export8() {
  [ -f "$MODEL_DIR/config.json" ] || { log "no $MODEL_DIR — run download first"; exit 1; }
  cd "$REPO_ROOT/MNN/transformers/llm/export"
  log "int8 MNN export on the board (patched tree) -> $OUT8_DIR"
  "$PY" llmexport.py --path "$MODEL_DIR" --export mnn --quant_bit 8 \
    --quant_block 64 --dst_path "$OUT8_DIR" \
    --mnnconvert "$REPO_ROOT/MNN/build/MNNConvert"
  log "exported: $(du -sh "$OUT8_DIR" | cut -f1)"
}

do_logitsdiff() {
  local bin="$REPO_ROOT/MNN/build/llm_logits_diff"
  [ -f "$INT4_DIR/config.json" ] || { log "no $INT4_DIR — board int4 model missing"; exit 1; }
  [ -f "$OUT8_DIR/config.json" ] || { log "no $OUT8_DIR — run export8 first"; exit 1; }
  local prompt="$WORK/fid-prompt.txt"
  printf 'Explain what a tokenizer does in a language model, in two sentences.\n' > "$prompt"
  log "teacher-forced logits diff: VL int4 (A) vs VL int8 (B), 64 steps"
  taskset -c 0-3 "$bin" "$INT4_DIR/config.json" "$OUT8_DIR/config.json" "$prompt" 64 \
    | tee "$RES/fidelity_lfm3vl_logitsdiff_int4_int8.txt" | tail -4
  log "read vs family reference (LFM3-1.7B int4-vs-int8): mean KL 0.206, 9/56 argmax disagreements"
}

run_step() {
  case "$1" in
    setup)      do_setup ;;
    download)   do_download ;;
    hfppl)      do_hfppl ;;
    export8)    do_export8 ;;
    logitsdiff) do_logitsdiff ;;
    all)        do_setup; do_download; do_hfppl; do_export8; do_logitsdiff ;;
    *) echo "usage: $0 {setup|download|hfppl|export8|logitsdiff|all} [more steps...]"; exit 2 ;;
  esac
}
for step in "${@:-all}"; do run_step "$step"; done
