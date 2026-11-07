#!/usr/bin/env bash
# minicpm5_um890.sh — convert openbmb/MiniCPM5-2B to MNN on an x86_64 host (Minisforum UM890 PRO).
#
# MiniCPM5-2B is plain LlamaForCausalLM (model_type=llama, GQA 16:2, rope_theta 5e6,
# head_dim 128) — no custom code, no MNN patch needed. This mirrors scripts/lfm3_um890.sh
# but without the patch step; run on the UM890:
#   git pull                                    # remote `board` = this repo on the QCS8300
#   scripts/minicpm5_um890.sh all               # or: setup download export test push
#
# What it does:
#   setup     reuse the lfm3-work venv + MNN build if present (same tested revision);
#             otherwise same install/clone/venv/build flow as lfm3_um890.sh, minus the patch.
#   download  hf-download openbmb/MiniCPM5-2B (~5 GB, single safetensors, no custom code).
#   export    correctness gate (--test, HF-vs-torch trace) then llmexport int4 (block 64).
#   export8   llmexport int8 (block 64)  -> <out>-int8
#   exporthq  llmexport int4 + HQQ       -> <out>-int4hqq
#   test      llm_demo smoke from the exported model (test8/testhq for variants).
#   push      copy exported variants to the board (MINICPM5_PUSH_TARGET / MINICPM5_PUSH_PATH).
#
# Requires: Fedora (dnf) or Debian/Ubuntu (apt), ~12 GB free disk, network. Everything is idempotent.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Same work dir as the LFM3 port: the MNN build + python venv are model-agnostic. A fresh
# UM890 gets a fresh setup there; an existing lfm3-work is reused as-is (its lfm3 export
# patch, if applied, is export-side only and does not affect llama exports).
WORK="${MINICPM5_WORK:-${LFM3_WORK:-$REPO_ROOT/lfm3-work}}"
MNN_URL=https://github.com/alibaba/MNN.git
MNN_REV=17236ad20b4798a1faccc761c80f283283bec30e   # revision this repo benchmarks against
MODEL_ID="${MINICPM5_MODEL:-openbmb/MiniCPM5-2B}"
MODEL_DIR="$WORK/models/$(basename "$MODEL_ID")"
OUT_DIR="$WORK/models/$(basename "$MODEL_ID")-MNN"
VENV="$WORK/venv"
PY="$VENV/bin/python"
log() { echo "[minicpm5] $*"; }

install_deps() {
  local SUDO=""
  [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && SUDO=sudo
  if command -v dnf >/dev/null 2>&1; then
    log "dnf: installing toolchain + python3-pip"
    $SUDO dnf install -y gcc gcc-c++ make cmake git python3-pip python3-devel curl
  elif command -v apt-get >/dev/null 2>&1; then
    log "apt: installing toolchain + python3-venv"
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq build-essential cmake git python3-venv python3-pip curl
  else
    log "no dnf/apt found — assuming cmake, compiler, git, python3 are already present"
  fi
}

do_setup() {
  if [ -x "$WORK/MNN/build/llm_demo" ] && [ -x "$PY" ]; then
    log "reusing $WORK (MNN build + venv from lfm3 setup)"
    git -C "$WORK/MNN" fetch --quiet origin "$MNN_REV" 2>/dev/null || true
    git -C "$WORK/MNN" checkout --quiet "$MNN_REV"
    return
  fi
  install_deps
  mkdir -p "$WORK/models"

  if [ ! -d "$WORK/MNN/.git" ]; then
    log "cloning MNN @ $MNN_REV"
    git clone "$MNN_URL" "$WORK/MNN"
  fi
  git -C "$WORK/MNN" fetch --quiet origin "$MNN_REV" 2>/dev/null || true
  git -C "$WORK/MNN" checkout --quiet "$MNN_REV"
  # NOTE: no patch step — MiniCPM5-2B is stock llama arch, supported upstream.

  if [ ! -x "$PY" ]; then
    log "python venv + deps (torch CPU; transformers only needs to load stock llama)"
    python3 -m venv "$VENV"
    "$PY" -m pip install -q --upgrade pip
    "$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    "$PY" -m pip install -q 'transformers>=4.51' 'huggingface_hub[cli]' \
      sentencepiece onnx onnxslim onnxruntime tqdm numpy yaspin requests Pillow datasets
  fi

  if [ ! -x "$WORK/MNN/build/llm_demo" ]; then
    log "building MNN (x86, llm+lowmem+transformer_fuse+converter) — a few minutes"
    # Strict ISO C++17 (GCC 16 __int128 redefinition, see lfm3_um890.sh note).
    cmake -S "$WORK/MNN" -B "$WORK/MNN/build" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_EXTENSIONS=OFF \
      -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
      -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_TOOLS=ON
    cmake --build "$WORK/MNN/build" -j"$(nproc)"
  fi
  log "setup done"
}

do_download() {
  export HF_HUB_DISABLE_PROGRESS_BARS=1
  "$WORK/venv/bin/hf" download "$MODEL_ID" --local-dir "$MODEL_DIR" \
    || "$VENV/bin/huggingface-cli" download "$MODEL_ID" --local-dir "$MODEL_DIR"
  [ -s "$MODEL_DIR/config.json" ] || { log "download incomplete"; exit 1; }
  log "downloaded $MODEL_ID -> $MODEL_DIR ($(du -sh "$MODEL_DIR" | cut -f1))"
}

do_export() {
  cd "$WORK/MNN/transformers/llm/export"
  log "correctness gate: HF model vs exported torch graph (single greedy prompt)"
  "$PY" llmexport.py --path "$MODEL_DIR" --test "Q: What is the capital of France? A:" | tail -8
  do_export_variant "" "$OUT_DIR"                # int4 b64 baseline
}

do_export_variant() { # $1=suffix-label $2=dst_dir $3...=extra llmexport quant flags
  local label="$1" dst="$2"; shift 2
  cd "$WORK/MNN/transformers/llm/export"
  log "export ${label:-int4} (block 64) $* -> $dst"
  "$PY" llmexport.py --path "$MODEL_DIR" --export mnn \
    --quant_block 64 "$@" --dst_path "$dst" \
    --mnnconvert "$WORK/MNN/build/MNNConvert"
  log "exported: $(du -sh "$dst" | cut -f1)"
}

do_test() { # $1 = exported dir (default int4 baseline)
  local dir="${1:-$OUT_DIR}"
  local cfg="$dir/config.json"
  [ -f "$cfg" ] || { log "no $cfg — run the matching export step first"; exit 1; }
  log "llm_demo smoke $(basename "$dir") (greedy, max 512 tokens)"
  sed -i 's#"sampler_type".*#"sampler_type": "greedy",#' "$cfg" 2>/dev/null || true
  # Prompt as FILE arg (stdin REPL never exits, see lfm3_um890.sh note). MiniCPM5 is a
  # thinking-style model: expect a reasoning preamble before the answer — that is fine.
  printf 'Name the three smallest prime numbers.\n' > "$WORK/prompt.txt"
  timeout 900 "$WORK/MNN/build/llm_demo" "$cfg" "$WORK/prompt.txt" 512 2>&1 | tail -60
  log "smoke done — check the answer stops on <|im_end|> (stop list: eos 1 + 130073)"
}

PUSH_TARGET="${MINICPM5_PUSH_TARGET:-arduino@localhost}"
PUSH_PATH="${MINICPM5_PUSH_PATH:-/home/arduino/Code/ventunoq-mnn/minicpm5-models}"
do_push() {
  local d name
  ssh "$PUSH_TARGET" "mkdir -p '$PUSH_PATH'"
  local have_rsync=no
  command -v rsync >/dev/null 2>&1 && ssh "$PUSH_TARGET" "command -v rsync" >/dev/null 2>&1 && have_rsync=yes
  for d in "$OUT_DIR" "$OUT_DIR-int8" "$OUT_DIR-int4hqq"; do
    [ -f "$d/config.json" ] || { log "skip $(basename "$d") (not exported)"; continue; }
    name="$(basename "$d")"
    log "pushing $name ($(du -sh "$d" | cut -f1)) -> $PUSH_TARGET:$PUSH_PATH"
    if [ "$have_rsync" = yes ]; then
      rsync -a --delete "$d/" "$PUSH_TARGET:$PUSH_PATH/$name/"
    else
      ssh "$PUSH_TARGET" "rm -rf '$PUSH_PATH/$name'"
      scp -q -r "$d" "$PUSH_TARGET:$PUSH_PATH/"
    fi
  done
  log "push done — on the board: scripts/quality_matrix_mnn.sh q_minicpm5_all"
}

# accept one or more steps, e.g. `setup export test`
run_step() {
  case "$1" in
    setup)    do_setup ;;
    download) do_download ;;
    export)   do_export ;;
    export8)  do_export_variant int8 "$OUT_DIR-int8" --quant_bit 8 ;;
    exporthq) do_export_variant int4hqq "$OUT_DIR-int4hqq" --hqq ;;
    test)     do_test ;;
    test8)    do_test "$OUT_DIR-int8" ;;
    testhq)   do_test "$OUT_DIR-int4hqq" ;;
    push)     do_push ;;
    all)      do_setup; do_download; do_export; do_test ;;
    *) echo "usage: $0 {setup|download|export|export8|exporthq|test|test8|testhq|push|all} [more steps...]"; exit 2 ;;
  esac
}
for step in "${@:-all}"; do run_step "$step"; done
