#!/usr/bin/env bash
# lfm3_um890.sh — convert LiquidAI LFM3 to MNN on an x86_64 host (Minisforum UM890 PRO).
#
# The QCS8300 board is too small for torch-side export; run this on the UM890:
#   git clone <this repo> && cd ventunoq-mnn
#   scripts/lfm3_um890.sh all          # or: setup download export test
#
# What it does:
#   setup     clone alibaba/MNN at the tested revision, apply patches/mnn-lfm3-export.patch,
#             create a python venv (torch-CPU + transformers>=5.12 for LFM3 custom code),
#             build MNN (llm + low-memory + transformer-fuse + converter).
#   download  hf-download LiquidAI/LFM3-1.7B (safetensors + custom modeling code).
#   export    correctness gate (--test, HF-vs-torch trace) then llmexport int4 (block 64).
#   export8   llmexport int8 (block 64)  -> <out>-int8
#   exporthq  llmexport int4 + HQQ       -> <out>-int4hqq
#   test      llm_demo smoke generation from the exported model (test8/testhq for variants).
#   push      copy exported variants to the QCS8300 board for the quality matrix
#             (LFM3_PUSH_TARGET / LFM3_PUSH_PATH env, rsync if both ends have it else scp).
#
# Requires: Fedora (dnf) or Debian/Ubuntu (apt), ~10 GB free disk, network. Everything is idempotent.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${LFM3_WORK:-$REPO_ROOT/lfm3-work}"
MNN_URL=https://github.com/alibaba/MNN.git
MNN_REV=17236ad20b4798a1faccc761c80f283283bec30e   # revision this repo benchmarks against
PATCH="$REPO_ROOT/patches/mnn-lfm3-export.patch"
MODEL_ID="${LFM3_MODEL:-LiquidAI/LFM3-1.7B}"
MODEL_DIR="$WORK/models/$(basename "$MODEL_ID")"
OUT_DIR="$WORK/models/$(basename "$MODEL_ID")-MNN"
VENV="$WORK/venv"
PY="$VENV/bin/python"
log() { echo "[lfm3] $*"; }

install_deps() {
  # root on the UM890 is common; only use sudo when not root and sudo exists
  local SUDO=""
  [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && SUDO=sudo
  if command -v dnf >/dev/null 2>&1; then
    # Fedora: gcc-c++ replaces build-essential; python3-pip provides venv/ensurepip
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
  install_deps
  mkdir -p "$WORK/models"

  if [ ! -d "$WORK/MNN/.git" ]; then
    log "cloning MNN @ $MNN_REV"
    git clone "$MNN_URL" "$WORK/MNN"
  fi
  git -C "$WORK/MNN" fetch --quiet origin "$MNN_REV" 2>/dev/null || true
  git -C "$WORK/MNN" checkout --quiet "$MNN_REV"
  if git -C "$WORK/MNN" apply --reverse --check "$PATCH" 2>/dev/null; then
    log "lfm3 patch already applied"
  else
    log "applying $(basename "$PATCH")"
    # a stale/older patch revision may sit in the tree — reset export utils first
    git -C "$WORK/MNN" checkout -- transformers/llm/export 2>/dev/null || true
    git -C "$WORK/MNN" apply "$PATCH"
  fi

  if [ ! -x "$PY" ]; then
    log "python venv + deps (torch CPU, transformers>=5.12 for LFM3 custom code)"
    python3 -m venv "$VENV"
    "$PY" -m pip install -q --upgrade pip
    "$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    "$PY" -m pip install -q 'transformers>=5.12' 'huggingface_hub[cli]' \
      sentencepiece onnx onnxslim onnxruntime tqdm numpy yaspin requests Pillow datasets \
      torchvision
  fi

  if [ ! -x "$WORK/MNN/build/llm_demo" ]; then
    log "building MNN (x86, llm+lowmem+transformer_fuse+converter) — a few minutes"
    # Strict ISO C++17: Fedora 44's GCC 16 libstdc++ defines std::abs(__int128) and
    # __is_integer<__int128> twice when the legacy extension-int list (GNU dialect only)
    # is also active -> redefinition errors in every -std=gnu++XX target. -std=c++XX
    # disables the legacy path (bundled protobuf, CXX_EXTENSIONS OFF, built fine).
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
  # custom code must be present (trust_remote_code export)
  for f in modeling_lfm3.py configuration_lfm3.py; do
    [ -s "$MODEL_DIR/$f" ] || { log "MISSING $f in $MODEL_DIR — repo needs custom code"; exit 1; }
  done
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
  # Must pass the prompt as a FILE arg: with stdin-only, llm_demo enters its
  # chat() REPL which never checks getline EOF — it re-prefills the growing
  # conversation forever. File mode runs each prompt once and exits.
  printf 'Name the three smallest prime numbers.\n' > "$WORK/prompt.txt"
  timeout 900 "$WORK/MNN/build/llm_demo" "$cfg" "$WORK/prompt.txt" 512 2>&1 | tail -60
  log "smoke done — compare vs HF reference in research/lfm3-port.md"
}

PUSH_TARGET="${LFM3_PUSH_TARGET:-arduino@localhost}"
PUSH_PATH="${LFM3_PUSH_PATH:-/home/arduino/Code/ventunoq-mnn/lfm3-models}"
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
  log "push done — on the board: scripts/quality_matrix_mnn.sh q_lfm3_all"
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
