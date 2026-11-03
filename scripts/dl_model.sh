#!/usr/bin/env bash
# Download a taobao-mnn model dir into models/ (delete-after-suite policy!), append the
# sha256 + df DOWNLOAD entry to results/DISK_LOG.md. Resumable — F11 recipe.
# Usage: scripts/dl_model.sh Qwen3.5-0.8B-MNN [more names…]
set -euo pipefail
cd "$(dirname "$0")/.."

FILES_FALLBACK=(config.json llm_config.json llm.mnn llm.mnn.json llm.mnn.weight tokenizer.txt)
# visual.* intentionally omitted — text-only policy (F9); use scripts/text_only.py.

for name in "$@"; do
  # tree API decides the file list: exports differ per model (9B adds embeddings_bf16.bin,
  # F15) and a hardcoded list silently downloads "Entry not found" stubs from missing
  # repos (F16). Fall back to the classic six if the API is unreachable.
  FILES=($(curl -s --max-time 30 "https://huggingface.co/api/models/taobao-mnn/$name/tree/main" \
    | python3 -c "import json,sys
print(' '.join(f['path'] for f in json.load(sys.stdin)
      if f['path'].split('.')[-1] in ('json','txt','bin','mnn','weight')
      and not f['path'].startswith('visual')))" 2>/dev/null))
  [ ${#FILES[@]} -ge 6 ] || FILES=("${FILES_FALLBACK[@]}")
  for f in "${FILES[@]}"; do
    curl -sL -C - --retry 3 --speed-limit 200000 --speed-time 30 \
      --create-dirs -o "models/$name/$f" \
      "https://huggingface.co/taobao-mnn/$name/resolve/main/$f"
    # F16 guard: HF serves 404s as a 200 with a 15-byte "Entry not found" body.
    if [ -s "models/$name/$f" ] && [ "$(wc -c < "models/$name/$f")" -lt 100 ] \
       && grep -q "not found" "models/$name/$f" 2>/dev/null; then
      echo "FATAL taobao-mnn/$name: '$f' missing upstream (HF 'Entry not found' stub) — repo is an empty placeholder (F16)" >&2
      exit 1
    fi
  done
  {
    echo
    echo "### $(date -u '+%F %H:%M UTC') — DOWNLOAD: taobao-mnn/$name ($(du -sh "models/$name" | cut -f1))"
    echo '```'
    sha256sum "models/$name/llm.mnn.weight"
    df -h / | tail -1
    echo '```'
  } >> results/DISK_LOG.md
  echo "OK models/$name — delete after its suite and log sha256+df (policy header in DISK_LOG.md)"
done
