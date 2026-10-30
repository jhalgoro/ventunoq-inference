#!/usr/bin/env bash
# Download a taobao-mnn model dir into models/ (delete-after-suite policy!), append the
# sha256 + df DOWNLOAD entry to results/DISK_LOG.md. Resumable — F11 recipe.
# Usage: scripts/dl_model.sh Qwen3.5-0.8B-MNN [more names…]
set -euo pipefail
cd "$(dirname "$0")/.."

FILES=(config.json llm_config.json llm.mnn llm.mnn.json llm.mnn.weight tokenizer.txt)
# visual.* intentionally omitted — text-only policy (F9); use scripts/text_only.py.

for name in "$@"; do
  for f in "${FILES[@]}"; do
    curl -sL -C - --retry 3 --speed-limit 200000 --speed-time 30 \
      --create-dirs -o "models/$name/$f" \
      "https://huggingface.co/taobao-mnn/$name/resolve/main/$f"
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
