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
  TREE=$(curl -s --max-time 30 "https://huggingface.co/api/models/taobao-mnn/$name/tree/main" \
    | python3 -c "import json,sys
for f in json.load(sys.stdin):
    if f['path'].split('.')[-1] in ('json','txt','bin','mnn','weight') \
       and not f['path'].startswith('visual'):
        print(f['path'], f.get('size', 0))" 2>/dev/null) || true
  declare -A RSIZE=()
  FILES=()
  while read -r p s; do [ -n "$p" ] && FILES+=("$p") && RSIZE["$p"]=$s; done <<< "$TREE"
  [ ${#FILES[@]} -ge 6 ] || { FILES=("${FILES_FALLBACK[@]}"); RSIZE=(); }
  for f in "${FILES[@]}"; do
    # F18: never re-touch a byte-complete file. `-C -` on a file that text_only.py had
    # rewritten (smaller than upstream, via cp -al hardlinks) APPENDED upstream tail
    # bytes after the JSON — engine then read default config (embeddings_bf16.bin,
    # Reshape 18400->18336). Size-equal = done; oversize = clobbered, full re-download;
    # undersize = interrupted, resume (F11).
    dst="models/$name/$f" remote=${RSIZE[$f]:-0} have=0
    [ -f "$dst" ] && have=$(wc -c < "$dst")
    if [ "$remote" -gt 0 ] && [ "$have" -eq "$remote" ]; then continue; fi
    [ "$remote" -gt 0 ] && [ "$have" -gt "$remote" ] && rm -f "$dst"
    curl -sL -C - --retry 3 --speed-limit 200000 --speed-time 30 \
      --create-dirs -o "$dst" \
      "https://huggingface.co/taobao-mnn/$name/resolve/main/$f"
    # F16 guard: HF serves 404s as a 200 with a 15-byte "Entry not found" body.
    if [ -s "$dst" ] && [ "$(wc -c < "$dst")" -lt 100 ] \
       && grep -q "not found" "$dst" 2>/dev/null; then
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
