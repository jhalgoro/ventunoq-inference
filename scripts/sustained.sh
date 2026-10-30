#!/usr/bin/env bash
# 10-min sustained pp+tg loop (L3) — the passive-cooling throttle curve.
# Usage: scripts/sustained.sh <gguf> <cpu|npu> <tag> [minutes]
#   cpu → v0.8.0 prebuilt, board-pinned;  npu → hexagon runner + service tuning.
# Pair with: scripts/power_probe.py <tag> 5   (start the probe FIRST, stop both together)
# Emits results/sustained_<tag>.csv — one row per minute with the window's tg t/s.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

GGUF=${1:?usage: sustained.sh <gguf> <cpu|npu> <tag> [minutes]}
ENGINE=${2:?cpu|npu}; TAG=${3:?tag}; MIN=${4:-10}
OUT=results/sustained_$TAG.csv

if [ "$ENGINE" = cpu ]; then
  BIN="$LLAMA/llama-cli"; NGL=0; EXTRA="$CPU_PIN"
else
  BIN="$HEX/bin/llama-cli"; NGL=100; EXTRA="-t 4 --cpu-mask 0x0f -b 1024 -ub 512 -fa on"
fi
PROMPT=$(python3 -c "print('The quick brown fox jumps over the lazy dog. '*64)")

echo "minute,ts,tg_tps" > "$OUT"
echo "sustained: $ENGINE / $(basename "$GGUF") / ${MIN} min → $OUT"
start=$(date +%s)
min=0
while :; do
  now=$(date +%s); elapsed=$(( now - start ))
  [ $elapsed -ge $(( MIN * 60 )) ] && break
  # one ~60 s window: pp 512 + tg until the minute rolls over
  line=$(taskset -c 0-3 "$BIN" -m "$GGUF" -ngl "$NGL" $EXTRA -p "$PROMPT" -n 2048 \
           -c 4096 --no-display-prompt -t 4 2>&1 | grep -E "evaluate.*tokens per second|print.*tokens per second" | tail -1) || true
  tg=$(echo "$line" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
  min=$(( min + 1 ))
  echo "$min,$(date -u +%FT%TZ),${tg:-NA}" >> "$OUT"
  echo "  min $min: tg=${tg:-NA} t/s"
done
echo "→ $OUT  (throttle Δ% = last vs first; correlate with results/power_$TAG.csv)"
