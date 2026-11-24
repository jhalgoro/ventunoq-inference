#!/bin/bash
# Retry K2-Horizon-4B: original eval_k2.sh used repo-stem as file-stem, but the
# repo is K2-Horizon-3.7B-GGUF while the file is K2-Horizon-4B-Q4_K_M.gguf (404).
# Waits for the live 7B cycle to finish (k2_done), then runs the corrected 4B.
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=$PWD/llama.cpp
B=./llama.cpp/llama-bench
LOG=results/DISK_LOG.md
disklog() {
  echo "" >> $LOG
  echo "### $(date -u +'%F %R UTC') — $1: \`$2\`" >> $LOG
  [ -f "$2" ] && echo '```' >> $LOG && sha256sum "$2" >> $LOG && echo '```' >> $LOG
  echo '```' >> $LOG; df -h / | tail -1 >> $LOG; echo '```' >> $LOG
}
delmodel() { disklog "DELETE (tests finished)" "$1"; rm -f "$1"; disklog "AFTER delete" "$1"; }
fits() { local need_kb=$(( $1 / 1024 + 2*1024*1024 )); local avail_kb=$(df -k / | awk 'NR==2{print $4}'); [ "$avail_kb" -ge "$need_kb" ]; }

echo "waiting for 7B cycle (k2_done)..."
while [ ! -f results/k2_done ]; do sleep 60; done
echo "7B done, retrying 4B $(date -u +%FT%TZ)"

F=models/K2-Horizon-4B-Q4_K_M.gguf
URL="https://huggingface.co/IFM/K2-Horizon-3.7B-GGUF/resolve/main/K2-Horizon-4B-Q4_K_M.gguf"
SIZE=$(curl -sSLI -m 30 "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | head -1)
echo "4B size: ${SIZE:-?} bytes"
fits "${SIZE:-3200000000}" || { echo "4B no longer fits disk policy"; echo "SKIPPED disk" > results/k2h4B_status; echo "K2-4B RETRY DONE" > results/k2_4b_done; exit 0; }
curl -sSL --fail -m 3600 -o "$F" "$URL" || { echo "download failed 4B" > results/k2h4B_status; echo "K2-4B RETRY DONE" > results/k2_4b_done; exit 0; }
disklog "DOWNLOAD (K2-4B retry)" "$F"
$B -m "$F" -p 512,2048,8192 -n 32,128 -r 2 -o csv > results/bench_k2h_4B_main.csv 2>&1
bash scripts/serve.sh "$F" || { echo "serve failed 4B" > results/k2h4B_status; delmodel "$F"; echo "K2-4B RETRY DONE" > results/k2_4b_done; exit 0; }
python3 scripts/gsm8k_eval.py 8181 50 K2-Horizon-4B-Q4_K_M > results/gsm8k_k2h_4B.log 2>&1
python3 scripts/humaneval_eval.py 8181 20 K2-Horizon-4B-Q4_K_M > results/humaneval_k2h_4B.log 2>&1
pkill -f llama-server; sleep 3
delmodel "$F"
echo "done" > results/k2h4B_status
echo "K2-4B RETRY DONE" > results/k2_4b_done
