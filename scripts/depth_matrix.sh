#!/usr/bin/env bash
# L1/L2/L5 depth matrix — one driver, sequential stages, delete-after-row policy.
# Regime (comparable to parent §2): -p 512,8192 -n 32,128 -r 2
#   NPU  = $HEX2 (llamacpp/20261029, lfm3-capable) --device HTP0, service tuning
#          (OPBATCH 2048, MBUF 256,256 full / 128,128 partial — parent convention)
#   CPU  = same $HEX2 runner, -ngl 0 (scope rule: only the llama.cpp CPU+NPU arch is
#          evaluated — the v0.8.0 CPU-only stack and MNN are out of scope)
# Resume-safe: a stage's rows are skipped if the CSV already carries the build fingerprint.
# Usage: scripts/depth_matrix.sh [stage ...]   (default: all, in file order)
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh

LOG=results/DISK_LOG.md
DL=/tmp/opencode/dl
ARGS="-p 512,8192 -n 32,128 -r 2"
SVC_NPU="-t 4 --cpu-mask 0x0f -b 1024 -ub 512 -fa on"
OKPAT='"7df8b876b"'        # HEX2 build_commit (b11996)

hex_env() { # $1 = runner dir — per-runner lib path (never mix 0.27 and 0.28 libs)
  export LD_LIBRARY_PATH=/tmp/opencode/libfix:$1/lib
  export ADSP_LIBRARY_PATH=$1/lib
  export GGML_HEXAGON_OPBATCH=2048
}

disklog() {
  echo "" >> $LOG
  echo "### $(date -u +'%F %R UTC') — $1: \`$2\`" >> $LOG
  [ -f "$2" ] && echo '```' >> $LOG && sha256sum "$2" >> $LOG && echo '```' >> $LOG
  echo '```' >> $LOG; df -h / | tail -1 >> $LOG; echo '```' >> $LOG
}
delmodel() { disklog "DELETE (tests finished)" "$1"; rm -f "$1"; }
fetch() { # url -> $F, resume + stall guard (parent F6 recipe); skip if already complete
  local want
  want=$(curl -sIL --fail -L "$1" | tr -d '\r' | awk 'tolower($0) ~ /^x-linked-size:/{split($0,a,": ");print a[2]}')
  if [ -f "$F" ] && [ -n "$want" ] && [ "$(stat -c%s "$F")" = "$want" ]; then
    echo "$(date -u +%T) fetch $TAG already complete" >> results/matrix_status.log
    disklog "REUSE ($TAG)" "$F"; return 0
  fi
  curl -sSL --fail -C - -m 3600 --retry 3 --speed-limit 200000 --speed-time 30 -o "$F" "$1" \
    && disklog "DOWNLOAD ($TAG)" "$F" && return 0
  echo "FAIL download $TAG" >> results/matrix_status.log; return 1
}

bench_npu() { # tag ngl [extra flags] — full uses MBUF 256,256; partial 128,128 (parent rule)
  local R=${HEX2} out="results/bench_${1}_npu_ngl${2}.csv"
  [ -s "$out" ] && grep -q "$OKPAT" "$out" && { echo "$(date -u +%T) npu $1 ngl=$2 SKIP (have rows)" >> results/matrix_status.log; return 0; }
  hex_env "$R"
  if [ "$2" = 100 ]; then export GGML_HEXAGON_MBUF=256,256; else export GGML_HEXAGON_MBUF=128,128; fi
  timeout 3600 $R/bin/llama-bench -m "$F" --device HTP0 -ngl "$2" $SVC_NPU $ARGS ${3:-} -o csv \
    > "$out" 2> "${out%.csv}.err"
  local ok=$(grep -c "$OKPAT" "$out")
  echo "$(date -u +%T) npu $1 ngl=$2 rows=$ok" >> results/matrix_status.log
  [ "$ok" -gt 0 ]
}
npu_ctrl() { # CPU control = SAME runner/binary, -ngl 0, repo pinning convention (scope: CPU+NPU arch only)
  local out="results/bench_${1}_hex2cpu.csv"
  [ -s "$out" ] && grep -q "$OKPAT" "$out" && { echo "$(date -u +%T) hex2cpu $1 SKIP (have rows)" >> results/matrix_status.log; return 0; }
  hex_env "$HEX2"
  unset GGML_HEXAGON_MBUF GGML_HEXAGON_OPBATCH
  timeout 7200 $HEX2/bin/llama-bench -m "$F" -ngl 0 -t 4 --cpu-mask 0x0f --cpu-strict 1 $ARGS ${2:-} -o csv \
    > "$out" 2> "${out%.csv}.err"
  local ok=$(grep -c "$OKPAT" "$out")
  echo "$(date -u +%T) hex2cpu $1 rows=$ok" >> results/matrix_status.log
}
HF=https://huggingface.co
LFM3=$HF/LiquidAI/LFM3-1.7B-GGUF/resolve/main
UNS=$HF/unsloth
K2=$HF/IFM

# ---------------- stages ----------------
s_lfm3_q4km() { # L1 headline + L4/L6 hosts (model kept until L6/L4 rows done)
  TAG=LFM3-17B-Q4KM; F=$DL/LFM3-1.7B-Q4_K_M.gguf
  [ -f "$F" ] || fetch "$LFM3/LFM3-1.7B-Q4_K_M.gguf" || return
  bench_npu "$TAG" 100 && npu_ctrl "$TAG"
  npu_ctrl "$TAG"
}
s_lfm3_qad() {
  TAG=LFM3-17B-QAD; F=$DL/LFM3-1.7B-QAD-Q4_0.gguf
  fetch "$LFM3/LFM3-1.7B-QAD-Q4_0.gguf" || return
  bench_npu "$TAG" 100 && npu_ctrl "$TAG"
  npu_ctrl "$TAG"
  delmodel "$F"
}
s_lfm3_q6k() { # L2 headline: Q6_K keeps the NPU?
  TAG=LFM3-17B-Q6K; F=$DL/LFM3-1.7B-Q6_K.gguf
  fetch "$LFM3/LFM3-1.7B-Q6_K.gguf" || return
  bench_npu "$TAG" 100 || true
  npu_ctrl "$TAG"
  delmodel "$F"
}
s_lfm3_q80() {
  TAG=LFM3-17B-Q80; F=$DL/LFM3-1.7B-Q8_0.gguf
  fetch "$LFM3/LFM3-1.7B-Q8_0.gguf" || return
  bench_npu "$TAG" 100 || true
  npu_ctrl "$TAG"
  delmodel "$F"
}
s_lfm3_f16() {
  TAG=LFM3-17B-F16; F=$DL/LFM3-1.7B-F16.gguf
  fetch "$LFM3/LFM3-1.7B-F16.gguf" || return
  npu_ctrl "$TAG"   # NPU: predicted ❌ (3.3 GB ≫ 1.75 GiB) — skip the certain crash
  delmodel "$F"
}
s_lfm3_kv() { # L2 KV-dtype rows (needs Q4_K_M still on disk — runs before its delete)
  F=$DL/LFM3-1.7B-Q4_K_M.gguf
  [ -f "$F" ] || return
  npu_ctrl "LFM3-17B-Q4KM-kvq8"  "-ctk q8_0 -ctv q8_0"
  npu_ctrl "LFM3-17B-Q4KM-kvq4"  "-ctk q4_0 -ctv q4_0"
  bench_npu "LFM3-17B-Q4KM-kvq8" 100 "-ctk q8_0 -ctv q8_0" || true
  bench_npu "LFM3-17B-Q4KM-kvq4" 100 "-ctk q4_0 -ctv q4_0" || true
  delmodel "$F"   # LFM3 Q4_K_M released here (after L1+L2+KV rows)
}
s_lfm25() {
  TAG=LFM25-26B-Q4KM; F=$DL/LFM2.5-2.6B-Q4_K_M.gguf
  fetch "$HF/LiquidAI/LFM2.5-2.6B-GGUF/resolve/main/LFM2.5-2.6B-Q4_K_M.gguf" || return
  bench_npu "$TAG" 100 && npu_ctrl "$TAG"
  npu_ctrl "$TAG"
  delmodel "$F"
}
s_q08() {
  TAG=Qwen35-08B-Q4KM; F=$DL/Qwen3.5-0.8B-Q4_K_M.gguf
  fetch "$UNS/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf" || return
  bench_npu "$TAG" 100 && npu_ctrl "$TAG"
  npu_ctrl "$TAG"
  delmodel "$F"
}
s_q2b() {
  TAG=Qwen35-2B-Q4KM; F=$DL/Qwen3.5-2B-Q4_K_M.gguf
  fetch "$UNS/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf" || return
  bench_npu "$TAG" 100 || true   # 1.22 GB + KV + 512 MiB MBUF — F3 edge for Q4_K_M
  npu_ctrl "$TAG"
  delmodel "$F"
}
s_q4b_q4km() {
  TAG=Qwen35-4B-Q4KM; F=$DL/Qwen3.5-4B-Q4_K_M.gguf
  fetch "$UNS/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf" || return
  npu_ctrl "$TAG"
  for N in 100 36 30 24; do bench_npu "$TAG" $N || true; done
  delmodel "$F"
}
s_q4b_q40() { # L5 core: full -ngl sweep on the known-good repack
  TAG=Qwen35-4B-Q40; F=$DL/Qwen3.5-4B-Q4_0.gguf
  fetch "$UNS/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_0.gguf" || return
  npu_ctrl "$TAG"
  for N in 12 18 24 30 36 42 100; do bench_npu "$TAG" $N || true; done
  delmodel "$F"
}
s_q9b_q4km() {
  TAG=Qwen35-9B-Q4KM; F=$DL/Qwen3.5-9B-Q4_K_M.gguf
  fetch "$UNS/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf" || return
  npu_ctrl "$TAG"
  for N in 100 32 24 16; do bench_npu "$TAG" $N || true; done
  delmodel "$F"
}
s_q9b_q40() {
  TAG=Qwen35-9B-Q40; F=$DL/Qwen3.5-9B-Q4_0.gguf
  fetch "$UNS/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_0.gguf" || return
  for N in 24 16 8; do bench_npu "$TAG" $N || true; done
  delmodel "$F"
}
s_k2_4b() {
  TAG=K2H-4B-Q4KM; F=$DL/K2-Horizon-4B-Q4_K_M.gguf
  fetch "$K2/K2-Horizon-3.7B-GGUF/resolve/main/K2-Horizon-4B-Q4_K_M.gguf" || return
  npu_ctrl "$TAG"
  for N in 100 30 24 18; do bench_npu "$TAG" $N || true; done   # 18 = F4 disaster re-test on new runner
  delmodel "$F"
}
s_k2_4b_q40() { # L5 F3 question: does the Q4_0 repack fit where Q4_K_M didn't?
  TAG=K2H-4B-Q40; F=$DL/K2-Horizon-3.7B-Q4_0.gguf
  fetch "$HF/bartowski/K2-Horizon-3.7B-GGUF/resolve/main/K2-Horizon-3.7B-Q4_0.gguf" || return
  for N in 100 30 24 18; do bench_npu "$TAG" $N || true; done
  delmodel "$F"
}
s_k2_7b() {
  TAG=K2H-7B-Q4KM; F=$DL/K2-Horizon-7B-Q4_K_M.gguf
  fetch "$K2/K2-Horizon-7B-GGUF/resolve/main/K2-Horizon-7B-Q4_K_M.gguf" || return
  npu_ctrl "$TAG"
  for N in 100 40 32 24; do bench_npu "$TAG" $N || true; done
  delmodel "$F"
}
s_gemma_e2b() { # gemma4 additions 2026-10-30 — b11996 enumerates the gemma4 arch
  TAG=Gemma4-E2B-Q4KM; F=$DL/gemma-4-E2B-it-Q4_K_M.gguf
  fetch "$UNS/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf" || return
  npu_ctrl "$TAG"
  bench_npu "$TAG" 100 || { for N in 30 24 18; do bench_npu "$TAG" $N || true; done; }
  delmodel "$F"
}
s_gemma_e4b() { # 4.98 GB Q4_K_M ⇒ F3 sweep expected; unsloth also ships MTP heads (L9)
  TAG=Gemma4-E4B-Q4KM; F=$DL/gemma-4-E4B-it-Q4_K_M.gguf
  fetch "$UNS/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf" || return
  npu_ctrl "$TAG"
  for N in 100 40 30 24 18; do bench_npu "$TAG" $N || true; done
  delmodel "$F"
}

STAGES="s_lfm3_q4km s_lfm3_qad s_lfm3_q6k s_lfm3_q80 s_lfm3_f16 s_lfm3_kv s_lfm25 s_q08 s_q2b s_q4b_q4km s_q4b_q40 s_q9b_q4km s_q9b_q40 s_k2_4b s_k2_4b_q40 s_k2_7b s_gemma_e2b s_gemma_e4b"
for S in ${@:-$STAGES}; do
  echo "=== stage $S $(date -u +%FT%TZ)" | tee -a results/matrix_status.log
  $S
done
echo "=== matrix done $(date -u +%FT%TZ)" | tee -a results/matrix_status.log
