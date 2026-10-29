#!/bin/bash
# Waits for the running llama-bench to finish, then runs the rest of the CPU suite.
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=$PWD/llama.cpp
B=./llama.cpp/llama-bench
Q4KM=models/LFM3-1.7B-Q4_K_M.gguf
QAD=models/LFM3-1.7B-QAD-Q4_0.gguf
LFM25=models/LFM2.5-2.6B-Q4_K_M.gguf

# 1) wait for the main bench to finish
while pgrep -f "llama-bench -m models/LFM3-1.7B-Q4_K_M" >/dev/null; do sleep 20; done

# 2) thread sweep + affinity (big cores = cpu0-3 => mask 0xf)
$B -m $Q4KM -p 512 -n 128 -t 2,4,6,8 -r 2 -o csv > results/bench_q4km_threads.csv 2>&1
$B -m $Q4KM -p 512 -n 128 -t 4 -C 0xf --cpu-strict 1 -r 2 -o csv > results/bench_q4km_pinbig.csv 2>&1

# 3) flash-attn on/off
$B -m $Q4KM -p 2048 -n 128 -fa on,off -r 2 -o csv > results/bench_q4km_fa.csv 2>&1

# 4) QAD-Q4_0 comparison
$B -m $QAD -p 512,2048,8192 -n 32,128 -r 2 -o csv > results/bench_qad_main.csv 2>&1

# 5) LFM2.5-2.6B baseline (blog claims LFM3 faster + better w/ 35% fewer params)
$B -m $LFM25 -p 512,2048,8192 -n 128 -r 2 -o csv > results/bench_lfm25_main.csv 2>&1

# 6) KV memory + long-context scaling
bash scripts/kv_memory.sh $Q4KM > results/kv_memory_q4km.txt 2>&1
$B -m $Q4KM -p 16384,32768 -n 0 -r 1 -o csv > results/bench_q4km_longctx.csv 2>&1

echo "SUITE DONE" > results/suite_done
