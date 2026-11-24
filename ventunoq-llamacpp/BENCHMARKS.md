# Benchmarks — what works, and how fast

All numbers measured **on this board** (Arduino Ventuno Q, QCS8300). Nothing here is
quoted from vendor sheets; raw data pointers are given per table. The chronological
log with exact command lines is `docs/REPORT.md`.

**Method notes that matter:**
- CPU baselines (§1–2, §4) ran on llama.cpp **v0.8.0 b11972** (official Ubuntu-arm64
  prebuilt), CPU backend `armv8.2_2` (runtime-dispatched; board lacks SVE2/i8mm).
- NPU runs (§3) use Arduino's **`llamacpp/20261020` hexagon runner** (ggml 0.27.0,
  upstream Hexagon backend). Its CPU controls were re-run **on that same binary** with
  `-ngl 0` and the service's own tuning (`-t 4 --cpu-mask 0x0f -b 1024 -ub 512 -fa on`) —
  the §1 table's un-pinned 8-thread numbers are *not* the NPU comparison baseline.
- Quality harnesses (`scripts/gsm8k_eval.py`, `scripts/humaneval_eval.py`) are pure-stdlib
  Python against a local `llama-server` `/completion`, greedy/low-temp, strict scoring.
- `pp` = prompt eval t/s, `tg` = token generation t/s.

## 1. CPU throughput (llama-bench, `-ngl 0`)

Raw: `results/bench_*_main.csv`, `bench_q4km_*.csv`.

### 1.1 LFM3 vs the field

| Model (Q4_K_M unless noted) | pp512 | pp8192 | tg128 | GSM8K | HumanEval |
|---|---|---|---|---|---|
| **LFM3-1.7B** (1.1 GB) | 49.0 | 43.3 | **11.7** | 60.0% | 65% |
| LFM3-1.7B **QAD-Q4_0** (1.0 GB) | **87.4** | **62.6** | 10.2 | — | — |
| LFM2.5-2.6B (1.7 GB) | 32.0 | 28.0 | 7.3 | — | — |
| unsloth Qwen3.5-2B (1.2 GB) | 49.1 | 41.3 | 7.9 | 22.0% | 40% |
| unsloth Qwen3.5-4B (2.4 GB) | 19.8 | 16.9 | 4.2 | 72.0% | 85% |
| unsloth Qwen3.5-9B (5.4 GB) | 11.6 | 10.2 | 2.7 | 85.0%* | 80%* |
| K2-Horizon-4B (3.2 GB) | 19.8 | 11.5 | 4.3 | 60.0% | 75% |
| K2-Horizon-7B (5.6 GB) | 11.2 | 8.3 | 3.0 | 75.0%* | **100%*** |

\* reduced sample (GSM8K N=20, HumanEval 10 tasks) — 9B/7B decode is swap-adjacent, full
N was not worth the wall-clock. All others: GSM8K N=50, HumanEval first 20 tasks.

Key readings:
- **LFM3-1.7B beats LFM2.5-2.6B on both axes** (prefill ×1.5–1.6, decode ×1.4–1.6)
  with 35% fewer params — Liquid's announcement claim holds on real hardware.
- **QAD-Q4_0 is a prefill monster** (×1.8): Q4_0 dequant is far cheaper than Q4_K_M's
  block-unpacking on a NEON-only core. Cost: ~5% perplexity (29.5 → 31.1 on gatsby,
  `results/perplexity_*.txt`). For prompt-heavy use, run QAD.
- **Sub-3B quality winner is LFM3 outright** (60/65 vs Qwen3.5-2B's 22/40). Qwen3.5-2B's
  GSM8K collapse was verified as real behaviour, not harness error (11/39 misses were
  prompt-echo rambles; even discounting those ≈34%).
- **K2-Horizon-7B is the board's best coder** (10/10 HumanEval slice) at Qwen-9B speed.

### 1.2 Threads & core pinning (LFM3-1.7B Q4_K_M)

Raw: `results/bench_q4km_threads.csv`, `bench_q4km_pinbig.csv`.

| Config | pp512 | tg128 |
|---|---|---|
| `-t 2` | 29.0 | 13.1 |
| `-t 4` | 49.0 | 19.1 |
| `-t 6` | 46.5 | 13.8 |
| `-t 8` (default) | 47.3 | 11.1 |
| `-t 4 --cpu-mask 0xf --cpu-strict 1` | **49.6** | **19.4** |

**Rule: 4 threads pinned to the big cores.** Spreading to the A55 cluster *hurts*
decode (−43% at `-t 8`). This is the single biggest CPU tuning win on this board.

## 2. Long context & KV memory (LFM3-1.7B)

Raw: `results/bench_q4km_longctx.csv`, `results/kv_memory_q4km.txt`.

- Prefill degrades gracefully: pp16384 = **37.8 t/s**, pp32768 = **30.8 t/s** (`-c 33792`).
- Measured RSS 2.49 GB @2K ctx → 3.62 GB @128K ctx ⇒ **≈1.14 GB KV for 128K tokens**
  (~8.9 KB/token) — confirms the tri-hybrid architecture claim (256-token SWA + short
  convs); Qwen3.5-4B-class caches are 2.6–3.8× this size.
- RAM ceiling: 9B/7B-class models (5.4–5.7 GB weights) fit but leave the machine
  swap-adjacent; ≤4B is the comfortable zone.

## 3. NPU — Hexagon V75 via llama.cpp's upstream Hexagon backend

Runner: Arduino's `llamacpp-npu-runner` release `llamacpp/20261020` (fetch + sha256 +
env recipe in `research/bricks-faster-inference.md`; only runtime gap was the
unversioned `libcdsprpc.so` symlink). `llama-cli --list-devices` → `HTP0: Hexagon`
as the plain user. Raw: `results/bench_hex_*_npu.csv` (NPU) vs `bench_hex2_*_cpu.csv`
(same-binary pinned CPU control).

| Model (Q4) | NPU pp512 | NPU pp8192 | NPU tg128 | pinned-CPU pp512 / tg128 | speed-up (pp8192 / tg128) |
|---|---|---|---|---|---|
| Qwen3.5-0.8B-Q4_0 | **1366** | 1233 | **43.0** | 148.7 / 31.9 | **13.6× / 1.35×** |
| Qwen3.5-2B-Q4_0 | **863** | 807 | **24.3** | 69.8 / 16.1 | **14.9× / 1.51×** |
| LFM2.5-2.6B-Q4_K_M | **679** | 572 | **17.7** | 40.1 / 13.1 | **21.1× / 1.35×** |
| Qwen3.5-4B-Q4_0 **partial** (`-ngl 24`) | 70.8 | 53.7 | 10.3 | *(v0.8.0 unpinned Q4_K_M: 19.8/4.2)* | ~3× / ~2.4× |
| K2-Horizon-4B-Q4_K_M full | ✗ VA ceiling (FAILURES F3) | — | — | — | — |

- **Prefill 9–21× faster on the HTP; decode 1.35–1.5×.** Decode is bandwidth-bound on
  every engine of this SoC (one shared LPDDR — `research/memory-map.md`).
- **q4_K→NPU repack works as advertised** (llama.cpp PR #28994): Q4_K_M files run
  NPU-side unmodified; q2_K/q3_K and token embeddings stay on CPU.
- **Partial offload can backfire**: K2-4B at `-ngl 18` decoded at **2.6 t/s vs the
  CPU's 7.9** — per-token CPU↔NPU ping-pong. Rule: offload fully, or keep ≥⅔ of
  layers on the NPU (see FAILURES F4).
- **Interactive quality ceiling moves 2B → 4B**: Qwen3.5-4B-Q4_0 at `-ngl 24`
  (fits the ~1.75 GiB DSP VA budget) decodes **10.3 t/s with 54 t/s prefill @8k**
  vs 4.2 t/s on CPU.
- **LFM3 itself does not run on this runner** — it predates the `lfm3` arch (ggml
  0.27.0). Rebuild recipe: ROADMAP R1.

### 3.1 Quality parity, NPU vs CPU (same binary, same model, same sampler)

LFM2.5-2.6B-Q4_K_M, GSM8K N=20, `llama-server` `-ngl 100` vs `-ngl 0`:

| Engine | GSM8K |
|---|---|
| NPU (`--device HTP0 -ngl 100`, ctx 8192) | **12/20 = 60.0%** |
| CPU control (`-ngl 0`) | 11/20 = 55.0% |

(`results/gsm8k_hexq_{npu,cpu}.log`, JSONs alongside.) One question apart at N=20 —
treat 55–60% as one band. The q4_K repack + DSP recurrent-state path does not change
answers. **The NPU is the default engine on this board for prompt-heavy work up to
the ~1.75 GiB NPU-side budget.**

## 4. Quality evals (CPU, Q4_K_M, this machine)

Raw: `results/gsm8k_*.json`, `humaneval_*.json` (+ per-run `.log` transcripts).

| Model | GSM8K (strict last-number) | HumanEval (sandboxed exec) |
|---|---|---|
| LFM3-1.7B | **30/50 = 60.0%** | **13/20 = 65%** |
| Qwen3.5-2B | 11/50 = 22.0% | 8/20 = 40% |
| Qwen3.5-4B | 36/50 = 72.0% | 17/20 = 85% |
| Qwen3.5-9B | 17/20 = 85.0%* | 8/10 = 80%* |
| K2-Horizon-4B | 30/50 = 60.0% | 15/20 = 75% |
| K2-Horizon-7B | 15/20 = 75.0%* | 10/10 = 100%* |

Failure anatomy (LFM3): HumanEval misses are edge-case handling (empty strings,
numeric-word parsing, off-by-one), not syntax; GSM8K misses are arithmetic drift,
never format failures. Perplexity (gatsby, n_ctx 2048): Q4_K_M **29.54 ± 0.49**,
QAD-Q4_0 **31.14 ± 0.52**.

## 5. Engine selection guide

| Workload | Use | Command shape |
|---|---|---|
| Interactive chat/code ≤2B | **CPU, pinned** | `-t 4 --cpu-mask 0xf --cpu-strict 1` |
| Interactive 4B-class | **NPU partial** `-ngl 24` (Q4_0) | 10.3 tg / 54 pp @8k |
| RAG / agents / long prompts | **NPU full offload** (≤1.75 GiB NPU-side) | `-ngl 100 -b 1024 -ub 512 -fa on` |
| LFM3 (any size) | **CPU only** today | NPU runner predates `lfm3` (ROADMAP R1) |
| K2-4B and larger on NPU | **don't** — CPU beats partial offload | see FAILURES F3/F4 |
| Anything on GPU | **don't** (yet) | see FAILURES F2 + ROADMAP R2 |
