# Benchmarks — llama.cpp CPU + NPU depth (measured on this board)

**Scope:** only the **llama.cpp CPU+NPU hexagon runner** is evaluated here. CPU rows are
that runner's own `-ngl 0` path (same binary as NPU rows); the v0.8.0 CPU-only build and
MNN are out of scope — §1 below is inherited context, kept for history, never extended.
§1–2 are **inherited baselines** — the regimes every new measurement in this repo must
compare against. Raw CSVs live in
[`../ventunoq-llamacpp/results/`](../ventunoq-llamacpp/results/) (not duplicated here);
method notes: parent [BENCHMARKS.md](../ventunoq-llamacpp/BENCHMARKS.md) header.
Everything from §3 on is this repo's own output — fill from the
[ROADMAP](ROADMAP.md) items, keep this file numbers-only.

## 1. Inherited CPU-only baselines (v0.8.0 b11972 — out of scope, context only)

| Model (Q4_K_M unless noted) | pp512 | pp8192 | tg128 |
|---|---|---|---|
| LFM3-1.7B | 49.0 | 43.3 | 11.7 |
| LFM3-1.7B QAD-Q4_0 | **87.4** | **62.6** | 10.2 |
| LFM2.5-2.6B | 32.0 | 28.0 | 7.3 |
| Qwen3.5-0.8B-Q4_0 | 148.7 | — | 31.9 |
| Qwen3.5-2B | 49.1 | 41.3 | 7.9 |
| Qwen3.5-4B | 19.8 | 16.9 | 4.2 |
| Qwen3.5-9B | 11.6 | 10.2 | 2.7 |
| K2-Horizon-4B | 19.8 | 11.5 | 4.3 |
| K2-Horizon-7B | 11.2 | 8.3 | 3.0 |

Threads/pinning rule (`results/bench_q4km_{threads,pinbig}.csv`): `-t 4 --cpu-mask 0xf
--cpu-strict 1` is optimal (49.6/19.4 LFM3); spreading to the A55s costs **−43% decode**.
Long ctx (LFM3, `results/bench_q4km_longctx.csv`): pp16384 = 37.8, pp32768 = 30.8 t/s;
KV ≈ 8.9 KB/token (tri-hybrid), Qwen-class 2.6–3.8× that.

## 2. NPU baselines (hexagon runner `llamacpp/20261020`, `-ngl 100`)

Rows 1–5 are inherited; their `pinned-CPU control` column is v0.8.0-era context — L8 is
replacing it with same-runner `-ngl 0` controls for every row.

| Model (Q4) | NPU pp512 | NPU pp8192 | NPU tg128 | pinned-CPU control | speed-up (pp8192 / tg128) |
|---|---|---|---|---|---|
| Qwen3.5-0.8B-Q4_0 | **1366** | 1233 | **43.0** | 148.7 / 31.9 | **13.6× / 1.35×** |
| Qwen3.5-2B-Q4_0 | **863** | 807 | **24.3** | 69.8 / 16.1 | **14.9× / 1.51×** |
| LFM2.5-2.6B-Q4_K_M | **679** | 572 | **17.7** | 40.1 / 13.1 | **21.1× / 1.35×** |
| Qwen3.5-4B-Q4_0 partial (`-ngl 24`) | 70.8 | 53.7 | 10.3 | *(unpinned: 19.8/4.2)* | ~3× / ~2.4× |
| K2-Horizon-4B-Q4_K_M full | ✗ F3 VA ceiling | — | — | — | — |
| **LFM3-1.7B-Q4_K_M** ² | **1033.2 ±0.5** | 905.9 | 25.6 | 46.8 / 19.7 ² | **20.9× / 1.30×** |

² LFM3 row only: runner `llamacpp/20261029` (ggml 0.28.0, b11996-7df8b876b) — 20261020
can't load `lfm3` (F5). Control = same runner, `-ngl 0`, t4 mask 0x0f (strict-0 pass;
strict-1 re-run in progress, L8).
`results/bench_LFM3-17B-Q4KM_{npu_ngl100,hex2cpu}.csv`. Regime `-p 512,8192 -n 32,128
-r 2 -b 1024 -ub 512 -fa on`, OPBATCH=2048, MBUF 256/256.

Rules already banked (parent §3): prefill 9–21× on HTP, decode only 1.35–1.5×
(bandwidth-bound); **offload fully or ≥⅔ layers** (F4 ping-pong); DSP-side budget
~1.75 GiB (F3); quality parity NPU↔CPU confirmed (GSM8K 60 vs 55% @N=20 — one band).
LFM3: F5 **resolved 2026-10-30** via the public 20261029 asset → L1 ✅ measured,
21× prefill confirms the prediction; GSM8K parity check still open.

## 3. Dtype & quant matrix (L2) — LFM3-1.7B reference, CPU pinned + NPU-budget check

Predictions = bandwidth scaling (decode ∝ 1/weight-bytes) + the QAD lesson (prefill is
*dequant-cost* bound: Q4_0 ×1.8 vs Q4_K_M). Measure, don't assume.

| Config | Weight size | tg128 | pp512 | ppl (gatsby; Q4_K_M = 29.5) | NPU ≤1.75 GiB? |
|---|---|---|---|---|---|
| Q4_K_M (baseline) | 1.14 GB | 11.7 | 49.0 | — | ✅ |
| QAD-Q4_0 (baseline) | 1.0 GB | 10.2 | 87.4 | 31.1 | ✅ |
| Q6_K | ~1.5 GB | ⏳ ~8.6 | ⏳ | ⏳ ≈F16 | ⏳ barely |
| Q8_0 | ~2.0 GB | ⏳ ~6.7 | ⏳ | ⏳ lossless | ⏳ ❌ predicted |
| F16 (`--type f16`) | ~3.4 GB | ⏳ ~3.5 | ⏳ | reference | ❌ |
| Q4_K_M + `-ctk q8_0 -ctv q8_0` | +KV | ⏳ | ⏳ | ⏳ | ⏳ |
| Q4_K_M + `-ctk q4_0 -ctv q4_0` | −KV | ⏳ | ⏳ | ⏳ | ⏳ |

**Headline question:** is Q6_K the only upgrade that keeps the 9–21× NPU prefill?

## 4. NPU offload boundaries (L5) — `-ngl` sweeps on F3/F4-class models

| Model | -ngl | NPU-side MiB | pp512 | tg128 | verdict |
|---|---|---|---|---|---|
| Qwen3.5-4B-Q4_0 | 24 (known-good) | ~1.7 GiB | 70.8 | 10.3 | ✅ baseline row |
| Qwen3.5-4B-Q4_0 | 12 / 18 / 30 / 36 | ⏳ | ⏳ | ⏳ | ⏳ F4 re-test on a *Q4_0* repack |
| K2-Horizon-4B-Q4_K_M | 18 (F4 disaster: 2.6 t/s) | ⏳ | ⏳ | ⏳ | ⏳ is the ping-pong quant-dependent? |
| K2-Horizon-4B-Q4_0 | full | ⏳ | ⏳ | ⏳ | ⏳ F3 is weight-bytes-bound → does Q4_0 fit? |

## 5. Sustained / thermal (L3) — 10-min loops, `scripts/sustained.sh`

Passive cooling ⇒ the per-minute curve, not the burst mean, is deployment truth.

| Engine/config | t/s min 1 → min 10 (tg) | Δ% | plateau? | peak °C (cpuss/nsp) |
|---|---|---|---|---|
| CPU pinned (LFM3) | ⏳ | ⏳ | ⏳ | ⏳ |
| NPU (LFM2.5, `-ngl 100`) | ⏳ | ⏳ | ⏳ | ⏳ |

## 6. Energy (L4) — INA232 hwmon2, `scripts/power_probe.py`; J/token = W ÷ t/s

Spot checks so far: idle ≈ 8.5–9.5 W; 8× `yes` → avg 11.0 / max 11.8 W (memory-light —
not a ceiling). The ~15 W hunch stays open until the NPU rows exist.

| Engine/config | avg W (pp) | avg W (tg) | J/tok pp | J/tok tg |
|---|---|---|---|---|
| CPU ctrl (`-ngl 0`, same runner) | ⏳ | ⏳ | ⏳ | ⏳ |
| NPU `-ngl 100` | ⏳ | ⏳ | ⏳ ⏳ ← hypothesis: tiny | ⏳ ⏳ ← hypothesis: ≪ CPU |

## 7. Serving & concurrency (L7) — `llama-server -np 2/4`

| Engine | clients | per-client tg | aggregate tg | TTFT p50/p95 |
|---|---|---|---|---|
| CPU pinned | 1 / 2 / 4 | ⏳ | ⏳ | ⏳ |
| NPU | 1 / 2 / 4 | ⏳ | ⏳ | ⏳ |
