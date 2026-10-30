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

Rows 1–5 are inherited; their `pinned-CPU control` column is v0.8.0-era context. L8 is
done — same-runner controls now exist for the whole Q4_K_M family set in §4 (the §2
0.8B/2B rows are Q4_0 on the older build, so they keep their inherited controls as context).

| Model (Q4) | NPU pp512 | NPU pp8192 | NPU tg128 | pinned-CPU control | speed-up (pp8192 / tg128) |
|---|---|---|---|---|---|
| Qwen3.5-0.8B-Q4_0 | **1366** | 1233 | **43.0** | 148.7 / 31.9 | **13.6× / 1.35×** |
| Qwen3.5-2B-Q4_0 | **863** | 807 | **24.3** | 69.8 / 16.1 | **14.9× / 1.51×** |
| LFM2.5-2.6B-Q4_K_M | **679** | 572 | **17.7** | 40.1 / 13.1 | **21.1× / 1.35×** |
| Qwen3.5-4B-Q4_0 partial (`-ngl 24`) | 70.8 | 53.7 | 10.3 | *(unpinned: 19.8/4.2)* | ~3× / ~2.4× — **crashes on b11996** (§4) |
| K2-Horizon-4B-Q4_K_M full | ✗ F3 VA ceiling | — | — | — | — (confirmed §4) |
| **LFM3-1.7B-Q4_K_M** ² | **1033.2 ±0.5** | 905.9 | 25.6 | 65.5 / 19.7 ² | **16.7× / 1.30×** |

² LFM3 row only: runner `llamacpp/20261029` (ggml 0.28.0, b11996-7df8b876b) — 20261020
can't load `lfm3` (F5). Control = same runner, `-ngl 0 -t 4 --cpu-mask 0x0f --cpu-strict 1`
(L8 complete): 65.5 / 54.2 / 19.7 (pp512/pp8192/tg128).
`results/bench_LFM3-17B-Q4KM_{npu_ngl100,hex2cpu}.csv`. Regime `-p 512,8192 -n 32,128
-r 2 -b 1024 -ub 512 -fa on`, OPBATCH=2048, MBUF 256/256.

Rules already banked (parent §3), now **measured in-arch on b11996** (§3–§4): prefill
15–18× (pp8192, full-offload family set) with decode only **1.30–1.45×** (bandwidth-bound);
**offload fully or at exactly one partial depth** (F4 — the usable window is one `-ngl`
wide, §4); F3 re-derived as a per-buffer VA-mapping limit, not a flat 1.75 GiB budget (§4,
FAILURES F3); quality parity NPU↔CPU confirmed (GSM8K 60 vs 55% @N=20 — one band).
LFM3: F5 **resolved 2026-10-30** via the public 20261029 asset → L1 ✅ measured;
GSM8K parity check still open.

## 3. Dtype & quant matrix (L2) — LFM3-1.7B every dtype, NPU `-ngl 100` vs same-runner CPU ctrl

Regime `-p 512,8192 -n 32,128 -r 2`, build b11996-7df8b876b. NPU = `--device HTP0 -ngl 100
-b 1024 -ub 512 -fa on -t 4 --cpu-mask 0x0f`, OPBATCH=2048, MBUF 256/256; CPU ctrl = same
binary `-ngl 0 -t 4 --cpu-mask 0x0f --cpu-strict 1`. CSVs `results/bench_LFM3-17B-*_{npu_ngl100,hex2cpu}.csv`.

| Config | Weights | NPU pp512 · pp8192 · tg128 | CPU ctrl pp512 · pp8192 · tg128 | ratio (pp8192 / tg128) | NPU fit |
|---|---|---|---|---|---|
| Q4_K_M | 1.14 GB | **1033.2 · 905.9 · 25.6** | 65.5 · 54.2 · 19.7 | 16.7× / 1.30× | ✅ |
| QAD-Q4_0 | 1.00 GB | **1036.5 · 901.4 · 29.7** | 85.5 · 66.4 · 21.3 | 13.6× / 1.39× | ✅ |
| Q6_K | ~1.5 GB | 1033.0 · 899.2 · 24.3 | 50.2 · 43.0 · 14.7 | 20.9× / **1.65×** | ✅ |
| Q8_0 | ~1.94 GB | 1027.2 · 896.1 · 19.0 | 74.0 · 59.4 · 13.4 | 15.1× / 1.42× | ✅ — **overturns the F3 "1.75 GiB" prediction** |
| F16 | ~3.4 GB | — (not attempted) | 27.8 · 25.8 · 7.3 | — | ❌ predicted, skipped the certain crash |
| Q4_K_M + KV q8_0 | +KV | 1025.7 · 798.7 · 25.9 | 62.3 · 39.1 · 19.8 | 20.4× / 1.31× | ✅ |
| Q4_K_M + KV q4_0 | −KV | 💥 **F20** (core dump) | 60.9 · 33.6 · 19.6 | — | crash |

Measured rules (these replace the bandwidth-scaling predictions that headed this section):

1. **NPU prefill is quant-independent**: 1027–1037 t/s from Q4_K_M to Q8_0 — HTP compute
   bound, not weight-bandwidth bound. The headline question ("is Q6_K the only upgrade that
   keeps the ~1030 prefill?") answers: **all of them do**.
2. **Decode is where quant matters**: 29.7 (QAD) → 19.0 (Q8_0) t/s on NPU; QAD is the decode
   king and also the fastest CPU control (21.3).
3. **CPU prefill hates K-quants**: Q6_K 50.2 < Q4_K_M 65.5 < Q8_0 74.0 < QAD 85.5 —
   dequant cost, not bytes; that's why Q6_K's decode ratio looks best (1.65× — its CPU
   control suffers most).
4. **KV q8_0 is nearly free on NPU** (pp8192 −12%, decode unchanged); KV q4_0 is F20 on NPU
   and −38% pp8192 on CPU — nobody should run q4_0 KV.
5. Quality (ppl/GSM8K) on these dtype rows is not yet measured — L1 parity item stays open.

## 4. NPU offload boundaries (L5) — `-ngl` sweeps, build b11996

Full-offload family table (all Q4_K_M, `-ngl 100` MBUF 256/256 vs same-runner ctrl;
`results/bench_{*,*_npu_ngl100}.csv`):

| Model | NPU pp512 · pp8192 · tg128 | CPU ctrl | ratio (pp8192 / tg128) |
|---|---|---|---|
| Qwen3.5-0.8B | 1397.5 · 1256.5 · **42.0** | 124.4 · 81.5 · 30.6 | 15.4× / 1.37× |
| Qwen3.5-2B | 880.1 · 824.9 · 22.1 | 57.9 · 46.6 · 15.2 | 17.7× / **1.45×** |
| LFM2.5-2.6B | 684.4 · 574.4 · 17.8 | 40.3 · 34.2 · 13.2 | 16.8× / 1.35× |
| LFM3-1.7B | 1033.2 · 905.9 · 25.6 | 65.5 · 54.2 · 19.7 | 16.7× / 1.30× |

(LFM2.5 confirms build-to-build stability vs §2's 20261020 row: 684.4 vs 679.)

Partial-offload sweeps (MBUF 128/128; `—` = F3 crash before any row; ctrl = same-runner
`-ngl 0` pp512 · pp8192 · tg128):

| Model | -ngl | pp512 | pp8192 | tg128 | verdict |
|---|---|---|---|---|---|
| Q35-4B-Q4_K_M *(ctrl 23.0 · 17.7 · 7.0)* | 100 / 36 | — | — | — | 💥 F3 |
| | **30** | 135.4 | 134.3 | 8.4 | ✅ sweet spot (7.6× / 1.20×) |
| | 24 | 69.5 | 53.7 | 8.9 | ✅ usable (3.0× / 1.27×) |
| Q35-4B-Q4_0 *(ctrl 27.5 · 20.3 · 7.7)* | 100/42/36/24 | — | — | — | 💥 F3 — **non-monotonic: 30 works, 24 doesn't** |
| | **30** | 136.9 | 126.1 | 10.4 | ✅ sweet spot (6.2× / 1.35×) |
| | 18 | 44.4 | 38.8 | **2.6** | 💥 **F4 collapse** (decode 0.34× CPU) |
| | 12 | 38.5 | 28.7 | 6.1 | ⚠️ decode below CPU |
| Q35-9B-Q4_K_M *(ctrl 12.1 · 10.6 · 4.0)* | 100/32/24/16 | — | — | — | ❌ **no usable depth** (890 MB block fails even at 16) |
| Q35-9B-Q4_0 | 24 / 16 | — | — | — | 💥 F3 |
| | 8 | 18.7 | 15.0 | 2.2 | ❌ decode 0.55× CPU — not usable |
| K2-4B-Q4_K_M *(ctrl 24.1 · 12.1 · 7.9)* | 100 | — | — | — | 💥 F3 |
| | 30 | 96.9 | — | — | 💥 crashed mid-run (one row survived) |
| | **24** | 59.3 | 29.0 | 8.4 | ✅ only usable window (2.4× / 1.06×) |
| | 18 | 39.9 | 20.9 | **3.5** | 💥 F4 collapse (0.44×) |
| K2-4B-Q4_0 *(same ctrl)* | 100 / 30 | — | — | — | 💥 F3 |
| | 24 / 18 | 48.1 / 46.7 | 29.6 / 22.1 | **3.1 / 3.2** | ⚠️ fits, but decode ~0.40× CPU — **repack answer: fitting ≠ usable** |
| K2-7B-Q4_K_M *(ctrl 13.0 · 8.5 · 4.4)* | 100/40/32/24 | — | — | — | ❌ **no usable depth** (898 MB block at ngl 24) |

Boundary rules (measured — these replace the inherited ~1.75 GiB heuristic):

1. **The usable window is exactly one `-ngl` depth per model** (4B → 30, K2-4B → 24). One
   step below = F4 collapse zone (decode 2.6–3.5 t/s = 0.3–0.45× CPU); two below = decode
   ≤ CPU with nothing to show for it.
2. **≥6B has no usable NPU config on this board** — 9B and K2-7B fail the F3 mapping at
   *every* depth tried; 9B-Q4_0 `-ngl 8` decodes at 2.2 t/s. ≥6B = CPU-only.
3. **F3 is per-buffer VA mapping, not a weight budget** — fingerprint `HTP0 buffer mapping
   failed : domain_id 3 size <N> error 0x00000001`; the failing single buffer ranges
   134 MB (K2-4B ngl 30) → 898 MB (K2-7B ngl 24); non-monotonic in `-ngl` (4B-Q4_0: 24
   fails, 30 works); Q8_0-LFM3 fits 1.94 GB of weights outright. Depends on allocator /
   graph layout — and on build: §2's known-good 4B-Q4_0 `-ngl 24` row (20261020) crashes on
   b11996.
4. **F4 confirmed live**: collapse caught at 4B-Q4_0 `-ngl 18` (2.6 vs CPU 7.7 t/s) and
   K2-4B `-ngl 18` (3.5 vs 7.9). Quant shifts the sweet-spot depth, not the collapse's
   existence; K2 (hybrid attention) is the worst NPU citizen — even its best window only
   ties CPU decode (1.06×).

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
