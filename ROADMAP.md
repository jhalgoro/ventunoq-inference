# Roadmap — llama.cpp CPU+NPU depth program (L1–L9)

**Scope (2026-10-30):** this project evaluates **one architecture — llama.cpp CPU+NPU on
the hexagon runner**. CPU-only builds (v0.8.0 prebuilt) and MNN (sibling repos) are *not*
evaluated here; every CPU row is the runner's own `-ngl 0` path, same binary as the NPU
rows. Parent survey items are **R** ([../ventunoq-llamacpp/ROADMAP.md](../ventunoq-llamacpp/ROADMAP.md)).
L1–L4 port parent R1/R6/R7/R8 — the *how* lives there, this file keeps the *here-now*
plan and success criteria; L5–L7 are new axes the survey had no room for; L8–L9 are
2026-10-30 measurements that reopened the baseline question.

| # | Parent | Question | Effort | Blocks on |
|---|---|---|---|---|
| L1 | R1 | Rebuild the hexagon runner so **LFM3 runs on the NPU** | ~~low~~ **done 2026-10-30** | — (was: workflow access; not needed, see L1) |
| L2 | R6 | dtype matrix: Q6_K / Q8_0 / F16 / KV-dtype — CPU cost + NPU budget | low–med | disk policy |
| L3 | R7 | Deep suites: sustained/thermal, TTFT tails, NIAH, grounded-QA, full-N | medium | — |
| L4 | R8 | Energy: W and J/token per engine from INA232 (sudo-free) | low | — |
| L5 | — | NPU offload boundaries: `-ngl` sweeps, F3/F4 rule edges | low | F3 (measurement only) |
| L6 | — | CPU micro-tuning beyond `-t 4` pinning | low | — |
| L7 | — | Serving & concurrency: `llama-server -np 2/4`, both engines | medium | — |
| L8 | — | **In-arch CPU controls** (same runner, `-ngl 0`, strict pinning) for every NPU row — §1 is out-of-scope context with missing raw CSVs | low | — |
| L9 | — | **MTP speculative decode** (`--spec-type draft-mtp`) for Qwen3.5 — merged in b11996 | med | — |

> **Build freshness is a first-class variable here** (2026-10-30 lesson). Arduino ships a
> `llamacpp/<date>` release **every day, no GitHub auth needed** — plain release assets
> (`llamacpp-hexagon-<date>.tar.gz` on `arduino/app-bricks-py`), sha256 in the body; fetch
> recipe in `scripts/env.sh`. Twelve days (0.27.0→0.28.0) bought
> **16 hexagon-backend commits** (MUL_MAT_ID row-skip, multicore HMX FA, FA VA slicing,
> `GGML_HEXAGON_VTCM_SIZE`) *and* doubled small-model CPU prefill (Qwen3.5-2B Q4_0 pp512
> 23.7 → 54.2). **Every cross-build comparison must name both builds.**

## State of play (2026-10-30)

| Model | CPU ctrl (§1 inherited, v0.8.0 ctx) | NPU (HTP0, `-ngl 100`) | Notes |
|---|---|---|---|
| LFM3-1.7B Q4_K_M | 49.6 / 19.4 t/s (§1) | **1033.2 pp512 · 25.6 tg128** ✅ L1 | was ⛔ F5 — now unblocked, 0.28.0 nightly |
| LFM2.5-2.6B Q4_K_M | 32.0 / 7.3 | 297.7 / 11.5 | |
| Qwen3.5-0.8B Q4_0 | 148.7 / 31.9 | 1487.7 / 40.7 | |
| Qwen3.5-2B Q4_0 | 49.1 / 7.9 (§1; newer builds ≥54) | 1034.6 / 27.7 | |
| Qwen3.5-4B Q4_0 | 19.8 / 4.2 | 666.6 / 27.7 | |
| Qwen3.5-9B Q4_0 | 11.6 / 2.7 | 239.8 / 14.6 | |
| K2-Horizon-4B Q4_K_M | 19.8 / 4.3 | 385.2 / 14.1 | |
| K2-Horizon-7B Q4_K_M | 11.2 / 3.0 | 197.8 / 9.7 | |

Everything runs, and the NPU wins prefill by **9–21×** — but decode still lands *at or
below* the CPU (F4), F3 caps what fits, and the CPU column above is inherited v0.8.0
context, not this arch. The matrix (`scripts/depth_matrix.sh`) is re-measuring all of it
on 0.28.0: NPU rows + same-runner `-ngl 0` controls with strict pinning (L8).

---

## L1 — LFM3 on the NPU ✅ measured (parity check open)

**Was (parent F5):** runner `llamacpp/20261020` ships ggml 0.27.0 — predates the `lfm3`
arch. LFM3 (the board's best sub-3B model, 60% GSM8K / 65% HumanEval) was the *only*
interesting model missing the 9–21× NPU prefill.

**Resolved without any workflow access (2026-10-30):** LFM3 merged upstream 2026-10-27
(ggml PR #31252), and the daily `llamacpp/20261029` release asset is **public** — no
Actions rerun needed. `$HEX2` = that build (ggml 0.28.0, b11996-7df8b876b, sha256
`fb587ac4…`); it enumerates HTP0 and runs `lfm3` end-to-end.

**Measured** — LFM3-1.7B-Q4_K_M, `-ngl 100 -b 1024 -ub 512 -fa on -t 4 --cpu-mask 0x0f`,
OPBATCH=2048, MBUF 256/256, r=2 (`results/bench_LFM3-17B-Q4KM_npu_ngl100.csv`):

| | pp512 | pp8192 | tg32 | tg128 |
|---|---|---|---|---|
| NPU 0.28.0 | **1033.2 ±0.5** | 905.9 ±0.3 | 25.8 | 25.6 |
| CPU ctrl 0.28.0 (same runner, `-ngl 0`) | 46.8 | 43.3 | 20.2 | 19.7 |

**≈22× prefill, 1.30× decode** vs the same-runner control — top of the family band, as
predicted. Decode 25.6 sits between 0.8B (40.7) and 2.6B (11.5), same ordering as CPU.
(Inherited §1 lists LFM3 tg128 = 11.7 on the out-of-scope v0.8.0 build — do not ratio
against it; L8.) Control row is strict-0 pending its strict-1 re-run — numbers move a few %.

**Still open:** quality parity — GSM8K N=20 NPU vs CPU (parent `scripts/gsm8k_eval.py`);
a wrong-on-NPU model is a bug report, not a result. Acceptance: |Δ| ≤ 1 question, else
upstream issue.

## L2 — dtype matrix (R6): how much quality per t/s and per MB

**Do:** for the two archs that matter (Qwen3.5-2B, LFM3-1.7B) sweep
**Q4_K_M → Q6_K → Q8_0 → F16** on *both* engines, plus KV dtype (`--cache-type-eq f16`,
q8_0) — the V64 32-head KV is the memory hog at 8k.
**Acceptance:** one table (BENCHMARKS §3) with columns
`model | quant | size | cpu pp512/tg128 | npu pp512/tg128 | fits-F3? (Y/N)`;
one sentence per arch: "the quality-per-MB knee is at ____".
F16 LFM3 = 3.3 GB → **won't fit F3** — that row is a documented *N* (and an F3 datapoint),
not a failure. **Matrix stages s_lfm3_kv / s_q352b_kv running.**

## L3 — depth suites (R7): from "fast" to "dependable"

**Do, in this order:**
1. **Sustained/thermal** (R1): 10-min continuous pp+tg loop, per-minute t/s + INA232 W →
   is there throttling on this fanless-ish board? Both engines. `scripts/sustained.sh` ready.
2. **TTFT tail**: 20× pp512 and 20× first-token-at-8k — report p50/p95, not mean.
3. **Full-N evals**: GSM8K N=100 (from N=20), Qwen3.5-2B CPU vs NPU parity.
4. **NIAH 8k** + **grounded-QA** (parent harnesses exist: `scripts/niah.py`,
   `scripts/grounded_qa.py`) — NPU path first, since it's the new variable.
**Acceptance:** BENCHMARKS §4 has a throttling verdict (Δ<5% or the curve), and a
quality-parity row per engine.

## L4 — energy (R8): the NPU's actual claim to fame

**Do:** INA232 rail power (`scripts/power_probe.py`, sudo-free hwmon2) sampled during
steady-state pp512 and tg128 for CPU vs NPU on LFM3 + Qwen3.5-2B →
`J/prefill-token` and `J/decode-token` per engine.
**Acceptance:** BENCHMARKS §5 table + the one-line headline this whole repo exists for:
**"NPU is N× more energy-efficient per token at equal t/s"** (or it isn't — measure).
**Harness `scripts/energy.sh` ready** (window-filtered, idle-baseline subtracted).

## L5 — offload boundaries: where F3/F4 actually bite

**Do:** `-ngl` sweeps on Qwen3.5-4B (0,8,16,24,32) and K2-Horizon-4B (0,9,18,27,34) —
the layer count is printed at load; record the F3 VA limit hit and the F4 decode collapse
**with** the OPBATCH=2048 fix. Partial offload is the only way K2-7B can use the NPU at
all (6.4 GB > F3), so its real usable point must be found, not assumed.
**Data already in hand (2026-10-30):** parent re-test on build 11754 confirms the F4
collapse persists — 4B ngl24 tg128 **10.3** (vs 27.7 full), K2-4B ngl18 tg128 **2.6**
(vs 14.1 full) — `../ventunoq-llamacpp/results/bench_hex2_*_ngl{24,18}.csv`. The 0.28.0
matrix stages (s_q354b_ngl / s_k24b_ngl / s_k27b_ngl) add the intermediate points.
**Acceptance:** BENCHMARKS §6 curve per model + a "minimum usable `-ngl`" rule of thumb
(refine F4's "two-thirds" with the measured knee).

## L6 — CPU micro-tuning (cheap, do it in the gaps)

**Do:** on LFM3 + Qwen3.5-2B CPU: `-b 2048 -ub 1024` vs baseline; `--numa none`;
`-fa on` vs off. (Already tested: t2/4/6/8, big-only, A55, strict, `--mlock`.)
**Acceptance:** one row in BENCHMARKS §7 — "CPU flags: nothing beats / X beats `-t 4` by N%".
**Harness `scripts/cpu_flags.sh` ready.**

## L7 — serving & concurrency (the Bricks use-case)

**Do:** `llama-server` with `-np 2` and `-np 4` on Qwen3.5-0.8B/2B, both engines;
`scripts/serve_load.py -c 1/2/4` → per-client tg t/s, TTFT p50/p95, slot fairness.
**Acceptance:** BENCHMARKS §7 serving table + one-line guidance: "for N concurrent users,
run ____ on ____".
**Harnesses `scripts/serve_load.py` + `scripts/serve_matrix.sh` ready.**

## L8 — in-arch CPU controls for every NPU row (new 2026-10-30)

**Why:** the scope rule (runner CPU path only) makes the inherited §1 table *context*,
not baseline — and it has provenance cracks that disqualify it as one anyway:
1. **Missing raw data:** §1's appendix cites `bench_{lfm3,qwen35-*}_v080.csv` — those
   files are **not in the parent results/ dir**. Only `bench_q4km_{threads,pinbig}.csv`
   survive, and pinbig's CSV records `--cpu-mask 0x0f` with **strict=0**, while §1's
   header claims `--cpu-strict 1`.
2. **CPU path moved:** 0.28-era runners measure LFM3 tg128 **19.7** where §1 says 11.7,
   and build 11754 measures Qwen3.5-2B Q4_0 pp512 **54.2** vs §1's 23.7 — nobody knows
   which build made which row. Ratios against §1 are meaningless.
3. **Ratio inflation:** the README's "9–21×" mixes 0.28-era NPU numbers with v0.8.0-era
   CPU numbers on some rows.

**Do:** the matrix's `*_hex2cpu` stages measure every model in the NPU tables on b11996
with the repo pinning (`-t 4 --cpu-mask 0x0f --cpu-strict 1`); BENCHMARKS gets a
same-runner `pp512/pp8192/tg128` column beside each NPU table (§2/§3/§4); §1 stays marked
"inherited context, out of scope"; rewrite the README ratio column as NPU ÷ same-runner-CPU.
**Acceptance:** every CPU number in this repo traces to a CSV that exists, with the build
hash in the row, produced by the runner binary.

## L9 — MTP speculative decode for Qwen3.5 (new 2026-10-30)

**Why:** b11996 ships `--spec-type draft-mtp` **and** `llama_model_qwen35::graph_mtp`
(symbols in libllama.so) — the Qwen3.5 MTP head is merged upstream. Decode is the NPU's
weak axis (F4: tg128 ≈ CPU); if the MTP draft accepts at ~2+, decode could be the *next*
9–21× story, on both engines. Unsloth ships matching files: `unsloth/Qwen3.5-{0.8B,2B,
4B,9B}-MTP-GGUF` (full quant set, single-file).

**Do:** Qwen3.5-2B-MTP Q4_K_M, CPU and NPU, `--spec-type draft-mtp` vs none, `-p 512
-n 128`; record tg t/s, accept rate, draft tokens; then repeat at `-np 1` server. Check
whether the MTP head survives quantization on the NPU path (draft on HTP = extra VA
pressure — an F3 datapoint).
**Acceptance:** BENCHMARKS §8: decode Δ% per engine, accept-rate, and a verdict:
"MTP on/off by default for qwen35". If draft-mtp crashes on the HTP path → upstream
issue with the log.

---

## Execution notes

- **Order:** matrix (running) → L8 write-up → L4 energy → L3 sustained → L6 flags →
  L7 serving → L1 GSM8K parity → L9 MTP (needs its own downloads).
  One heavy thing at a time; benchmarks don't share a board with their own harness.
- **Runner hygiene:** `$HEX` (0.27) and `$HEX2` (0.28) must never share
  `LD_LIBRARY_PATH`/`ADSP_LIBRARY_PATH` — `scripts/depth_matrix.sh:hex_env` sets both
  per invocation. Violating this loads mismatched `libggml-htp-v2` + signed skels.
- **Disk:** the matrix deletes each GGUF right after its rows (DISK_LOG.md). Re-fetch
  costs ~70 s at ~16 MB/s from hf-mirror — plan L3/L4/L6 reruns around that.
- **Every row** appends to `results/` via `llama-bench --output-format csv`; failures
  leave a `.err` + a DISK_LOG line, never a silent skip.
- **Upstream watch:** hexagon backend is moving fast (16 commits/12 d). Re-pull the
  nightly before any "the NPU can't do X" conclusion; open hexagon issues worth tracking:
  #30534 (Qwen3.5-9B QAT-Q8_0 crashes HTP), #30720 (K2-Horizon-7B NPU prefill regression).
