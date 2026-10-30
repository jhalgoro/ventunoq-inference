# Roadmap — llama.cpp CPU+NPU depth program (L1–L7)

Ordered by value/effort on this board. Prefix **L** is this repo's; parent survey items
are **R** ([../ventunoq-llamacpp/ROADMAP.md](../ventunoq-llamacpp/ROADMAP.md)), MNN items
are **M** ([../ventunoq-mnn/ROADMAP.md](../ventunoq-mnn/ROADMAP.md)).
L1–L4 port parent R1/R6/R7/R8 — the *how* lives there, this file keeps the *here-now*
plan and success criteria; L5–L7 are new axes the survey had no room for.

| # | Parent | Question | Effort | Blocks on |
|---|---|---|---|---|
| L1 | R1 | Rebuild the hexagon runner so **LFM3 runs on the NPU** | low | GitHub workflow access |
| L2 | R6 | dtype matrix: Q6_K / Q8_0 / F16 / KV-dtype — CPU cost + NPU budget | low–med | disk policy |
| L3 | R7 | Deep suites: sustained/thermal, TTFT tails, NIAH, grounded-QA, full-N | medium | — |
| L4 | R8 | Energy: W and J/token per engine from INA232 (sudo-free) | low | — |
| L5 | — | NPU offload boundaries: `-ngl` sweeps, F3/F4 rule edges | low | F3 (measurement only) |
| L6 | — | CPU micro-tuning beyond `-t 4` pinning | low | — |
| L7 | — | Serving & concurrency: `llama-server -np 2/4`, both engines | medium | — |

---

## L1 — LFM3 on the NPU (highest value, fully in our control)

**Gap (parent F5):** runner `llamacpp/20261020` ships ggml 0.27.0 — predates the `lfm3`
arch. LFM3 (the board's best sub-3B model, 60% GSM8K / 65% HumanEval) is the *only*
interesting model that misses the 9–21× NPU prefill.

**Fix** — rebuild Arduino's own artifact against a newer ref (full recipe: parent
`research/bricks-faster-inference.md`):

```bash
gh workflow run build-llamacpp.yml -R <your-fork>/app-bricks-py \
  -f llama_cpp_ref=<master-ish containing lfm3 support> -f release_version=<date>
# → release tag llamacpp/<release_version>, asset llamacpp-hexagon-<release_version>.tar.gz
```

**Verify:** `llama-cli --list-devices` → `HTP0`; then LFM3-1.7B-Q4_K_M `-ngl 100` bench
(expect the tri-hybrid conv/SWA layers to map like LFM2.5 → 21×-class pp512, target
≥ ~400 t/s) + GSM8K parity re-run vs the CPU control. New runner → update `HEX` in
`scripts/env.sh` (both runners coexist; keep the old one for baseline comparability).

## L2 — dtype & quant matrix (BENCHMARKS §3)

LFM3-1.7B as the reference model, CPU pinned; every row also gets the **NPU ≤1.75 GiB
check** (F3). The interesting row is **Q6_K**: near-F16 quality that (barely) still fits
the DSP window — the only quality upgrade path that keeps the NPU prefill win. KV dtypes
(`-ctk/-ctv q8_0 / q4_0`) ride along: they change the bandwidth-bound decode *and* the
long-ctx RSS slope (§1's 8.9 KB/token). Predictions in BENCHMARKS §3 are bandwidth
scaling + the QAD dequant-cost lesson — all five must be measured, not assumed.
Disk: Q8_0/F16 are ~2–3.4 GB each → one at a time, delete-after-row, DISK_LOG.

## L3 — Deep suites (R7 port; BENCHMARKS §5)

- **Sustained/thermal** — `scripts/sustained.sh` (10-min pp+tg loop, per-minute t/s +
  temps) for CPU-pinned and NPU. Passive cooling ⇒ the curve is the deployment number.
- **TTFT tails** — pp8192/32768 repeated runs, report p50/p95, not single means.
- **NIAH on our own corpus** — haystack = parent's `docs/REPORT.md` + `BENCHMARKS.md` +
  `FAILURES.md`; needles at 0/25/50/75/100% depth, 8K/32K/128K.
- **Grounded-QA** — questions about *our own numbers* ("which engine hits 679 t/s?"),
  golden answers scripted from `results/*.csv`; exact-match scoring.
- **Full-N** — GSM8K 234 + HumanEval 164 on LFM3 + Qwen3.5-4B, both engines where the
  F3 budget allows (absorbs parent R5).
- No new model downloads beyond §3's matrix — delete-after-suite policy unchanged.

## L4 — Energy: W and J/token per engine (BENCHMARKS §6)

Probe ready: **`scripts/power_probe.py`** (INA232 hwmon2 `power1_input` µW + nsp/cpuss/
ddrss temps, sudo-free; verified 2026-10-29 — note hwmon2 has *no* temp files, temps live
in the separate `*_thermal` hwmons, the probe handles it). Conditions per engine: idle →
loaded-idle → pp burst → tg burst → 10-min sustained (same loop as L3, run them together).
Headline = **J/token = W ÷ t/s** for pp and tg. Hypothesis to beat: NPU decode looks only
1.35–1.5× in wall-clock but should look *big* in J/token; if NPU pp is 21× at similar W,
engine choice is the single biggest energy lever on this board. Open question to settle:
the ~15 W ceiling under sustained NPU+CPU+DDR load. (Sibling repo's M6 runs the same
probe against MNN — coordinate tags, don't duplicate rows.)

## L5 — NPU offload boundaries (BENCHMARKS §4)

F3 (~1.75 GiB DSP VA) and F4 (ping-pong: offload fully or ≥⅔ layers) are two measured
points with a whole design space between them. Sweep `-ngl {12,18,24,30,36}` on
Qwen3.5-4B-Q4_0 logging NPU-side MiB (from the runner's load output) against pp/tg, and
re-test F4's disaster config (`-ngl 18`) on a **Q4_0 repack** — is the ping-pong
quant-dependent? Then the F3 question from the other side: K2-4B **Q4_0** (fewer weight
bytes) — does it fit where Q4_K_M didn't? Every answer rewrites the "which model at what
quant on the NPU" row of the engine-selection guide.

## L6 — CPU micro-tuning (beyond the `-t 4` pin rule)

One-flag-one-run matrix on LFM3 + Qwen3.5-2B, CPU pinned: `-b/-ub {1024,2048}/{512,1024}`,
`-fa on/off`, `-ctk/-ctv` (shared with L2), mmap on/off, `--numa`-equivalent first-touch
(none here — single socket, skip), and `GGML_CPU_ARM_ARCH` rebuild variants only if a
flag-level win <5% (then stop — the parent's kernel audit says the instruction ceiling,
not flags, is the real limit, F8). The service's own tuning (`-b 1024 -ub 512 -fa on`) is
a baseline row, not an assumption.

## L7 — Serving & concurrency (BENCHMARKS §7)

`scripts/serve.sh`'s deployment shape was never load-tested. `llama-server -np {1,2,4}`
on both engines (NPU within the F3 budget), measure per-client tg, aggregate tg, TTFT
p50/p95; the interesting curve is where aggregate throughput *saturates* — at 4×A78C +
1 HTP this board likely caps at 2 concurrent chats, and knowing that beats discovering it
in a demo. Replay a few real multi-turn tool-call transcripts against the same server for
per-turn latency + KV growth against the measured ~8.9 KB/token.

---

## Definition of done (this repo's slice of the parent's "everything works")

- [ ] **L1** — LFM3 `-ngl 100` on the rebuilt runner: pp512 ≥ ~400 t/s, GSM8K parity.
- [ ] **L2** — BENCHMARKS §3 complete; Q6_K-on-NPU answered; quant rule updated.
- [ ] **L3** — throttle curve + NIAH recall + grounded-QA published (LFM3 + Qwen3.5-4B).
- [ ] **L4** — J/token table per engine; the 15 W ceiling confirmed or corrected.
- [ ] **L5** — `-ngl` boundary curve for 4B-class; F4 rule refined or confirmed.
- [ ] **L6** — every flag row measured; final CPU recipe frozen in README.
- [ ] **L7** — concurrency saturation point known for both engines.
