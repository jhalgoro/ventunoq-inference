# ventunoq-llamacpp-cpu+npu — llama.cpp depth program for QCS8300

Deep-dive on **one architecture**: llama.cpp's **CPU+NPU** path on the **Arduino Ventuno Q
(QCS8300)** — Arduino's hexagon runner (`llamacpp/<date>` nightly, current: 20261029),
running `--device HTP0` for the NPU and the *same binary* at `-ngl 0` as the CPU control.
Out of scope by design: the v0.8.0 CPU-only prebuilt and MNN (sibling repos cover those).
The survey said *which* engines run ([`../ventunoq-llamacpp`](../ventunoq-llamacpp)); this
repo answers *how far this one goes*: dtype matrix, long context, sustained/thermal
behaviour, energy (W, J/token), NPU offload boundaries, and CPU micro-tuning.

## What inspired this repo — the sources, with links

- **The NPU path itself**: Arduino **“Bricks”** —
  <https://docs.arduino.cc/software/app-lab/bricks/about-bricks> — whose `arduino:llm` brick
  ships the `llamacpp-npu-runner` image for *this exact board* (ventunoq), i.e. the
  `llamacpp/20261020` Hexagon runner everything in §2 measures. Parent research doc:
  [`../ventunoq-llamacpp/research/bricks-faster-inference.md`](../ventunoq-llamacpp/research/bricks-faster-inference.md).
  The same repo's `build-llamacpp.yml` workflow is what **L1** pulls.
- **The five Round-3 NPU articles** (surveyed in parent
  [research/alt-engines.md](../ventunoq-llamacpp/research/alt-engines.md) Round 3 — none
  names an engine runnable here, three shape this repo's method):
  - **llm.npu (ASPLOS ’25)** — <https://arxiv.org/html/2407.05858v2> (code:
    <https://github.com/UbiquitousLearning/mllm>) — the only LLM-on-Qualcomm-NPU system
    published; its out-of-order CPU/NPU subgraph scheduling is the design reference for
    **L5** (F4 ping-pong), its shadow-outlier W8A8 the accuracy-recovery note for any future
    NPU quant path.
  - *LLM Inference at the Edge under sustained load* — <https://arxiv.org/html/2603.23640v1>
    — the sustained-load/thermal methodology **L3** adopts (their phones throttle hard; this
    board is passively cooled).
  - *NPU Hardware Evaluation v1.0* — <https://arxiv.org/html/2609.13166> — corroborates
    “one DSP job at a time” (the F4 rule) and shows the IQ-9075 running the same Kryo/A78C
    cluster as this board.
  - *localaimaster 2026 NPU comparison* — <https://localaimaster.com/blog/npu-comparison-2026>
    and *Benchmarking Edge AI Platforms (USC)* — <https://arxiv.org/html/2409.14803v1> — the
    negative screen: every Qualcomm cell is QNN → parent F1; the Intel/Apple parts are
    foreign hardware. Kept as evidence that **no external engine beats the llama.cpp path here**.
- **The pinning rule's origin**: llama.cpp — <https://github.com/ggml-org/llama.cpp> +
  llama-bench — <https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench>;
  the `-t 4 --cpu-mask 0xf --cpu-strict 1` rule this repo inherits was measured in the
  parent's thread/pinning study (`../ventunoq-llamacpp/results/bench_q4km_{threads,pinbig}.csv`)
  — on the v0.8.0 CPU-only build, which is **context only**: this repo never benchmarks it.

## Status at a glance

Spun out of the parent's ROADMAP: **L1–L9 here ≙ R1, R6, R7, R8 there** plus the
NPU-boundary and serving work the survey had no room for.

| Track | Status |
|---|---|
| Inherit pinned-CPU + NPU baselines | ✅ [BENCHMARKS.md](BENCHMARKS.md) §1–2 (raw CSVs stay in the parent) |
| L1 — LFM3 on NPU (runner rebuild) | 🟡 measured (pp512 1033 t/s) — GSM8K parity open; [ROADMAP](ROADMAP.md) L1 |
| L2 — dtype matrix (Q6_K/Q8_0/F16/int8, KV dtypes) | 🟡 matrix running |
| L3 — deep suites (sustained, NIAH, grounded-QA, full-N) | ⏳ L3 |
| L4 — energy: INA232 W + J/token per engine | ⏳ L4 — probe ready: `scripts/power_probe.py` |
| L5 — NPU offload boundaries (`-ngl` sweeps, F3/F4 rules) | 🟡 matrix running |
| L6 — CPU micro-tuning beyond `-t 4` pinning | ⏳ L6 |
| L7 — serving & concurrency (`llama-server -np`) | ⏳ L7 |
| L8 — same-runner CPU controls for every NPU row | 🟡 matrix `*_hex2cpu` stages |
| L9 — MTP spec-decode for Qwen3.5 (`draft-mtp`, new in b11996) | ⏳ L9 |

## Binary strategy — no duplication

One runner does both engines. This repo vendors nothing: the hexagon runner is fetched
sha256-pinned to `/tmp/opencode/` (recipe in `scripts/env.sh`; the `llamacpp/<date>`
assets are public, no GitHub auth), and `env.sh` sets the library paths + service tuning:

```bash
source scripts/env.sh          # sets $HEX2 (current runner, lfm3 ✓) and $HEX (0.27 control)
$HEX2/bin/llama-bench -m model.gguf -ngl 0 -t 4 --cpu-mask 0xf --cpu-strict 1 -p 512 -n 128 -r 3   # CPU ctrl
$HEX2/bin/llama-cli --list-devices   # must print HTP0: Hexagon
```

Never mix `$HEX` and `$HEX2` lib dirs in one process (mismatched `libggml-htp-v2` + skels).
The parent's v0.8.0 prebuilt (`$LLAMA`) stays exported in `env.sh` for cross-referencing
inherited docs only — it is not benchmarked here.

## Inherited baselines (the numbers every new measurement compares against)

Regime: `-t 4 --cpu-mask 0xf --cpu-strict 1` (pinned big cores) vs `-ngl 100` on the HTP.
CPU column below is **inherited v0.8.0 context** (out of scope, L8 re-measures the same
model set on the runner itself).

| Model (Q4) | CPU pp512 / tg128 *(v0.8.0, ctx)* | NPU pp512 / tg128 | NPU speed-up (pp / tg) |
|---|---|---|---|
| Qwen3.5-0.8B-Q4_0 | 148.7 / 31.9 | 1366 / 43.0 | 13.6× / 1.35× |
| Qwen3.5-2B-Q4_0 | 69.8 / 16.1 | 863 / 24.3 | 14.9× / 1.51× |
| LFM2.5-2.6B-Q4_K_M | 40.1 / 13.1 | 679 / 17.7 | 21.1× / 1.35× |
| LFM3-1.7B-Q4_K_M | 46.8 / 19.7 *(runner `-ngl 0`)* | 1033 / 25.6 ✅ L1 | 22× / 1.30× |

Full inherited tables (threads/pinning, long ctx, partial offload, quality parity):
[BENCHMARKS.md](BENCHMARKS.md) §1–2.

## The board (condensed — full story in the parent repo)

4×Cortex-A78C (cpu0‑1 @2112 MHz, cpu2‑3 @2361.6 MHz) + 4×A55 (cpu4‑7); ARMv8.2 with
`dotprod` + FP16-storage, **no i8mm, no bf16, no SVE, no FMLAL/FMLSL-vector** (parent F8);
~14 GiB usable RAM; kernel `6.8.0-1084-qcom`. NPU = Hexagon V75 via the upstream Hexagon
backend (FastRPC/FBCL — **not** QNN: every QNN path dies at `HAP_mmap2`, parent F1); DSP VA
budget ~1.75 GiB (parent F3). Passively cooled (`emc2305` fan faulted) — the L3 sustained
curve matters. Energy telemetry: **INA232 at hwmon2, sudo-free** (`power1_input` µW).

## Project conventions (inherited from the parent)

- **One architecture under test**: llama.cpp CPU+NPU on the hexagon runner. CPU rows =
  the runner's own `-ngl 0` path. v0.8.0 CPU-only and MNN are **not evaluated here** —
  sibling repos own them; inherited §1 numbers are context, never a baseline.
- **Evidence or it didn't happen**: every number in a `.md` has a raw log/CSV in `results/`
  and an exact command line. Repro first, conclusions second.
- **Disk policy**: model files exist only during their own test; delete right after, log
  sha256 + `df` into [results/DISK_LOG.md](results/DISK_LOG.md) (target / < 50%).
- **Global numbering across the ventunoq family**: parent owns F1–F8 / R1–R8;
  `ventunoq-mnn` owns F9–F19 / M1–M9; **this repo reserves F20+ (failures) and L1–L7+
  (roadmap)**.
- **Baselines are pinned**: every CPU control is `-t 4 --cpu-mask 0xf --cpu-strict 1`
  on the same binary the NPU run uses; un-pinned numbers are never the comparison base.
- **Sudo-free**, always.

## Repository map

```
README.md            this file
BENCHMARKS.md        inherited baselines (§1–2) + planned depth sections (§3+)
FAILURES.md          F1–F8 inheritance table + this repo's F20+
ROADMAP.md           L1–L9 — the depth program
scripts/             env.sh · depth_matrix.sh · power_probe.py · energy.sh · sustained.sh
                     · cpu_flags.sh · serve_matrix.sh · serve_load.py
results/             raw logs + DISK_LOG.md audit trail
models/              gitignored — transient, delete-after-suite policy
```

## Related projects

- [`../ventunoq-llamacpp`](../ventunoq-llamacpp) — the engine survey (parent): F1–F8,
  R1–R8, all vendored binaries and inherited raw CSVs.
- [`../ventunoq-mnn`](../ventunoq-mnn) — MNN engine work; **out of scope for this repo**,
  listed only because its energy harness reuses this repo's `power_probe.py`.
