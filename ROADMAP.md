# Roadmap — MNN depth program (M1–M9)

Ordered by value/effort on this board. Prefix **M** is this repo's; parent survey items are
**R** ([../ventunoq-llamacpp/ROADMAP.md](../ventunoq-llamacpp/ROADMAP.md)), llama.cpp depth
items are **L** ([../ventunoq-llamacpp-cpu+npu/ROADMAP.md](../ventunoq-llamacpp-cpu+npu/ROADMAP.md)).

| # | Question | Effort | Blocks on |
|---|---|---|---|
| M1 | Is MNN int4 as *good* as it is fast? (quality parity) | low | — |
| M2 | How far up the size range does the win hold? (4B, int8) | low | — |
| M3 | Does `gguf2mnn` make our existing GGUFs first-class? | low | — |
| M4 | Kernel knobs: KleidiAI, ARM82-off, OpenCL/Vulkan probes | low | F2 for GPU |
| M5 | Long context: pp curves + KV memory vs llama.cpp's 8.9 KB/tok | low | — |
| M6 | Sustained + energy: throttle curve, W and J/token vs llama.cpp | low | reuse L4 probe |
| M7 | Upstream LFM2/LFM3 into `model_mapper` (F12) | **high** | upstream appetite |
| M8 | Serving shape: subprocess driver, or concede to llama-server? | medium | — |
| M9 | Decision table: when MNN, when llama.cpp (the repo's product) | trivial | M1–M6 |

---

## M1 — Quality parity: the number that decides adoption

MNN's int4 carries fp16 per-block scales (≈Q4_K-class on paper) — but the whole +30% decode
claim is worthless if answers degrade. Measure, same corpora as the parent repo:

- **GSM8K** via `scripts/gsm8k_mnn.py` (drives `llm_demo` as a subprocess, F14; same
  questions/strict scorer as parent `scripts/gsm8k_eval.py`, N=50).
- **Perplexity** on `bench/gatsby.txt` (parent corpus — copy it) for a continuous signal;
  MNN ships no ppl tool, so score logprobs via the engine (or drop this leg and lean on
  HumanEval-style behavioral diffs on 20 tasks).
- Compare: MNN int4 vs llama.cpp **Q4_0** (speed rival) **and Q4_K_M** (quant rival),
  same models 0.8B/2B.

**Success =** a row in BENCHMARKS §4.1 per model; if MNN quality ≤ Q4_0 − noise, the CPU
recommendation stays llama.cpp regardless of speed.

## M2 — Size range: `taobao-mnn/Qwen3.5-4B-MNN` (+ int8 export)

- 4B-MNN (~2.4 GB weights int4) fits RAM comfortably; expect the bandwidth-bound crossover
  to shrink the win toward 0 (trend so far: +30% → +19% decode at 2B).
- Also export **int8** (`llmexport.py --weight_quant_bits 8` — needs a host with torch; if
  no x86 host is handy, run on-board with the venv deps, disk-policy bound) and compare
  speed/quality vs int4 and vs llama.cpp Q8_0 (parent L2 runs the same dtype question —
  cross-reference, don't duplicate measurements).
- 9B only if disk <50% allows during the suite (delete-after-suite, DISK_LOG).

## M3 — `gguf2mnn.py`: our GGUFs → MNN

- Convert `unsloth/Qwen3.5-2B-GGUF` Q4_K_M → MNN; compare vs `taobao-mnn` export (same
  weights, different quantizer): speed should match; quality delta quantizes the exporter.
- Attempt `LiquidAI/LFM3-1.7B-GGUF` → **expect hard failure** (F12); capture the traceback
  as evidence for the M7 upstream issue.
- Note the arch map is `general.architecture`-driven (`llama` branch) — document exactly
  where qwen3.5 GGUFs land (`Arch:` print + which template).

## M4 — Kernel & backend probes (each is a build flag + one bench run)

| Probe | Build | Expect |
|---|---|---|
| KleidiAI on | `-DMNN_KLEIDIAI=ON -DMNN_KLEIDIAI_DEFAULT_ON=ON` | i8mm kernels must **not** dispatch (no i8mm); dotprod-only variants may; verify via runtime probe line + speed delta |
| ARM82 off (fp32 kernels) | `-DMNN_ARM82=OFF` | should land near `-c 1` numbers; quantifies the fp16 kernel contribution beyond the flag |
| OpenCL / Vulkan | defaults | quick probe only: `-a opencl` / `-a vulkan` on 0.8B; expect no usable driver or F2-class faults — record, don't fight |

## M5 — Long context

`llm_bench -p {2048,8192,16384,32768}` + `-pg` reuse runs on 0.8B/2B; RSS per KV token via
`/usr/bin/time -v` at 2K vs 32K ctx (`max_new_tokens`/`ctx` in `config.json`). Compare
against llama.cpp's measured ≈8.9 KB/token (LFM3) and Qwen-class 2.6–3.8× that. Also: does
MNN spill/realloc or fail at 32K? (parent pp32768 = 30.8 t/s CPU is the bar.)

## M6 — Sustained + energy (don't reinvent — reuse the L4 probe)

Run the sibling repo's `../ventunoq-llamacpp-cpu+npu/scripts/power_probe.py` (INA232 hwmon2
+ nsp/cpuss/ddrss temps) alongside a 10-min MNN fp16 loop; report avg W and **J/token** for
pp and tg next to llama.cpp-pinned and NPU rows. Hypothesis: MNN fp16 does *less work per
joule* than llama.cpp f32-compute — the decode win should look even bigger in J/token.

## M7 — Upstream: LFM2/LFM3 in `model_mapper` (highest ceiling, real work)

What it takes: map LFM's blocks (`feed_forward_network`, conv-gate / short conv attention
of LFM2/LFM3) onto MNN geometry ops + a `gguf2mnn` template; validate logits vs llama.cpp
token-by-token on a shared prompt (the parent's gsm8k parity method). First move: upstream
issue citing F12 evidence + this repo's numbers (an engine that beats llama.cpp on ARM82
adding Liquid support is a sellable ask). **Timebox**: spike the GGUF→mapper path for
LFM2.5-2.6B (simpler arch) before committing to LFM3's tri-hybrid.

## M8 — Serving: concede or build?

`llm_demo` has no HTTP API (F14). Options: (a) thin FastAPI-free stdlib socket wrapper
around a persistent `llm_demo` process — medium effort, fragile stdin framing; (b) declare
serving a llama.cpp capability and keep MNN for batch/bench/CLI. Decide after M1 lands;
default is (b).

## M9 — The decision table (repo output, not a task)

Current draft — M1–M6 refine the numbers, nothing else changes it:

| Workload | Engine | Command shape |
|---|---|---|
| CPU chat/code, Qwen-class ≤2B | **MNN fp16** | `taskset -c 0-3 llm_bench/llm_demo -t 4 -c 2` |
| CPU, LFM2/LFM3 | llama.cpp (MNN lacks arch, F12) | `-t 4 --cpu-mask 0xf --cpu-strict 1` |
| Anything NPU (prefill-heavy ≤1.75 GiB) | llama.cpp Hexagon runner (F13) | `-ngl 100` recipe in parent |
| HTTP serving / concurrency | llama.cpp `llama-server` | parent repo §5 |
