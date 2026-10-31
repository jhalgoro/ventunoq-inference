# ventunoq-mnn — MNN as this board's CPU engine

Deep-dive research on [alibaba/MNN](https://github.com/alibaba/MNN) (3.x, `MNN_BUILD_LLM`)
on the **Arduino Ventuno Q (QCS8300)**: quality parity, more models, kernel/backend knobs,
long context, sustained behaviour, energy, and the LFM3 upstream question.

Spun out of [`../ventunoq-llamacpp`](../ventunoq-llamacpp) Round 2
([research/alt-engines.md](../ventunoq-llamacpp/research/alt-engines.md)), where MNN became
the first engine measured **faster than llama.cpp's CPU** on this board.

## What inspired this repo — the sources, with links

- **The article that started it**: Allen Kuo, *“Edge Inference Engine: One Native API Across
  CPU, GPU, NPU, and RISC-V”* — <https://allenkuo.medium.com/edge-inference-engine-one-native-api-across-cpu-gpu-npu-and-risc-v-8e4166d1816f>.
  Its “ten providers” table was the Round-2 candidate screen; **MNN was the only provider
  that runs LLMs on this board** (the rest died on F1/QNN, F2/Vulkan, missing hardware, or
  missing model coverage — full screen in parent `research/alt-engines.md` Round 2).
- **The engine**: alibaba/MNN — <https://github.com/alibaba/MNN> (built on-board via
  `scripts/build_mnn.sh`; LLM tooling docs under `transformers/llm.md`).
- **The pre-converted models** that made round one possible without a torch host: HF org
  **taobao-mnn** — [`Qwen3.5-0.8B-MNN`](https://huggingface.co/taobao-mnn/Qwen3.5-0.8B-MNN) ·
  [`Qwen3.5-2B-MNN`](https://huggingface.co/taobao-mnn/Qwen3.5-2B-MNN) (int4 + fp16 scales;
  M2 adds [`Qwen3.5-4B-MNN`](https://huggingface.co/taobao-mnn/Qwen3.5-4B-MNN)).
- **The F8 landmine it had to dodge**: the mistral.rs SIGILL
  ([parent F8](../ventunoq-llamacpp/FAILURES.md)) that made a pre-run kernel audit
  ([research/kernel-audit.md](research/kernel-audit.md)) mandatory before executing anything.

## Status at a glance

| Track | Status |
|---|---|
| Build (on-board, ~4 min) | ✅ `scripts/build_mnn.sh` |
| Speed vs llama.cpp CPU | ✅ **+9% pp / +30% tg @0.8B**, +8% / +19% @2B (fp16, big-core pinned) |
| Quality parity vs llama.cpp | 🛠️ harness ready (`scripts/quality_matrix_mnn.sh`) — numbers ⏳ [M1](ROADMAP.md) |
| Model matrix (4B+, int8, GGUF import) | ⏳ [M2, M3](ROADMAP.md) |
| LFM2/LFM3 | ❌ not in `model_mapper` (only `lfm2_audio`) — llama.cpp stays the only LFM3 engine ([F12](FAILURES.md)); upstream work: [M7](ROADMAP.md) |
| NPU (MNN Hexagon backend) | ❌ Android + QNN SDK → parent **F1** (`smmu_proxy` kernel gap) ([F13](FAILURES.md)) |
| GPU (OpenCL / Vulkan) | ❌ parent F2 territory; quick probe only ([M4](ROADMAP.md)) |
| Serving (HTTP API) | ⏳ [M8](ROADMAP.md) — llama.cpp `llama-server` remains the serving shape |

## Why MNN is safe on this SoC (the F8 context)

The mistral.rs release binary **SIGILLs** here because QCS8300 does not implement the
FMLAL/FMLSL (vector) instructions its build assumes
([parent F8](../ventunoq-llamacpp/FAILURES.md)). MNN cannot step on that mine by accident:

- ARM82 fp16 hot kernels are hand-written `.S` using only `fmla / fmul / fcvt / sdot` —
  every one verified executable on QCS8300.
- The only FMLAL code (`vfmlalq_lane_*` in `source/backend/arm82/Arm82Functions.cpp`) is
  **compile-time gated** behind `__ARM_FEATURE_FP16_FML`, which the default
  `-march=armv8.2-a+fp16` build does not enable.
- KleidiAI (needs i8mm — absent on this SoC) is `MNN_KLEIDIAI=OFF` by default.
- Runtime probe agrees: `The device supports: i8sdot:1, fp16:1, i8mm:0, sve2:0, sme2:0`.

Full source audit + repro one-liners: [research/kernel-audit.md](research/kernel-audit.md).

## Quick start

```bash
scripts/build_mnn.sh                 # clone + cmake + ninja llm_demo llm_bench (~4 min)

# model (pre-converted int4+fp16-scale, no torch needed) + text-only patch:
scripts/dl_model.sh Qwen3.5-0.8B-MNN            # taobao-mnn/* on HF; logs sha256+df
scripts/text_only.py models/Qwen3.5-0.8B-MNN    # is_visual:false (F9) — llm_bench skips this

# the standard bench (same semantics as llama-bench -pg 512,128, big-core pinned):
scripts/bench_mnn.sh models/Qwen3.5-0.8B-MNN q08

# interactive sanity:
echo "What is the capital of France?" | taskset -c 0-3 "$MNN/build/llm_demo" models/q08-text/config.json

# quality suite (M1) — same suites/items as ../ventunoq-llamacpp-cpu+npu:
python3 scripts/fetch_bench.py                    # bench/ corpora (committed once)
scripts/quality_matrix_mnn.sh q_qwen08_smoke      # plumbing check, every suite at tiny n
scripts/quality_matrix_mnn.sh                     # full matrix (hours; model deleted after)
```

`-c 2` = fp16 ("Low") is the whole story: fp32 decode is **−65%** ([F10](FAILURES.md)).
4 threads on the big cores (`taskset -c 0-3`) is the decode optimum; 8 threads raises
prefill but drags decode below llama.cpp's.

## Headline numbers (fp16, t4, `-pg 512,128`; llama.cpp v0.8.0 b11972 `-t 4 --cpu-mask 0xf --cpu-strict 1`)

| Model | llama.cpp Q4_0 | llama.cpp Q4_K_M | **MNN int4 (fp16)** | gain vs Q4_0 |
|---|---|---|---|---|
| Qwen3.5-0.8B pp512 | 148.7 | 106.2 | **162.7 ± 0.1** | **+9.4%** |
| Qwen3.5-0.8B tg128 | 31.9 | 27.0 | **41.5 ± 0.1** | **+30%** |
| Qwen3.5-2B pp512 | 69.8 | — | **75.7 ± 0.1** | **+8.4%** |
| Qwen3.5-2B tg128 | 16.1 | — | **19.2 ± 0.0** | **+19%** |

Peak RSS 930 MB for the 449 MB model (2.1×). Details & sweeps: [BENCHMARKS.md](BENCHMARKS.md).

## The board (condensed — full story in the parent repo)

4×Cortex-A78C (cpu0‑1 @2112 MHz, cpu2‑3 @2361.6 MHz) + 4×A55 (cpu4‑7); ARMv8.2 with
`dotprod` + FP16-storage, **no i8mm, no bf16, no SVE, no FMLAL/FMLSL-vector** (F8);
~14 GiB usable RAM; kernel `6.8.0-1084-qcom`. NPU (Hexagon V75) is reachable only through
llama.cpp's Hexagon runner — every QNN-runtime path (including MNN's own Hexagon backend)
dies at `HAP_mmap2` for lack of the `smmu_proxy` kernel service (parent F1). Vulkan broken
(parent F2). Toolchain present: gcc/g++ 13.3, cmake, ninja, gdb. All work runs **sudo-free**.

## Project conventions (inherited from the parent)

- **Evidence or it didn't happen**: every number in a `.md` has a raw log/CSV in `results/`
  and an exact command line. Repro first, conclusions second.
- **Disk policy**: model files exist only during their own test; delete right after, log
  sha256 + `df` into [results/DISK_LOG.md](results/DISK_LOG.md) (`scripts/dl_model.sh`
  appends the download entry for you).
- **Global numbering across the ventunoq family**: parent `ventunoq-llamacpp` owns
  F1–F8 / R1–R8; this repo reserves **F9–F19** (failures) and **M1–M9** (roadmap);
  `ventunoq-llamacpp-cpu+npu` reserves F20+ / L1+.
- **Baselines are pinned**: llama.cpp comparisons use `-t 4 --cpu-mask 0xf --cpu-strict 1`;
  MNN's equivalent is `taskset -c 0-3` (verified equivalent regime, see BENCHMARKS method).

## Repository map

```
README.md            this file
BENCHMARKS.md        measured numbers + method equivalences
FAILURES.md          F9+ — what breaks and why (MNN-specific)
ROADMAP.md           M1–M9 — the deep-dive agenda
bench/               quality corpora (scripts/fetch_bench.py; same items as sibling repo)
scripts/             build_mnn.sh · dl_model.sh · text_only.py · bench_mnn.sh
                     fetch_bench.py · llm_eval.cpp · quality_eval_mnn.py ·
                     quality_matrix_mnn.sh · quality_parity.py
results/             raw logs + DISK_LOG.md audit trail
research/            kernel-audit.md (F8-safety source audit)
MNN/                 gitignored — source clone + build (scripts/build_mnn.sh)
models/              gitignored — transient, delete-after-suite policy
```

## Related projects

- [`../ventunoq-llamacpp`](../ventunoq-llamacpp) — the engine survey (llama.cpp CPU+NPU,
  MNN/mistral.rs/Strata screening, FAILURES F1–F8, ROADMAP R1–R8).
- [`../ventunoq-llamacpp-cpu+npu`](../ventunoq-llamacpp-cpu+npu) — the llama.cpp depth
  program (dtype matrix, long-ctx, sustained/thermal, energy, NPU offload boundaries).
