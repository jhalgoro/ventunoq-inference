# ventunoq-inference

**Can a single edge board run real LLMs well — and on which engine, at what speed, at what
energy cost, and where does each path break?**

This is the umbrella repo for a measured, evidence-first survey of **LLM inference engines on
the Arduino Ventuno Q** (Qualcomm QCS8300 "Monza": 4×Cortex-A78C + 4×A55, Hexagon V75 NPU,
Adreno 623 GPU via Turnip/Vulkan, 16 GiB unified LPDDR, Ubuntu 24.04 aarch64 — all work
runs **sudo-free**). Every claim in these projects is backed by a raw log/CSV in `results/`
and an exact command line: *evidence or it didn't happen.*

## The family

Four subprojects share one numbering system for failures (F#) and roadmap tracks (R/M/L#),
and each owns a slice of the question:

| Directory | Role | Owns |
|---|---|---|
| [`ventunoq-llamacpp/`](ventunoq-llamacpp/README.md) | **The engine survey of record.** llama.cpp on every accelerator: CPU, Hexagon NPU (two stacks), Adreno GPU — plus screening of alt engines (MNN, mistral.rs, Strata). Two engines work and are benchmarked; two paths are blocked with reproducible root causes. | F1–F8 / R1–R8 |
| [`ventunoq-llamacpp-cpu+npu/`](ventunoq-llamacpp-cpu+npu/README.md) | **llama.cpp depth program.** How far the CPU+NPU path (Arduino's hexagon runner) goes: dtype matrix, long context, sustained/thermal behaviour, energy (W, J/token via INA232), NPU offload boundaries, serving & concurrency. | F20+ / L1–L9 |
| [`ventunoq-mnn/`](ventunoq-mnn/README.md) | **MNN depth program.** alibaba/MNN as the board's CPU engine — the first engine measured *faster than llama.cpp CPU* here (+9% pp / +30% tg @0.8B): builds, quality parity, kernel-safety audit, model matrix, LFM3 upstream question. | F9–F19 / M1–M9 |
| [`ventunoq-mnn-tandem/`](ventunoq-mnn-tandem/DESIGN.md) | **Tandem design study (no builds).** Feasibility analysis of CPU+GPU+NPU cooperative inference in MNN: all three backends exist, but MNN binds one forward type per session — so "tandem" is a scheduler to build on the Module API, and it only pays in throughput, model capacity, or structural asymmetry. | architecture |

## Headline findings

- **The NPU is real**: LFM2.5-2.6B-Q4_K_M prefills at **679 t/s** on the Hexagon runner vs
  40 t/s on pinned CPU — **21×** — with GSM8K quality parity. LFM3-1.7B on NPU: 1033 t/s pp512.
- **Decode is bandwidth-bound everywhere**: all engines share one 16 GiB LPDDR (no VRAM, no
  NPU RAM), so the NPU buys only 1.35–1.5× on decode. The NPU's win is prefill and
  long-context RAG.
- **llama.cpp is not automatically fastest**: MNN's ARM82 fp16 path beats it on CPU
  (Qwen3.5-0.8B: 162.7/41.5 vs 148.7/31.9 t/s) — but LFM3 and the NPU belong to llama.cpp alone.
- **LFM3-1.7B is the best sub-3B model on this board**: 49 pp / 11.7 tg t/s, 60% GSM8K /
  65% HumanEval — near-triple Qwen3.5-2B's quality at equal prefill.
- **Two hard walls, fully root-caused**: QNN/QAIRT dies at `HAP_mmap2` because the kernel
  lacks the vendor `smmu_proxy` service (F1); the Vulkan/Turnip GPU path needs
  `storageBuffer8BitAccess` the driver doesn't advertise and segfaults instead of erroring (F2).
- **One hard NPU limit**: the CDSP maps through a 32-bit IOMMU window measured at
  **~1.75 GiB** — full offload works up to ~2B-class Q4; partial offload usually *backfires*
  (ping-pong, F4): offload fully or keep ≥⅔ of layers on the NPU.
- **Pinning rule**: `-t 4 --cpu-mask 0xf --cpu-strict 1` (+75% decode vs default `-t 8`) is
  the baseline regime across all repos.

## Shared conventions

- **Evidence or it didn't happen** — every number in a `.md` has a raw log/CSV in `results/`.
- **Disk policy** — model weights (`*.gguf`, `*.mnn.weight`, exported model dirs) are
  transient: fetched one at a time, deleted right after their suite, with a sha256+`df` audit
  trail in each repo's `results/DISK_LOG.md`. They are therefore *not* in git history.
- **Baselines are pinned** — CPU controls always run pinned to the big cores on the same
  binary the NPU run uses.
- **Sudo-free, always** — the FastRPC nodes are open to the `arduino` user.

## Repository layout

```
ventunoq-llamacpp/          engine survey — llama.cpp CPU/NPU/GPU + alt-engine screening
ventunoq-llamacpp-cpu+npu/  llama.cpp depth — dtype, thermal, energy, offload boundaries
ventunoq-mnn/               MNN depth — CPU engine, quality parity, kernel audit
ventunoq-mnn-tandem/        tandem CPU+GPU+NPU feasibility design (DESIGN.md)
```

Each subproject follows the same internal map: `README.md` (story) · `BENCHMARKS.md`
(numbers) · `FAILURES.md` (root causes) · `ROADMAP.md` (fix paths) · `scripts/`
(reproducible harnesses) · `results/` (raw data) · `research/` (sourced notes).

## License

MIT (scripts, docs, results). Fetched code and model weights remain under their upstream
terms — see each subproject's README for details.
