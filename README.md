# ventunoq-llamacpp

**llama.cpp on the Arduino Ventuno Q (Qualcomm QCS8300) — every accelerator, measured,
with every failure root-caused.**

Started as a Liquid AI **LFM3** evaluation (announced 2026-10-27) and grew into a
full engine survey of the board: CPU, Hexagon NPU (two independent stacks), and the
Adreno GPU via Vulkan/Turnip. Two engines work and are benchmarked; two paths are
blocked, each with a reproducible root cause, evidence, and a concrete fix path.

## Engine status at a glance

| Engine | Stack | Status | Evidence |
|---|---|---|---|
| **CPU** (4×A78 + 4×A55, NEON+dotprod) | llama.cpp v0.8.0 prebuilt (`armv8.2_2`) | ✅ **works** — baseline for everything | [BENCHMARKS §1–2](BENCHMARKS.md) |
| **NPU** (Hexagon V75 HTP) | Arduino `llamacpp/20261020` runner — upstream **Hexagon backend** over classic fastrpc | ✅ **works** — prefill **9–21×** CPU, decode 1.35–1.5×, quality parity, ≤ ~1.75 GiB NPU-side | [BENCHMARKS §3](BENCHMARKS.md) |
| **NPU via QNN/QAIRT** | Qualcomm AI Runtime 2.51, HTP skel | ❌ **blocked** — kernel lacks the vendor `smmu_proxy` service (`HAP_mmap2` → `paddr 0x0`) | [FAILURES F1](FAILURES.md) |
| **GPU** (Adreno 623, Turnip/Mesa 25.2.8, Vulkan 1.3) | llama.cpp v0.8.0 `-DGGML_VULKAN=ON` | ❌ **blocked** — shaders demand `storageBuffer8BitAccess` the device doesn't advertise; driver segfaults instead of erroring | [FAILURES F2](FAILURES.md) |

Headline results (details in [BENCHMARKS.md](BENCHMARKS.md)):

- **LFM3-1.7B is the best sub-3B model on this board**: 49 pp / 11.7 tg t/s,
  60% GSM8K / 65% HumanEval — near-triple Qwen3.5-2B's quality at equal prefill.
- **The NPU is real and worth it**: LFM2.5-2.6B-Q4_K_M prefills at **679 t/s**
  (572 @8k) vs 40 t/s on a pinned CPU core set — 21×; and GSM8K parity holds
  (NPU 60.0% vs CPU 55.0%, N=20 — one question apart).
- **Decode is bandwidth-bound on every engine** (one shared 16 GiB LPDDR):
  the NPU buys 1.35–1.5× on decode, no more.
- **One hard NPU limit**: the CDSP maps through a **32-bit IOMMU window measured
  at ~1.75 GiB** (`npu/cumap_probe.c`) — full offload works up to ~2B-class Q4;
  K2-Horizon-4B aborts mid-load.

## Quick start (what works today)

```bash
# CPU — llama-server on the vendored v0.8.0 binaries (LFM3 needs ≥ v0.8.0):
scripts/serve.sh models/LFM3-1.7B-Q4_K_M.gguf          # port 8181

# CPU — best interactive flags on this board (big-core pinning, +75% decode):
llama.cpp/llama-bench -m <gguf> -ngl 0 -t 4 --cpu-mask 0xf --cpu-strict 1

# NPU — Arduino's hexagon runner (fetch + env recipe: research/bricks-faster-inference.md):
export LD_LIBRARY_PATH=/tmp/opencode/libfix:$PKG/lib ADSP_LIBRARY_PATH=$PKG/lib
export GGML_HEXAGON_OPBATCH=2048 GGML_HEXAGON_MBUF=256,256
$PKG/bin/llama-server -m <gguf> -ngl 100 -c 8192 -t 4 --cpu-mask 0x0f -b 1024 -ub 512 -fa on

# Reproduce the NPU benchmark cycles / quality parity check:
scripts/test_hexagon.sh && scripts/test_hexagon2.sh && scripts/eval_hex_quality.sh
```

Model sources used (Q4 quants; official `Qwen/*-GGUF` are gated, so **unsloth** mirrors
were used for Qwen): `LiquidAI/LFM3-1.7B-GGUF`, `LiquidAI/LFM2.5-2.6B-GGUF`,
`unsloth/Qwen3.5-{0.8B,2B,4B,9B}-GGUF`, `IFM/K2-Horizon-{3.7B,7B}-GGUF`.
Disk policy: every GGUF deleted immediately after its tests — full sha256+`df` audit
trail in `results/DISK_LOG.md`.

## The board

| | |
|---|---|
| SoC | Qualcomm QCS8300 "Monza" (Arduino Ventuno Q), Linux 6.8.0-1084-qcom, Ubuntu 24.04.5 aarch64 |
| CPU | 4× Cortex-A78-class @2.36 GHz + 4× Cortex-A55 @1.96 GHz; ARMv8.2 dotprod, **no** SVE/i8mm/bf16 |
| NPU | Hexagon **V75** HTP; FastRPC nodes open to the `arduino` user (no sudo anywhere in this project) |
| GPU | Adreno 623 via **Turnip** (Mesa 25.2.8, Vulkan 1.3.318, headless on `renderD128`) |
| RAM | 16 GiB unified LPDDR (~14.9 GiB to Linux) — **no VRAM, no NPU RAM**; memory topology in `research/memory-map.md` |

## Repository map

```
README.md            this file
BENCHMARKS.md        everything that works, with numbers + raw-data pointers
FAILURES.md          every failed pathway: symptom → root cause → evidence → status
ROADMAP.md           what is needed to make each blocked thing work (per engine)
docs/REPORT.md       the original chronological field log (2026-10-28/29), kept as-is
scripts/             working harnesses: bench suites, GSM8K/HumanEval evals, NPU/GPU cycles
results/             raw CSV/JSON/logs + run markers + DISK_LOG.md audit trail
npu/                 QNN bring-up log (NOTES.md) + C probes: cumap_probe.c (VA ceiling),
                     mmap_probe.c (fastrpc MMAP works), soc_scan.c, spoof_soc.c
research/            sourced research notes: LFM3 announcement, Bricks/NPU runner recipe,
                     memory map, K2-Horizon
llama.cpp/           llama.cpp v0.8.0 b11972 Ubuntu-arm64 binaries (CPU+RPC) — on disk,
                     git-ignored (fetch recipe in ROADMAP.md)
bench/               eval datasets (GSM8K test.jsonl, HumanEval.jsonl, gatsby.txt)
vendor/              websockets (vendored — the board has no pip)
```

## Further documentation

- [BENCHMARKS.md](BENCHMARKS.md) — CPU & NPU throughput matrices, thread/pinning study,
  long-context KV economics, quality evals, NPU↔CPU parity
- [FAILURES.md](FAILURES.md) — QNN `smmu_proxy` wall, Turnip 8-bit-storage wall, DSP VA
  ceiling, partial-offload backfire, build gotchas, harness artifacts
- [ROADMAP.md](ROADMAP.md) — LFM3-on-NPU rebuild recipe, the llama.cpp #6401 gate,
  kernel/vendor requirements, and what "everything works" would take
- `npu/NOTES.md` — full QNN forensics (DSP-side `.farf` logs, ioctl probes)
- `docs/REPORT.md` — the original dated log everything above was distilled from
