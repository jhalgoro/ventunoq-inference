# Alternative inference engines — Strata, mistral.rs, MNN, and the article sweeps (2026-10-29/30)

Round 1 (2026-10-29): are [`niko1221/Strata`](https://github.com/niko1221/Strata) and
[`EricLBuehler/mistral.rs`](https://github.com/EricLBuehler/mistral.rs) worth testing on this
board, in the same regime as the llama.cpp survey? One is not testable at all; the other runs,
crashes in its own release binary, and loses everywhere it doesn't crash.

Round 2 (2026-10-30): the same question for the engines in
[“Edge Inference Engine: One Native API Across CPU, GPU, NPU, and RISC-V”](https://allenkuo.medium.com/edge-inference-engine-one-native-api-across-cpu-gpu-npu-and-risc-v-8e4166d1816f)
(Allen Kuo, 2026-08-05) — ONNX Runtime, TensorRT, ncnn, **MNN**, IREE, LiteRT, bare QNN,
DirectML, Vulkan(-Video). **One of them beats llama.cpp on this board's CPU — MNN, by up to
+30% decode** (see §Round 2 below). That changes the project's "which engine" answer for
Qwen-class CPU inference, though not for LFM3 or the NPU.

Round 3 (2026-10-30): five NPU-inference articles (localaimaster's 2026 NPU comparison, two
arXiv hardware/benchmark studies, USC's edge-platform benchmark, and llm.npu @ ASPLOS '25),
asked specifically whether any of them names an **NPU engine worth exploring** here. Answer:
no — every Qualcomm-NPU route in them is QNN/SNPE, which this kernel blocks (F1). llm.npu is
still the most useful read of the set, as a design blueprint for the Hexagon path we already
run (§Round 3 below).

## Verdict

| Engine | Verdict | Why |
|---|---|---|
| **Strata** | ❌ **not testable** | No engine build for this hardware: CUDA/HIP/SYCL/Metal only; its CPU kernels are x86 AVX2/AVX-512; smallest ARM-runnable model is 84 GB (board: 14 GiB RAM) |
| **mistral.rs** | ⚠️ **works, not competitive** | CPU-only path (no NPU/GPU), default release binary **SIGILLs on QCS8300** (F8), 2–2.6× slower than llama.cpp on prefill, LFM3 not in any release |
| **MNN** | ✅ **the one engine worth adopting for CPU** | ARM82 fp16 kernels: **+9% prefill / +30% decode vs llama.cpp Q4_0 at 0.8B**, +8%/+19% at 2B; F8-safe by construction; no LFM3/LFM2, no NPU here (QNN=Android, F1) |
| ONNX Runtime / TensorRT / DirectML / LiteRT / bare QNN / ncnn / IREE | ❌ see §Round 2 matrix | Blocked by F1 (all QNN paths), F2 (Vulkan), no hardware (CUDA/NVIDIA/Windows), or no LLM/arch coverage (ncnn, ORT) |
| Round 3 (NPU articles): llm.npu/MLLM, hailo-ollama, MLC-LLM, OpenVINO, MLX, SNPE | ❌ **nothing to test** — see §Round 3 | Every Qualcomm-NPU path in them is QNN/SNPE → **F1**; the rest is non-Qualcomm hardware or GPU/CPU routes already covered |

## Board constraints this survey keeps hitting (recap)

4×Cortex-A78C + 4×Cortex-A55, ARMv8.2 with `dotprod` + FP16-storage/`asimdhp`, **no i8mm, no
bf16, no SVE**; ~14 GiB usable RAM; NPU usable only through llama.cpp's Hexagon runner (F1);
Vulkan broken (F2). Any engine must therefore deliver: ARMv8.2 CPU backend + GGUF + ≤14 GiB
models — and can only beat llama.cpp by being *faster on CPU*, since the NPU is llama.cpp-only.

## Strata — closed before the first download

Evidence gathered from the repo (C++, MIT, ~36k★, created 2026-09-24; engine 0.1.54 / app 0.2.8):

- **Backends:** CUDA, HIP, SYCL, Metal. The "CPU" code is an *expert* path for x86:
  `src/kernels/cpu/iq_avx512.cpp`, `STRATA_ISA_FLOOR` — AVX2/AVX-512 gates, no NEON.
- **arm64 ≠ this board.** arm64 packaging exists solely for NVIDIA DGX Spark (GB10: CUDA GPU +
  128 GB unified memory). `docs/DGX_SPARK_LINUX.md` says it directly: *"Not built for ARM: … the
  HIP, SYCL and **CPU-only engines**, and the Q2_0 and Gyro packs, whose CPU kernels are x86-only."*
  The `linux-aarch64` engine packages require NVIDIA driver 580+ / CUDA 13 / glibc ≥2.39.
  Release assets otherwise: windows-x64, linux-x86_64, macos-arm64.
- **Model floor.** Three families only (DeepSeekV4.1-Flash 239–280 GB, Qwen3.8-Flash-Next,
  Qwen3.6-35B-A3B). Smallest documented footprint anywhere is the Gyro-S pack at **27.6 GB** —
  x86-only. The ARM-runnable option (Qwen3.8-Flash-Next IQ3_S) is an **84 GB download**.
  No Vulkan backend exists to fall back on.

Nothing to build, nothing to boot, nothing ≤14 GiB to load. Not tested — untestable here.

## mistral.rs — what it takes to make it run at all

v0.9.5 (2026-10-19), Rust on Candle. The board-relevant backend is **Candle CPU**: no Vulkan,
no Hexagon, and the CUDA aarch64 builds are for NVIDIA SoCs. GGUF Q/K-quant support ✔,
IQ quants ✘, K2-Horizon ✘, PagedAttention auto-off on CPU.

### F8 in one paragraph (full entry in FAILURES.md)

The release `mistralrs-cpu-aarch64-unknown-linux-gnu.tar.gz` (`-C target-feature=
+dotprod,+fp16,+fhm`) loads Qwen3.5-0.8B-Q4_0 fine and dies with **SIGILL before the first
token**, in `attention::backends::cpu::single_q::compute_group_range::<f16>`. gdb + a 5-line
`.s` repro show why: **QCS8300 does not implement `FMLAL`/`FMLSL` (vector)** — architecturally
*optional* FP16 widening instructions — even though the kernel advertises `asimdhp` in
HWCAP/`/proc/cpuinfo`. `fmla v0.4h`, `sdot`, `fcvt h0` all execute fine; `fmlal v0.4s, v2.4h,
v3.4h` traps on **both** clusters. Any `-C target-feature=+fp16` Rust binary is unsound here.

Two workarounds, both used for the numbers below:
1. the second release asset, **`…-v8_0.tar.gz`** (ARMv8.0 baseline, NEON-only) — runs;
2. the default dotprod build with **`MISTRALRS_CPU_KV_F32=1`** — f32 KV avoids the f16
   attention kernel that contains the illegal `fmlal`.

### Numbers (2026-10-29, board idle, big-core pinning `CANDLE_CPU_MASK=0-3`, pp512/tg128, d1, 1 iter + 1 warmup)

| Model (Q4) | engine | pp512 t/s | tg128 t/s | peak RSS |
|---|---|---|---|---|
| Qwen3.5-0.8B-Q4_0 | **mistral.rs 0.9.5** (dotprod build, f32 KV) | **80.1** | **12.5** | 2.03 GB |
| | llama.cpp v0.8.0 pinned (baseline) | **148.7** | **31.9** | mmap'd weights |
| | ratio | 0.54× | **0.39×** | — |
| LFM2.5-2.6B-Q4_K_M | **mistral.rs 0.9.5** (dotprod build, f32 KV) | **19.0** | **12.1** | 5.45 GB |
| | llama.cpp v0.8.0 pinned (baseline) | **40.1** | **13.1** | mmap'd weights |
| | ratio | 0.47× | 0.92× | — |

Supporting runs: `-v8_0` build on Qwen: 67.4 pp / 12.1 tg (d2) — so `+dotprod` buys mistral.rs
~+14–19% prefill and ~0% decode; the dotprod build with default f16 KV: SIGILL (F8).

Reading:

- **Prefill loses ~2×** — generic Candle kernels vs llama.cpp's hand-written NEON/dotprod
  GGML kernels. LFM2.5's short-conv layers amplify it (0.47×).
- **Decode is nearly parity on LFM2.5** (12.1 vs 13.1) — decode is bandwidth-bound (see
  `memory-map.md`), so the engine barely matters at 2.6B; at 0.8B the weights fit in cache and
  kernel quality dominates again (0.39×).
- **RSS is 3–4× the model size** (private pages, no mmap): 5.45 GB for a 1.67 GB model. On a
  14 GiB board that directly shrinks the workable model class.
- **The NPU moat is untouched**: llama.cpp's Hexagon runner does 679 t/s prefill on this model
  (BENCHMARKS §3); mistral.rs has no path to it.

### LFM3 — the actual target model — is not supported

`mistralrs bench -f LFM3-1.7B-Q4_K_M.gguf` on v0.9.5: `Error: Unknown normal-model GGUF
architecture 'lfm3'`. LFM3 support merged **2026-10-28** (PR #2515, backend-generic model +
CPU conv/rms-norm kernels, so Candle CPU should work) — *after* v0.9.5. Releases ship every
~3–4 weeks with no nightly/pre-release channels, so the options are a ~1 h on-board source
build (rustup + `cargo build --release`, no sudo needed) or waiting for v0.9.6. **Not done**:
the measured 2–2.6× CPU gap is structural (Candle vs GGML kernels), so a source build would
only prove LFM3 loads — it cannot change the verdict, and llama.cpp already runs LFM3 on both
CPU *and* NPU.

### Repro recipe

```bash
# build variants: release assets mistralrs-cpu-aarch64-unknown-linux-gnu{,-v8_0}.tar.gz
CANDLE_CPU_MASK=0-3 MISTRALRS_CPU_KV_F32=1 \
  ./mistralrs bench -f Qwen3.5-0.8B-Q4_0.gguf --cpu --prompt-len 512 --gen-len 128 \
  --depth 1 --iterations 1 --warmup 1            # 80.1 pp / 12.5 tg
CANDLE_CPU_MASK=0-3 ./mistralrs bench -f … --cpu  # without f32 KV → SIGILL (F8)
# FMLAL probe (both clusters SIGILL):
printf '.global _start\n_start:\n  fmlal v0.4s, v2.4h, v3.4h\n  mov x8,94\n  svc 0\n' > i.s
as -march=armv8.2-a+fp16fml -o i.o i.s && ld -o i i.o && taskset -c 0-3 ./i; echo $?   # 132
```

## Round 2 — the article's "ten providers" on this board (2026-10-30)

The article benchmarks **vision** workloads (YOLOv8n, RIFE) on desktop/Jetson/Snapdragon-8Gen3
(Android+root)/RISC-V — different problem space, but its provider list is exactly the candidate
set for "another engine on this board". Screening it against this board's facts (F1: any QNN
runtime dies at `HAP_mmap2` — kernel-side, engine-agnostic; F2: Turnip Vulkan; no NVIDIA/Windows):

| Provider (article) | On QCS8300 for LLMs | Reason |
|---|---|---|
| author's engine itself | ✘ n/a | Vision pipeline (YOLO/RIFE/codec), no LLM/GGUF story |
| ONNX Runtime CPU / CUDA | ✘ | No LLM inference (vision graphs only); CUDA: no hardware |
| TensorRT · NvSci | ✘ | NVIDIA-only |
| DirectML (D3D) | ✘ | Windows-only |
| bare QNN · LiteRT(+NPU) · ORT-QNN EP | ✘ | All ride the QNN runtime → **F1** (missing `smmu_proxy`); article's QNN success is Android-with-root, not this Linux kernel |
| ncnn CPU/Vulkan | ✘ | LLM set = llama3.2/qwen2.5/qwen3-0.6B/minicpm — **no Qwen3.5, no LFM, no GGUF import**; Vulkan: F2 + article itself flags MNN-Vulkan miscompute on desktop |
| IREE | ✘ | No NPU backend (article: "no NPU of any kind"); LLM export needs an x86 Python toolchain; no GGUF |
| LiteRT-LM | ✘ | `.task`/`.litertlm` via Python converter; Qwen ✔ but Qwen3.5 ✘; NPU = Qualcomm → F1 |
| **MNN** | ✅ **tested — wins** | Below |

### Why MNN was safe to try where mistral.rs died (F8)

Source audit before running anything: the ARM82 fp16 hot kernels are hand-written `.S` using
`fmla/fmul/fcvt/sdot` — all verified executable on QCS8300 — and the only `FMLAL` code
(`vfmlalq_lane_*` in `source/backend/arm82/Arm82Functions.cpp`) is **compile-time gated** behind
`__ARM_FEATURE_FP16_FML`, which the default `-march=armv8.2-a+fp16` build does not enable.
KleidiAI (i8mm!) is `MNN_KLEIDIAI=OFF` by default. Runtime probe agrees:
`i8sdot:1, fp16:1, i8mm:0, sve2:0, sme2:0` — correct detection, no SIGILL.

### Setup (all on-board, no sudo)

```bash
git clone --depth 1 https://github.com/alibaba/MNN && cd MNN
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_LLM=ON -DMNN_ARM82=ON -DMNN_LOW_MEMORY=ON -DMNN_BUILD_SHARED_LIBS=OFF
ninja -C build llm_demo llm_bench          # ~4 min on this SoC
# models: taobao-mnn/Qwen3.5-{0.8B,2B}-MNN on HF (pre-converted int4+fp16-scale, no torch needed)
taskset -c 0-3 ./llm_bench -m <dir>/config.json -a cpu -t 4 -c 2 -pg 512,128 -rep 3
# -c 2 = fp16 ("Low"); -pg mirrors llama-bench -pg exactly (pp then tg reusing the KV cache)
```

### Numbers (big-core pinned `taskset -c 0-3`, 4 threads, fp16; llama.cpp v0.8.0 b11972 `-t 4 --cpu-mask 0xf --cpu-strict 1`)

| Model | llama.cpp Q4_0 | llama.cpp Q4_K_M | **MNN int4 (fp16)** | MNN gain vs Q4_0 |
|---|---|---|---|---|
| Qwen3.5-0.8B pp512 | 148.7 | 106.2 | **162.7 ± 0.1** | **+9.4%** |
| Qwen3.5-0.8B tg128 | 31.9 | 27.0 | **41.5 ± 0.1** | **+30%** |
| Qwen3.5-2B pp512 | 69.8 | — | **75.7 ± 0.1** | **+8.4%** |
| Qwen3.5-2B tg128 | 16.1 | — | **19.2 ± 0.0** | **+19%** |

Config sweep (0.8B, per the article's "sweep the knob" rule): fp16 is worth +65% decode
(41.5 vs 25.1 fp32); 4 big-core threads is the decode optimum (8 threads: pp 170.7 but tg
drops to 31.5 — the A55s drag decode). Peak RSS 930 MB for the 449 MB model (2.1×, vs
mistral.rs's 4× and llama.cpp's mmap).

Reading:

- The win is **kernel + quant format together**: MNN's int4 weights carry fp16 per-block scales
  (≈Q4_K-class quality) and compute in ARM82 fp16. llama.cpp's own Q4_K_M — the like-for-like
  quant — is *slower* than its Q4_0 here (106 vs 149 pp), so MNN is 1.5× over quant-matched
  llama.cpp at 0.8B. Decode gain shrinks 30%→19% from 0.8B→2B, exactly the bandwidth-bound
  crossover from `memory-map.md`.
- **What MNN does not change**: LFM2/LFM3 are not in its model mapper (`lfm2_audio` only), so
  llama.cpp remains the only LFM3 engine; and its Hexagon LLM backend is Android + QNN SDK →
  **F1 blocks it here** — llama.cpp's Hexagon runner (679 t/s prefill) stays the only NPU path.
  `is_visual` exports also need `visual.mnn` for `llm_demo` (text-only: patch `is_visual:false`,
  or just use `llm_bench`, which skips it).

## Round 3 — NPU-path articles (2026-10-30)

Five "NPU inference" articles asked about specifically for **NPU engines worth exploring** on
this board: [localaimaster's 2026 NPU comparison](https://localaimaster.com/blog/npu-comparison-2026),
[NPU Hardware Evaluation v1.0](https://arxiv.org/html/2609.13166) (Covision Lab),
[LLM Inference at the Edge under sustained load](https://arxiv.org/html/2603.23640v1),
[Benchmarking Edge AI Platforms](https://arxiv.org/html/2409.14803v1) (USC), and
[llm.npu (ASPLOS '25)](https://arxiv.org/html/2407.05858v2). Verdict: **none names an engine
runnable here** — but llm.npu is the best design document available for improving the NPU path
we already have.

| Article | Engines named | On QCS8300? |
|---|---|---|
| localaimaster NPU comparison 2026 | OpenVINO · ORT-QNN EP · AI Engine Direct · LiteRT ("in progress") · Core ML/MLX | ✘ laptop parts (Panther Lake / X2-on-Windows / Ryzen AI / Apple); every Qualcomm cell is QNN → F1 |
| NPU Hardware Evaluation v1.0 | Hailo-8/10H · Axelera · EdgeCortix · SiMa · QCS6490/8550/IQ-9075 via **AI Engine Direct + SNPE/QNN** · TensorRT | ✘ vision-only suite (12 CV models, no LLM); Qualcomm stack → F1; ASIC chips = foreign hardware |
| Edge LLM under sustained load | **hailo-ollama**+HailoRT (RPi5+Hailo-10H) · MLC-LLM/TVM OpenCL (S24 **GPU**) · MLX Swift (iPhone, "Neural Engine not utilised") · vLLM | ✘ Hailo-10H is a separate PCIe M.2 module with a vendor kernel driver (no-sudo ✘); the S24 (Snapdragon 8 Gen 3!) is benchmarked on the **GPU** — no NPU stack is used anywhere in the paper |
| USC edge-platform benchmark | OpenVINO + Intel NPU library (LLaMA on Core Ultra) | ✘ Intel-only; no Qualcomm content |
| **llm.npu (ASPLOS '25)** | **MLLM + QNN**, Android (Xiaomi 14 / Redmi K60 Pro); baselines llama.cpp · TFLite · MNN · MLC-LLM · PowerInfer-v2 | ✘ as an engine (QNN → F1, Android-only); ✅ as a **design reference** — below |

### llm.npu — not testable, but the blueprint for our Hexagon path

The only LLM-on-Qualcomm-NPU system among the five: >1,000 t/s prefill on 1B-class models,
22.4× prefill / 30.7× energy savings on average vs llama.cpp/MNN/TFLite baselines (SD 8 Gen 3,
W8A8). Its three techniques map directly onto levers for our llama.cpp Hexagon runner
(679 t/s pp512, §3 BENCHMARKS):

1. **Chunk-sharing graphs** — QNN graph build+optimize costs seconds (360 ms + 11.5 s for
   Gemma-2B), so per-prompt graph prep kills NPU prefill; llm.npu pre-builds fixed-length
   chunk subgraphs and shares the static (Linear/LayerNorm) ones across chunks. Our
   ggml-Hexagon path pays no per-prompt graph-prep (which is why it can show a 21× prefill
   win at all), but the chunking logic is the model for handling variable prompts on any
   future NPU graph API here.
2. **Shadow outlier execution** — per-tensor W8A8 on NPU + the 0.1–0.3% outlier channels
   computed in float on CPU in parallel. This is the accuracy-recovery trick for NPU-friendly
   quants; relevant if a W8A8 NPU quant path is ever pursued here (mobile NPUs have no
   group-quantized MatMul — their Table 1 — same reason our NPU path needs the q4_K→NPU
   repack, PR #28994).
3. **Out-of-order CPU/NPU subgraph scheduling** — attention/LayerNorm stay on CPU, linears on
   NPU; naive overlap leaves 37% bubbles, so subgraphs are reordered to keep the NPU (the
   critical path) fed. Directly relevant to our F4 finding (partial offload regresses decode
   to 7.9 t/s via ping-pong — offload fully or keep ≥⅔ of layers on the NPU).

Code: <https://github.com/UbiquitousLearning/mllm> (MLLM; QNN backend is Android-only).

### Corroborations picked up along the way

- 2609: Hexagon DSP platforms "execute one inference job at a time" — consistent with our
  F4/full-offload rule; and the IQ-9075's CPU is Kryo Gen 6 (A78C @ 2.1–2.36 GHz), i.e. the
  same cluster as this board's.
- 2603: sustained-load failure modes on phones (iPhone loses ~½ throughput in 2 iterations;
  S24 hits an OS-enforced GPU frequency floor) — the right methodology for ROADMAP R7
  (sustained/thermal suites; our own thermal behaviour is unmeasured); their Hailo-10H number
  (6.9 t/s decode at <2 W, bandwidth-bound on-module DDR) is another data point for our
  "decode is bandwidth-bound on every engine" rule.
- 2603 explicitly: MLX does not use Apple's Neural Engine — the "NPU" in most consumer NPU
  marketing rarely runs LLMs; consistent with Round 2's finding that the article's QNN wins
  were Android-with-root, not reachable from this Linux kernel.

## Revisit when…

- …MNN adds LFM2/LFM3 to `llmexport`/`model_mapper` — that would move LFM3 CPU inference to a
  ~30% faster engine; and
- …mistral.rs v0.9.6+ ships LFM3 (expect the ~2× Candle deficit anyway); and
- …anyone ships a QNN path that doesn't need `smmu_proxy` (kernel work, F1) — until then every
  NPU claim in the article's QNN/LiteRT/MNN-Hexagon providers, and llm.npu/MLLM, is unreachable
  on this board; and
- …llm.npu's chunk-graph / shadow-outlier ideas become implementable on the ggml Hexagon
  backend (W8A8 NPU quant + outlier shadow ops) — that's the upgrade path the paper sketches.

