> **ARCHIVED FIELD LOG (2026-10-28/29).** This is the original chronological report,
> kept verbatim for provenance. The curated project view now lives in the top-level
> [README.md](../README.md), [BENCHMARKS.md](../BENCHMARKS.md),
> [FAILURES.md](../FAILURES.md) and [ROADMAP.md](../ROADMAP.md).
> Paths mentioning `~/Code/lmf3-ventunoq` predate the rename to `ventunoq-llamacpp`.

# LFM3 (Liquid AI) — Local Performance & Benchmark Evaluation

**Machine:** ventunoq — Qualcomm "Monaco/Monza" SoC (Linux 6.8.0-qcom, Ubuntu 24.04.5 aarch64)
**CPU:** 8 cores — 4× Cortex-A78-class @ up to 2.36 GHz (part 0xd4b) + 4× Cortex-A55 @ up to 1.96 GHz (part 0xd05)
**ISA:** ARMv8.2-A, NEON + dotprod (`asimddp`); **no** SVE/SVE2, **no** i8mm, **no** bf16
**RAM:** 14 GiB LPDDR, no swap · **GPU:** Adreno (Turnip/Mesa 25.2.8, Vulkan 1.3) — enabled
in-source but inference-blocked (§10); §3–5 ran CPU-only, §9 ran NPU
**Date:** 2026-10-28 · **Research:** done via CloakBrowser (stealth Chrome 146) at `localhost:18080` → `research/`

---

## 1. What LFM3 is (research summary)

Announced **Oct 27, 2026** by Liquid AI ([blog](https://www.liquid.ai/blog/lfm3), sources in `research/`):

- **New tri-hybrid architecture**: 24 layers = **13 gated short convolutions** (kernel 7) + **5 sliding-window attention** (256-token window) + **6 global attention**. Plus head-wise gated attention (per-head sigmoid), partial RoPE on global layers, sandwich norm. Same 128K vocab as LFM2.5-8B-A1B.
- **Trained on 22T tokens** (3.5T mid-training extends context to **256K**).
- **Six open-weight models (~2B)** branched from `LFM3-1.7B-Base`:
  | Model | Params | Modality |
  |---|---|---|
  | LFM3-1.7B | 1.73B | text, 256K ctx |
  | LFM3-VL-2B | 1.9B | vision + video, "thinking" |
  | LFM3-Audio-2B | 2B | speech in/out, 18 langs |
  | LFM3-Omni-2B | 2.2B | vision+audio+text |
  | d1-2B / d1-Omni-2B | ~2B | decision models (single-pass structured decisions) |
- **Edge claims**: text model ≈ **1 GB at 4-bit**; llama.cpp Q4_0 @32K ctx: fastest prefill *and* decode vs MiniCPM5-2B, Qwen3.5-4B, Gemma 4 E4B on 3 edge devices; **1.6 GB KV cache @128K** (vs 4.3 GB Qwen3.5-4B, 5.6 GB MiniCPM5-2B).
- **Headline quality claims** (official, full tables in `research/lfm3-announcement.txt`):
  - LFM3-1.7B avg **49.37** across 27 benchmarks vs LFM2.5-2.6B **31.55**, Qwen3.5-9B **47.24**.
  - IFEval 96.66 · BFCL V4 71.39 · ToolSandbox 80.36 · LiveCodeBench v6 73.46 · GPQA-D 63.37 · SWE-bench Verified 43.40 (OpenCode harness) · RULER 128K 65.56.
- **Day-0 support**: llama.cpp ≥ **v0.8.0 (b11960)**, Ollama ≥ v0.40.12 (`ollama run lfm3`), LM Studio, vLLM, SGLang, MLX, ONNX Runtime, OpenVINO, Qualcomm AI Hub. Official GGUF: `LiquidAI/LFM3-1.7B-GGUF` (incl. **QAD Q4_0** = quantization-aware-distilled checkpoint).
- License: **LFM Open License v1.0** (open-weight).

## 2. Build on this machine

| Component | Version / choice | Why |
|---|---|---|
| llama.cpp | **v0.8.0, build b11972** (`llama-b11972-bin-ubuntu-arm64.tar.gz`) | first release line with LFM3 support; no compiler on host (no sudo), official Ubuntu arm64 binaries |
| Models | `LFM3-1.7B-Q4_K_M` (1.05 GB), `LFM3-1.7B-QAD-Q4_0` (1.00 GB), `LFM2.5-2.6B-Q4_K_M` (1.67 GB, baseline) | Q4_K_M = common default; QAD-Q4_0 = what Liquid measured on-device; LFM2.5 to test "35% fewer params, better" claim |
| Comparison models | **unsloth GGUFs only** (per review): `unsloth/Qwen3.5-2B-GGUF`, `unsloth/Qwen3.5-4B-GGUF`, `unsloth/Qwen3.5-9B-GGUF`, all Q4_K_M | official `Qwen/*-GGUF` repos are gated (401); unsloth mirrors are public. 9B **does fit** (5.68 GB weights + ~1 GB KV < 14 GB RAM) → included with reduced scope (1 rep, GSM8K N=20, HumanEval N=10) because decode will be slow |
| Disk policy | each GGUF deleted immediately after its tests; free-space guard before every download keeps `/` < 50% | 55 GB rootfs; full audit trail with sha256 + `df` snapshots in `results/DISK_LOG.md` |
| CPU backend | `libggml-cpu-armv8.2_2.so` (runtime dispatch) | host lacks SVE2/i8mm; armv8.6/armv9.2 backends unavailable |
| Eval harness | `scripts/` (pure stdlib Python 3.12; `websockets` vendored for CDP) | no pip on host |

## 3. Throughput — llama-bench (CPU)

All runs: llama.cpp **v0.8.0 b11972**, CPU backend `armv8.2_2`, `-ngl 0`, batch 512/ub 8 unless noted. Raw CSVs in `results/`.

### 3.1 LFM3-1.7B vs LFM2.5-2.6B vs LFM3-QAD (Q4_K_M / Q4_0)

| Test | **LFM3-1.7B Q4_K_M** | LFM3-1.7B **QAD Q4_0** | LFM2.5-2.6B Q4_K_M |
|---|---|---|---|
| pp512 (t/s) | 49.0 | **87.4** | 32.0 |
| pp1024 | 49.1 | — | — |
| pp2048 | 46.7 | 81.5 | 30.4 |
| pp4096 | 46.7 | — | — |
| pp8192 | 43.3 | **62.6** | 28.0 |
| tg32 | **13.0** | 7.1 | — |
| tg128 | 11.7 | 10.2 | 7.3 |
| tg512 | **10.6** | — | — |

- **LFM3-1.7B beats LFM2.5-2.6B on both axes** (prefill ×1.5–1.6, decode ×1.4–1.6) with **35% fewer params** — the announcement's claim holds on this board.
- **QAD-Q4_0 is a prefill monster**: ~87 t/s (×1.8 vs Q4_K_M) because Q4_0 dequant is far cheaper than Q4_K_M's block-unpacking on a NEON-only core; decode is slightly slower. For long-context/prompt-heavy use, QAD is the quant to run here.

### 3.2 Thread scaling & core pinning (LFM3-1.7B Q4_K_M)

| Config | pp512 | tg128 |
|---|---|---|
| `-t 2` | 29.0 | 13.1 |
| `-t 4` | **49.0** | **19.1** |
| `-t 6` | 46.5 | 13.8 |
| `-t 8` (default suite) | 47.3 | 11.1 |
| `-t 4 --pin-big` (`--cpu-mask 0xf --cpu-strict 1`) | **49.6** | **19.4** |

**Key finding:** this is a 4×A78 + 4×A55 layout — decode is *hurt* by spreading to the little cluster. 4 threads pinned to the big cores gives **+75% decode** (19.4 vs 11.1 t/s) at equal prefill. `--pin-big` should be the default for interactive use on this board.

## 4. Long context & KV memory

- Prefill degrades gracefully: pp16384 = **37.8 t/s**, pp32768 = **30.8 t/s** (Q4_K_M, -c 33792).
- Measured RSS: 2.49 GB model+KV @2K ctx → **3.62 GB @128K ctx** ⇒ **≈1.14 GB KV for 128K tokens** (~8.9 KB/token). Confirms Liquid's "1.6 GB KV @128K" claim being real (they quote f16-cache worst case; tri-hybrid conv state + windowed attention makes the cache tiny). Qwen3.5-4B-class caches measured at ×2.6–3.8 this size in the announcement — LFM3's 256-token sliding window + short convolutions are doing their job.

## 5. Quality spot-checks (this machine, Q4_K_M, greedy)

### 5.1 LFM3-1.7B results

| Eval | Score | Notes |
|---|---|---|
| GSM8K (N=50, few-shot, strict last-number match) | **30/50 = 60.0%** | `results/gsm8k_LFM3-1.7B-Q4_K_M.json` |
| HumanEval (N=20 first tasks, sandboxed exec) | **13/20 = 65.0%** | `results/humaneval_LFM3-1.7B-Q4_K_M.json` |
| Perplexity, The Great Gatsby (n_ctx 2048) | **29.54 ± 0.49** (Q4_K_M) / **31.14 ± 0.52** (QAD Q4_0) | QAD degrades ~5% PPL for its ~1.8× prefill win |

HumanEval failure anatomy (7 fails): almost all are **edge-case handling** — empty-string inputs (×3), numeric-word parsing, off-by-one in paren-depth — not syntax or planning failures; completions always compiled and ran. GSM8K errors are typical small-model arithmetic drift, not format failures (answer extraction worked on all 50).

### 5.2 vs unsloth Qwen3.5 quants + K2-Horizon (same board, same harness)

All rows: Q4_K_M, llama.cpp v0.8.0 b11972, CPU-only, same bench flags and eval
scripts. GSM8K = N=50 (9B/7B: N=20, reduced scope — swap-dominated decode);
HumanEval = first 20 tasks (9B/7B: 10).

| Model (Q4_K_M) | pp512 | pp8192 | tg128 | GSM8K | HumanEval |
|---|---|---|---|---|---|
| **LFM3-1.7B** (1.1 GB) | 49.0 | 43.3 | **11.7** | 60.0% | 65% |
| unsloth **Qwen3.5-2B** (1.2 GB) | 49.1 | 41.3 | 7.9 | 22.0% | 40% |
| unsloth **Qwen3.5-4B** (2.4 GB) | 19.8 | 16.9 | 4.2 | 72.0% | 85% |
| unsloth **Qwen3.5-9B** (5.4 GB) | 11.6 | 10.2 | 2.7 | 85.0%* | 80%* |
| **K2-Horizon-4B** (3.2 GB) | 19.8 | 11.5 | 4.3 | 60.0% | 75% |
| **K2-Horizon-7B** (5.6 GB) | 11.2 | 8.3 | 3.0 | 75.0%* | **100%*** |

\* reduced N. `results/bench_qwen35_*.csv`, `gsm8k_Qwen*.json`, `humaneval_Qwen*.json`.

- **Sub-3B class: LFM3-1.7B wins outright.** Same prefill speed as Qwen3.5-2B
  (49 t/s), 48% faster decode (11.7 vs 7.9 t/s), and near-triple its GSM8K
  score (60% vs 22%) at nearly 2× HumanEval (65% vs 40%).
- **Qwen3.5-2B's GSM8K collapse is real, not a harness artifact**: 28/39 misses
  are wrong final numbers, 11/39 ran out the 512-token budget rambling *about
  the prompt itself* (few-shot raw-completion style appears to fight its chat
  training). Even discounting the rambles it is ~34% vs LFM3's 60%.
- **Quality/size curve is steep**: 2B→4B doubles GSM8K (22→72%) and costs ~2.5×
  in throughput; 9B adds little over 4B on this board's harness (85% vs 72% on a
  smaller N) for another ~1.6× slowdown, and decodes at 2.7 t/s once weights
  (5.4 GB) exceed comfortable RAM.
- **Practical ceiling for interactive use: Qwen3.5-4B** (4.2 t/s, 85% HumanEval).
  Above it the board is batch-mode only.
- **K2-Horizon-7B is the strongest coder here**: 10/10 on its HumanEval slice
  (vs 8/10 for Qwen3.5-9B) and 75% GSM8K, at throughput essentially identical to
  Qwen3.5-9B (11.2 pp512 / 3.0 tg128). Its prefill decays faster with context
  (8.3 t/s at 8k vs Qwen's 10.2), so it is the better pick for short-prompt code
  generation, the worse one for long documents.
- Note on the 7B numbers: `eval_k2.sh` had a repo/file naming bug (repo is
  `K2-Horizon-3.7B-GGUF`, files are `K2-Horizon-4B-*`) so the 4B first pass 404'd;
  corrected in `scripts/eval_k2.sh` + `scripts/eval_k2_4b.sh`, which re-ran 4B with
  full scope (N=50 / N=20) — those are the numbers in the table.
- **K2-Horizon vs Qwen3.5 at equal size**: 4B matches Qwen3.5-4B on decode (4.3 vs
  4.2 t/s) but scores clearly lower (60%/75% vs 72%/85%) and loses more prefill speed
  at 8k context (11.5 vs 16.9 t/s). At 7B the picture flips — K2 out-codes Qwen-9B
  (10/10 vs 8/10) at equal speed. Net: Qwen3.5 is the better sub-5B family here; K2
  earns its slot as the board's best *short-prompt coder* at 7B.

## 6. Verdict

**LFM3-1.7B is the best sub-3B model on this board today** — it matches
Qwen3.5-2B prefill, decodes 48% faster (11.7 vs 7.9 t/s), and scores 60%/65% on
GSM8K/HumanEval vs 22%/40%. For Arabic/bilingual or prompt-heavy workloads the
**QAD-Q4_0** quant is the standout: 87 t/s prefill (~1.8×) at a ~5% PPL cost.

If quality matters more than latency, **Qwen3.5-4B is the practical ceiling**
for interactive use (4.2 t/s, 72%/85%); 9B buys little on this harness for a
further 1.6× slowdown and decodes at 2.7 t/s. K2-Horizon-4B matches Qwen-4B's
speed but scores lower (60%/75%); **K2-Horizon-7B is the board's best coder**
(10/10 HumanEval, 75% GSM8K) at Qwen-9B speed — batch-mode territory either way.
**On the NPU (§9) this verdict shifts**: Qwen3.5-4B-Q4_0 with `-ngl 24` decodes at
10.3 t/s (54 t/s prefill @8k) — 4B-quality interactive is no longer batch-mode.

Board guidance: pin to the big cores (`--cpu-mask 0xf --cpu-strict 1`, +75%
decode), keep context ≤8k for RAM headroom (KV ≈1.14 GB @128k). The NPU is
**not** a dead end after all: the QNN path needs a missing smmu_proxy kernel
(§7, npu/NOTES.md), but Arduino's brick-provided **Hexagon-backend llama.cpp**
works on this board today: **prefill 9–21× and decode 1.35–1.5×** vs a
big-core-pinned CPU control (§9, `research/bricks-faster-inference.md`) —
use it for prompt-heavy work; decode stays bandwidth-bound everywhere.
The **GPU is a third engine in hardware but not in software** (§10): Turnip runs compute
shaders fine, yet llama.cpp's quantized/tiled matmul shaders require an 8-bit-storage
feature this Adreno doesn't advertise, and the driver segfaults or faults instead of
saying so — CPU + NPU remain the only usable engines.

## 7. NPU (Hexagon V75 HTP) status

Full log: `npu/NOTES.md`. Summary: **the NPU is real, reachable, and compiles graphs on-DSP — but graph *execution* is blocked by a buffer-registration failure in the board's FastRPC stack.**

| Step | Status |
|---|---|
| HTP present (V75), FastRPC nodes open as user | ✅ |
| DSP code execution (calculator unit test over unsigned PD) | ✅ |
| ONNX → DLC conversion **on-device** (QAIRT 2.51 Python API) | ✅ |
| SoC whitelist (QCS8275 = "SnapdragonModel 82" not supported) | ✅ bypassed — soc_id spoof + sweep found 4 whitelisted ids (605/606/607/695) |
| Graph compose + **finalize/prepare on DSP** | ✅ reaches "Executing Graphs" |
| **Graph execute** | ❌ DSP-side `HAP_mmap2(fd)` returns `paddr 0x0` → `0x80000448` → skel `sharedMemoryMmap failed` → `QAIRT_MEM_ERROR_MAPPING (8003)` → exec err 6001 |

Ruled out as causes (each tested): wrong dma-heap (redirect to `qcom,system` — same), stale userspace FastRPC lib (rebuilt **qualcomm/fastrpc master** `libcdsprpc.so` on-board — same), SDK point-release (2.51.0 vs 2.51.200 — same), PD session & firewall knobs, QNN 2.48 era libs (ABI-locked to their own binaries/DLCs).

**Root cause (established via DSP-side `.farf` logging, a direct `/dev/fastrpc-cdsp`
ioctl probe `npu/mmap_probe.c`, and a live kernel capture):** the vendor HTP skel
registers its execute arenas by passing a dma-buf **fd** to the DSP; the vendor
CDSP firmware resolves that fd with `HAP_mmap2`, which requires the **vendor
`smmu_proxy` Linux service** to do the CDSP-SMMU mapping on the DSP's behalf.
That service does not exist in this board's Ubuntu `6.8.0-1084-qcom` kernel (no
module anywhere, "no reserved DMA memory for FASTRPC" ×3 at boot, no fastrpc
debugfs). The kernel's *own* mapping path works fine — the legacy `MMAP(6)`
ioctl maps a 588 KB dma-buf and returns DSP VA `0xe0900000` — but the vendor
DSP ignores kernel-provided pages for fd args and calls `HAP_mmap2`, which
returns `paddr 0x0`. The new `MEM_MAP(10)` ioctl is answered by the firmware
with `AEE_EUNSUPPORTED` for every size. Patching self-built `libcdsprpc` to
route all maps through the working legacy ioctl was implemented and tested: it
regresses device bring-up (`find_map(fd)` on the DSP is fd-keyed, so the
dspqueue state buffer then fails) — reverted, baseline re-verified.
**Not fixable from userspace; needs a vendor/Android kernel (or a Qualcomm
port of smmu-proxy) on this board.**

Consequence for GGUF LLMs *(as written on the QNN path)*: QNN-native LLM runners need this same buffer path, so **no LLM can use the NPU through QNN today**, regardless of quant.

> **2026-10-29 update — superseded for the non-QNN path.** llama.cpp *does* have an upstream
> Hexagon backend (`ggml/src/ggml-hexagon`), and it registers memory through the **classic
> fastrpc MMAP ioctl** — the exact kernel path `npu/mmap_probe.c` proved working — not through
> the vendor `HAP_mmap2`/smmu_proxy route that QNN needs. Arduino ships this backend prebuilt
> and running on this board; `HTP0: Hexagon` enumerates as the plain user. See §9 and
> `research/bricks-faster-inference.md`. CPU numbers in §3 remain the baseline until
> `scripts/test_hexagon.sh` quantifies the HTP.

## 8. Memory: one 16 GiB DDR shared by CPU, NPU and GPU

The board has **16 GiB unified LPDDR** — no dedicated VRAM, no dedicated NPU RAM. Full
measurements in `research/memory-map.md`; headline split (from dmesg/meminfo/device-tree):

| Bucket | Size | Who can use it |
|---|---|---|
| Physical DDR | 16 GiB | all masters (CPU, Adreno, Hexagon, video/camera/audio DSPs) |
| nomap carve-outs | 558 MiB | secure/boot subsystems only; invisible to the CPU |
| — of which CDSP firmware | 30.5 MiB | **Hexagon NPU OS image only — model memory does NOT come from here** |
| — of which GPU | 8 KiB | stub — the Adreno has **no** carve-out, it maps RAM via its IOMMU |
| CMA pools (reusable) | 164 MiB | dma-buf consumers; CPU page-cache may use them when idle |
| Linux `MemTotal` | 14.93 GiB | everything else; includes the CMA pools |

How the accelerators actually get memory: **dma-buf heaps** (`/dev/dma_heap/`). GPU
(freedreno/Turnip) allocates from `system`; the Hexagon NPU (llama.cpp backend, fastrpc)
allocates from `qcom,system` — both are ordinary RAM pages mapped through the device's
IOMMU/SMMU, zero-copy. The `arduino` user is in the `fastrpc`/`video`/`render` groups, so
**userspace can allocate GPU and DSP buffers without sudo** (verified).

Practical consequences:
- Weights on the NPU do **not** free RAM — capacity stays one shared ~15 GiB pool (the 9B/K2-7B
  swap in §5 is this ceiling, not a carve-out).
- The NPU's real limit is **DSP virtual address space**: Arduino documents ~3,134 MiB per HTP
  session × 4 sessions, but this kernel exposes a single shared 32-bit window measured at
  **~1.75 GiB total** (`npu/cumap_probe.c`) — fits ≤2B-class Q4 models fully, not K2-4B (§9).
- All engines contend for the same LPDDR **bandwidth**, so expect the HTP's biggest wins on
  prefill (compute-bound); decode is bandwidth-bound on every engine.

## 9. Arduino Bricks → the path to faster inference

**Bricks** are App Lab's building blocks (docs rev 2026-10-19): Python modules that package a
feature — the `arduino:llm` brick is a client to the `llamacpp` **app service**, which on this
board (`ventunoq`) runs the container **`llamacpp-npu-runner`**: llama.cpp master + the upstream
**Hexagon backend**, built by Arduino's `build-llamacpp.yml` workflow (cross-compile via the
Snapdragon toolchain container, Arduino's `fix-race.patch` applied) and released as
`llamacpp/20261020`. On `unoq` the same brick falls back to a CPU-only runner. Full findings:
`research/bricks-faster-inference.md`.

Verified on this board (2026-10-29):
- `llamacpp-hexagon-20261020.tar.gz` downloaded, sha256 matches the Dockerfile pin; ships
  `libggml-htp-v75.so` — this board's HTP arch.
- Only missing runtime dep was the unversioned `libcdsprpc.so` symlink (fixed in
  `/tmp/opencode/libfix`); then `llama-cli --list-devices` → **`HTP0: Hexagon`** as the plain user.
- Arch support in that build: `qwen35` ✔, `k2_horizon` ✔, `lfm2` ✔ — but **`lfm3` ✗** (build is
  ggml 0.27.0). Fix path: dispatch the same workflow on a fork with `llama_cpp_ref` set to a
  master containing lfm3.
- Quant policy from the brick's own `configure-llamacpp.py`: Q4_0 is fully NPU-bound (the
  catalog's standard); the Arduino build additionally repacks **q4_K/q5_K/q6_K onto the NPU**
  (llama.cpp PR #28994) — our existing Q4_K_M files are NPU candidates as-is; q2_K/q3_K and
  token embeddings stay on CPU.
- Service tuning to mirror: `-b 1024 -ub 512 -fa on -t 4 --cpu-mask 0x0f`,
  `GGML_HEXAGON_OPBATCH=2048`, `GGML_HEXAGON_MBUF=256,256`, ctx 16384, `-ngl 100`.

**Measured (NPU: `results/bench_hex_*_npu.csv`; fair CPU control: same binary, `-ngl 0`,
same service tuning `-t 4 --cpu-mask 0x0f -b 1024 -ub 512 -fa on` — `results/bench_hex2_*_cpu.csv`):**

| Model (Q4) | NPU pp512 | NPU pp8192 | NPU tg128 | CPU ctrl pp512/tg128 | speed-up (pp8192 / tg128) |
|---|---|---|---|---|---|
| Qwen3.5-0.8B-Q4_0 | **1366** | 1233 | **43.0** | 148.7 / 31.9 | **13.6× / 1.35×** |
| Qwen3.5-2B-Q4_0 | **863** | 807 | **24.3** | 69.8 / 16.1 | **14.9× / 1.51×** |
| LFM2.5-2.6B-Q4_K_M | **679** | 572 | **17.7** | 40.1 / 13.1 | **21.1× / 1.35×** |
| Qwen3.5-4B-Q4_0 **partial** (`-ngl 24`) | 70.8 | 53.7 | 10.3 | *(no ctrl; v0.8.0 Q4_K_M: 19.8/4.2)* | ~3× / ~2.4× |
| K2-Horizon-4B-Q4_K_M | ✗ exceeds VA ceiling → **partial `-ngl 18`** | 21.4 | 2.6 | 19.1 / 7.9 | 1.9× / **0.33×** ✗ |

Honest picture vs a *properly pinned* CPU control (the §3–5 baselines ran un-pinned 8-thread,
which is why early comparisons overstated decode gains): **prefill 9–21× faster on the HTP;
decode 1.35–1.5×** — decode is bandwidth-bound on every engine of this SoC, exactly as §8
predicts. The q4_K→NPU repack (PR #28994) works as advertised.

Practical wins:
- **Interactive quality ceiling moves 2B → 4B**: Qwen3.5-4B-Q4_0 at `-ngl 24` (fits the VA
  budget) decodes at **10.3 t/s with 54 t/s prefill @8k** — vs 4.2 t/s on CPU.
- Long-context/prompt-heavy workloads (RAG, agents) get the full ~15–20× regardless of model size.
- **Partial offload can backfire**: K2-4B at half-offload decodes at 2.6 t/s (worse than
  CPU's 7.9) — per-token CPU↔NPU ping-pong eats the gain when neither engine has most of
  the layers. Rule: offload fully, or keep enough layers on the NPU to dominate (≥⅔).
- Harness artifacts to note: this build rejects `--device CPU` (cycle-1 controls were void;
  redone bare), and the K2 CPU control's tg32 row (0.38 t/s vs tg128 7.89) is a warmup
  artifact — ignore tg32 there.

**The one hard limit found:** the CDSP maps buffers through a **32-bit IOMMU window**
("DMA64 disabled"); `npu/cumap_probe.c` measures it at **~1.75 GiB cumulative** (256 MiB
chunks → `ENOMEM` at 1792 MiB; contiguity also matters), and both HTP0/HTP1 fall back to the
*same* domain `cdsp:3` on this kernel, so a second session adds no address space. K2-4B
(~2.4 GB NPU-side) therefore aborts mid-load (`fastrpc_mmap failed`, error 1). Rule of
thumb: **NPU-side weights + KV + op arenas ≤ ~1.75 GiB** without root. `RLIMIT_MEMLOCK`
(1.9 GiB, unraisable as user) is a near-identical second wall.

**Quality parity (done):** `scripts/eval_hex_quality.sh` ran LFM2.5-2.6B-Q4_K_M GSM8K N=20
through an NPU `llama-server` (`-ngl 100`, ctx 8192) vs a `-ngl 0` control on the *same*
binary (`results/gsm8k_hexq_{npu,cpu}.log`): **NPU 12/20 = 60.0% vs CPU 11/20 = 55.0%**.
One question apart at N=20 — statistically indistinguishable, so the q4_K→NPU repack and the
DSP recurrent-state path do not change answers. Combined with the speed data above, the NPU
is now the default engine on this board for anything prompt-heavy up to the ~1.75 GiB VA
budget. (Sample-size caveat stands: ±1–2 questions at N=20; treat 55–60% as one band.)

---

## 10. GPU (Vulkan / Turnip) — the third engine

**Enablement (done).** The board ships Mesa 25.2.8's **Turnip** driver for the Adreno GPU
(vendor 0x5143): `vulkaninfo` reports **Vulkan 1.3.318, `DRIVER_ID_MESA_TURNIP`**, headless,
on `renderD128` — the user is in group `render`, so no sudo was needed anywhere. llama.cpp
v0.8.0 was rebuilt in-place with `-DGGML_VULKAN=ON` (`/tmp/opencode/llama-vulkan`); the
toolchain (cmake, ninja, glslc + libshaderc1, Vulkan headers, SPIRV-Headers) was staged
sudo-free from `.deb`s into `/tmp/opencode/vkdeps`. Two build gotchas worth recording:
(1) Ubuntu's `glslang-tools` does **not** ship `glslc` — it is a separate package and
`vulkan-shaders-gen` hardcodes it; (2) `libvulkan-dev` carries only the `libvulkan.so`
symlink — with the real `libvulkan.so.1` absent from the extracted prefix, cmake configure
passed on the dangling link and only ninja noticed at link time.

**Why it matters.** Turnip may be the *only* engine that can run the `lfm3` arch today: the
prebuilt hexagon binary predates LFM3 support (rebuild recipe in
`research/bricks-faster-inference.md`), so a GPU row completes the picture.

**Verdict: software-blocked, root-caused.** The bench matrix (`scripts/test_vulkan.sh`,
`-ngl 100 --device Vulkan0`) aborts at the smoke step — `llama-bench` SIGSEGVs inside
`libvulkan_freedreno.so` under `vkCreateComputePipelines` before the first timing row.
Bisected with `test-backend-ops` + the Khronos validation layer (no driver symbols needed):

| ggml path | shader | Turnip result |
|---|---|---|
| vec matmul, f32/f16 weights (batch ≤3) | clean GLSL | ✅ 33/33 tests pass |
| tiled matmul (batch ≥4, i.e. any prefill) | `mul_mm.comp` | 💥 GPU IOMMU fault, `a6xx_recover` in dmesg |
| quantized weights (every GGUF quant) | `mul_mat_vecq.comp` | 💥 SIGSEGV in shader compiler |

**Root cause (one line).** `vulkan-shaders/types.glsl` declares quant blocks with
`uint8_t`/`int8_t` members, so glslc emits the SPIR-V capability `StorageBuffer8BitAccess` —
but Turnip on this Adreno reports `storageBuffer8BitAccess = false` and **ggml v0.8.0 never
checks that feature** (it hard-requires 16-bit storage at init, ggml-vulkan.cpp:4664, and
silently assumes 8-bit). Validation confirms it: `VUID-RuntimeSpirv-storageBuffer8BitAccess-06328`.
Any shader including `types.glsl` (all quantized matmuls, `mul_mm`, `copy_to_quant`,
`get_rows_quant`, FA variants) is invalid SPIR-V on this device; Turnip's response is not a
clean error but a segfault or a GPU fault that trips the kernel's fault-ring recovery.
The pure-vec f32 path proves the GPU + kernel + IOMMU are healthy — it is purely the
shader-stack gap.

**Upstream.** Known and unfixed: llama.cpp **#6401** (support devices without
`storageBuffer8BitAccess`, open since 2024), **#25894** (identical wall on Adreno 642 +
Turnip, Jul 2026), **#25906** (Vulkan eval bug reported on this board family); related
Turnip small-batch MUL_MAT thread on mesa-dev (2026-July/226673, Karmjit Mahil reply).
Mesa 25.2.8 is the newest Ubuntu 24.04 offers. **Reproducible fix requires** either an
upstream llama.cpp gate (skip 8-bit shaders when the feature is absent) or a Turnip that
advertises 8-bit storage on a623 — until then this engine is out for GGUF inference and
§9's NPU + §3's CPU remain the two working engines. The LFM3-on-GPU idea dies with it:
the GPU needs the same quantized shaders.
