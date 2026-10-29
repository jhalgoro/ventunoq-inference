# Failed pathways — documented, root-caused

Every dead end hit during this project, with symptom → investigation → root cause →
evidence. Fix paths live in [ROADMAP.md](ROADMAP.md) (references like R1 below).

**Status legend:** 🔒 needs someone else (kernel/vendor/upstream) · 🛠️ buildable fix
exists, not yet done · 📝 artifact/pitfall, workaround in place

---

## F1 — QNN/QAIRT NPU inference: graph execution blocked 🔒

**The path:** Qualcomm AI Runtime (QAIRT 2.51 community SDK) → QNN HTP graph → Hexagon
V75. The "official" way to run NPU workloads.

**Symptom.** Everything up to execution works: HTP detected (V75), DSP code execution
proven (calculator unit test on unsigned PD), ONNX→DLC conversion *on-device*, SoC
whitelist bypassed (`libsocspoof.so` + sweep → ids 605/606/607/695), graph compose +
finalize on DSP reaches "Executing Graphs" — then every execute fails:

```
DSP: HAP_mmap2(fd) → paddr 0x0 → 0x80000448
skel: sharedMemoryMmap failed → QAIRT_MEM_ERROR_MAPPING (8003) → exec err 6001
```

**Investigation** (full forensics in `npu/NOTES.md`):
- DSP-side `.farf` logging localized the failure to buffer registration.
- `npu/mmap_probe.c`: the kernel's own fastrpc path **works** — legacy `MMAP(6)` ioctl
  maps a 588 KB dma-buf, DSP VA `0xe0900000`. The new `MEM_MAP(10)` ioctl is answered
  `AEE_EUNSUPPORTED` by this firmware for every size.
- The vendor HTP skel ignores kernel-provided pages for fd args and calls `HAP_mmap2`,
  which requires the vendor **`smmu_proxy` Linux service** to do the CDSP-SMMU mapping.
  That service **does not exist in this board's Ubuntu `6.8.0-1084-qcom` kernel**: no
  module anywhere, `no reserved DMA memory for FASTRPC` ×3 at boot, no fastrpc debugfs.
- Ruled out (each tested): dma-heap choice, stale userspace FastRPC (rebuilt
  qualcomm/fastrpc master on-board), SDK point-release (2.51.0 vs 2.51.200), PD session
  & firewall knobs, QNN 2.48-era libs. A patched self-built `libcdsprpc` routing all
  maps through the working legacy ioctl was implemented — it regresses device bring-up
  (DSP `find_map(fd)` is fd-keyed) and was reverted.
- Upgrading userspace (Ubuntu 26.04) does not help — the gap is kernel-side
  (`npu/NOTES.md` §"Would upgrading…").

**Root cause.** Vendor CDSP firmware expects the `smmu_proxy` kernel service for
fd-based buffer registration; the board's mainline-ish kernel doesn't ship it.
**Not fixable from userspace.** → **R3** for what would fix it.

**Consequence.** No LLM can use the NPU *through QNN* on this board. (llama.cpp's
upstream Hexagon backend sidesteps the entire QNN stack via the classic fastrpc MMAP
path — which is why BENCHMARKS §3 works.)

## F2 — GPU / Vulkan (Turnip): quantized & tiled matmuls blocked 🔒🛠️

**The path:** llama.cpp v0.8.0 `-DGGML_VULKAN=ON` on Turnip (Mesa 25.2.8, Vulkan
1.3.318, Adreno 623). The GPU is fully enumerated headless (`vulkaninfo`, group
`render`, no sudo) — the third engine of the trio.

**Symptom.** `llama-bench --device Vulkan0` SIGSEGVs inside `libvulkan_freedreno.so`
under `vkCreateComputePipelines` before the first timing row (gdb backtrace:
`ggml_vk_create_pipeline_func ← ggml_vk_load_shaders ← ggml_vk_mul_mat_vec_q_f16`).
`GGML_VK_DISABLE_F16=1`, `GGML_VK_DISABLE_MMVQ=1` don't help.

**Bisection** (`test-backend-ops -b Vulkan0` + Khronos validation layer; no driver
symbols needed — Ubuntu ships no ddebs for mesa on arm64, ports.ddebs 404):

| ggml path | shader | Turnip result |
|---|---|---|
| vec matmul, f32/f16 weights (batch ≤3) | clean GLSL | ✅ 33/33 (`results/vk_tbo_health.log`) |
| tiled matmul (batch ≥4 — any prefill) | `mul_mm.comp` | 💥 GPU IOMMU fault, `a6xx_recover` in dmesg (`results/vk_tbo_mulmat.log`) |
| quantized weights (every GGUF quant) | `mul_mat_vecq.comp` | 💥 SIGSEGV in shader compiler |

**Root cause (one line).** `vulkan-shaders/types.glsl` declares quant blocks with
`uint8_t`/`int8_t` members → glslc emits the SPIR-V capability
`StorageBuffer8BitAccess` → this Turnip reports `storageBuffer8BitAccess = false` →
**ggml v0.8.0 never checks that feature** (it hard-requires 16-bit storage at
ggml-vulkan.cpp:4664 and silently assumes 8-bit). Validation confirms it:
`VUID-RuntimeSpirv-storageBuffer8BitAccess-06328` (`results/vk_val.log`-style repro:
run any llama-bench with `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`).
Every shader including `types.glsl` (`mul_mat_vecq`, `mul_mm`, `copy_to_quant`,
`get_rows_quant`, FA variants) is invalid SPIR-V here; Turnip's response is a
segfault or a GPU fault that trips kernel fault-ring recovery — never a clean error.
The passing vec-f32 path proves GPU + kernel + IOMMU are healthy.

**Upstream.** llama.cpp **#6401** (support devices without `storageBuffer8BitAccess`,
open since 2024), **#25894** (identical wall, Adreno 642 + Turnip, Jul 2026),
**#25906** (Vulkan eval bug on this board family); mesa-dev thread
2026-July/226673 (Turnip/Adreno 702 small-batch MUL_MAT). Mesa 25.2.8 is Ubuntu
24.04's newest. → fix paths in **R2**.

**Consequence.** GPU is out for GGUF inference (all quants need `types.glsl`), which
also kills "run LFM3 on GPU while the NPU runner lacks lfm3". Working engines: CPU + NPU.

## F3 — NPU full offload of bigger models: 32-bit DSP VA ceiling 🔒

**Symptom.** K2-Horizon-4B-Q4_K_M (~2.4 GB NPU-side) with `-ngl 100` aborts mid-load:
`fastrpc_mmap failed` (error 1).

**Root cause.** This kernel maps CDSP buffers through a **32-bit IOMMU window**
("DMA64 disabled"). `npu/cumap_probe.c` measures it at **~1.75 GiB cumulative**
(256 MiB chunks → `ENOMEM` at 1792 MiB; contiguity also matters). HTP0/HTP1 both fall
back to the *same* domain `cdsp:3`, so a second session adds no address space.
`RLIMIT_MEMLOCK` = 1.9 GiB (unraisable as user) is a near-identical second wall.
Arduino documents ~3,134 MiB × 4 sessions — that requires the vendor stack.

**Budget rule of thumb:** NPU-side weights + KV + op arenas ≤ **~1.75 GiB** without
root. Fits: ≤2B-class Q4 full offload, 4B-Q4_0 at `-ngl 24`, ctx ≤8k
(`GGML_HEXAGON_MBUF=128,128` saves 256 MiB for weights). → **R4** for what lifts it.

## F4 — Partial NPU offload of K2-4B: slower than CPU 📝

**Symptom.** K2-Horizon-4B at `-ngl 18` (half offload, fits the VA budget): decode
**2.6 t/s vs 7.9 t/s pure-CPU** (pp21.4 does beat CPU's 19.1).

**Cause.** Per-token CPU↔NPU ping-pong when neither engine holds most of the graph.
**Rule: offload fully, or keep ≥⅔ of layers on the NPU.** Partial-offload ladder data:
`results/bench_hex2_K2H-4B-Q4KM_npu_ngl18.csv`; the `-ngl 24` Qwen-4B success shows
the rule's other side.

## F5 — LFM3 on the NPU runner: arch not supported 🛠️

Arduino's prebuilt `llamacpp/20261020` runner supports `qwen35` ✔ `k2_horizon` ✔
`lfm2` ✔ — but **not `lfm3`** (its ggml is 0.27.0; LFM3 landed in v0.8.0/b11960).
Not a failure so much as a version skew with a clean fix: re-run Arduino's own
`build-llamacpp.yml` workflow against a newer llama.cpp ref → **R1**.

## F6 — Environment gotchas (each cost real time; all solved sudo-free) 📝

| Gotcha | Resolution |
|---|---|
| Ubuntu's `glslang-tools` does **not** ship `glslc` — but `vulkan-shaders-gen` hardcodes it | separate `glslc` + `libshaderc1` packages (now installed system-wide) |
| `libvulkan-dev` ships only the `libvulkan.so` **symlink**; extracted prefix had a dangling link — cmake configure passed, ninja died at link | symlink prefix → system `libvulkan.so.1.3.275` |
| Hexagon runner: missing unversioned `libcdsprpc.so` | symlink dir on `LD_LIBRARY_PATH` (`/tmp/opencode/libfix`) |
| Hexagon runner **rejects `--device CPU`** (doesn't enumerate CPU) — cycle-1 "controls" were void | correct control = no `--device` + `-ngl 0` |
| No compiler/cmake/pip on the board, no sudo (during the run) | official prebuilt binaries; toolchain staged from `.deb`s into `/tmp/opencode/vkdeps`; `websockets` vendored in `vendor/` |
| Official `Qwen/*-GGUF` repos gated (401) | **unsloth** mirrors, per project rule: compare only against unsloth quants |
| No mesa ddebs on Ubuntu ports (arm64) — 404 on both pool paths | symbolication skipped; validation-layer + op-bisection route instead (F2) |

## F7 — Harness/data artifacts (don't trust these numbers blindly) 📝

- **tg32 is warmup garbage** on the hexagon runner (K2 CPU control: 0.38 t/s at tg32
  vs 7.9 at tg128) — use tg128+ rows.
- **K2 repo/file naming trap:** repo is `K2-Horizon-3.7B-GGUF` but files are
  `K2-Horizon-4B-*` — first 4B pass 404'd; fixed in `scripts/eval_k2_4b.sh`.
- **`test-backend-ops` grep traps:** `-o` regex matches the bare op name (`MUL_MAT`),
  not the full test string — `MUL_MAT\(.*` silently runs **0/0 tests and prints OK**;
  and `: OK` lines are ANSI-colored, so naive greps report 0 passes on green runs.
- **`npu/NOTES.md` early claim "GPU ruled out — renderD128 bound to msm_dpu, Turnip
  cannot bind"** was wrong — later disproven: Turnip enumerates fine (F2). Kept as
  written for the record; this file is the correction.
- §1's CPU baselines ran un-pinned 8-thread; NPU comparisons (§3) use re-run pinned
  controls on the same binary — never compare across those two regimes.
