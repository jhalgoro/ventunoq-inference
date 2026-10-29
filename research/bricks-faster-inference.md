# Arduino "Bricks" — how they are built, and the faster-inference path they open

Research date 2026-10-29 (docs via CloakBrowser; code via GitHub raw/API; runtime verified on-board).

## 1. What Bricks are (Arduino concept)

Per `docs.arduino.cc/software/app-lab/bricks/about-bricks` (rev 2026-10-19):
**Bricks are the modular building blocks of an Arduino App Lab application** — reusable
components that package complex features (AI models, web UIs, databases, sensors) so an app
author wires them up in a few lines of Python. Technically a Brick is a Python module that
runs either **in-process (pure Python)** or as a **client to a background service** (a Docker
container). The IDE/CLI resolves which service a brick needs *per board*.

Repo: `arduino/app-bricks-py` (MPL-2.0) — 27 brick modules under
`src/arduino/app_bricks/` (`llm`, `cloud_llm`, `asr`, `object_detection`, `ocr`, …), backing
services under `src/arduino/app_services/` (`llamacpp`, …), container recipes under
`containers/`, model catalog under `models/models-list.yaml` + `models-handlers.yaml`.

## 2. The LLM brick pipeline on Ventuno Q

```
arduino:llm (brick)  →  app_services/llamacpp (service)  →  container image
   llamacpp-npu-runner   (ventunoq — this board)      llama.cpp + Hexagon backend
   llamacpp-runner       (unoq — CPU only, 4 thr)
```

`service_compose.yaml` (default = ventunoq) for `llamacpp-models-runner`:
`-b 1024 -ub 512`, flash-attn on, `-c 16384`, `-t 4`, `--cpu-mask 0x0f` (big cores only),
`GGML_HEXAGON_OPBATCH=2048`, `GGML_HEXAGON_MBUF=256,256`, `mem_limit 12g`, `cpuset 0-3`,
devices `/dev/dma_heap/system` + `/dev/fastrpc-cdsp`, memlock unlimited, models dir
`/var/lib/arduino-app-cli/models/llamacpp/` mounted at `/models`. (unoq override swaps to the
CPU runner image: 2.5 GB limit, 4 threads, no DSP devices.)

### How the binary is built (`.github/workflows/build-llamacpp.yml`)
- Manual-dispatch workflow in app-bricks-py: clones **`ggml-org/llama.cpp` master** (or any
  fork/ref you pass as input), applies `fix-race.patch` (Arduino's fix for a `QURT_ETIMEDOUT`
  stall in `ggml/src/ggml-hexagon/htp/main.c`), and cross-compiles from amd64 inside
  `ghcr.io/snapdragon-toolchain/arm64-linux:v0.8` with preset `arm64-linux-snapdragon-release`.
- A second job builds a plain CPU variant (`-mcpu=cortex-a53`, debian trixie arm64).
- Reproducible tarballs published as GitHub release tag `llamacpp/<date>`; current pin
  **`llamacpp/20261020`**, asset `llamacpp-hexagon-20261020.tar.gz`
  (sha256 `35e7956c…98417`, pinned by `--checksum` in the runner Dockerfile).
- **Because the workflow takes `llama_cpp_repo`/`llama_cpp_ref` inputs, anyone can rebuild the
  exact NPU-enabled llama.cpp from a newer master by dispatching the workflow on a fork.**

### Runtime wiring (`run-model-router.sh` + `configure-llamacpp.py`)
- `configure-llamacpp.py` parses each GGUF header (never the weights) to compute (a) bytes
  that land on the NPU and KV size, hence (b) how many **HTP sessions** are needed
  (`--print-ndev`) and any context cap (`--print-ctx`).
- Sessions: `GGML_HEXAGON_DEVICES=n` → `--device HTP0[,HTP1,…] -ngl 100 --load-mode none
  --models-preset /models/models.ini`. `LLAMACPP_DISABLE_NPU_SUPPORT=true` → `-ngl 0`.
- From `SESSION_ALLOCATION.md`: a session = a DSP protection domain, 4 per process,
  **~3,134 MiB of DSP address space each**; KV cache lives on the DSP; quantized KV is unsafe
  across sessions; compute buffers overlap across sessions (measured gemma-4-E4B: 2,600 MiB
  DSP footprint for 1,620 MiB of weights+KV).
- From `configure-llamacpp.py` comments: upstream llama.cpp keeps **all K-quants on the CPU**
  in the Hexagon backend, but **the Arduino build (llama.cpp PR #28994) repacks q4_K/q5_K/q6_K
  onto the NPU**; only q2_K/q3_K stay CPU-side. `get_rows`-only tensors (token embeddings)
  always stay on CPU. Hybrid models (Qwen3.5, LFM2) keep a recurrent state on the NPU
  (256 MiB allowance).
- Catalog (`models-list.yaml`, handler `hf-handler`): Q4_0 is the house quant —
  `Qwen3.5-0.8B-Q4_0` (507 MB), `gemma-3-1b-it-Q4_0` (722 MB), `MiniCPM5-2B-Q4_K_M` (1.5 GB),
  `gemma-4-E2B/E4B_q4_0` (3.4/5.3 GB, ventunoq-only). Bring-your-own: App Lab UI or
  `arduino-app-cli model install llamacpp:unsloth/<repo>:<QUANT>`.

## 3. Why this is the faster-inference path (§7's blocker does NOT apply)

§7/npu/NOTES.md killed the **QNN/QAIRT** path: its vendor HTP skel registers execute arenas by
fd, resolved DSP-side via `HAP_mmap2`, which needs the vendor **smmu_proxy** kernel service —
absent from this Ubuntu kernel.

The upstream **Hexagon backend** (`ggml/src/ggml-hexagon`, what this build ships) uses
**dspqueue over the classic fastrpc `MMAP` ioctl** — exactly the kernel path proven working by
`npu/mmap_probe.c` (588 KB dma-buf → DSP VA `0xe0900000`). No QNN, no smmu_proxy, no soc_id
spoofing.

### Verified on this board (2026-10-29)
1. Downloaded `llamacpp-hexagon-20261020.tar.gz`, sha256 matches the Dockerfile pin; extracted
   to `/tmp/opencode/pkg-snapdragon/` (`llama-server`, `llama-bench`, `libggml-hexagon.so`,
   `libggml-htp-v{73,75,79,81,85}.so` — **v75 = this board's HTP**).
2. The only missing runtime dep was the unversioned `libcdsprpc.so` symlink (board ships
   `libcdsprpc.so.1`): `ln -s … /tmp/opencode/libfix/libcdsprpc.so` + `LD_LIBRARY_PATH`.
3. `llama-cli --list-devices` as the user → **`HTP0: Hexagon` enumerated**. `/dev/fastrpc-cdsp`
   is world-rw; the DSP dma-heaps carry an ACL granting the `fastrpc` group (user `arduino` is
   a member). **No sudo anywhere in the path.**

### Architecture support in this build (ggml 0.27.0, symbols in `libllama.so`)
| Our model | Arch in hexagon build | NPU-testable? |
|---|---|---|
| Qwen3.5 (2B/4B/9B) | `qwen35`/`qwen35moe` ✔ | **yes** |
| K2-Horizon (4B/7B) | `k2_horizon` ✔ | **yes** |
| LFM2.5-2.6B | `lfm2`/`lfm2moe` ✔ (hybrid, state on DSP) | **yes** |
| **LFM3-1.7B** | **absent** (needs llama.cpp ≥ v0.8.0 / ggml 0.28) | **no** — until a rebuild |

**LFM3 gap → concrete path:** dispatch `build-llamacpp.yml` on a fork of app-bricks-py with
`llama_cpp_ref` = a master that includes the `lfm3` arch (our v0.8.0 baseline works); the
workflow produces the same signed-by-checksum tarball the runner Dockerfile can consume:

```bash
gh repo fork arduino/app-bricks-py --default-branch-only
gh workflow run build-llamacpp.yml -R <you>/app-bricks-py \
  -f llama_cpp_repo=ggml-org/llama.cpp -f llama_cpp_ref=<ref-with-lfm3> \
  -f release_version=$(date +%Y%m%d)
# → release tag llamacpp/<release_version>, asset llamacpp-hexagon-<release_version>.tar.gz
```

Workflow inputs confirmed from the saved copy (`/tmp/opencode/bricks/build-llamacpp.yml`):
`llama_cpp_repo` (owner/repo, optional), `llama_cpp_ref` (branch/tag/SHA, optional),
`release_version` (empty ⇒ no release); jobs run on `ubuntu-latest` (amd64) inside
`ghcr.io/snapdragon-toolchain/arm64-linux:v0.8`, so **no board-side toolchain is needed**.

### Quantization guidance (from the brick's own code)
- Q4_0 = every weight byte NPU-bound (the catalog's choice; LFM3's QAD Q4_0 quant would fit
  this profile perfectly once the arch is supported).
- Q4_K_M/Q5_K_M/Q6_K = NPU-bound **only in the Arduino build** (PR #28994 repack) — so our
  existing Q4_K_M files are candidates as-is.
- q2_K/q3_K and token embeddings stay on CPU → mixed execution either way.

## 4. Results (cycle 1, 2026-10-29) — the NPU is dramatically faster

`scripts/test_hexagon.sh`: per model download → `llama-bench --device HTP0 -ngl 100`
(`-t 4 --cpu-mask 0x0f -b 1024 -ub 512 -fa on`, `GGML_HEXAGON_OPBATCH=2048`,
`GGML_HEXAGON_MBUF=256,256`) → delete GGUF. CSVs: `results/bench_hex_*_npu.csv`.

| Model on **HTP0** (Q4) | pp512 | pp8192 | tg128 | CPU ctrl* (pp512 / tg128) | speed-up (pp8192 / tg128) |
|---|---|---|---|---|---|
| Qwen3.5-0.8B-Q4_0 | **1366.3** | 1233.3 | **43.0** | 148.7 / 31.9 | **13.6× / 1.35×** |
| Qwen3.5-2B-Q4_0 | **863.3** | 806.7 | **24.3** | 69.8 / 16.1 | **14.9× / 1.51×** |
| LFM2.5-2.6B-Q4_K_M | **679.2** | 571.7 | **17.7** | 40.1 / 13.1 | **21.1× / 1.35×** |
| Qwen3.5-4B-Q4_0, `-ngl 24` partial | 70.8 | 53.7 | 10.3 | — | (v0.8.0-CPU Q4_K_M: 19.8/4.2) |
| K2-Horizon-4B-Q4_K_M, `-ngl 18` partial | 37.4 | 21.4 | **2.6** ✗ | 19.1 / 7.9 | 1.9× / **0.33×** |

\* same-binary control (`scripts/test_hexagon2.sh`, bare `-ngl 0` — this build does not
enumerate `CPU` as a `--device`), same service tuning `-t 4 --cpu-mask 0x0f -b 1024 -ub 512
-fa on`. The pinned control is much faster than the un-pinned 8-thread §3–5 baselines
(pinning alone nearly doubles decode), so these are the honest speed-ups.

The q4_K→NPU repack (Arduino PR #28994) is confirmed working in practice — LFM2.5's Q4_K_M
ran fully offloaded. Prefill gains (9–21×) dwarf decode (1.35–1.5×), exactly matching the
shared-LPDDR bandwidth analysis in `research/memory-map.md`. Partial offload only pays when
the NPU holds most layers (Qwen-4B `-ngl 24`: 10.3 t/s decode, 2.4× over CPU); a half/half
split (K2-4B `-ngl 18`) is *slower than CPU* (2.6 vs 7.9 t/s) — per-token CPU↔NPU
ping-pong dominates. Quality-parity check of the repacked path: `scripts/eval_hex_quality.sh`
(LFM2.5 GSM8K N=20 NPU vs CPU, `results/gsm8k_hexq_*.log`).

### K2-Horizon-4B failure — root cause: 32-bit DSP VA aperture ≈ 1.75 GiB

Crash during load: `ggml-hex: HTP0 buffer mapping failed : domain_id 3 size 561401856
fd 19 error 0x00000001` → `fastrpc_mmap failed` → abort. Direct reproduction:
`npu/cumap_probe.c` replays the backend's exact registration path
(`ALLOC_DMA_BUFF` + legacy `MMAP` ioctl 6 — the `MEM_MAP` ioctl 10 is unsupported here and
returns AEE_EBADPARM `0x8000040E` as a raw negative return, which is why `errno` reads 0):

- 256 MiB buffers map fine up to **1792 MiB cumulative**; the next fails `ENOMEM`.
  DSP VAs descend from `0xd0000000` — a **32-bit window** (log: "DMA64 disabled").
- A second 536 MiB buffer fails at only 536 MiB cumulative → the allocator also needs
  *contiguous* VA holes (fragmentation).
- `RLIMIT_MEMLOCK` (soft=hard 1910 MiB, unraisable without root) is a second wall at about
  the same height, but the VA aperture binds first.
- `-ngl 100` with 2 devices does **not** help: `FASTRPC_GET_DOMAINS` fails (`0x6c`) and the
  static fallback puts **HTP0 and HTP1 in the same domain `cdsp:3`** — one shared window,
  so the documented "4 sessions × 3134 MiB" never materializes on this kernel. (Arduino's
  container also sets `memlock unlimited`, which wouldn't fix the aperture either.)

K2-4B needs ≈2.4 GB mapped (3.16 GB file minus ~0.75 GB CPU-side embeddings) → exceeds
1.75 GiB → died on the 535 MiB tensor. **Practical rule: NPU-side weights + KV + op-arenas
(MBUF 256+256 MiB) must fit ~1.75 GiB.** `scripts/test_hexagon2.sh` therefore runs partial
ladders (`-ngl 100→24/18→…`, `MBUF=128,128`) for Qwen3.5-4B and K2-4B, plus valid CPU
controls (no `--device` flag) for all cycle-1 models.

### What this means for the product path

- Sub-2B-class models (the catalog's Qwen3.5-0.8B/2B Q4_0, gemma-3-1B) run **fully offloaded
  with huge margins** — this is why the catalog is Q4_0-centric and small.
- 4B-class works **today via partial offload** (Qwen3.5-4B-Q4_0 `-ngl 24`: 10.3 t/s decode,
  54 t/s prefill @8k) — but full offload needs Arduino's larger-VA environment (64-bit-DSP
  fastrpc/driver, or genuinely separate-domain sessions). K2-4B is not a partial-offload
  candidate (decode regresses below CPU at half-offload).
- LFM3 still needs the rebuild (arch absent from ggml 0.27.0), **and** will then fit easily
  (1.06 GB weights → NPU-side well under the ceiling).

## 5. Secondary path found along the way (unexplored)

GPU: board exposes `/dev/dri/renderD128` + Mesa 25.2.8 Turnip (Adreno a6xx) + user in
`render`. llama.cpp's Vulkan backend could be built from source (needs sudo apt
glslang-tools libvulkan-dev). Expected value after cycle 1: **low for decode** (the GPU
shares the same LPDDR bandwidth that now caps HTP decode too) — keep it only as a prefill
fallback for models above the DSP VA ceiling.
Memory sharing model: see `research/memory-map.md`.
