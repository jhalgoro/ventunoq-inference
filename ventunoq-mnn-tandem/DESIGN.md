# MNN in Tandem on the Arduino VENTUNO Q
### CPU + GPU + NPU cooperative inference — feasibility analysis and architecture
**Status:** research / architecture deliverable (no builds) · **MNN:** 3.6.2 (`include/MNN/MNNDefine.h`) · **Date:** 2026-11-09

---

## 0. Verdict up front

**Feasible, but not for the reason usually assumed.**

All three engines exist and all three have a real MNN backend. Nothing here is speculative:

| Engine | Silicon | MNN backend | Forward type | Build flag |
|---|---|---|---|---|
| CPU | Octa-core Arm (QCS8275) | `source/backend/cpu` (+ `arm82` FP16) | `MNN_FORWARD_CPU = 0` | always on, `MNN_ARM82=ON` |
| GPU | Adreno (QCS8275) | `source/backend/opencl`, `source/backend/vulkan` | `MNN_FORWARD_OPENCL = 3`, `MNN_FORWARD_VULKAN = 7` | `MNN_OPENCL`, `MNN_VULKAN` |
| NPU | **Hexagon Tensor AI Processor, up to 40 dense TOPS** | `source/backend/hexagon` (native HTP) **and** `source/backend/qnn` | `MNN_FORWARD_HEXAGON = 10`, `MNN_FORWARD_QNN = 16` | `MNN_HEXAGON`, `MNN_QNN` |

The hard part is not device support. It is that **MNN binds exactly one forward type per Session/Module**. There is no `backendMask`, no multi-device scheduler, no cross-device tensor router anywhere in the tree — `grep -rn "backendMask" include/ source/` returns **zero hits**. MNN's multi-device story is:

1. **One** primary backend per session, plus
2. a **single** `backupType` fallback (`ScheduleConfig::backupType`, default `MNN_FORWARD_CPU`) used only when an op is *unsupported* — see `source/core/Schedule.cpp:136`: `"Can't Find type=%d backend, use %d instead"`.

That is *fault-tolerant fallback*, not tandem execution. So "in tandem" is not a configuration option. **It is a scheduler we build on top of the Module API.** That scheduler is this project.

**And the honest performance caveat**, which §3 develops quantitatively: for a *single small model at batch 1*, splitting layers across three devices usually **loses** to running the whole model on the best single device, because cross-device handoff is expensive relative to per-layer compute. Tandem pays off in **throughput** (batch ≥ 2, pipeline parallelism), **model capacity** (model doesn't fit one device's memory), and **structural asymmetry** (LLM prefill/decode, ensembles, multi-model pipelines). A design that ignores this produces a demo that is slower than `-forward_type 0`. Everything below is organised around earning the win rather than assuming it.

---

## 1. Target hardware — verified

Source: `docs.arduino.cc/hardware/ventuno-q/` (HTTP 200). Arduino's own framing: App Lab is *"the converging point of development across **NPU, CPU, GPU, and MCU** platforms."*

**MPU — Qualcomm Dragonwing™ IQ8 (QCS8275)**, running **Ubuntu Linux** (Debian support "coming soon"):
- Octa-core Arm® Cortex®
- **Adreno GPU/VPU**
- **Hexagon Tensor AI Processor (NPU): up to 40 dense TOPS**
- Qualcomm Spectra 692 ISP
- Docs state *"Adreno GPU/VPU: Arm® Cortex® A623 at 877 MHz"* — this sentence is garbled marketing copy (it conflates the GPU/VPU with a CPU core name and a clock). **Do not trust it.** Core topology, big/mid/little split, L2 sharing and actual clocks must be read from `/proc/cpuinfo` and `/sys/devices/system/cpu/` on the board (§10).

**MCU — STMicroelectronics STM32H5F5**, Arm Cortex-M33 @ 250 MHz, 4 MB flash, 1.5 MB RAM, RPC library to the MPU.
*Deliberately out of scope for inference* — 1.5 MB RAM cannot hold a useful tensor arena. It belongs in the design as a **sensor/actuator sidecar** (pre-trigger, gating, always-on wake) so the Linux side can stay parked. See §7.4.

**I/O:** 2.5 GbE, HDMI muxed w/ MIPI DSI, USB-C (host/device + video), 2× USB 3.0 A, Wi-Fi 6, BT 5.3, M.2 NVMe Gen4, UNO shield headers, Qwiic, 40-pin RPi HAT header.

**Why this matters to the design:** QCS8275 is a **big-accessible-memory, shared-SoC** target — CPU, GPU and NPU all hang off the same DRAM controller. That is the *good* case for layer-splitting: a handoff is a DMA copy inside one coherent-ish memory system, not a PCIe hop. It also means the three engines **contend for the same bandwidth**, which caps realistic tandem gain (§3.3).

---

## 2. Software target — MNN 3.6.2, verified inventory

Clone: `/home/user/projects/mnn-ventuno-q/mnn` (361 MB, 8594 files).
`source/backend/` → `arm82 cpu coreml cuda hexagon hiai metal musa neuropilot nnapi opengl opencl qnn rknn tensorrt vulkan`

The two that make this board interesting:

### 2.1 Native Hexagon HTP backend (`MNN_HEXAGON`)
`source/backend/hexagon/` — `backend/`, `execution/`, `htp-ops-lib/`, `schema/`, `tests/`, `README.md`.

From `README.md`, verbatim workflow:
```bash
cd source/backend/hexagon/htp-ops-lib
bash build.sh v79          # DSP arch: v73 / v75 / v79 ... (default v79, build.sh:6)
# -> outputs/libMNN_htpops.so        (aarch64 stub, runs on CPU)
# -> outputs/libMNN_htpops_skel.so   (Hexagon skeleton, runs on NPU/DSP)
cmake .. -DMNN_HEXAGON=ON   # README: "Don't need Hexagon SDK."
```
Deployment (README): `LD_LIBRARY_PATH` for the stub, `ADSP_LIBRARY_PATH` for the skeleton, e.g. `/vendor/lib/rfsa/adsp`.

Runtime internals (`HexagonRuntime.cpp`): `rpcmem_init()` (:725), `rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, size)` (:408), `rpcmem_to_fd()` (:413), `dlopen("libcdsprpc.so")` with error `"Open libcdsprpc.so failed"` (:502), `dlopen(HTP_OPS_DL_PATH)` (:506). Compiled with `-march=armv8.2-a+fp16` (`CMakeLists.txt`). Separate allocators for static / command / weight / lazy-weight buffers (`HexagonRuntime.hpp:103-107`).

Related knobs: `MNN_HEXAGON_ASAN` (memory-consistency checks), `MNN_HEXAGON_APP_POWER_HOLD` (hold DSP power vote across flushes — **relevant to us**, §8).

### 2.2 QNN backend (`MNN_QNN`)
`MNN_FORWARD_QNN = 16`, `MNN_QNN_ONLINE_FINALIZE` (ON by default), `MNN_QNN_OFFLINE_CONTEXT`.
Offline path: `tools/cpp/MNN2QNNModel.cpp` →
```
MNN2QNNModel <qnnSDKPath> <socId> <hexagonArch> <srcMNNPath> <outputDir> [shapes...]
```
emits a config JSON containing `soc_id` and `dsp_arch: "v<N>"` (`generateConfigFile`, :23/:76/:77). Also `tools/cpp/compilefornpu.cpp`. Debug aid: `MNN_QNN_DUMP_INTERMEDIATE_OUTPUTS` + `MNN_QNN_DUMP_DIR` dumps every native activation — **the single most useful accuracy-parity tool we have** (§9.2).

`MNN_NPU_BACKENDS_SHARED` (`CMakeLists.txt:293`) builds NPU backends as **explicitly-loaded shared plugins** — the mechanism for co-loading NPU runtimes in one process without symbol clashes.

> **Decision: Hexagon-native as primary NPU leg, QNN as fallback.** Hexagon-native gives MNN-owned graph construction (contiguous command streams, MNN's own allocators → we control the buffers we need for zero-copy handoff). QNN's graph-finalize model is more opaque and its offline path wants a host-side SDK + fixed `soc_id`, which fights a model-agnostic autotuner. Keep QNN because it covers ops the HTP op-lib lacks.

---

## 3. The real problem: handoff economics

### 3.1 What a boundary actually costs
A cut between device A and device B pays, per inference:
```
T_boundary = T_rpc_or_launch  +  T_layout  +  T_memcpy  +  T_stall
```
- `T_rpc_or_launch` — FastRPC round-trip to the Hexagon skeleton. Order **10–50 µs**, independent of size. This is the tax that kills fine-grained splitting.
- `T_memcpy` — activation bytes / effective DMA bandwidth. A 224×224×64 fp32 feature map is **12.8 MB**; at a realistic ~5–10 GB/s single-stream DMA that is **1.3–2.6 ms** — often *more than the layer's compute time*.
- `T_layout` — MNN CPU defaults to NC4HW4; OpenCL uses image/buffer layouts; QNN/HTP wants NHWC. A boundary crossing two layouts triggers a transpose that MNN implements as a gather — routinely as slow as the memcpy.
- `T_stall` — the sync bubble if the consumer isn't already queued.

### 3.2 Why naive layer-splitting loses, and the fix
Model takes `T` on the best single device. Split into `S` segments with per-boundary cost `c`:
```
T(S) = T/S + (S-1)·c        →  optimal S* = sqrt(T/c),  T(S*) ≈ 2·sqrt(T·c) − c
```
MobileNetV2-class model, `T ≈ 5 ms`, `c ≈ 0.2 ms` → `S* ≈ 5`, best case `≈ 2.0 ms` — a **2.5× speed-up ceiling**, and only if the segments are perfectly balanced *and* `c` is genuinely 0.2 ms. Halve `c` and you gain 1.4×; double it and tandem is a wash.

**Three design rules fall out of this arithmetic:**
1. **Few, fat segments.** Target 2–4 boundaries, never per-layer. The design must make cut *count* the primary budget, not cut *placement*.
2. **Cut where activations are small, not where compute is even.** Activation size scales as `H·W·C`. The cheapest cuts in any CNN are **deep and narrow** (late stages, 7×7×1280 ≈ 0.46 MB fp32) and the most expensive are **early and wide** (112×112×32 ≈ 1.6 MB, 56×56×64 ≈ 0.8 MB). The cost model must be driven by **tensor bytes**, not FLOPs.
3. **Never cut around a downsample.** Stride-2/pool boundaries are where `H·W` collapses — the single best cut sites. Conversely, never cut immediately before an upsampling/decoder stage.

### 3.3 The ceiling: shared DRAM bandwidth
CPU, Adreno and Hexagon share one memory controller. Once all three are saturated the aggregate is **bandwidth-bound, not compute-bound**, so `1/T_tandem < 1/T_cpu + 1/T_gpu + 1/T_npu`. Realistic expectation for a bandwidth-heavy vision net: **1.3–2×**, not 3×. Where tandem legitimately approaches the sum is **compute-bound, weight-stationary** work (large FC/attention blocks resident in TCM), and where the model **doesn't fit** one device's usable arena — then tandem is not an optimisation but a requirement.

**Therefore the harness must be able to report "tandem is not worth it here" and prove it.** A partitioner that always returns `S>1` is a broken partitioner.

---

## 4. MNN extension points we will build on (all verified)

| Need | API | Location |
|---|---|---|
| Slice graph to a sub-range | `ScheduleConfig::Path{inputs, outputs, Mode}` with `Op=0` / `Tensor=1` | `Interpreter.hpp:35-55` |
| Build a module over a subgraph | `Module::extract(inputs, outputs, fortrain, subGraph)`; `struct SubGraph{inputs, outputs}` | `Module.hpp`, `Executor.hpp` |
| Pin a module to a device | `Module::Config::BackendInfo{MNNForwardType type; BackendConfig* config}` | `Module.hpp:51-59` |
| Share one runtime across modules | `Module::load(..., std::shared_ptr<RuntimeManager>, config)` | `Module.hpp` |
| Multiple runtimes in one process | `RuntimeInfo = std::pair<std::map<MNNForwardType, std::shared_ptr<Runtime>>, std::shared_ptr<Runtime>>` | `Interpreter.hpp` |
| Explicit runtime per type | `Executor::_getOrCreateRuntime(MNNForwardType, BackendConfig*, numThread)`; `Executor::getRuntime()` | `Executor.hpp` |
| Probe op-level cost | `runSessionWithCallBackInfo(session, before, after, TensorCallBackWithInfo)` with `OperatorInfo{name(), type(), flops() /*MFlops*/}` | `Interpreter.hpp:426`, `:87` |
| Session-level cost | `getInfo(SessionInfoCode)` → `MEMORY=0` (MB), `FLOPS=1` (MFlops), `BACKENDS=2`, `RESIZE_STATUS=3` | `Interpreter.hpp:444-458` |
| User-owned I/O buffers (for zero-copy) | `Session_Input_User=3`, `Session_Output_User=5` | `Interpreter.hpp:140-146` |
| Device tensor + fence | `Tensor::createDevice(...)`, `Tensor::map/unmap(MapType,...)`, `Tensor::wait(MapType mtype, bool finish)` | `Tensor.hpp:91-101, 300-306` |
| Shape-only pass (plan, don't run) | `Module_Forward_Separate=16` / `Module_Forward_Combine=17`; `Session_Resize_Defer=7`; `traceOrOptimize()` | `Interpreter.hpp:173-174` |
| Per-op fallback policy | `Session_Backend_Fix=8` / `Session_Backend_Auto=9`; `ScheduleConfig::backupType` | `Interpreter.hpp:152-153` |
| Op-availability probe | `RuntimeManager::isBackendSupport(const std::vector<MNNForwardType>&)` | `Executor.hpp:143` |
| Device introspection | `RuntimeManager::getDeviceInfo(const std::string& key, MNNForwardType, std::string& value)` | `Executor.hpp:153` |
| GPU tuning budget | `HintMode::MAX_TUNING_NUMBER=0`, `OP_ENCODER_NUMBER_FOR_COMMIT=9` | `Interpreter.hpp:209, 243` |
| CPU pinning / topology | `HintMode::CPU_CORE_IDS=14`, `CPU_LITTLECORE_DECREASE_RATE=6` | `Interpreter.hpp:227, 255` |
| Weight/featuremap staging | `ExternalPathType`: `EXTERNAL_FEATUREMAP_DIR=1`, `EXTERNAL_WEIGHT_DIR=2`, `EXTERNAL_NPU_FILE_DIR=3` | `Interpreter.hpp:269-282` |
| GPU kernel-cache reuse | `RuntimeManager::setCache(file)` + `updateCache()` | `Executor.hpp:121` |

Existing tools to reuse rather than rewrite: `GetMNNInfo`, `MNNV2Basic`, `ModuleBasic`, `backendTest`, `getPerformance`, `Profiler`, `fuseTest`, `modelCompare`, `checkInvalidValue`, `SequenceModuleTest` (all in `tools/cpp/`).

**The load-bearing constraint:** `createRuntimeManager(std::vector<ScheduleConfig>&)` is documented **"Deceperate, the same as createRuntimeManager(configs[0])"** (`Executor.hpp:110-114`). Multi-config session scheduling is *not* a thing. The only sound route to true multi-device execution is **one `RuntimeManager` + one `Module` per device, glued by us at the tensor level.** That is the architecture in §5.

---

## 5. Architecture — the Tandem Partitioner

```
                        ┌──────────────────────────────────────┐
                        │  1. PROFILE  (offline, once/model)   │
                        │  per-op t̂ on CPU|GPU|NPU  (ms, bytes)│
                        │  op-compat matrix via backup probe   │
                        └───────────────────┬──────────────────┘
                                            │  DAG + cost + compat
                        ┌───────────────────▼──────────────────┐
                        │  2. PARTITION  (DP over cut points)  │
                        │  min makespan s.t. #cuts ≤ K,        │
                        │  bytes(boundary), layout/quant agree │
                        └───────────────────┬──────────────────┘
                                            │  segments → devices
 ┌──────────────┐  ┌──────────────┐  ┌──────▼───────────┐
│ Mod_cpu      │  │ Mod_gpu      │  │ Mod_npu          │   3. EXECUTE
│ BackendInfo  │  │ BackendInfo  │  │ BackendInfo      │   DAG order,
│ {CPU}        │  │ {OPENCL}     │  │ {HEXAGON}        │   1 thread each,
│ RuntimeMgr   │  │ RuntimeMgr   │  │ RuntimeMgr       │   double-buffered
└──────┬───────┘  └──────▲───────┘  └──────┬───────────┘   ring, fences
       │  shm ring ──────┘                 │               4. HANDOFF
       └───────────────► shm ring ◄────────┘               dmabuf/ion,
                                                            Tensor::wait()
                        ┌───────────────────┴──────────────────┐
                        │  5. MEASURE & RETUNE (online)        │
                        │  realised vs predicted, re-solve     │
                        └──────────────────────────────────────┘
```

### 5.1 Stage 1 — Profile (per model, once)
- Walk ops with `runSessionWithCallBackInfo` on a **CPU-only** session to get the canonical op list: `OperatorInfo::name()`, `type()`, `flops()`.
- For each candidate device, build a full-model session on that device and take **per-op wall time** from the same callback (before/after pair) → `t_op[op][dev]`. `Session_Debug` mode is required for callbacks to fire.
- **Compatibility matrix** `support[op][dev]`: probe by forcing `Session_Backend_Fix` and watching for the `Schedule.cpp:136` fallback print, or by building single-op micro-modules and checking `isBackendSupport`. Anything unsupported is pinned to a device that supports it, or the segment is split further.
- **Activation bytes** per candidate boundary: enumerate tensor `size()` at the cut. This — not FLOPs — is the boundary cost driver (§3.2).
- Record **layout class** per device (`NC4HW4` CPU / image-buffer OpenCL / NHWC HTP) so the partitioner can *price* a layout conversion instead of discovering it at runtime.
- Use `Session_Resize_Defer` + `Module_Forward_Separate` to get shapes/allocations without paying compute, and `getInfo(MEMORY/FLOPS)` for the session-level totals.
- Persist as a model-signed artifact (mirroring `MNN2QNNModel`'s `_<socId>_<hexagonArch>` signature convention) so profiling is amortised.

### 5.2 Stage 2 — Partition (dynamic programming)
For a linearised DAG (MNN's execution order is already a topological sequence), with `N` ops, `K ≤ 4` cuts:

```
minimise   makespan = max_dev( Σ t_op[op][dev] )  +  Σ_bytes(b)/BW  +  Σ c_layout(b) + Σ c_rpc(b)
s.t.       #cuts ≤ K
           contiguous(op ranges) assigned to one device
           support(op,dev)=1  ∀ op in segment      (else insert CPU fallback)
           segment memory ≤ arena[dev]
```
Exact DP in `O(N²K·D)` — trivially fast at `N≈150, K≤4, D≤3`; emit the top-`k` plans, not one, so the runtime can pick by measured latency. Enforce the §3.2 rules as hard constraints: **prefer post-pool cuts**, **cap boundary bytes** (`bytes(b) ≤ θ`, default `θ = 1 MB`), and **allow the trivial plan `K=0`** (single device) to win.

Output: `vector<Segment{ops, device, inputs, outputs}>` where `inputs/outputs` are tensor names → directly feeds `ScheduleConfig::Path` with `Mode::Tensor` (semantics: *"find the pipeline that computes outputs from inputs"*), or `Module::extract(...)` + `SubGraph`.

### 5.3 Stage 3 — Build one module per segment
```cpp
// per device, once
MNN::BackendConfig bc; bc.power = MNN::BackendConfig::Power_Normal; // see §8
MNN::ScheduleConfig sc;
sc.type = MNN_FORWARD_HEXAGON;          // / CPU / OPENCL
sc.numThread = 1;                       // CPU leg only: pin to its own big cores
sc.path = {inNames, outNames, ScheduleConfig::Path::Tensor};
sc.backupType = MNN_FORWARD_CPU;        // per-op safety net inside the segment
auto rt = Executor::RuntimeManager::createRuntimeManager(sc);
rt->setCache(cachePathFor(sc.type));    // GPU kernel autotune cache — cold-start fix
rt->setHint(CPU_CORE_IDS, ...);         // CPU leg: pin to cores no one else uses
MNN::Express::Module::Config mc;
MNN::Express::Module::BackendInfo bi; bi.type = sc.type; bi.config = &bc;
mc.backend = &bi; mc.shapeMutable = false;   // fixed shapes ⇒ no per-infer resize
auto mod = Module::load(inNames, outNames, modelBuffer, len, rt, &mc);
```
`shapeMutable=false` is important: it stops `resizeSession` firing every call, which on GPU also re-triggers tuning.

### 5.4 Stage 4 — Handoff (where the project is won or lost)
- **Buffers:** a fixed ring of `dmabuf`/`ion`-backed allocations shared by all three legs. Rationale is concrete: `HexagonRuntime.cpp:408` already allocates via `rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED)` and exposes `rpcmem_to_fd()` (:413) — **the NPU leg already hands us an fd.** Passing that same fd to the CPU (mmap) and to Adreno (`clImportMemoryARM` / `VK_EXT_external_memory_dma_buf`) is what turns a memcpy into a pointer swap. This is the single highest-leverage piece of engineering in the project.
- **Fences:** `Tensor::wait(MapType, bool finish)` (`Tensor.hpp:306`) is the documented wait primitive; build producer→consumer ordering on it rather than on `std::condition_variable` alone, otherwise we synchronise the CPU while the GPU command stream is still in flight.
- **User-owned I/O:** `Session_Input_User` / `Session_Output_User` so segment buffers outlive the session and can be reused across iterations — mandatory for double buffering.
- **Layout normalisation at boundaries:** convert **once, on the producer side, into the consumer's preferred layout**, so the copy and the transpose fuse into one DMA. If MNN inserts its own conversion *and* we do ours, we pay twice — detectable in the profile as a mystery `Raster`/`ConvertTensor` op at every boundary.
- **Fallback:** if a boundary's layout class is unknown, downgrade to `copyFromHostTensor` and let the tuner learn its true cost.

### 5.5 Stage 5 — Execute and retune
- One OS thread per leg, **pinned** (`CPU_CORE_IDS`), so the CPU leg cannot be starved by the MNN thread pool. On an octa-core, give the CPU leg the big cores and park the rest.
- Pipeline parallelism across **independent inferences** (batch/stream) is what actually fills all three devices; a single inference leaves devices idle at every boundary. Expose `--queue-depth N`.
- Online: compare realised per-segment latency to prediction, re-solve the DP when drift exceeds a threshold, hot-swap the plan (modules are already built for the neighbours).

---

## 6. The model-agnostic benchmark harness (deliverable shape)

```
ventuno-tandem/
├── core/
│   ├── dag_profile.{hpp,cpp}     # op list, t̂[op][dev], support[op][dev], boundary bytes
│   ├── cost_model.{hpp,cpp}      # c_rpc, BW, layout-class pricing; persists model-signed artifact
│   ├── partition.{hpp,cpp}       # DP over cut points; emits top-k plans; K=0 always legal
│   ├── device_ring.{hpp,cpp}     # dmabuf/ion ring, fd import per backend, fences
│   ├── scheduler.{hpp,cpp}       # pinned leg threads, double buffering, Tensor::wait
│   └── plan.hpp                  # serialised partition plan (the tunable artifact)
├── tools/
│   ├── vq_probe.cpp              # enumerate backends, isBackendSupport matrix, getDeviceInfo
│   ├── vq_bench.cpp              # one model × {cpu,opencl,vulkan,hexagon,qnn,tandem} × {1,2,4,8}
│   ├── vq_cutcost.cpp            # measure c per candidate boundary → calibrate cost model
│   └── vq_partsearch.cpp         # offline DP sweep, dump top-k plans
├── cmake/                        # per-target toolchains, §7 build matrix
└── reports/                      # latency, throughput, JSD/parity, energy-per-infer
```
Model-agnostic in the sense the brief asked for: **any `.mnn` is a layer DAG**; nothing keys on MobileNet or LLM structure. `vq_bench` is the instrument; the partitioner is the product.

Headline metrics: **p50/p95 latency**, **throughput @ queue depth N**, **energy/inference** (from rails if exposed), and **accuracy parity vs CPU fp32** — reported per plan, including the "don't split" baseline.

---

## 7. Build matrix

### 7.1 On-board (Ubuntu, QCS8275) — the real target
```bash
cmake -B build-vq -S mnn \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=cmake/qcs8275.cmake \
  -DMNN_ARM82=ON -DMNN_OPENCL=ON -DMNN_VULKAN=ON \
  -DMNN_HEXAGON=ON -DMNN_QNN=ON -DMNN_NPU_BACKENDS_SHARED=ON \
  -DMNN_GPU_TIME_PROFILE=ON -DMNN_BUILD_TOOLS=ON
```
Plus the HTP op-lib first: `bash htp-ops-lib/build.sh v<arch>` (**arch to be determined, §10**), then `ADSP_LIBRARY_PATH` + `LD_LIBRARY_PATH` per the Hexagon README.

`MNN_NPU_BACKENDS_SHARED=ON` so QNN loads as an explicit plugin rather than clashing with the Hexagon backend in one process.

### 7.2 Host proxy (this machine) — CPU-leg validation only
Native aarch64, 4× Cortex-A76, `asimdhp` (FP16) + `asimddp` (int8 dot-product) → **`MNN_ARM82=ON` is legitimately exercisable here**, and the DP partitioner, cost model, ring buffer and scheduler are all device-agnostic C++ that this box can test honestly.
**It has no GPU compute and no NPU:** `libvulkan.so.1` + `libOpenCL.so.1` are *loaders only* — no ICD, no `/dev/dri`. So GPU/NPU legs must be **mocked** as scripted-latency devices with a pluggable `Device` interface. That is exactly the right seam: the partitioner is then testable against synthetic `t̂`/bytes profiles and only validated for real on the board.

### 7.3 Cross-compile fallback
`aarch64-linux-gnu-gcc` is present if an on-board build proves too slow. Note MNN's Hexagon `CMakeLists` wants `-march=armv8.2-a+fp16` on `HexagonBackend.cpp` — fine on both paths.

### 7.4 MCU (STM32H5F5) — explicitly not an inference target
1.5 MB RAM. Use it via the RPC/Bridge library as a **gating coprocessor**: wake-word/motion pre-trigger decides *whether* the Linux side runs at all. On a 40 TOPS part, avoiding inferences is a bigger energy win than any partition.

---

## 8. System-level levers (often worth more than the partition)

1. **DVFS / thermal.** `MNN_HEXAGON_APP_POWER_HOLD` exists precisely to keep the DSP power vote across command flushes (`hexagon/CMakeLists.txt`). Without it, every boundary re-votes. Also set the CPU governor to `performance` and pin legs to cores that share no L2.
2. **GPU cold-start.** `MAX_TUNING_NUMBER` + `RuntimeManager::setCache()/updateCache()` — OpenCL autotuning on first run is minutes; caching it is the difference between a usable and an unusable demo.
3. **Quantisation.** The NPU's 40 TOPS is an **int8** (or int16) figure. A fp32/fp16 model on the HTP gets a fraction of it. Expect the *largest single speed-up* to come from quantising, not from splitting. `DYNAMIC_QUANT_OPTIONS`, `ATTENTION_OPTION` (int8 Q/K/V, flash attention), `ConvertToFullQuant.hpp`.
4. **Memory staging.** `EXTERNAL_WEIGHT_DIR` / `EXTERNAL_FEATUREMAP_DIR` let weights/feature maps live in a file-backed arena — relevant when the combined working set across three legs exceeds what fits comfortably.
5. **Don't oversubscribe.** `numThread` on the CPU leg must account for the GPU/NPU submit threads. `CPU_LITTLECORE_DECREASE_RATE` matters once topology is known.

---

## 9. Validation plan

### 9.1 Correctness before speed
Per segment, compare against a single-device CPU reference at three granularities: whole model (`modelCompare`), per-segment boundary tensors (via `saveTensors` / `ScheduleConfig::saveTensors`), and per-op (QNN's intermediate dump). Gate on **relative L2 ≤ 1e-2 fp32 / documented int8 budget**, and on `checkInvalidValue` for NaN/Inf. **A fast wrong answer is the expected failure mode of hand-rolled cross-device handoff** — layout or stride skew shows up here first.

### 9.2 Parity tools already in-tree
`modelCompare`, `checkInvalidValue`, `fuseTest` (spurious `Raster`/`ConvertTensor` insertion — our boundary tax, §5.4), `getPerformance`, `Profiler`, `backendTest` (per-device op coverage), and `MNN_QNN_DUMP_INTERMEDIATE_OUTPUTS` for exhaustive NPU activation dumps.

### 9.3 Acceptance criteria
| # | Criterion | Bar |
|---|---|---|
| A1 | Each leg runs the model end-to-end at parity | CPU=OpenCL=Vulkan=Hexagon, within quant budget |
| A2 | Cost model predicts measured segment time | MAPE ≤ 20 % |
| A3 | Tandem throughput ≥ best single device | ≥ 1.15× at queue depth ≥ 2, else report why |
| A4 | Boundary cost measured, not assumed | `vq_cutcost` calibration within 2× of predicted |
| A5 | Partitioner returns `K=0` where correct | ≥ 1 model where it declines to split |
| A6 | p95 latency stable | p95/p50 ≤ 1.3 |

A5 is deliberately an acceptance criterion. It is what stops this becoming a scheduler that splits for the sake of splitting.

---

## 10. Open risks (severity-ranked, each with a probe)

| # | Risk | Sev | Probe / mitigation |
|---|---|---|---|
| R1 | **Hexagon arch version unknown.** `build.sh` defaults **v79** (Snapdragon 8-Gen-3-era). QCS8275's HTP is likely older (v73/v75). Wrong arch → skeleton won't load. | **Blocker** | On board: read `/sys/devices/soc0/soc_id`, `qmi`/`getDeviceInfo`, and whether the BSP ships `libMNN_htpops_skel*.so` or a matching `fastrpc` stack. Build for the discovered arch; keep `MNN_QNN` (which takes explicit `<socId> <hexagonArch>`) as the deterministic fallback. |
| R2 | **Ubuntu userspace may lack the Android-flavoured FastRPC bits** the backend assumes: `libcdsprpc.so` (:502), `rpcmem_*` (:408/:725), `ADSP_LIBRARY_PATH`, `/vendor/lib/rfsa/adsp`. | **Blocker** | `ldconfig -p \| grep -Ei 'cdsprpc\|adsprpc\|rpcmem'`; `ls /dev/*rpc* /dev/*adsprpc*`. Qualcomm Linux BSPs do normally ship `libcdsprpc`; if `rpcmem` is absent, replace with `dma-heap` (`/dev/dma_heap/system`) + `memfd` and patch the allocator — a contained change in `HexagonRuntime.cpp`. |
| R3 | **Cross-device handoff eats the win** (§3). | High | `vq_cutcost` before writing any scheduler; enforce `θ` boundary-bytes cap; `K=0` legal. |
| R4 | **Layout skew** NC4HW4 ↔ image ↔ NHWC inserting hidden transposes. | High | `fuseTest` + callback op-dump at boundaries; normalise on producer side. |
| R5 | **Op coverage gaps on HTP** force CPU fallbacks mid-segment, serialising the pipeline. | Med | `support[op][dev]` matrix up-front; `backendTest`; `backupType`; choose cuts around fallback islands. |
| R6 | **Shared DRAM bandwidth caps aggregate gain** (§3.3). | Med | Measure single-device roofline first; set expectations to 1.3–2× for bandwidth-bound nets. |
| R7 | **GPU tuning cold start** (minutes) poisons benchmarks. | Med | `setCache`/`updateCache`, warm-up iterations excluded from timing. |
| R8 | **fp32 model on an int8-tops NPU** → NPU looks slow, tandem looks bad, wrong conclusions. | Med | Quantise before any tandem claim; report fp32/fp16/int8 as separate rows. |
| R9 | **CPU topology unknown** (docs sentence is garbled, §1) → wrong pinning. | Low | `/proc/cpuinfo`, `/sys/.../cpu_capacity`; then `CPU_CORE_IDS`. |
| R10 | MNN 3.6.2 Hexagon backend is young (`MNN_HEXAGON_ASAN` exists; `tests/` present). | Low | Enable `MNN_HEXAGON_ASAN` in bring-up builds; upstream any findings. |

---

## 11. Phased plan

| Phase | Deliverable | Exit test |
|---|---|---|
| **P0 — Board truth** (probe only) | `vq_probe` report: cores/clocks, GPU vendors (OpenCL + Vulkan), `libcdsprpc`/`rpcmem` presence, **Hexagon arch + soc_id**, `isBackendSupport` matrix | R1 + R2 answered go/no-go |
| **P1 — Per-device baselines** | MNN built with CPU/Arm82/OpenCL/Vulkan/Hexagon(+QNN); each leg runs reference models at parity | **A1** |
| **P2 — Cost model & cut costing** | `vq_cutcost`, `dag_profile`, persisted model-signed artifact; DP partitioner on host with mocked devices | **A2**, host unit tests green |
| **P3 — Tandem, 2 devices** | CPU+Hexagon, then CPU+GPU. `device_ring` with **memcpy first**, zero-copy second | **A3** at depth 2 |
| **P4 — Zero-copy handoff** | dmabuf/fd sharing across all three legs; fences via `Tensor::wait` | **A4**, boundary cost ≥3× below memcpy |
| **P5 — 3-device + autotuner** | CPU+GPU+NPU, online re-solve, top-k plan hot-swap | **A3, A5, A6** |
| **P6 — Workload sweeps** | CNN / transformer / LLM-prefill-decode; energy-per-infer; documented "when not to split" | Report shipped |

**Critical path: P0.** R1/R2 are the only items that can invalidate the whole project, and both are answered by one short probe session on a board.

---

## 12. Recommendation

Proceed, in this order of expected payoff per unit of risk:

1. **P0 probe** — cheap, decisive, unblocks R1/R2.
2. **Quantise + per-device baselines (P1)** — this alone typically captures **most** of the available speed-up. The 40 TOPS figure is an int8 number; leaving the model in fp32 and then reaching for a multi-device scheduler is the classic way to build a slow "accelerated" system.
3. **Then** the tandem partitioner — where the genuine, defensible, novel value is, and where the throughput and capacity wins live.

Set expectations explicitly: **tandem is a throughput and capacity technology, not a single-inference latency technology.** For one small model at batch 1, the best partition is frequently `K=0`. A harness that can *prove* that — and then find the regimes (batched streams, multi-model pipelines, LLM prefill/decode, oversized models) where three engines really do beat one — is a far more useful artifact than a demo that hand-waves a 3× number.
