# Roadmap — what it takes to make everything work

Each item maps a blocked pathway ([FAILURES.md](FAILURES.md)) to the concrete change
that would unblock it, who can make that change, and how to verify. Ordered by
value/effort on this board.

| # | Unblocks | Needs | Owner | Effort |
|---|---|---|---|---|
| R1 | LFM3 on NPU | rebuild of Arduino's hexagon runner | **us** (GitHub workflow) | low |
| R2 | GPU inference | llama.cpp 8-bit-storage gate **or** Turnip feature support | upstream llama.cpp / Mesa | medium–hard |
| R3 | QNN path | vendor `smmu_proxy` kernel service | Arduino/Qualcomm | out of our hands |
| R4 | NPU > 1.75 GiB | 64-bit CDSP VA window (DMA64) | kernel/vendor | out of our hands |
| R5 | Statistical confidence | bigger eval N | us, wall-clock | trivial |
| R6 | Quality/perf beyond Q4 | quant & dtype matrix: Q6_K / Q8_0 / F16 / int8, KV dtype | us, wall-clock | low |
| R7 | Benchmark depth | longer/harder speed & quality suites, incl. evals on **our own project text** | us, wall-clock | medium |
| R8 | Energy profile | INA232 power sampling (sudo-free) → J/token per engine, then reduce | us | low |

---

## R1 — LFM3 on the NPU (highest value, fully in our control)

**Gap (F5):** runner `llamacpp/20261020` ships ggml 0.27.0 — predates the `lfm3`
arch (needs llama.cpp ≥ v0.8.0/b11960). LFM3 currently runs CPU-only, missing the
9–21× prefill the NPU gives everything else.

**Fix — rebuild Arduino's own artifact against a newer ref** (full recipe:
`research/bricks-faster-inference.md`):

```bash
gh workflow run build-llamacpp.yml -R <your-fork>/app-bricks-py \
  -f llama_cpp_ref=<master-ish containing lfm3 support> \
  -f release_version=<date>
# → release tag llamacpp/<release_version>, asset llamacpp-hexagon-<release_version>.tar.gz
```

The workflow cross-compiles with the Snapdragon toolchain container + Arduino's
`fix-race.patch`; nothing else about the runner changes.

**Verify:** `llama-cli --list-devices` still shows `HTP0`; then
`scripts/test_hexagon.sh`-style bench of `LFM3-1.7B-Q4_K_M` (`-ngl 100`) — expect the
tri-hybrid conv/SWA layers to map like LFM2.5 did (21× class prefill), plus a GSM8K
parity re-run (`scripts/eval_hex_quality.sh`).

## R2 — GPU (Vulkan/Turnip) inference

**Gap (F2):** quantized + tiled matmul shaders require `StorageBuffer8BitAccess`;
Turnip on a623 says no; ggml never checks. Three ways out, in order of realism:

1. **llama.cpp gate (the right fix, upstream).** Implement #6401: at device init,
   require `storageBuffer8BitAccess` *or* generate 8-bit-free variants of everything
   that includes `types.glsl` (quant blocks re-expressed as `uint` + unpack — the
   shaders already unpack via `unpack8s4`-style helpers, so the container change is
   mechanical in `vulkan-shaders-gen`). Then `GGML_VK_*`-free operation on Turnip.
   - Verification here: `scripts/test_vulkan.sh` is wired end-to-end (build → smoke →
     4-model matrix → `results/vk_done`) — rerun it, expect the smoke step to pass.
   - Note: even with the gate, the *tiled* `mul_mm.comp` path must be rebuilt from the
     same clean container (it faults the GPU today, F2 table, row 2).
2. **Mesa/Turnip advertises 8-bit storage on a6xx.** A byte-addressed buffer-load
   question for the freedreno folks (mesa-dev thread 2026-July/226673 is the anchor).
   If it ever lands, this board gets GPU support with a mesa update alone — no llama.cpp
   change. Ubuntu 24.04 is pinned at Mesa 25.2.8; would need a newer mesa (oibaf-style
   PPA does not build arm64; source build into a local prefix is feasible sudo-free).
3. **Proprietary Adreno driver** (Android-style): advertises 8-bit storage on
   comparable silicon — but shipping it on this Ubuntu image is Arduino's firmware
   call, not ours.

**Until one lands, do not schedule GPU work** — CPU (§1) and NPU (§3) cover the board.

## R3 — QNN/QAIRT path (needs the vendor kernel)

**Gap (F1):** vendor HTP skel registers execute arenas via `HAP_mmap2(fd)`, which
requires the `smmu_proxy` kernel service; `6.8.0-1084-qcom` has no such module and the
firmware answers the modern `MEM_MAP(10)` ioctl with `AEE_EUNSUPPORTED`.

**What would fix it:** Arduino (or Qualcomm) shipping a kernel/firmware pair with
`smmu_proxy` (or firmware that honors the in-kernel fastrpc mappings). Ask upstream;
provide `npu/NOTES.md` — it contains the full ioctl-level evidence.
**Not actionable from userspace** (a `libcdsprpc` patch routing maps through the
working legacy `MMAP(6)` ioctl was implemented and regresses bring-up — NOTES.md).

Practical stance: the upstream llama.cpp Hexagon backend already delivers NPU
inference without QNN (BENCHMARKS §3); QNN only matters for non-llama.cpp HTP
workloads (DLC graphs, QNN API pipelines).

## R4 — More than ~1.75 GiB on the NPU

**Gap (F3):** 32-bit CDSP IOMMU window ("DMA64 disabled"), single shared `cdsp:3`
domain, `RLIMIT_MEMLOCK` 1.9 GiB.

- **Real fix:** kernel/firmware with DMA64 enabled → 64-bit DSP VA; that's vendor
  territory (same owners as R3).
- **Root-level partial relief:** raising `RLIMIT_MEMLOCK` only moves the wall from
  1.75 → 1.9 GiB — not worth it.
- **What works today:** budget NPU-side ≤1.75 GiB — full offload for ≤2B Q4;
  Qwen3.5-4B-Q4_0 at `-ngl 24` (interactive 10.3 t/s); `GGML_HEXAGON_MBUF=128,128` +
  ctx ≤8k to claw back arena space. Respect the ≥⅔-layers-on-NPU rule (F4).

## R5 — Eval rigor

N=20 GSM8K parity (§3.1) has ±1–2 question noise; N=50+ NPU runs and full HumanEval
(164 tasks) would tighten the quality claims. Pure wall-clock: ~1 h/model on the NPU.
`scripts/gsm8k_eval.py <port> <N> <tag>` — nothing blocking.

## R6 — Quantization & dtype sweep: Q6/Q8 up, F16/int8 down

Everything measured so far is Q4 (K_M/Q_0/QAD). The board's two hard constraints —
**decode is bandwidth-bound** and the **~1.75 GiB NPU budget (F3)** — make weight dtype
the biggest untested lever on both quality and speed. Predictions below are bandwidth
scaling (decode ∝ 1/weight-bytes) plus the measured QAD lesson (prefill is *dequant-cost*
bound, dtype-dependent: Q4_0 was ×1.8 vs Q4_K_M); they must be measured, not assumed.

### What to run (LFM3-1.7B as the reference model, CPU pinned `-t 4 --cpu-mask 0xf --cpu-strict 1`)

| Config | Weight size | Predicted tg128 | Predicted pp512 | Quality vs Q4_K_M (ppl 29.5) | NPU ≤1.75 GiB? |
|---|---|---|---|---|---|
| Q4_K_M (baseline, measured) | 1.14 GB | **11.7** | 49.0 | — | ✅ fits |
| **Q6_K** | ~1.5 GB | ~8.6 (×0.74) | ≈45–49 | ≈ F16 (Δ ppl ~−0.1…0.2) | ✅ fits, barely |
| **Q8_0** ("int8") | ~2.0 GB | ~6.7 (×0.57) | ≈45–50 | effectively lossless | ❌ over F3 budget |
| **F16** (`--type f16`) | ~3.4 GB | ~3.5 (×0.30) | ≈40–49 | reference point | ❌ way over |

- **Q6_K** is the interesting one: buys near-F16 quality and *still* fits the DSP window
  — the only upgrade path that keeps the 9–21× NPU prefill. Verify with a
  `test_hexagon.sh`-style `-ngl 100` run; if it aborts mid-load, F3 bit again.
- **F16 on this CPU is all cost, no compute win**: ggml dequantizes to f32 and computes
  in f32 regardless of storage dtype, so the A78's FEAT_FP16 arithmetic never gets used —
  you pay 3× decode for ~zero quality gain over Q6_K. bf16 is worse: the core lacks
  FEAT_BF16, so it adds a costlier dequant on top. **Verdict to verify: skip both.**
- **"int8" in llama.cpp means `q8_0`** (int8 weights + per-block f32 scales, dotprod
  `vec_dot` path): half the bytes of F16 at ~lossless quality — the honest sweet spot
  when quality > speed. True **W8A8 int8 compute does not exist** in llama.cpp (activations
  are always f32), and the Hexagon backend's HMX kernels consume repacked q4_0/q4_K/q8_0
  *weights* only — so no engine here can exploit int8 activations.
- **KV-cache dtype** (independent of weights): `-fa on -ctk q8_0 -ctv q8_0` should halve
  the measured 1.14 GB @128K KV to ~0.6 GB — big for long-context NPU runs where the
  arena competes with weights (R4 relief we *can* do from userspace). Measure the quality
  drift at N=50.

### Method

```bash
# one model, four dtypes — quantize locally from the safetensors/GGUF we already vet:
llama-quantize LFM3-1.7B-Q4_K_M.gguf lfm3-q6_k.gguf  Q6_K
llama-quantize LFM3-1.7B-Q4_K_M.gguf lfm3-q8_0.gguf  Q8_0
llama-bench    -m lfm3-q6_k.gguf -ngl 0 -t 4 --cpu-mask 0xf --cpu-strict 1 -pg 512,8192 -n 128
./llama-perplexity -m lfm3-q6_k.gguf -f bench/gatsby.txt -c 2048     # vs baseline 29.54
scripts/gsm8k_eval.py <port> 50 q6k                                  # quality sanity
# NPU: Q6_K -ngl 100 (fits?) and Q8_0 of the 0.8B (budget test); KV: rerun 32K ctx with -ctk/-ctv q8_0
```

Disk policy unchanged: each variant exists only during its own suite (F16 is the largest
file we would ever fetch — delete first). Output to `results/bench_dtype_*` + `DISK_LOG.md`.
**Success =** a per-workload dtype recommendation in BENCHMARKS §5 replacing the current
"Q4_K_M for interactive / Q4_0 for prefill" rule, with the Q6_K-on-NPU budget question
answered either way.

## R7 — Deeper benchmark suites: longer, harder, and on our own corpus

R5 tightens N on the existing evals; R7 adds whole axes we never measured — sustained
behavior, long-context *quality*, and evals grounded in **this project's own text**
(the repo corpus is free, versioned, domain-relevant, and already vetted by us).

### Speed axes not yet touched

- **Sustained load / thermals** — this board is passively cooled and every number so far
  is a short burst mean. Run a 10-min continuous pp+tg loop, log t/s per minute →
  throttle curve; that curve, not the burst, is what a real deployment gets.
- **Long-context prefill curves** — pp up to 128K on both engines, reported as
  TTFT p50/p95 over repeated runs (single-shot means hide the tail).
- **Concurrency** — `llama-server -np 2/4`: per-client tg and aggregate throughput.
  `scripts/serve.sh`'s deployment shape was never load-tested.
- **Agentic replay** — replay real multi-turn tool-call transcripts (this project
  generated plenty) against `llama-server`, measure per-turn latency and KV growth
  against the measured ~8.9 KB/token (§2).

### Quality axes not yet touched

- **Full-N suites** — GSM8K full 234 questions, HumanEval full 164, both engines
  (this absorbs R5).
- **Harder reasoning/code** — MUSR + BBH-style subsets and MBPP+; pure-stdlib scorers in
  the `gsm8k_eval.py` style, datasets vendored into `bench/`.
- **Long-context recall on our corpus (NIAH)** — needle-in-a-haystack at 8K/32K/128K
  where the haystack is our own `docs/REPORT.md` + `BENCHMARKS.md` + `FAILURES.md`
  concatenated, needles planted at 0/25/50/75/100% depth. Directly answers "can this
  board do RAG over exactly the kind of text it produces".
- **Grounded-QA / hallucination** — questions about our *own numbers* ("what tg128 did
  K2-Horizon-4B reach?", "which engine hits 679 t/s?"), golden answers generated from
  `results/*.csv` by script; score = exact number match. Cheap, brutal, and self-referential.
- **Domain perplexity** — extend beyond `gatsby.txt`: llama.cpp source excerpts and this
  repo's docs as corpora → fitness for on-board RAG over this domain.
- **Summarization rubric** — compress `docs/REPORT.md` sections; stdlib scorer checks
  keyword/number coverage against the originals (no LLM judge — keep it reproducible).

### Method sketch

```bash
scripts/eval_deep.sh <gguf> <engine>   # chains: sustained → NIAH → grounded-QA → full-N
# → results/deep_*.{csv,json} + results/deep_done marker
# budget: ~0.5 day CPU wall-clock for LFM3 + Qwen3.5-4B; NPU variants within the F3 budget
# no new model downloads — disk policy (delete-after-suite) unaffected
```

**Success =** a throttle curve, NIAH recall curves per engine, and grounded-QA scores for
every sub-3B model in the matrix; anything that changes a recommendation gets promoted
into BENCHMARKS §5 (engine selection guide).

## R8 — Energy: measure watts and J/token, then reduce them

**Status:** zero power telemetry went into any measurement so far — every number in this
project is a *speed* number. The working assumption "we never draw more than ~15 W" is
untested. Good news found on 2026-10-29: **the board already ships a power meter we can
read without sudo.**

### Telemetry that exists on this board (verified)

| Source | What it gives |
|---|---|
| `/sys/class/hwmon/hwmon2` = **ina232** | `power1_input` (µW), `in1_input` (mV), `curr1_input` (mA) on the 20 V input rail — whole-board draw incl. PMIC losses; `update_interval` configurable |
| 46 hwmon thermal zones | per-block temps incl. **`nsp_0_*` (the Hexagon NSP!)**, `cpuss_*`, `ddrss_*`, `gpuss_*` — throttle correlation without guessing |
| `emc2305` (hwmon0) | fan controller; `fan*_fault` set → **passively cooled**, confirms the R7 thermal concern |
| `ucsi-source-psy-*` (USB-C PD) | `voltage_now`/`current_now` read 0 — dead end; use the INA232 |

**Spot check (2026-10-29):** idle ≈ 8.5–9.5 W; 8× `yes` CPU stress → avg 11.0 W,
**max 11.8 W**. Consistent with the ≤15 W hunch — but `yes` is memory-light, DDR is
under-tested, and **NPU and GPU load were never measured at all**. Treat 15 W as an
open question until an all-engines stress says otherwise.

### R8a — measure

- `scripts/power_probe.py`: sample `power1_input` + the `nsp/cpuss/ddrss` zones at 2–10 Hz
  alongside any existing suite → `results/power_<tag>.csv` (same marker style as the suites).
- Conditions per engine (CPU-pinned / NPU / GPU-when-R2-lands): idle → loaded-idle →
  pp burst → tg burst → 10-min sustained (reuse the R7 loop).
- Headline metric: **J/token = W ÷ t/s** for prefill and decode, per engine. Hypothesis to
  test: the NPU's decode win looks small in wall-clock (1.35–1.5×) but should look *big*
  in J/token — HTP MACs plus a shorter high-power window beat NEON dequant+dot.
- Caveat: the INA232 reads the input rail (the right number for deployment); supply-side
  losses upstream of the USB-C input stay invisible.

### R8b — reduce (only after R8a has a baseline)

- **Already banked:** big-core pinning is an energy win too (race-to-idle — the work
  finishes sooner); expect it to show up in J/token.
- **Quant/dtype × energy:** rerun the probe across the R6 matrix — Q4_0/QAD do less
  dequant work per token; `-ctk/-ctv q8_0` halves KV → less DDR energy in the
  bandwidth-bound decode regime.
- **Engine choice as an energy knob:** if NPU prefill is 21× faster at similar watts,
  it is ~21× cheaper per prompt token — the single biggest energy lever here; verify.
- **Governor/cpufreq tuning** needs root → flag for the board owner, not part of this
  project's sudo-free scope.
- **If throttling shows** (rising `nsp_*`/`cpuss_*` + falling t/s): revisit `-t 4`,
  `-ngl` splits, and `GGML_HEXAGON_OPBATCH` as *thermal* knobs.

**Success =** a W and J/token table per engine (LFM3 + Qwen-4B minimum) published as a new
BENCHMARKS §6, and the 15 W ceiling either confirmed under all-engines sustained load or
corrected with the real peak.

---

## "Everything works" definition of done

- [ ] **CPU** — done (§1–2): pinned-4-thread recipe is the board default.
- [ ] **NPU (llama.cpp Hexagon)** — done (§3): default for prompt-heavy ≤1.75 GiB.
- [ ] **NPU + LFM3** — R1 rebuild lands; LFM3 pp512 ≥ ~400 t/s on HTP0; GSM8K parity.
- [ ] **GPU** — R2 lands; `scripts/test_vulkan.sh` reaches `results/vk_done` with a
      full 4-model matrix (no SIGSEGV, no `a6xx_recover` in dmesg).
- [ ] **QNN** — R3: `qairt-net-run` executes one HTP graph end-to-end (then reassess
      whether it beats the Hexagon backend at all).
- [ ] **Big models on NPU** — R4: K2-4B `-ngl 100` loads and beats its CPU control.
- [ ] **Dtype matrix** — R6: Q6_K / Q8_0 / F16 measured on LFM3 (CPU + NPU budget check);
      Q6_K-on-NPU answered; per-workload quant rule updated in BENCHMARKS §5.
- [ ] **Depth** — R7: throttle curve + NIAH recall curve + grounded-QA score published
      for at least LFM3-1.7B and Qwen3.5-4B, on both engines where the budget allows.
- [ ] **Energy** — R8: J/token table per engine in BENCHMARKS §6; the ~15 W ceiling
      confirmed or corrected under sustained all-engines load.

## Reproducing this project's environment from scratch

```bash
sudo apt-get install -y cmake ninja-build glslang-tools libvulkan-dev spirv-headers \
                        glslc libshaderc1 gdb        # GPU build + forensics
# CPU stack:  llama.cpp v0.8.0 b11972 Ubuntu-arm64 prebuilt → ./llama.cpp/ (vendored)
# NPU stack:  gh release download llamacpp/20261020 -R arduino/app-bricks-py \
#               -p 'llamacpp-hexagon-*.tar.gz'   # sha256 pin: research/bricks-faster-inference.md
#             + unversioned libcdsprpc.so symlink (F6)
# Vulkan build:  cmake -S llama.cpp -B build -DGGML_VULKAN=ON && cmake --build build
#             (toolchain now system-wide; the /tmp/opencode/vkdeps prefix in
#              build-vk's cache can be dropped for a fresh configure)
# Models:  see README "Quick start" URLs; delete each after its tests (results/DISK_LOG.md)
```
