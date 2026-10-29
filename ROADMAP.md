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
