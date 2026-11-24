# Ventuno Q memory map — how the 16 GiB DDR is shared by CPU, NPU, GPU

Measured on-board 2026-10-29 from `dmesg`, `/proc/meminfo`, `/proc/device-tree/reserved-memory`,
`/dev/dma_heap`. The board has **16 GiB** of unified LPDDR (earlier "8 GB" notes were wrong;
`free` reports 15,651,832 kB MemTotal).

## Headline

There is **no dedicated VRAM and no dedicated NPU RAM**. CPU, Adreno GPU and Hexagon NPU
all master the *same* physical DDR. What separates them is (a) boot-time carve-outs the CPU
is forbidden to touch, (b) IOMMU/SMMU mappings, and (c) shared memory pools (CMA, dma-buf
heaps). The constraint every engine actually shares is **memory bandwidth**, not capacity.

Kernel boot line (dmesg):

```
Memory: 15400920K/16140796K available (..., 571940K reserved, 167936K cma-reserved)
```

| Bucket | Size | Notes |
|---|---|---|
| Physical DDR | 16 GiB | single LPDDR stack-group, all masters share it |
| Kernel-visible | 15.39 GiB (16,140,796 KiB) | remainder held by secure subsystems below the kernel's map |
| nomap carve-outs | **558 MiB** (571,940 KiB) | removed from CPU page tables entirely (table below) |
| CMA pools | **164 MiB** (CmaTotal) | *reusable*: counted in MemTotal, CPU may use them when idle |
| `MemTotal` (Linux) | **14.93 GiB** (15,651,832 KiB) | what `free`/OOM-killer reason about, CMA included |

## Carve-outs (nomap — invisible to the CPU)

| Region | Size | Owner / purpose |
|---|---|---|
| sail-ss-region | 256 MiB | Sail security subsystem |
| pil-adsp + q6-adsp-dtb | 30.5 MiB | ADSP (audio/sensor) firmware |
| pil-gpdsp + q6-gpdsp-dtb | 30.5 MiB | General-purpose DSP firmware |
| **pil_cdsp + q6-cdsp-dtb** | **30.5 MiB** | **CDSP firmware — the Hexagon NPU's OS image only, NOT model memory** |
| trusted-apps-region | 25 MiB | TrustZone TA storage |
| lpass-machine-learning | 15 MiB | audio-ML (LPASS-side) |
| firmware-region / tags-region | 8 + 8 MiB | firmware scratch, dm-log tags |
| adsp-rpc-remote-heap | 8 MiB | ADSP RPC remote heap |
| pil-camera / pil-cvp / pil-video | 7 MiB each | camera, CV, video DSPs |
| qtee-region | 5 MiB | QTEE |
| hyp-region | 6 MiB | Gunyah hypervisor |
| **pil-gpu-region** | **8 KiB** | **stub only — the Adreno GPU has no carve-out** |
| xbl/aop/smem/scmi/cpucp/tz misc | ~15 MiB | boot firmware & mailbox regions |

## Reusable pools (CMA — still part of MemTotal)

`mem-dump` 48 MiB, `secure-cdsp` 32 MiB, `qcom,qseecom` 20 MiB, `qcom,qseecom-ta` 16 MiB,
`qcom,audio-ml` 16 MiB (named pools; CmaTotal 164 MiB). These hold pages for DMA-capable
consumers but are handed to the page cache when unused (`CmaFree` 118 MiB at idle).

## dma-buf heaps (`/dev/dma_heap/`) — where accelerator buffers actually come from

| Heap | Access | Used by |
|---|---|---|
| `system` | ACL rw: audio, video, **fastrpc** (+ user arduino via video/fastrpc groups) | GPU (freedreno/Turnip), general dma-buf |
| `qcom,system` | ACL rw: audio, video, **fastrpc** | **CDSP/Hexagon (llama.cpp Hexagon backend, QNN, fastrpc mmap)** |
| `reserved`, `secure-pixel`, `secure-non-pixel`, `qseecom*`, `cma-secure-cdsp` | root:video | secure camera/DRM/TEE paths only |

The `arduino` user is in `fastrpc` + `video` + `render` groups, so **plain userspace can
allocate DSP and GPU buffers — no sudo needed**. (Confirmed: `llama-cli --list-devices`
enumerates `HTP0: Hexagon` as the user.)

## What each engine can reach

- **CPU**: the full `MemTotal` 14.93 GiB minus kernel/OS. All llama.cpp CPU benchmarks here
  ran within this; the 9B / K2-7B swap behaviour is this ceiling, not a carve-out.
- **NPU (Hexagon V75 on CDSP)**: the 30.5 MiB `pil_cdsp` carve-out is *only* the DSP firmware
  image. Model weights and KV cache are ordinary DDR pages allocated from the `qcom,system`
  heap and mapped through the CDSP SMMU. The real limit is the **DSP virtual address space**:
  Arduino documents ~3,134 MiB per HTP session (up to 4), but on this kernel the measured
  usable window is only **~1.75 GiB total** (32-bit, single shared domain — see
  `npu/cumap_probe.c` and Consequence 3 below). KV lives on the DSP next to the weights.
- **GPU (Adreno a6xx, freedreno/Turnip)**: no carve-out (the 8 KiB `pil-gpu` is a stub);
  buffers come from the `system` heap mapped via the GPU IOMMU. Turnip's advertised "VRAM"
  (~½ MemTotal) is a Vulkan heuristic, not a partition — it can consume free RAM on demand.

## Consequences for LLM inference on this board

1. **Capacity is one shared pool.** Putting weights on the NPU does not free RAM; it only
   moves compute. Total budget stays ~15 GiB (minus OS).
2. **Bandwidth is the shared bottleneck.** LLM decode is bandwidth-bound; CPU, GPU and NPU
   contend for the same LPDDR channels, so cross-engine speedups on token generation are
   smaller than on prefill (compute-bound), where the HTP's HVX/HMX shines.
3. **The NPU's ceiling is addressing, not memory — and it is lower than advertised here.**
   Arduino's `SESSION_ALLOCATION.md` documents ~3,134 MiB per HTP session × 4 sessions, but
   on *this* kernel `npu/cumap_probe.c` measures the CDSP mapping window at **~1.75 GiB
   cumulative** (32-bit IOMMU, "DMA64 disabled"; `ENOMEM` at 1792 MiB in 256 MiB chunks), and
   the driver's static-domain fallback puts **all HTP devices in one domain (`cdsp:3`)**, so
   extra sessions add no address space. `RLIMIT_MEMLOCK` (1.9 GiB here, `unlimited` in
   Arduino's container) is a second, similar-height wall. A 3.16 GB Q4_K_M like K2-Horizon-4B
   therefore cannot fully offload; KV cache still lives on the DSP.
4. **No secure/TEE involvement needed**: everything the llama.cpp Hexagon path uses
   (`/dev/fastrpc-cdsp`, `qcom,system` heap) is open to the `fastrpc` group.
