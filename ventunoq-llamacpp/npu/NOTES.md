# NPU (Hexagon HTP) bring-up log — QCS8275 / Arduino UNO Q

Goal: run real inference on the on-board Hexagon NPU (not CPU/GPU) with the
Qualcomm AI Runtime (QAIRT) community SDK, benchmarked against CPU.

## Confirmed hardware / software facts

| Item | Result |
|---|---|
| NPU core | Hexagon **V75** HTP (`qnn-platform-validator --coreVersion`) |
| FastRPC | working: `/dev/fastrpc-cdsp`, `/dev/fastrpc-gdsp0` user-openable, board `libcdsprpc.so` v1.2.188 |
| DSP unit test | **passed** — calculator sample ran *on the DSP* over unsigned skeletons (no signing wall) |
| GPU | ~~ruled out: `renderD128` bound to `msm_dpu` (display); Adreno lives on KGSL → Turnip/Vulkan cannot bind~~ **CORRECTED 2026-10-29: wrong** — Turnip enumerates and runs compute fine (Mesa 25.2.8, Vulkan 1.3.318); llama.cpp on Vulkan is blocked later, by the 8-bit-storage shader gap (see FAILURES.md F2) |
| QAIRT SDK | 2.51.0.260929 + 2.51.200.261021 libs (community, no-login download); no newer community release exists (2.52.x guesses → 404) |
| ONNX→DLC | **works on-device** (`qairt.api.converter` → `npu/test_convnet.dlc`, 1 MB) |
| HTP graph compose/finalize | **works on DSP** (prepare runs on-CDSP under spoof, reaches "Executing Graphs") |
| HTP graph execute | **blocked**: DSP-side buffer registration fails (see below) |

## SoC whitelist workaround (works)

`libQnnHtp.so` reads `/sys/devices/soc0/soc_id` → maps to a `SnapdragonModel`
enum → refuses SoC model 82 (QCS8275, soc_id 675). Workaround:

1. **LD_PRELOAD sysfs shim** (`npu/spoof_soc.c` → `npu/libsocspoof.so`): serves
   a fake soc_id from a memfd; `FAKE_SOC_ID` env selects it.
2. **soc_id sweep** (`npu/soc_scan.c`, sweep 2..1000, results in
   `/tmp/opencode/soc_map.txt`): **605, 606, 607, 695 → `OK-DEVICE`** (backend +
   device creation succeed; these are V75/V79-era whitelist entries).
   `FAKE_SOC_ID=605` used for all runs.
3. `QNN_FORCE_FIREWHEEL_SOCID` env: no effect. `pd_session` nsp1/2/3/compute:
   all fail. CPU backend `libQnnCpu.so` fails ("Check executable cache failure").

## DSP library path requirement (fixed)

`fastrpc_shell_unsigned_3` + `libc++.so.1`/`libc++abi.so.1` for the CDSP live in
**`/usr/lib/dsp/cdsp/`** on this image. `ADSP_LIBRARY_PATH` MUST include it —
without it the compute PD cannot spawn; with it, the run reaches
"Executing Graphs". (Also silence: "unexpected GraphHtpSettings option 66" is a
benign skel-side warning.)

## Execution blocker (root-cause chain, still open)

Under `FAKE_SOC_ID=605` spoof, qnn-net-run reaches *Executing Graphs* then:

```
<E> unexpected GraphHtpSettings option 66
Error 0x80000414: remote_handle64_invoke failed for module
      libdspqueue_rpc_skel.so, method 3 on domain 3 (sc 0x3010100)  ×7-9
<E> Error from rpc transport when doing buffer mapping, transportStatus = 0, map result: 8003
<E> Failed to map input buffer on NSP
<E> Graph test_convnet failed in execution with err 6001
```

- `0x80000414` = `DSP_AEE_EOFFSET(0x80000400)` + **20 = AEE_EBADPARM** — an
  *immediate* parameter rejection on the DSP side (~1 ms per call), not an IOMMU
  timeout. (Note: in current fastrpc master, method 3 of dspqueue_rpc is
  `is_imported`, only used at close — the firmware's IDL evidently differs;
  the failing calls happen during buffer registration.)
- `8003` = `QAIRT_MEM_ERROR_MAPPING` (`include/QAIRT/QairtMem/QairtMemCommon.h`).
- All kernel FastRPC ioctls succeed (MEM_MAP ×10 rc=0); buffers come from
  `/dev/dma_heap/system`; `GET_DSP_INFO` ioctl (0xc01c520d) returns a **zeroed
  capability struct** (only onChip=1 set); `GET_NOTIF_RESPONSE` unsupported
  (0x14); kernel boot log: "no reserved DMA memory for FASTRPC"; fastrpc
  compute-cb@5..12 attached to iommu groups 21-27 (SMMU proxy domains exist).
- `libdspqueue_rpc_skel.so` exists **nowhere on disk** → it is provided by the
  CDSP firmware image itself.

### Experiments run against the blocker

| Experiment | Result |
|---|---|
| Redirect `/dev/dma_heap/system` → `qcom,system` (LD_PRELOAD openat shim) | same 0x80000414 |
| pip `qnnruntime` **2.48** host libs + skel, 2.51 qnn-net-run + DLC | "Initialization failure" (ABI mismatch; 2.48 session API is context-binary-only, cannot load DLC; no 2.48 converter/binaries available) |
| Rebuild **latest qualcomm/fastrpc master** `libcdsprpc.so.1.0.0` on-board (gcc, no autotools; yaml.h shim + `-l:libyaml-0.so.2`) and dlopen it via `npu/compat/libcdsprpc.so` | new lib confirmed active (log format changed) — **same 0x80000414** → client-side ABI is NOT the issue; the firmware's dspqueue skel rejects the registration |
| skel/host version pairing | skels export versioned symbols (`qnn_2_51_0_skel_handle_invoke`) → host↔skel must be same version; 2.48↔2.51 mixing impossible |

## ROOT CAUSE (final, 2026-10-28 late) — kernel lacks vendor DSP memory-mapping service

Established via three independent probes (`npu/mmap_probe.c`, DSP-side `.farf`
logging, live `dmesg -w` capture):

1. **DSP-side farf logging** (`qnn-net-run.farf` = `0xFFFFFFFF` in the skel dir)
   forwards the vendor CDSP adsprpc logs: the exec failure is
   `fastrpc_invoke_fd_mmap_create failed for fd 24 of length 602368 bytes
   (map start paddr 0x0 ...)` ← `HAP_mmap2 failed ... prot 0x23` → skel
   `sharedMemoryMmap failed` → host `map result: 8003`
   (`QAIRT_MEM_ERROR_MAPPING`) → exec err 6001. The earlier `0x80000414`
   dspqueue errors are just this same failure's teardown noise.
2. **Direct ioctl probe** against `/dev/fastrpc-cdsp` (user is in `fastrpc`
   group — no sudo needed):
   - `MEM_MAP` (ioctl 10): fails for **every** size. With `flags=1` the vendor
     DSP init module returns `0x8000040E` (AEE_EBADPARM); with `flags=2`
     (`FASTRPC_MAP_FD`) it returns **`0x80000414` = AEE_EUNSUPPORTED** — the
     vendor firmware's `mem_map` service is simply not provided. Kernel logs
     `mem mmap error, fd N` (compute-cb@2), i.e. kernel-side SMMU mapping
     itself **succeeded**; the DSP rejected the follow-up message.
   - `MMAP` (legacy ioctl 6): **works** — kernel maps the 588 KB dma-buf through
     the CDSP compute-cb SMMU (iommu group 19, present and bound in DT) and the
     DSP init module accepts the physical-page list, returning DSP VA
     `0xe0900000`. Clean munmap too.
3. **Kernel forensics**: in-tree `fastrpc.ko` 6.8.0-1084-qcom (glink variant);
   **no `smmu_proxy` module exists anywhere** on the system; `no reserved DMA
   memory for FASTRPC` ×3 domains at boot; no fastrpc debugfs; kernel emits
   nothing during runs; no SMMU faults ever recorded.

**Mechanism.** The vendor HTP skel registers its execute arenas via the
`sharedMemoryMmap(fd, size)` FastRPC method. The mainline-style kernel passes
the dma-buf fd to the DSP over glink, and the vendor DSP-side adsprpc resolves
it with `HAP_mmap2(fd)` — which requires the **vendor `smmu-proxy` Linux
service** to perform the CDSP-SMMU mapping on the DSP's behalf. That service
does not exist in this Ubuntu qcom 6.8 kernel, so the resolution returns
`paddr 0x0` and the map fails. The kernel's own mapping (which works — see
MMAP(6)) is ignored by the vendor DSP for fd-based args.

**Userspace workarounds ruled out experimentally:**
- Patching self-built master `libcdsprpc` to route `fastrpc_mmap` through the
  working legacy `MMAP(6)` ioctl: compiles and maps fine, but the DSP-side map
  table is not fd-keyed, so `dspqueue_rpc_init_process_state`'s
  `find_map(fd)`/`HAP_mmap2(fd)` still fails (`0x80000448`) and device
  creation dies even earlier (`err 1002`, "Failed to parse default platform
  info"). Reverted; baseline behavior re-verified (fails identically at
  `sharedMemoryMmap fd 24`).
- No config/env knob disables the shared-memory arena path in `libQnnHtp`.
- MEM_MAP with `vaddrin`=host-VA, second-session registration, all dma-heaps,
  sizes 4 KB–4 MB: all rejected (`EUNSUPPORTED`/`EBADPARM`).

**Verdict: HTP graph EXECUTION is blocked by a missing kernel-side component
(`smmu_proxy` / vendor FASTRPC memory service) in the board's 6.8.0-1084-qcom
kernel. Not fixable from userspace.** Everything up to and including
on-DSP graph compose + finalize works (calculator unit test passes; skel loads
and executes methods on V75). Needs a vendor/Android-Base kernel, or Qualcomm
porting `smmu_proxy` (or DSP firmware accepting kernel-provided pages for fd
args) for this board.

## Artifacts

- `npu/qairt-sdk/` — pruned SDK (aarch64-ubuntu + hexagon-v75 only, ~1.3 GB)
- `npu/sdk200/` — 2.51.200 libs range-extracted over HTTP — same whitelist verdict
- `npu/compat/` — RPC lib symlinks; currently `libcdsprpc.so{,.1}`,
  `libxdsprpc.so` → **self-built master build** `npu/compat/libcdsprpc.so.1.0.0`
  (from `/tmp/opencode/fastrpc`, github.com/qualcomm/fastrpc)
- `npu/spoof_soc.c` / `libsocspoof.so` — sysfs spoof (`FAKE_SOC_ID`)
- `npu/soc_scan.c` — backend/device probe for the whitelist sweep
- `npu/test_convnet.dlc`, `npu/input_list.txt`, `npu/in0.raw` — test graph
- `npu/mmap_probe.c` / `mmap_probe` — direct `/dev/fastrpc-cdsp` ioctl probe
  (ALLOC_DMA_BUF / MEM_MAP / MMAP / MUNMAP sweep; no sudo needed)
- `/tmp/opencode/heapredirect.so` — dma_heap redirect shim (`DMAHEAP_REDIRECT=1`)
- `/tmp/opencode/fastrpc` — fastrpc master clone + on-board build
  (`rebuild.sh`); `src/fastrpc_mem.c.orig` pristine, MEM_MAP→MMAP64 patch
  experiment documented above (reverted)
- `/tmp/opencode/dmesg.log` — live kernel capture during runs (root, 644)
- `/tmp/opencode/debugfs.txt` — sudo debugfs/SMMU dump
- `scripts/zip_range_extract.py` — remote-zip HTTP-range extractor

## Env recipe (reproduce)

```bash
cd ~/Code/ventunoq-llamacpp
S=$PWD/npu/qairt-sdk/qairt/2.51.0.260929
export LD_LIBRARY_PATH=$PWD/npu/compat:$S/lib/aarch64-ubuntu-gcc9.4 QNN_SDK_ROOT=$S
p="$S/lib/hexagon-v75/unsigned;/usr/lib/dsp/cdsp;/vendor/dsp/cdsp;/usr/lib/rfsa/adsp"
export ADSP_LIBRARY_PATH=$p CDSP_LIBRARY_PATH=$p GDSP0_LIBRARY_PATH=$p
FAKE_SOC_ID=605 LD_PRELOAD=$PWD/npu/libsocspoof.so \
  $S/bin/aarch64-ubuntu-gcc9.4/qnn-net-run \
  --dlc_path npu/test_convnet.dlc --backend $S/lib/aarch64-ubuntu-gcc9.4/libQnnHtp.so \
  --input_list npu/input_list.txt --output_dir /tmp/opencode/npuout
```

## Would upgrading to Ubuntu 26.04 (resolute) fix the NPU? — No (2026-10-28)
Evidence (kernel-config dumps pulled via CloakBrowser from LP
~ubuntu-kernel-automation/+git/ubuntu-kernel-configs):
- The CDSP firmware on this board comes from Arduino's BSP package
  **linux-firmware-dragonwing** (20260825) at
  /usr/lib/firmware/updates/qcom/qcs8300/cdsp{0,1,2,3}.mbn — NOT Ubuntu's
  linux-firmware (which has no qcs8300/cdsp files at all). A distro upgrade
  keeps/replaces this BSP package → same firmware bits.
- resolute kernel = mainline 7.0.0; its configs repo has **no linux-qcom
  flavour** (noble has linux-qcom + linux-qcom-realtime, latest 6.8.0-1084.89
  = exactly the board's current kernel family). Ubuntu 26.04 ships NO kernel
  with dragonwing/qcs8300 BSP support at all.
- CONFIG_QCOM_SMMU_PROXY: absent from BOTH noble linux-qcom 6.8.0-1084.89
  arm64 config and resolute linux 7.0.0-38.38 arm64 config (only
  CONFIG_QCOM_FASTRPC=m in each). The smmu_proxy driver was never merged
  upstream and Ubuntu doesn't carry it → the root-cause service is missing
  on every Ubuntu kernel, 24.04 and 26.04 alike.
- Conclusion: 26.04 would REMOVE the only working kernel (Arduino BSP
  6.8.0-1084-qcom) without adding the missing piece. The fix can only come
  from Arduino's BSP channel (new linux-image-qcom / linux-firmware-dragonwing
  carrying QCOM's smmu_proxy work). Watch: apt list --upgradable | grep -E
  'qcom|dragonwing'. Raw dumps: /tmp/opencode/cfg_qcom_noble.txt, cfg_resolute.txt
