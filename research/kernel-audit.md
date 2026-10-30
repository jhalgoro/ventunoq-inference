# MNN kernel audit — why it can't step on F8 (and what would change that)

Date: 2026-10-30, MNN `master` @ clone in `/tmp/opencode/MNN` (406 MB tree, kept until
cleaned; rebuild: `scripts/build_mnn.sh`). Question: after the mistral.rs release binary
SIGILLed on `fmlal v0.4s, v2.4h, v3.4h` (parent **F8** — QCS8300 lacks FMLAL/FMLSL *vector*
despite HWCAP `asimdhp`; probe one-liner in the parent F8 entry, exit 132 on both clusters),
could MNN's ARM82 path execute the same class of instruction?

## Findings

1. **`.S` kernels use only proven-executable instructions.** Census of
   `source/backend/arm82/asm/arm64/low_memory/MNNGemmInt8AddBiasScale_ARMV82_*_FP16.S`
   (the int4/int8 GEMM hot loop): `fmul` (12), `fadd` (12), `fcvtl/fcvtl2` (16), `fcvtn/fcvtn2`
   (6), `fmin/fmax` (10), `fmla` (1) — plus `sdot` in the dotprod variants. All verified
   executable on QCS8300 (parent F8 probe: `fmla v0.4h`, `sdot`, `fcvt h` all exit 0).
   **No `fmlal`/`fmlsl` anywhere in the assembly.**
2. **The only FMLAL C++ code is compile-gated off.** `grep -rli fmlal` hits exactly one
   file: `source/backend/arm82/Arm82Functions.cpp`, lines ~671–722
   (`MNNPackedMatMulRemainFP16WithSme2PackedB_Fmlal_E` using `vfmlalq_lane_low/high_f16`),
   wrapped in `#if defined(__ARM_FEATURE_FP16_FML) || defined(vfmlalq_lane_low_f16)`.
   `__ARM_FEATURE_FP16_FML` requires `-march=…+fp16fml`; the ARM82 build uses
   `-march=armv8.2-a+fp16` (see `cmake/generate_podspec.cmake` and `cmake/KleidiAI.cmake`
   flag lines) → macro undefined → code not compiled.
3. **KleidiAI is off by default** — `CMakeLists.txt:274–275`:
   `option(MNN_KLEIDIAI "Enable KLEIDIAI" OFF)` and `MNN_KLEIDIAI_DEFAULT_ON OFF`. Its
   i8mm kernel variants (`cmake/KleidiAI.cmake`: `-march=armv8.2-a+i8mm+dotprod+fp16`)
   would be *runtime-dispatched*, but this SoC reports `i8mm: 0` — M4 probes what actually
   dispatches if someone flips it ON.
4. **Runtime detection is honest** (unlike mistral.rs's vacuous "extensions: none"):
   `llm_bench`/`llm_demo` print `The device supports: i8sdot:1, fp16:1, i8mm:0, sve2:0,
   sme2:0, perfCores:0` — matches the board's real feature set (parent F8 AT_HWCAP
   `0x119fff`).

## Verdict

Safe **as built by `scripts/build_mnn.sh`**. Re-run this audit if any of these change:
build gains `+fp16fml`/`+i8mm` march flags, `MNN_KLEIDIAI=ON`, an upstream refactor moves
`vfmlalq_*` out of the gate, or the model routes to a backend other than CPU-ARM82.

## Repro

```bash
cd MNN
grep -rn "fmlal\|FMLAL" source/backend/arm82/ | head          # → one gated C++ file only
grep -oE "^\s+[a-z0-9]+" source/backend/arm82/asm/arm64/low_memory/*_FP16.S \
  | awk '{print $2}' | sort | uniq -c | sort -rn | head -20    # instruction census
grep -n "MNN_KLEIDIAI" CMakeLists.txt | head -2                # both OFF
# F8 ground truth on this board (expect 132 = SIGILL):
printf '.global _start\n_start:\n  fmlal v0.4s, v2.4h, v3.4h\n  mov x8,94\n  svc 0\n' > i.s
as -march=armv8.2-a+fp16fml -o i.o i.s && ld -o i i.o && taskset -c 0-3 ./i; echo $?
```
