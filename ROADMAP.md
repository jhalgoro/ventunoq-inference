# Roadmap — MNN depth program (M1–M9)

Ordered by value/effort on this board. Prefix **M** is this repo's; parent survey items are
**R** ([../ventunoq-llamacpp/ROADMAP.md](../ventunoq-llamacpp/ROADMAP.md)), llama.cpp depth
items are **L** ([../ventunoq-llamacpp-cpu+npu/ROADMAP.md](../ventunoq-llamacpp-cpu+npu/ROADMAP.md)).

| # | Question | Effort | Blocks on |
|---|---|---|---|
| M1 | Is MNN int4 as *good* as it is fast? (quality parity) | low | — |
| M2 | How far up the size range does the win hold? (4B, int8) | low | — |
| M3 | Does `gguf2mnn` make our existing GGUFs first-class? + safetensors→MNN host | low | — |
| M3.5 | MNN-QNN: can MNN drive the QCS8300 Hexagon NPU? | medium | QNN SDK, arch support |
| M4 | Kernel knobs: KleidiAI, ARM82-off, OpenCL/Vulkan probes | low | F2 for GPU |
| M5 | Long context: pp curves + KV memory vs llama.cpp's 8.9 KB/tok | low | — |
| M6 | Sustained + energy: throttle curve, W and J/token vs llama.cpp | low | reuse L4 probe |
| M7 | LFM3 → MNN export port (F12) — text **done** + validated; VL-2B export **done**, all gates green (UM890 + board) | **high** | open upstream PR vs #5104 with `patches/mnn-lfm3-export.patch` |
| M8 | Serving shape: subprocess driver, or concede to llama-server? | medium | — |
| M9 | Decision table: when MNN, when llama.cpp (the repo's product) | trivial | M1–M6 |
| M10 | MiniCPM5-2B → MNN: **exported + board smoke/bench done** (zero patch) | low | R2 quality ladder → §4.1 row; llama.cpp Q4_K_M baseline |

---

## M1 — Quality parity: the number that decides adoption

MNN's int4 carries fp16 per-block scales (≈Q4_K-class on paper) — but the whole +30% decode
claim is worthless if answers degrade. Measure, same corpora as the parent repo:

- **Harness (done)**: `scripts/quality_matrix_mnn.sh` → `scripts/quality_eval_mnn.py`, a
  full port of the sibling `quality_eval.py` — gsm8k, humaneval, arc, hellaswag, winogrande,
  openbookqa, commonsenseqa, mmlu, niah, ppl. Greedy everywhere; `scripts/fetch_bench.py`
  uses the sibling's SPECS verbatim so `quality_parity.py` joins the two engines
  item-by-item. Generation goes through `llm_eval` (`scripts/llm_eval.cpp`, built by
  `build_mnn.sh`): one prompt file per item, greedy sampler, fresh KV per item, resume-safe.
- **MC scoring**: all six MC suites use the gen style (3-shot greedy letter +
  first-mention fallback). MNN exposes no logprobs, so the sibling's walk/raw-walk
  styles are impossible — noted wherever MNN rows face llama.cpp rows.
- **Perplexity**: MNN *does* ship a ppl tool after all — `tools/ppl_eval.cpp`, built by
  `scripts/build_mnn.sh` — run on `bench/gatsby.txt` (ROADMAP v1 said it didn't).
- **Status (2026-10-31)**: MNN side of 0.8B measured — `QWEN35-08B-INT4` in BENCHMARKS
  §4.1 (gsm8k 34%, niah 5/5, ppl 31.1; MC block at/below chance → gen-letter bias).
  **Ladder running 2026-11-01**: `quality_matrix_mnn.sh q_all` (detached) covers 2B/4B/9B
  + 0.8B/2B Base (`--mode base`, template-damage probe) + the Claude-distilled 0.8B–9B
  (recipe probe) — ~45 h, resume-safe, progress in results/quality_status.log.
  Remaining: llama.cpp Q4_0/Q4_K_M rows on the same items (sibling harness).
- Compare: MNN int4 vs llama.cpp **Q4_0** (speed rival) **and Q4_K_M** (quant rival),
  same models 0.8B/2B — llama.cpp rows from sibling-harness runs on the same items.

**Success =** a row in BENCHMARKS §4.1 per model; if MNN quality ≤ Q4_0 − noise, the CPU
recommendation stays llama.cpp regardless of speed.

## M2 — Size range: [`taobao-mnn/Qwen3.5-4B-MNN`](https://huggingface.co/taobao-mnn/Qwen3.5-4B-MNN) (+ int8 export)

- 4B-MNN (~2.4 GB weights int4) fits RAM comfortably; expect the bandwidth-bound crossover
  to shrink the win toward 0 (trend so far: +30% → +19% decode at 2B).
- Also export **int8** (`llmexport.py --weight_quant_bits 8` — needs a host with torch; if
  no x86 host is handy, run on-board with the venv deps, disk-policy bound) and compare
  speed/quality vs int4 and vs llama.cpp Q8_0 (parent L2 runs the same dtype question —
  cross-reference, don't duplicate measurements).
- 9B only if disk <50% allows during the suite (delete-after-suite, DISK_LOG).

## M3 — Model conversion on an x86 host: GGUF *and* safetensors → MNN

Source-probed 2026-11-01 (`MNN/transformers/llm/export/`). Three tools, two philosophies:

**A. `llmexport.py` — HF safetensors → MNN (canonical, what taobao-mnn ships).**
Full quantizer menu: `--quant_bit 2/3/4/8/16`, `--quant_block`, `--hqq/--awq/--gptq/
--smooth/--omni`, `--lora_path`, eagle/dflash drafts. Needs torch + a built `MNNConvert`.
This is the "make our own exports" path — any HF model MNN's `model_mapper` knows.

**B. `safetensors2mnn.py` — safetensors weight-graft onto a template.** Like gguf2mnn:
takes `--path <safetensors dir> --mnn_dir <template>` and quantizes fp weights itself
(default int4/block128 sym, `--scale_bit 16`). Use when a graph template already exists
and we want different weight quant than the template was built with.

**C. `gguf2mnn.py` — GGUF block transplant.** Also template-based (needs `llm.mnn.json`
+ `llm_config.json` of the *same arch* — e.g. a taobao-mnn dir), then transpiles GGUF
quant blocks straight into MNN's external-weight format. Verified supported types:
F16/F32, **Q4_0, Q4_1, Q4_K** (unpacks the 8 sub-block scales/mins into MNN int4+block32
asym — the same affine form, so the transcode is numerically faithful), Q8_0, Q5_0/Q5_1,
Q6_K. Q2_K/Q3_K/IQ\* assert. `--load_token` converts gpt2-BPE/llama tokenizers (Qwen's
BPE is gpt2-family → OK).

Why this matters for M1: a **Q4_K_M GGUF → MNN** dir runs the *same weights* llama.cpp
scores — engine-vs-engine parity with quant held constant, which taobao's HQQ-int4-from-
fp16 exports cannot give us. And the quant ladder (Q4_0 | Q4_K_M | MNN-int4-block64-hqq
on one arch) separates quant damage from engine damage on the flat-MC question.

**Conversion host:** the Minisforum UM890 Pro (8945HS, 8C/16T Zen4, ≤64 GB) or GMKtec
EVO-X2 (HX 370, 12C/24T) — both x86 with ample RAM (9B fp16 load ≈ 20–30 GB peak), where
torch/onnx export actually runs; the board does inference only. Flow: convert on host →
`scp` MNN dir to board `models/` → quality/bench harness unchanged → delete-after per
DISK_LOG policy.

- Attempt `LiquidAI/LFM3-1.7B-GGUF` → **expect hard failure** (F12); capture the traceback
  as evidence for the M7 upstream issue.
- Note the arch map is `general.architecture`-driven (`llama` branch) — document exactly
  where qwen3.5 GGUFs land (`Arch:` print + which template).

## M3.5 — NPU: the MNN-QNN backend (QCS8300 Hexagon)

MNN ships a Qualcomm path we haven't touched — probed in source 2026-11-01:

- **Build**: `-DMNN_QNN=ON` (+ `MNN_QNN_ONLINE_FINALIZE`), needs the QAIRT/QNN SDK
  (`include/QNN` + `lib`) via `QNN_SDK_ROOT`/`HEXAGON_SDK_ROOT` — see
  `MNN/docs/inference/npu.md`. Backend at `source/backend/qnn` (convertor + execution +
  `QnnBackendPlugin.cpp`).
- **Engine**: `llm.cpp` already accepts `backend_type: "qnn"` (and `"npu"`), with a
  `npu_model_dir` config key for offline-compiled artifacts.
- **Offline LLM flow**: `transformers/llm/export/npu/generate_llm_qnn.py` — uses
  `generateIO` to dump per-layer I/O, `npu_convert.py` to serialize QNN graphs; run on
  the x86 host, ship the artifacts to the board.
- **Board precedent**: the parent repo already executes on this QCS8300's Hexagon via
  llama.cpp's NPU runner (F13) — the HTP runtime is there; MNN-QNN is the second engine
  on the same silicon.

Steps: QAIRT SDK on the x86 host → cross-compile (or on-board build) with MNN_QNN →
`generate_llm_qnn.py` on the 0.8B export → push artifacts + `libQnn*.so` → `llm_bench`
with `backend_type:"qnn"` → quality harness re-run (NPU numerics differ; the harness is
already wired, and ppl gives a continuous drift signal).
Risks: QNN SDK Hexagon-arch support for QCS8300 (check v73/v75 libs); LLM-QNN flow
supports a subset of archs — Qwen3.5 support is an experiment, not a given; M5 KV numbers
do not transfer.
Value: prefill acceleration is where the parent repo's NPU runs won — if MNN-QNN matches,
M9's NPU row stops being llama.cpp-only.

## M4 — Kernel & backend probes (each is a build flag + one bench run)

| Probe | Build | Expect |
|---|---|---|
| KleidiAI on | `-DMNN_KLEIDIAI=ON -DMNN_KLEIDIAI_DEFAULT_ON=ON` | i8mm kernels must **not** dispatch (no i8mm); dotprod-only variants may; verify via runtime probe line + speed delta |
| ARM82 off (fp32 kernels) | `-DMNN_ARM82=OFF` | should land near `-c 1` numbers; quantifies the fp16 kernel contribution beyond the flag |
| OpenCL / Vulkan | defaults | quick probe only: `-a opencl` / `-a vulkan` on 0.8B; expect no usable driver or F2-class faults — record, don't fight |

## M5 — Long context

`llm_bench -p {2048,8192,16384,32768}` + `-pg` reuse runs on 0.8B/2B; RSS per KV token via
`/usr/bin/time -v` at 2K vs 32K ctx (`max_new_tokens`/`ctx` in `config.json`). Compare
against llama.cpp's measured ≈8.9 KB/token (LFM3) and Qwen-class 2.6–3.8× that. Also: does
MNN spill/realloc or fail at 32K? (parent pp32768 = 30.8 t/s CPU is the bar.)

## M6 — Sustained + energy (don't reinvent — reuse the L4 probe)

Run the sibling repo's `../ventunoq-llamacpp-cpu+npu/scripts/power_probe.py` (INA232 hwmon2
+ nsp/cpuss/ddrss temps) alongside a 10-min MNN fp16 loop; report avg W and **J/token** for
pp and tg next to llama.cpp-pinned and NPU rows. Hypothesis: MNN fp16 does *less work per
joule* than llama.cpp f32-compute — the decode win should look even bigger in J/token.

## M7 — LFM3 → MNN export port (highest ceiling, real work)

**Status 2026-11-06: verified end-to-end on the UM890 PRO** — torch gate passed, int4 export
970 MB, `llm_demo` greedy smoke answers correctly (prefill 132 / decode 44 tok/s). Two runtime
gaps found and fixed export-side (zero C++ changes): the o_proj gate-layout converter miss (gate
expanded to full width) and the chat-template gap (LFM3 ships `chat_template.jinja` only; a
simplified ChatML `jinja` override now ships in the patch). Full design + triage plan:
[research/lfm3-port.md](research/lfm3-port.md). The original GGUF-spike framing is dead —
`gguf2mnn.py` only understands llama/gpt2 tokenizers, so the path is **safetensors → `llmexport`**
with **zero C++ changes** (engine already has the mix two-plane mask, `FusedLinearAttention(short_conv)`,
gemma-style dual rotary, and tied-embed auto-detect). Delivered:
`patches/mnn-lfm3-export.patch` (additive edits across `model_mapper.py`/`model.py`/`transformers.py`
plus the ChatML `jinja` override in `llmexport.py`, applies to MNN `17236ad`) + `scripts/lfm3_um890.sh` (setup/download/export/test on x86). Key
correctness finding: HF partial RoPE zero-pads inv_freq and pairs `(i, i+head_dim/2)` across the full
head — MNN's `phi_rotary_pos` auto-detect path is numerically wrong here, so the patch keeps cos/sin at
head_dim width with zeroed frequencies. Done: `scripts/lfm3_um890.sh all` ran green on the UM890;
quant variants, R2 quality ladder (3 variants, zero failures) and pinned benchmarks
are in BENCHMARKS §1/§4.1 — int4 75.4/21.4 pp/tg (+54%/+83% vs llama.cpp Q4_K_M), HQQ rejected
(net negative), int8 kept as quality tier.
**PR scope (user decision 2026-11-07): ship LFM3-VL-2B too** — a text-only PR against issue
#5104 (which asks for VL-2B) would be apples and oranges. Scoping done: VL `text_config` is
byte-equivalent to the ported LFM3-1.7B (incl. sliding_attention layers); the vision tower
(14-layer conv/attn hybrid, axial rope) traces under the C++ `qwen2_vl` processor contract
(patches + (2,N) block-major position ids + block-diagonal mask — the C++ `Qwen::gridPositions`
already produces exactly the (2,N) axial rope input HF expects; no mrope: text graph takes 1-D
positions). **VL export implemented (2026-11-07)**: `Lfm3VlVision` in `utils/vision.py` with the
grid-free 9-gather rewrite of the gated depthwise conv (numpy-validated bitwise vs HF; frame-offset
fix keeps multi-frame taps per frame), MNN `Attention`+`VisionRotary` reuse for the attn layers,
HF projector/MLP verbatim, `regist_lfm3_vl` mapper, jinja override extended — all still zero C++
changes, patch regenerated (5 files, applies clean to `17236ad`). Driver `scripts/lfm3vl_um890.sh`
+ conv/onnx numerical gates written. **All gates green 2026-11-07**: UM890 convtest ≤9e-8, ORT
`visual.onnx` vs HF rel 3.78e-6, `--test` + `llm_demo` image smoke coherent (vision 0.84 s,
decode 43.9 tok/s); board (`ventunoq`) llm_demo rebuilt with `MNN_BUILD_OPENCV`+`MNN_IMGCODECS`
(vision support cmake gate) and smoke PASS (vision 1.59 s, decode 21.8 tok/s). **Only the PR vs
#5104 remains.**
VL chat template + tokenizer live in `chat_template.jinja` (same minja gap → extend the lfm3
jinja override).

## M8 — Serving: concede or build?

`llm_demo` has no HTTP API (F14). Options: (a) thin FastAPI-free stdlib socket wrapper
around a persistent `llm_demo` process — medium effort, fragile stdin framing; (b) declare
serving a llama.cpp capability and keep MNN for batch/bench/CLI. Decide after M1 lands;
default is (b).

## M9 — The decision table (repo output, not a task)

Current draft — M1–M6 refine the numbers, nothing else changes it:

| Workload | Engine | Command shape |
|---|---|---|
| CPU chat/code, Qwen-class ≤2B | **MNN fp16** | `taskset -c 0-3 llm_bench/llm_demo -t 4 -c 2` |
| CPU, LFM2/LFM3 | **MNN fp16** (M7 port: 75.4/21.4 pp/tg, +54%/+83% vs llama.cpp Q4_K_M) | same shape; `scripts/lfm3_um890.sh` to export |
| Anything NPU (prefill-heavy ≤1.75 GiB) | llama.cpp Hexagon runner (F13) | `-ngl 100` recipe in parent — M3.5 may add MNN-QNN |
| HTTP serving / concurrency | llama.cpp `llama-server` | parent repo §5 |

## M10 — MiniCPM5-2B → MNN (convert + validate)

`openbmb/MiniCPM5-2B` (2026, ~2.5B): plain `LlamaForCausalLM` — model_type `llama`, GQA 16:2,
head_dim 128, rope_theta 5e6, 42×2048/6144, 131K ctx, no custom code → **zero porting work**,
pure pipeline: export → smoke → board parity → pinned bench → R2 quality row. Risk checks done
(2026-11-07/08, all **verified on the board** against the live HF files):
- eos is a list `[1, 130073]` but lives in `generation_config.json`, which
  `utils/tokenizer.py` flattens into `stop_ids` (both via `tokenizer.generation_config` and a
  raw-JSON fallback) — `<|im_end|>` stops generation.
- Chat template ships as `chat_template.jinja` (tokenizer_config has no `chat_template` key,
  transformers 5.x style) with reasoning_content + XML tool blocks — **minja gate PASS**
  (2026-11-08, `MNN/build/apply_template` against the real template): renders correct ChatML,
  `<s>` bos, non-thinking `<|im_start|>assistant\n<|im_start|>\n\n<|im_end|>` form under
  `enable_thinking=false`. **No jinja override needed** — unlike lfm3/gemma4/glm_ocr/hunyuan_vl.
- Thinking-style model (`<|im_start|>`/`<|thought_begin|>` tokens) → score under the R2
  protocol like the LFM3/DIST2 rows.
- 5.04 GB single safetensors, `PreTrainedTokenizerFast`, untied embeddings (130560×2048).
Official GGUF repos exist → llama.cpp Q4_K_M baseline for the §1-style comparison.

Driver: `scripts/minicpm5_um890.sh` (reuses the lfm3-work venv + MNN build; steps
setup/download/export/export8/exporthq/test/push). Board: `q_minicpm5_smoke` then
`q_minicpm5_all` (int4 R2; int8/hqq stages exist but opt-in — LFM3 showed HQQ net negative).
