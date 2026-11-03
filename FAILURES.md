# Failures — MNN on QCS8300

Numbering continues the parent survey: [`../ventunoq-llamacpp/FAILURES.md`](../ventunoq-llamacpp/FAILURES.md)
owns **F1–F8**; this repo owns **F9–F19**. Legend (parent convention): 🔒 blocked outside our
control · 🛠️ actionable by us · 📝 gotcha (solved, don't re-learn it the hard way).

Inherited and still binding here:

| Parent | One-liner | Effect on this repo |
|---|---|---|
| F1 | QNN/QAIRT needs the `smmu_proxy` kernel service; `6.8.0-1084-qcom` has none | MNN Hexagon backend unreachable (F13) |
| F2 | Turnip Vulkan lacks `StorageBuffer8BitAccess`; quantized/tiled matmuls fault | MNN Vulkan backend = quick probe only (M4) |
| F8 | QCS8300 lacks FMLAL/FMLSL **vector** despite HWCAP `asimdhp` → mistral.rs SIGILL | MNN is safe by construction — audit in [research/kernel-audit.md](research/kernel-audit.md) |

---

## F9 — taobao-mnn Qwen3.5 exports are multimodal: `llm_demo` init fails 📝

`taobao-mnn/Qwen3.5-{0.8B,2B}-MNN` ship with `is_visual: true` (Qwen3-VL-style exports):

1. `llm_demo <dir>/config.json` → `Can't open file: …/visual.mnn` → `LLM init error`.
   (Downloading `visual.mnn` + `visual.mnn.weight` doesn't help either:)
2. → `Unsupported vision processor family: qwen3_vl` → `LLM init error` — the current
   engine build doesn't know that processor family.

**Workaround (text-only, what all benchmarks here use):** copy the dir, delete the visual
files, set `"is_visual": false` in `llm_config.json`, drop the `mllm` block from
`config.json` — `scripts/text_only.py` does it. `llm_bench` never touches `visual.mnn`, so
benchmarking works on the unpatched dir (it just prints the open-error per row).

## F10 — `-c` precision flag is inverted from intuition: `-c 1` is fp32 📝

`llm_bench -c <n>`: **1 = "High" = fp32, 2 = "Low" = fp16** — regardless of
`"precision": "low"` in `config.json` (the flag overrides the file). fp32 silently costs
**−65% decode** (25.1 vs 41.5 t/s at 0.8B) with *no warning*. Always pass `-c 2`
explicitly; `scripts/bench_mnn.sh` does.

## F11 — Hugging Face downloads stall mid-file 📝

Same as parent F6: plain `curl -O` hangs at partial sizes. Working recipe (baked into
`scripts/dl_model.sh`): `curl -L -C - --retry 3 --speed-limit 200000 --speed-time 30`,
launched with `setsid nohup … &` for long pulls. Throughput when healthy: 25+ MB/s.

## F12 — LFM2/LFM3 are not convertible: no model-mapper entry 🛠️

- `transformers/llm/export/utils/model_mapper.py` has `qwen3_5`, `qwen3_5_moe`, … but **no
  `lfm2`/`lfm3`** — only `lfm2_audio` in `utils/audio.py` (a different thing).
- `gguf2mnn.py` maps only a llama/qwen-style op template (`model == 'llama'` branch); an
  LFM GGUF has no path in.

Consequence: **llama.cpp remains the only LFM3 engine on this board** (and the only NPU
path). Upsert path: [ROADMAP M7](ROADMAP.md) — a `model_mapper` + `model.py` port of LFM's
conv-gate/short-conv blocks; start with an upstream issue, not a PR.

## F13 — MNN's Hexagon LLM backend is Android + QNN → unreachable here 🔒

The backend (docs `transformers/llm.md`, "Android Hexagon 后端") requires `backend_type:
hexagon`, a C4-fused export, and pushing `libMNN_htpops*.so` skel libs beside the model —
i.e. the **QNN runtime stack**. Every QNN runtime path on this board dies at `HAP_mmap2`
because the kernel lacks `smmu_proxy` (parent F1, ioctl-level evidence in parent
`npu/NOTES.md`). Not engine-specific; nothing userspace can do. llama.cpp's Hexagon runner
(bypasses QNN via FastRPC/FBCL) stays the only NPU path: 679 t/s prefill @LFM2.5-2.6B.

## F14 — `llm_demo` has no HTTP API; serving needs a driver script 🛠️

`llm_demo` is stdin/stdout interactive only — no OpenAI-compatible endpoint like
`llama-server`. Worse, its own batch mode (`llm_demo config prompt.txt`) reads **one prompt
per line** and streams every answer into a single stdout with no separator — unusable for
multi-line benchmark prompts. Delivered workaround (M1 harness): `scripts/llm_eval.cpp`, a
batch driver built into the clone by `scripts/build_mnn.sh` — one prompt file in, one answer
file out, greedy sampler, fresh KV per item, resume-safe — driven by
`scripts/quality_eval_mnn.py`. llama.cpp still owns any client/serving workload.

## F15 — 9B-MNN export splits embeddings into `embeddings_bf16.bin`; the hardcoded download list misses it 🛠️ (rerun on hold)

`taobao-mnn/Qwen3.5-9B-MNN` ships an extra **`embeddings_bf16.bin` (2.0 GB)** that the
0.8B/2B/4B repos don't have. The old `dl_model.sh` `FILES` list was the hardcoded classic
six → embeddings never downloaded → the model emits **`!!!!` repetition on every prompt**:
QWEN35-9B-INT4 scored 0.000 on all 10 suites (gsm8k `tail` = all `!`) and ppl = **248,300**
vs 16–48 on healthy rows.

Not a corrupt download: `sha256 llm.mnn.weight = 896bf5a7…` matches the HF LFS oid exactly
(DISK_LOG 2026-11-02 20:34) — it is a missing-file bug in our fetch list. **Fixed:**
`dl_model.sh` now derives the file list from the HF tree API (keeps the text-only `visual.*`
exclusion), so any per-model extra file is picked up automatically. 9B rerun parked by
choice — at ~19 h for the R1 suite it is the slowest row either way.

## F16 — `taobao-mnn/Qwen3.5-9B-…-Opus-Reasoning-Distilled-MNN` is an empty placeholder; HF serves 15-byte `Entry not found` bodies with HTTP 200 📝

The repo appears in the org API but its tree contains **only `.gitattributes`**.
`curl …/resolve/main/<file>` answers 200 with a 15-byte body `Entry not found` — the old
`dl_model.sh` "succeeded" (28K total), and the ladder died 2 s into `q_dist9b` with a
`JSONDecodeError` in `text_only.py`. Guard added: `dl_model.sh` aborts on any downloaded
file < 100 B containing "not found". Nothing to run until upstream uploads — recheck the
tree API before scheduling a 9B-DIST suite.

## F17 — Opus-distilled models think aloud and re-solve the few-shot prefix: R1 distilled rows score 0.000 on MC+humaneval+niah 📝 fixed (R2 protocol)

The Claude-Opus-Reasoning-Distilled exports reason aloud ("Let me analyze each option:
A… B… C…"), which breaks three sibling-pinned harness assumptions:

1. **MC (16-token budget)** runs out mid-analysis → all six MC suites 0.000 via `gen-miss`.
   Lifting the budget alone doesn't help: **the 3-shot prefix backfires** — the models
   continue the shot block and re-answer the example questions, so "last Answer: X" matches
   a *shot's* answer (probe: pred=B on 6/8, the shots' letters).
2. **humaneval (512 tok)** → prose reasoning ("Time complexity: O(n²)…") is executed as
   Python (SyntaxError 0.000); additionally `cut()` at the first ``` erased fenced code
   before extraction could see it.
3. **niah (24 tok)** → preamble only → 0/5.
   (gsm8k survived on its 700-token budget: 0.55/0.65.)

**R2 protocol** (`-DIST2` tags, `q_dist2b2`/`q_dist4b2`; plain-model rows untouched):
`--no-mc-shots` + `--mc-instruction` ("think briefly, end with a line 'Answer: X'") +
budgets MC 512 / HE 1536 / GSM 1024 / NIAH 512 (niah prompt also gains a "do NOT
enumerate the context" cue), via new `quality_eval_mnn.py` flags (`--mc-tokens
`--mc-instruction --no-mc-shots --he-tokens --gsm-tokens --niah-tokens`). Parsing is
reasoning-safe: explicit "answer is X" (last match) first; letter/choice-text heuristics
only on the tail; code extracted from the LAST fenced block / last `def <entry_point>(`
(reasoners sketch, then finalize); niah takes the last 5-digit number. Probe trail
(2B-DIST): arc n=8 — R1 0.000 → 96 tok 0.000 (shot-replay caught) → 192 tok+instr 0.000
(still shots) → zero-shot 256 tok 0.250 → +briefness instr 0.375 (chance 0.25) → budget
512. gsm8k n=6 @1024: 6/6 (R1 was 0.55). humaneval n=4: first-def extraction @1024 still
0/4 (truncated/mismatched code) → last-def + 1536 tok. niah n=2: enumeration @128 1/2 →
no-enumerate + 512.
