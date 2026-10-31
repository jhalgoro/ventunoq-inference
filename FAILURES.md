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
