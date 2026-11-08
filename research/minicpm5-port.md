# MiniCPM5-2B → MNN (M10) — zero-patch conversion record

**Status 2026-11-08: done.** `openbmb/MiniCPM5-2B` is plain `LlamaForCausalLM`
(model_type `llama`, GQA 16:2, head_dim 128, rope_theta 5e6, 42×2048/6144, 131K ctx,
untied 130560-vocab head, single 5.04 GB safetensors, `PreTrainedTokenizerFast`) — stock
`llmexport`, **no MNN changes at all**. Pipeline: UM890 export → board smoke/bench → R2.

## Pre-flight gates (verified on the board against the live HF files)

- **Chat template**: ships as `chat_template.jinja` only (tokenizer_config has no
  `chat_template` key — transformers 5.x layout, same as lfm3). But unlike lfm3/gemma4/
  glm_ocr/hunyuan_vl the real template **passes minja**: built the engine's
  `apply_template` and rendered it — correct ChatML, `<s>` bos, non-thinking
  `<|im_start|>assistant\n<think>\n\n</think>` form under `enable_thinking=false`.
  Heavy jinja features all parse (`namespace`, `messages[::-1]`, `tojson(ensure_ascii=False)`
  kwargs, `is string`, `is defined`). **No jinja override needed**; the exported
  `llm_config.json` carries the real template verbatim.
- **EOS stop list**: `eos_token_id` is a list `[1, 130073]` (`</s>`, `<|im_end|>`) in
  `generation_config.json`; `utils/tokenizer.py` flattens it into `stop_ids` via both
  `tokenizer.generation_config` and a raw-JSON fallback. Smoke confirmed clean stops.
- Gate tool: `apply_template <test.json> 1` now built by `scripts/build_mnn.sh` — use it
  to pre-check any `chat_template.jinja` before paying for a full export.

## Runs

- **UM890** (`scripts/minicpm5_um890.sh setup download export test push`, reuses lfm3-work):
  `--test` HF-vs-torch gate coherent (thinking preamble + correct answer), int4 b64 export
  1.7G, `llm_demo` smoke 35.8 tok/s x86, stops on `<|im_end|>`.
- **Board**: smoke PASS (correct "2, 3, 5", 482-tok decode incl. ~450-tok think preamble);
  pinned bench (`scripts/bench_mnn.sh`): pp512 **57.26 ± 0.29**, tg128 **18.38 ± 0.05**,
  1.18 GiB runtime. Below Qwen3.5-2B (76/19) — 2.49B params in 42 layers + untied 130K head.
- **Quality** (`q_minicpm5_smoke` → `q_minicpm5_r2`, R2 protocol — thinking-style):
  smoke showed the F17 pattern (4/6 MC suites 0.0 via 16-tok gen-miss on reasoning preambles);
  R2 ladder 10/10 in 2.8 h, extraction audit 0–1% gen-miss:
  gsm8k 76, humaneval **52** (best int4 row), arc 75, hellaswag 50, winogrande 51,
  obqa 50, csqa 47, mmlu 58.1, niah 5/5, ppl **31.93**. 8-task mean 57.4 ≈ 2B DIST2 56.8.
  Shape: reasoning/code/knowledge strong, commonsense genuinely weak (WinoGrande ≈ coin
  flip; LFM3-1.7B +15/+22/+6 on WG/OBQA/CSQA) — checkpoint behavior, not harness noise.

## Open

- llama.cpp Q4_K_M baseline (official GGUFs exist) for the §1 cross-engine cell — parent repo.
- int8 variant (opt-in `q_minicpm5_int8_r2`) if the quality tier is ever wanted; HQQ skipped
  (net negative on LFM3).
