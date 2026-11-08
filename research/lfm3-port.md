# LFM3 → MNN export port (M7)

**Status: verified end-to-end on the Minisforum UM890 PRO (2026-11-06).** torch gate passed, int4
export 970 MB, `llm_demo` greedy smoke answers the primes prompt correctly (prefill 132 / decode
44 tok/s, clean exit). Next: int8/hqq variants, quality-matrix rows, upstream PR.

History: first run (Fedora 44, GCC 16.2.1) hit a libstdc++-16 `__int128` redefinition bug on all
`-std=gnu++XX` targets (unrelated to the patch; protobuf with `CXX_EXTENSIONS OFF` built fine) —
fixed in `b436509` via strict `-DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_EXTENSIONS=OFF`; re-run pending.

**Runtime issue log.** (1) torch-side `--test` gate passed on first try (thinking trace + "Paris").
(2) llm_demo decode died with `Compute Shape Error for /layers.2/self_attn/o_proj` — layer 2 is the
first attention layer; the converter's `FuseTransformerC4::fuseAttentionOutputC4` gate pattern
requires the gate projection to carry `num_heads*head_dim` channels, but LFM3's per-head scalar
gate (`hidden→16` + broadcast/Tile) broke the matcher, leaving the Attention op with
`output_c4=false` and o_proj fed an NHWC tensor. Fixed export-side by expanding the gate weight
rows per head into a `hidden→hidden` Linear (numerically identical, ~+2 MB/layer int4). Lesson for
any gated-attention port: the gate must be a plain full-width projection, never a broadcast.
(3) after the gate fix, decode ran (gemm/attention/KV-cache all healthy in perf+gdb) but llm_demo
never exited: with stdin-only it enters `chat()`, whose REPL loop ignores getline EOF and re-prefills
the ever-growing conversation indefinitely. Harness fix: pass prompt via file arg
(`llm_demo cfg prompt.txt 512` → `eval()` path, terminates on EOS/max tokens).
(4) smoke then answered a generic greeting (11 prompt tokens) — no chat template reached the
engine: LFM3 ships its template only in `chat_template.jinja`, which neither the exporter nor
minja handles; fixed export-side with a simplified ChatML `jinja` override (details below).
Re-exported smoke answers correctly: 18 prompt tokens, 126 tokens at 44.1 tok/s, direct
non-thinking "2, 3, and 5".
(x86_64 — this QCS8300 board has no torch and is too small for the export).
**Artifacts:** [`patches/mnn-lfm3-export.patch`](../patches/mnn-lfm3-export.patch) (applies to MNN
`17236ad`, verified with `git apply --reverse --check`) ·
[`scripts/lfm3_um890.sh`](../scripts/lfm3_um890.sh) (`setup download export test all`).

## Why safetensors, not GGUF

`gguf2mnn.py` is a 528-line script that only understands llama/gpt2 tokenizer layouts and a couple of
architectures — LFM3's tokenizer (BPE + `Sequence[Split, ByteLevel]` pre-tokenizer, 128k vocab, Jinja
chat template with tool/reasoning parsing) and hybrid architecture don't fit it. The
safetensors → `llmexport` path reuses MNN's full tokenizer exporter (`export_mtok` handles exactly this
BPE+Sequence layout) and the whole engine-side feature set. **Zero C++ changes were needed.**

## LFM3-1.7B architecture vs. what MNN already had

24 layers, hidden 2048, 16/4 heads (head_dim 128), MLP 6144, vocab 128k, ctx 262144, tied embeddings.
`layer_types` ∈ {`conv`, `full_attention`, `sliding_attention`}; `norm_eps` 1e-5; norms carry
`variance_epsilon`; custom code via `trust_remote_code` (needs transformers ≥ 5.12).

| LFM3 feature | Engine support (already present) | Export-side gap (this patch) |
|---|---|---|
| `conv` layers — LFM2-style gated short conv (`in_proj` → B·x → depthwise conv(7) → C· → `out_proj`), `conv_L_cache=7` | `FusedLinearAttention(short_conv)` auto-selected when `conv_L_cache > 0` (`create_linear_attention`) | mapper entry (`regist_lfm3`) — conv structure is byte-identical to LFM2, maps as-is |
| `sliding_attention` (window 256) mixed with full layers | `attention_type='mix'`: two-plane mask, per-layer plane pick — auto-derived from `layer_types` in `config.py` | none — `sliding_window` just needs to reach the runtime config.json (llmexport already writes it) |
| Dual rope: full θ=1e6 partial 0.5, sliding θ=1e4 full | gemma3/gemma4 dual-rotary machinery in `model.py` (`rotary_full`/`rotary_sliding`, per-layer pick in `LlmModel.forward`) | add `'lfm3'` to the `_is_dual_rope` gate + **partial-rope fix** (below) |
| QK RMSNorm over head_dim, pre-rope, plain gamma | `Attention` q/k norm + FusedRoPE norm folding | canonicalize via `canonical_rms_norm` (eps from `variance_epsilon`) |
| Per-head sigmoid output gate (`gate_proj`: hidden→16 heads) | shared `attn_output * sigmoid(gate)` site exists (q_gate path) | broadcast gate (bsz,seq,16) → (bsz,seq,16·128) before that site |
| Sandwich norms (`post_operator_norm`, `post_ffn_norm`) on both branch outputs | — | optional-norm branch in `Decoder.forward` (getattr-guarded, off for every other model) |
| Tied embeddings | auto-detected by weight equality (`llmexport.py:174`) | none |

## The correctness trap: partial RoPE pairing

HF `Lfm3RotaryEmbedding` implements partial RoPE by **zero-padding inv_freq** and applying
`rotate_half` across the **full head_dim**: pairs are `(i, i + head_dim/2)`, and the trailing 64 dims
rotate with frequency 0 (cos=1, sin=0 — identity). MNN's auto-detected partial path (`phi_rotary_pos`,
used when `cos.shape[-1] < head_dim`) instead rotates the **first** `rotary_dim` dims with pairs
`(i, i + rotary_dim/2)` — a **different permutation**, numerically wrong for LFM3 even though it "runs."

The patch keeps the cos/sin tables at full `head_dim` width with the unused frequencies zeroed
(`model.py`, lfm3 branch), so the standard `llama_rotary_pos` / FusedRoPE path reproduces HF
bit-for-bit. Same table math as HF: `θ^(-2i/128)` for i < 32, zeros for i ≥ 32.

## Patch anatomy (5 edits, 4 files, export-side Python only)

1. **`utils/model_mapper.py` — `regist_lfm3()`**: config (`sliding_window`, `rope_parameters`,
   `layer_types`, `conv_L_cache`), model (`model.embedding_norm` → `final_layernorm`), decoder
   (`conv`/`operator_norm`/`ffn_norm` like lfm2 + the two sandwich norms), attention (q/k/v/o +
   `q_norm`/`k_norm`/`gate_proj`), linear_attention (in_proj/conv/out_proj).
2. **`utils/model.py`**: `'lfm3'` in `_is_dual_rope`; lfm3-specific zero-padded partial-rope table for
   `rotary_full` (θ from `rope_parameters.full_attention`, `rotary_dim` stays `head_dim`).
3. **`utils/transformers.py` — `Attention.__init__`**: canonicalize lfm3 q/k norms (plain gamma, eps =
   `variance_epsilon`) so FusedRoPE stores them as gamma.
4. **`utils/transformers.py` — `Attention.forward` / `Decoder.forward`**: `gate_proj` branch (per-head
   sigmoid gate, expand to head_dim) before the shared sigmoid site; `post_operator_norm` after the
   attention/conv branch and `post_ffn_norm` after the MLP, both getattr-guarded `is not None`. The
   per-head scalar gate is first expanded to a full-width `hidden→hidden` Linear (rows repeated per
   head, numerically identical) so the converter's gated-attention C4 fusion matches (see runtime
   issue log 2).
5. **`llmexport.py` — `lfm3` jinja override**: simplified ChatML chat template written into
   `llm_config.json` (see "Chat template gap" below).

`do_map` sets unmapped sources to `None`, so every hook is inert for non-lfm3 models; all edits are
additive branches. `py_compile` clean; patch reverse-applies against `17236ad`.

## Run + verification plan (UM890)

```bash
scripts/lfm3_um890.sh all        # setup → download → export → test
```

1. **Correctness gate** (built into `export` step): `llmexport.py --test "..."` runs the *patched torch
   trace* and prints token logits — compare its greedy continuation against
   `transformers` + `trust_remote_code` on the same prompt (same greedy sampling). Coherent, identical
   first tokens ⇒ rope pairing / gate broadcast / sandwich norms are right.
2. **`llm_demo` smoke** (test step): greedy short generation from the exported int4 MNN model. Expect
   coherent answer to the primes prompt; garbled output points at the mix-mask or conv-plane (check
   `attention_type: mix` + `sliding_window: 256` in the exported `config.json`).
3. If the gate passes, iterate variants: `--quant_bit 8`, `--hqq`, block-size sweep — same harness.

### Likely triage spots (in order)

- **transformers 5.x custom-code loading** — `Lfm3ForCausalLM` arrives via `auto_map`/`trust_remote_code`;
  MNN's `get_model_class` falls through to `AutoModelForCausalLM`, which should pick it up. If loading
  complains about tied weights (5.x API), mirror the `MobileMoEForExport` shim in `model.py:78`.
- **FusedRoPE partial-cos path** — if `--test` diverges from HF, first suspect the rope table; disable
  fused rope (`--transformer_c4 False`) to bisect.
- **mix-mask tracing** — the two-plane mask is built at runtime from config; a wrong `sliding_window`
  (0) silently degrades sliding layers to full attention (quality, not garbage).

## Chat template gap — smoke answered the wrong thing (2026-11-06)

Symptom: first UM890 smoke decoded cleanly (45.5 tok/s) but answered a generic greeting;
`prompt tokens num = 11` — far too few for template + question.

Root cause: LFM3 ships its chat template **only** in `chat_template.jinja` (12 KB: macros,
namespaces, custom `{% generation %}` tags). `tokenizer_config.json` has no `chat_template`
key, so the exporter's `get_chat_template()` found nothing, wrote no `jinja` block into
`llm_config.json`, and the engine's `Tokenizer::apply_chat_template` fell back to raw message
concatenation (`tokenizer.cpp:1022`). The model saw an untemplated, out-of-distribution prompt
— fluent but unconditioned output.

Template facts (from the raw HF file): turns are ChatML — `<|im_start|>{role}\n…<|im_end|>\n`
(pipe special tokens, both in `added_tokens`); bos is `<|startoftext|>`; the assistant's thinking block
uses **pipe-free text tags** `<think>`…`</think>` (plain text, BPE-split, not special tokens);
`enable_thinking` defaults true; the non-thinking form is an empty block `<think></think>`
directly after `<|im_start|>assistant\n`, and assistant history turns get the same empty block
prepended when thinking is not carried over. eos is `<|im_end|>`.

Fix (`llmexport.py`, same pattern as the `gemma4`/`glm_ocr` overrides): an `lfm3` branch writes
a simplified minja-parseable ChatML `jinja` block rendering the **non-thinking** form. Verified
by jinja2 render: bos + system/user/assistant turns + empty think block at the generation
prompt. Engine plumbing note: `Llm::setChatTemplate` consumes only `jinja.chat_template` and
`jinja.eos`; the `bos_token` jinja variable comes from `tokenizer.mtok` (populated from
`tokenizer_config.json`'s `bos_token`, which LFM3 does define), so `{{ bos_token }}` renders.

Result after re-export (2026-11-06): prompt tokens 18 (was 11) — `llm_demo`'s eval path templates
a single user message, no system block — and the answer names 2, 3, 5 directly, non-thinking form,
126 tokens at 44.1 tok/s.
Thinking mode = replace the empty block with the open text tag; parked for benchmark
comparability with the non-thinking matrix.

## Board-side validation (QCS8300, 2026-11-06)

Exported dirs pushed from the UM890 (`scripts/lfm3_um890.sh push` → `lfm3-models/`). The board's
MNN build is the same revision (`17236ad`) and needed no patch (engine-side lfm3 support is
upstream). Results:

- **ARM output is byte-identical to x86** on the greedy primes prompt — engine parity across
  architectures confirmed for all three quant variants.
- `llm_demo` decode (all 8 cores, greedy): int4 **21.7**, int4+HQQ **21.8**, int8 **12.6** tok/s
  (prefill 55–63). int8 pays the memory-bound decode tax at 2× size. Pinned `llm_bench -pg 512,128`
  regime (comparable with §1/§3 of BENCHMARKS): int4 **75.4 ± 0.2 / 21.4 ± 0.0** pp/tg, HQQ
  75.7 / 21.4, int8 77.6 / **12.4** — vs the llama.cpp Q4_K_M baseline 49.0/11.7 that is
  **+54% pp / +83% tg**, the largest cross-engine gap in the table (BENCHMARKS §1).
- **Quality protocol:** smoke under the plain sibling protocol scored 0.00 on arc/hellaswag/
  openbookqa/mmlu — LFM3 answers MC with a reasoning preamble ("The question asks…") and never
  reaches a letter inside the 16-token MC budget: the F17 reasoning-style failure verbatim, even
  with the non-thinking template. Rows are therefore scored under the **R2 protocol**
  (`q_lfm3_*_r2` stages: zero-shot MC + "Answer: X" cue + lifted budgets), matching the
  QWEN35-*-DIST2 rows. R2 probe: arc 5/5.
- Variant smokes (UM890): int8 1.8 GB, int4+HQQ 970 MB — both answer the primes prompt correctly;
  HQQ costs zero runtime vs plain int4.
- **Full R2 ladder done (2026-11-07, zero failures, ~12.4 h for all three variants).** Rows in
  BENCHMARKS §4.1. Verdicts: LFM3-1.7B beats 2B-DIST2 on 7/10 tasks (gsm8k 86 vs 60, mmlu 69.2
  vs 51), loses hellaswag (41 vs 52) and ppl (39.7 vs 33.7). HQQ **net negative** (worse on 7/10;
  humaneval +18 doesn't redeem gsm8k −8) — plain int4 stays the recipe. int8 is the quality tier
  (humaneval 38→72, mmlu 77.3, ppl 33.0 at 2B parity) for 2× size and −42% decode.

## Upstream angle

MNN issue [#5104](https://github.com/alibaba/MNN/issues/5104) (2026-11-04) requests LFM3-VL-2B; no
implementation exists. **User decision (2026-11-07): the PR must ship LFM3-VL-2B, not text-only** —
otherwise it is apples and oranges vs what #5104 asks. Text half is done and validated; VL scoping:

- **Text tower: free.** VL `text_config` == the ported LFM3-1.7B exactly (same `layer_types` incl.
  sliding_attention, same rope_parameters) — the existing lfm3 mapper/patch applies unchanged.
- **Data API is Qwen2-VL-shaped**: `pixel_values (N, 3·2·16²)` grouped by 2×2 merge block +
  `image_grid_thw`; patchify permutation matches MNN's `Qwen2Vision.vision_reshape` layout;
  prompt = `<|vision_start|>` + N×`<|image_pad|>` + `<|vision_end|>` (ids 125009/125030/125010,
  video 125029); mean/std 0.5, resample BICUBIC, smart_resize factor 32, pixels ∈ [65536, 16.7M].
- **Zero C++ changes looks achievable**: export `processor_family = "qwen2_vl"` → C++ `Qwen`
  processor: `Patch::prepare` = the same smart_resize (factor patch·merge, min/max pixels via
  config, INTER_CUBIC, temporal dup); `Qwen::gridPositions` emits **(2, N) block-major h/w coords
  = exactly** `Lfm3VlVisionRotaryEmbedding.forward`'s documented input; `attentionInputs` =
  block-diagonal bidirectional mask = HF's per-frame full attention. No mrope: `Lfm3VlModel.forward`
  passes 1-D position_ids through, and `Omni::gen_position_ids` falls back to plain positions when
  the graph's position input has dim[0]==1.
- **The one real gap**: the traced vision graph's contract is (patches, position_ids,
  attention_mask) — no `grid_thw` — but the vision tower's gated depthwise **2D conv** layers
  (`Lfm3VlVisionConv2d`, kernel 3, per-frame zero pad) need the frame grid. ONNX Reshape in the MNN
  converter takes a constant shape only → rewrite the conv as **9 shifted gathers** with
  neighbor indices/masks derived elementwise from position_ids (frame start = `i // (h·w)`,
  block-major offset formula) — mathematically identical to HF, fully dynamic, no data-dependent
  shapes. Verify numerically against the HF reference.
- Vision tower: 14 layers (conv,conv,attn,then alternating), hidden 1024, 16 heads / 8 kv,
  head_dim 64, GQA + per-head q/k RMSNorm, axial rope (inv_freq from head_dim//4, cat h|w,
  rotate_half over full 64), MLP silu-gate 2560; projector LayerNorm→Linear(4096)→GELU→Linear(2048)
  with 2×2 downsample. Model ships custom code (`modeling_lfm3_vl.py`, transformers ≥5.19-dev APIs:
  `vision_utils.get_vision_position_ids/get_vision_attention_seqlens`).
- Chat template: 12 KB `chat_template.jinja` → same minja gap as text; extend the llmexport lfm3
  jinja override to `lfm3_vl` (non-thinking + vision placeholder).

## VL implementation record (2026-11-07, export-side, zero C++ changes)

All pieces live in `patches/mnn-lfm3-export.patch` (regenerated, 5 files) — pure Python export:

- **`utils/vision.py` — `Lfm3VlVision(Vision)`** (registered `lfm3_vl`). Subclasses `Vision`
  directly, NOT `Qwen2Vision`: `Lfm3VlModel` uses plain 1-D text positions, and `Llm.get_position_ids`
  routes to `visual.get_position_ids` whenever it exists (Qwen2VL mrope) — absent attribute keeps the
  1-D fallback, which is also what `Omni::gen_position_ids` picks (module position input dim[0]==1).
  Qwen2Vision helpers reused by class-body aliasing: `refresh_image_resize_config`,
  `reset_multimodal_state`, `str_to_ids` (`<img>` prompt parsing), `vision_reshape` (temporal dup +
  merged flatten — identical to the HF processor's merge-block (c,tp,p,p) flatten),
  `vision_position_ids` (== C++ `Qwen::gridPositions`), `vision_attention_mask` (per-frame blocks),
  `images_forward`, `embed`, `smart_resize` (factor 32, [65536, 16.7M], BICUBIC == C++
  `qwenVlSmartResize` + `resizeImage(INTER_CUBIC)`).
- **`Lfm3VlVisionConv`** — the grid-free rewrite of `Lfm3VlVisionConv2d`: in_proj chunk(value,scale,key),
  gated=value·key, then nine shifted `index_select` gathers whose merge-block source index and validity
  masks come from position_ids arithmetic; `frame_base = floor(i/(grid_h·grid_w))·(grid_h·grid_w)` keeps
  multi-frame taps inside their frame. Output `out_proj(scale·convolved)` == HF.
  **Numerically validated bitwise-exact** (numpy port vs HF reference on C++ gridPositions order,
  t=1..3, grids 2×2…14×30: max abs diff 0). `scripts/lfm3vl_conv_test.py` repeats this in torch +
  ONNX/ORT round-trip on the UM890.
- **Attention layers** reuse MNN `Attention` (kv_cache/`past_key_value` inert in vision — never set;
  `export_fused_attn` False → HF `q_layernorm`/`k_layernorm` traced directly, norm-before-rope like HF)
  + `VisionRotary(rotary_dim=head_dim//2=32, theta=1e6)`: its cos/sin is exactly HF's axial
  recomposition cat(h_freqs|w_freqs) repeated ×2 over head_dim (verified elementwise).
- **Projector + MLP** reused as HF modules verbatim: LayerNorm → reshape(-1, 4096) (constant shape,
  Qwen2Vision-merger precedent) → linear_1 → gelu → linear_2; vision MLP is HF `Lfm3VlMLP` (w1/w3/w2).
- **Graph contract** (UM890-verified): base `Vision.export_onnx` for `patch_layout='merged'` replaces the
  patch input with **raw normalized frames** — inputs `pixel_values (temporal, 3, gh·p, gw·p)`,
  `position_ids (2,N)`, `attention_mask (1,N,N)`; the merge-block patchify is *inside* the graph
  (`Vision.export_inputs`, same permute as the HF processor). `get_config` writes
  `vision_processor = {family: qwen2_vl, input_layout: pixels, output_layout: tbc, patch/merge/temporal}`;
  C++ `Patch::prepare` resizes (INTER_CUBIC, mean/norm from config) and `_Concat`s the image
  `temporal` times, `Qwen::patchInputs` skips `packPatches` for `input_layout=pixels` and appends
  `gridPositions` + per-frame mask. Output `image_embeds (N/4, 1, llm_dim)` (tbc, no permute:
  `output_axes=None`). Main-config gets image_mean/norm=0.5·255 & 1/(0.5·255), size_unit 32, min/max
  pixels, token ids 125009/125010/125030. Placeholder strings are LFM3-specific —
  `<|image|>`/`<|video|>` (NOT qwen-style `*_pad`), resolved from the tokenizer's
  `added_tokens_decoder` by id. visual quant_bit 8 (class default).
- **`utils/model_mapper.py` `regist_lfm3_vl`**: deepcopy of the lfm3 map, config keys prefixed
  `text_config.`, model paths under `model.language_model.*`, `visual`→`model.vision_tower`,
  `multi_modal_projector`→`model.multi_modal_projector`. `init_models` discovers `regist_*`
  alphabetically, so `lfm3` exists before `lfm3_vl` deep-copies it.
- **`utils/model.py`**: MODEL_CLASS_MAPPING `lfm3_vl`→`AutoModelForImageTextToText` (custom code via
  auto_map); dual-rope + zero-padded partial-rotary conditions extended to `lfm3_vl`.
- **`llmexport.py`**: jinja ChatML override extended to `lfm3_vl`; `--test "<img>…</img>…"` runs the
  full torch-side VL pipeline (str_to_ids → vision forward → embed → text backbone).
- **Driver**: `scripts/lfm3vl_um890.sh` (setup/download/convtest/onnxexport/onnxtest/export/test/push)
  + `scripts/lfm3vl_onnx_test.py` (ORT `visual.onnx` vs HF `get_image_features().pooler_output` on one
  image: raw frames rebuilt by inverting the HF patchify permutation — bitwise self-checked — plus
  gridPositions ids and the per-frame mask, i.e. exactly the C++ `input_layout=pixels` feed).
  llm_demo image prompt syntax: `<img>/path/file.jpg</img>` in the prompt file
  (`kOmniMultimodalRegex`).
- Static checks pass (py_compile ×5, AST alias audit, conv math bitwise-exact); patch applies clean to
  `17236ad`. **Pending on UM890**: convtest → onnxexport → onnxtest → `--test` gate → int4 b64 export →
  llm_demo image smoke → board QA. Video math (frame_base, BlockDiagonal) is implemented but only
  images are tested in this pass.
