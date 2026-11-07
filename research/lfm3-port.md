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
implementation exists. This text-only port is the prerequisite half and a reasonable contribution
candidate once verified end-to-end on the UM890. VL-2B additionally needs the vision tower (mirror the
`lfm2_vl` pattern: `MODEL_CLASS_MAPPING` entry + visual mapper) — parked until 1.7B text works.
