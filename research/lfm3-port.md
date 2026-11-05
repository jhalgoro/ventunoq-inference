# LFM3 → MNN export port (M7)

**Status:** patch written + self-contained UM890 driver ready; execution pending on the Minisforum UM890 PRO
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

## Patch anatomy (4 edits, 3 files, export-side Python only)

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
   attention/conv branch and `post_ffn_norm` after the MLP, both getattr-guarded `is not None`.

`do_map` sets unmapped sources to `None`, so every hook is inert for non-lfm3 models; all four edits are
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

## Upstream angle

MNN issue [#5104](https://github.com/alibaba/MNN/issues/5104) (2026-11-04) requests LFM3-VL-2B; no
implementation exists. This text-only port is the prerequisite half and a reasonable contribution
candidate once verified end-to-end on the UM890. VL-2B additionally needs the vision tower (mirror the
`lfm2_vl` pattern: `MODEL_CLASS_MAPPING` entry + visual mapper) — parked until 1.7B text works.
