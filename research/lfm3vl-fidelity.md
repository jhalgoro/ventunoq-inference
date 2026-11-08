# LFM3-VL-2B → MNN conversion-fidelity review (M7)

**Question:** did the LFM3-VL-2B → MNN conversion actually work well? The open doubt is the
R2 ppl row: **151.77** on gatsby vs 39.65 for the text-only LFM3-1.7B int4 — while every task
score stayed healthy (8-task mean 62.0 vs 63.2). This page is the review package: what is
already proven, what the two decisive measurements are, and how to read them.

## Evidence layers

### L0 — numerical export gates (all green, 2026-11-06/07, UM890)

| Gate | What it pins | Result |
|---|---|---|
| `scripts/lfm3vl_conv_test.py` | 9-shifted-gather conv rewrite: exported torch graph vs HF module output + ONNX round-trip | rel err ≤ 9e-8 |
| `scripts/lfm3vl_onnx_test.py` | `visual.onnx` (ORT) vs HF `get_image_features` on a real image | `pooler_output` rel 3.78e-6 |
| `llmexport --test <img>` | HF model vs exported torch graph, image prompt, greedy | coherent, matches HF-style answer |
| board/UM890 image smoke | MNN runtime end-to-end generation on an image | coherent, on par with HF reference description |

These cover the port-specific code (the custom conv/attention + vision merge path) on the
torch→ONNX side with near-exact agreement. They do NOT cover int4 quantization or the
ONNX→MNN graph conversion — that is what L1/L2 measure.

### L1 — quantization-only divergence: `llm_logits_diff` (tool now built + validated)

`MNN/transformers/llm/engine/demo/llm_logits_diff.cpp` — teacher-forced per-step logits
comparison between two MNN exports: backend A drives the trajectory (its argmax token is fed
to both), so every step compares identical KV/history states. Reports argmax agreement,
maxAbsDiff, mean KL(A‖B), teacher-forced NLL.

Board measurements (2026-11-08, `results/fidelity_LFM3_logitsdiff_board.txt`):

| Run | Result | Meaning |
|---|---|---|
| VL int4 vs **itself** (sanity) | 0 disagreements, KL 0.000000 | tool + harness exact |
| LFM3-1.7B int4 vs int8 (family reference, 56 steps) | 9/56 argmax disagreements, mean KL **0.206**, NLL 0.475 vs 0.503 | what *healthy quantization noise* looks like for this architecture |

Pending: the same test on the VL model itself — `scripts/lfm3vl_um890.sh export8 logitsdiff`
(int4 vs int8, same machine, no push needed). **Read:** VL int4-vs-int8 KL in the same
~0.1–0.3 band → quantization on the VL text tower behaves like the text model's; an order of
magnitude higher → the VL export amplifies quantization, investigate.

### L2 — decisive checkpoint-vs-export test: HF perplexity on the same probe

`scripts/lfm3vl_hf_ppl.py` (UM890: `scripts/lfm3vl_um890.sh hfppl`) is a line-for-line twin
of MNN's `ppl_eval` (same raw-text tokenization, same stride-512/context-768 sliding window,
same per-chunk-mean averaging, same gatsby.txt) run on the ORIGINAL HF checkpoint (torch CPU,
bf16). MNN's 151.77 becomes interpretable for the first time:

| HF gatsby ppl | Verdict |
|---|---|
| ≈ 140–155 | the **checkpoint** is weak on 1925 literary prose — MNN conversion is faithful, ppl row stands as a model property |
| ≈ 40–60 | the **export/quantization** lost ~110 ppl — real bug, reopen the port (L1 KL will localize it) |
| between | partial artifact — cross-check L1 and an int8-MNN ppl rerun (`q_lfm3vl_int8_r2` exists) |

Context from the same ladder: the text-only 1.7B is already the table's ppl laggard (39.7 vs
Qwen-2B 21.3), and the truncated-ppl smoke probe moved the same direction (251.9 vs 142.5),
which is consistent with a checkpoint property — but only the HF-side number settles it.

### L3 — behavioral evidence (already in, §4.1 of BENCHMARKS)

R2 ladder 10/10: gsm8k 72, humaneval 40, arc 71, hellaswag 53, winogrande 68, obqa 76,
csqa 61, mmlu 55, niah 5/5. A conversion that silently destroyed the text tower could not
score 62.0 mean with 5/5 needle-in-haystack; and vision is pinned separately by L0's ORT gate.

## Running the review (UM890, ~2 h total)

```bash
git pull board master
scripts/lfm3vl_um890.sh export8      # int8 reference export (~5 min)
scripts/lfm3vl_um890.sh logitsdiff   # int4 vs int8 teacher-forced diff (~2 min)
scripts/lfm3vl_um890.sh hfppl        # HF gatsby ppl, CPU torch (~1 h)
```

Paste the three summaries back and the verdict writes itself from the matrix above.

## Honest gaps

* `llm_logits_diff` compares two MNN exports — it cannot see ONNX→MNN conversion errors
  common to both variants; L0's gates + L2's absolute HF number are what bound that.
* HF twin tokenizes with the HF tokenizer (MNN uses tokenizer.mtok): ±BOS token at most,
  negligible over ~50k scored tokens.
* HF runs bf16, MNN runs fp16-accum int4 — a few ppl points of dtype noise are normal; the
  decision margin in L2 is ~100 ppl, so it cannot flip the verdict.
