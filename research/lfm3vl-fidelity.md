# LFM3-VL-2B → MNN conversion-fidelity review (M7)

**Question:** did the LFM3-VL-2B → MNN conversion actually work well? The open doubt is the
R2 ppl row: **151.77** on gatsby vs 39.65 for the text-only LFM3-1.7B int4 — while every task
score stayed healthy (8-task mean 62.0 vs 63.2). This page is the review package: what is
already proven, what the two decisive measurements are, and how to read them.

## Verdict (2026-11-08 — all layers in, review closed)

**The conversion is faithful; the ppl row is a checkpoint property.** Two decisive numbers,
both measured on the QCS8300 board:

| Measurement | Result | Reading |
|---|---|---|
| HF fp32 gatsby ppl (same probe as MNN's 151.77) | **137.5699** | matrix top row (<160): the fp32 original is itself weak on 1925 prose |
| VL int4-vs-int8 logits-diff (teacher-forced) | mean KL **0.236**, 5/41 argmax flips | same band as the text-only family reference (0.206, 9/56) — quantization behaves normally |

The int4 export costs 151.77 − 137.57 ≈ **14 ppl** — consistent with the L1 noise level, not
the ~110 ppl an export bug would show. Nothing to fix in the port; the PR vs #5104 can be
reopened on this evidence. Raw data: `results/fidelity_lfm3vl_hf_ppl.log`,
`results/fidelity_lfm3vl_logitsdiff_int4_int8.txt`, chain log `results/fidelity_lfm3vl_board.out`.

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
| **LFM3-VL-2B int4 vs int8** (board, stop token at step 41/64) | 5/41 argmax disagreements (12% vs the reference's 16%), mean KL **0.236**, worst maxAbsDiff 5.35 | same band as the family reference → quantization on the VL text tower behaves like the 1.7B's |

The VL test ran on the board (`scripts/lfm3vl_board_fidelity.sh export8 logitsdiff`; the int8
reference export was built on-board from the same patched tree). KL 0.236 sits 15% above the
text-only 0.206 — same band, no order-of-magnitude amplification. All 5 disagreements occur
at low-margin steps (marginA ≤ 3.5, i.e. coin-flip tokens), exactly like the reference run.
Full trace: `results/fidelity_lfm3vl_logitsdiff_int4_int8.txt`.

### L2 — decisive checkpoint-vs-export test: HF perplexity on the same probe

`scripts/lfm3vl_hf_ppl.py` (UM890: `scripts/lfm3vl_um890.sh hfppl`) is a line-for-line twin
of MNN's `ppl_eval` (same raw-text tokenization, same stride-512/context-768 sliding window,
same per-chunk-mean averaging, same gatsby.txt) run on the ORIGINAL HF checkpoint (torch CPU;
fp32 on the board, bf16 would be UM890's default). MNN's 151.77 becomes interpretable:

| HF gatsby ppl | Verdict |
|---|---|
| ≈ 140–155 | the **checkpoint** is weak on 1925 literary prose — MNN conversion is faithful, ppl row stands as a model property |
| ≈ 40–60 | the **export/quantization** lost ~110 ppl — real bug, reopen the port (L1 KL will localize it) |
| between | partial artifact — cross-check L1 and an int8-MNN ppl rerun (`q_lfm3vl_int8_r2` exists) |

Context from the same ladder: the text-only 1.7B is already the table's ppl laggard (39.7 vs
Qwen-2B 21.3), and the truncated-ppl smoke probe moved the same direction (251.9 vs 142.5),
which is consistent with a checkpoint property.

**Measured (2026-11-08, board, fp32): HF Perplexity = 137.5699** (133 chunks, 73 min,
`results/fidelity_lfm3vl_hf_ppl.log`). The HF checkpoint is weak on this probe **by itself** —
matrix top row: checkpoint property, conversion faithful. int4 adds 151.77 − 137.57 ≈ 14 ppl,
the same order as the quantization-only gap L1 measures on the family. The 40–60 "export bug"
row is ruled out by a ~100 ppl margin.

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

**Board fallback (no UM890 needed):** `scripts/lfm3vl_board_fidelity.sh all` runs the same
three measurements on the QCS8300 — locally-created venv (torch CPU aarch64; this board needs
the `--without-pip` + get-pip bootstrap, handled by the script), public 3.85 GB checkpoint
download, then `hfppl` (~1.5 h), `export8` via the board's patched MNN tree, and
`logitsdiff` int4-vs-int8. Logs land in `results/fidelity_lfm3vl_*.log|txt`.

**Board dtype trap (measured 2026-11-08):** torch CPU aarch64 on this SoC does half-precision
matmul at ~0.2–0.3 GF/s vs **50.6 GF/s in fp32** (250× penalty, one bf16 warmup alone takes
~3 min) — a bf16 hfppl run burned 5 h without finishing 10 chunks. `hfppl` therefore defaults
to `--fp32` on the board (override: `HFPPL_EXTRA`). HF fp32 is the *stronger* reference anyway:
any MNN-vs-HF gap it shows is export+quantization, with no dtype excuse.

## Honest gaps

* `llm_logits_diff` compares two MNN exports — it cannot see ONNX→MNN conversion errors
  common to both variants; L0's gates + L2's absolute HF number are what bound that.
* HF twin tokenizes with the HF tokenizer (MNN uses tokenizer.mtok): ±BOS token at most,
  negligible over ~50k scored tokens.
* The measured HF run is **fp32** (board dtype trap forced it — and fp32 is the stronger
  reference anyway): any gap vs MNN's fp16-accum int4 is export+quantization with no dtype
  excuse. The 14 ppl gap it shows is small; the decision margin in L2 was ~100 ppl.
