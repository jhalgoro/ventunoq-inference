# Benchmarks — MNN on QCS8300 (measured on this board)

Nothing here is quoted from upstream tables; every number has a raw log in `results/` or a
`scripts/` repro. Provenance from the Round-2 survey runs (pre-repo, logs currently under
`/tmp/opencode/mnn_bench_*.log` — regenerate with `scripts/bench_mnn.sh`, which writes into
`results/` from now on).

## Method — how MNN knobs map onto llama.cpp's pinned regime

| MNN | llama.cpp v0.8.0 equivalent | notes |
|---|---|---|
| `llm_bench -pg 512,128` | `llama-bench -p 512 -n 128` | identical semantics: prefill 512, then generate 128 reusing the KV cache |
| `taskset -c 0-3` | `--cpu-mask 0xf --cpu-strict 1` | big cores only (A78C pair @2112/2361.6 MHz) |
| `-t 4` | `-t 4` | |
| `-c 2` | — | fp16 compute, printed as precision **Low**; `-c 1` = fp32 (**High**) — see F10 |
| `-rep 3` | `-r 3` | |
| model dir `config.json` | `-m file.gguf` | MNN model = dir (`llm.mnn` + `llm.mnn.weight` + tokenizer) |

llama.cpp baselines are the parent repo's: **v0.8.0 b11972** (official Ubuntu-arm64 prebuilt,
`armv8.2_2` dispatch), raw CSVs in
[`../ventunoq-llamacpp/results/`](../ventunoq-llamacpp/results/).

## 1. Headline — MNN int4 (fp16) vs llama.cpp, big-core pinned, `-pg 512,128`

| Model | llama.cpp Q4_0 (pp/tg) | llama.cpp Q4_K_M (pp/tg) | **MNN int4 fp16 (pp/tg)** | MNN vs Q4_0 |
|---|---|---|---|---|
| Qwen3.5-0.8B | 148.7 / 31.9 | 106.2 / 27.0 | **162.7 ± 0.1 / 41.5 ± 0.1** | +9.4% / **+30%** |
| Qwen3.5-2B | 69.8 / 16.1 | — | **75.7 ± 0.1 / 19.2 ± 0.0** | +8.4% / **+19%** |
| LFM3-1.7B | — | 49.0 / 11.7 | **75.4 ± 0.2 / 21.4 ± 0.0** | **+54% / +83% vs Q4_K_M** ¹ |
| LFM3-VL-2B | — | — | **75.3 ± 0.2 / 21.4 ± 0.0** | text parity with the 1.7B row ² |
| MiniCPM5-2B | — | pending ³ | **57.3 ± 0.3 / 18.4 ± 0.1** | — ³ |

¹ LFM3 has no llama.cpp Q4_0 baseline (§3), so the last cell compares against the like-model
Q4_K_M. Largest cross-engine gap in the table — plausible cause: llama.cpp's LFM3 support is
weeks old while MNN's conv/linear-attention ops are mature; int8 MNN trades tg for quality (§4.1).
² LFM3-VL-2B (2026-11-07, M7 VL): no llama.cpp baseline exists (no LFM3-VL support upstream).
Text pp/tg is at parity with the text-only 1.7B int4 row despite the bigger text tower and the
loaded vision module (953 MiB runtime) — text tokens never touch the vision path. Vision tower
(int8 visual, 0.21 MP image): 1.58 s on the board (0.132 MP/s), 0.84 s on UM890 x86 (0.248 MP/s).
³ MiniCPM5-2B (2026-11-08, M10): stock llama arch, zero export-side changes (UM890 export
`scripts/minicpm5_um890.sh`, int4 b64, 1.18 GiB runtime). pp/tg below Qwen3.5-2B (57/18 vs
76/19) — 2.49B params in 42 layers vs 2B, plus a 130K-vocab untied head. llama.cpp Q4_K_M
baseline pending (official GGUFs exist; parent-repo run). Thinking-style: decode budgets in
§4.1 must cover ~400-token reasoning preambles (R2 protocol).

- MNN's int4 weights carry fp16 per-block scales (≈Q4_K-class quality) and compute in ARM82
  fp16 — the win is kernel + quant format together.
- Quant-matched comparison matters: llama.cpp's Q4_K_M is *slower* than its own Q4_0 at 0.8B
  (106 vs 149 pp), so vs the like-for-like quant MNN is ~1.5× at 0.8B.
- Decode gain shrinks 30%→19% from 0.8B→2B — the bandwidth-bound crossover
  ([parent research/memory-map.md](../ventunoq-llamacpp/research/memory-map.md)).

## 2. Config sweeps (0.8B unless noted)

| Config | pp512 | tg128 | reading |
|---|---|---|---|
| t4 fp16 (`-c 2`) | 162.7 | **41.5** | board default for MNN |
| t4 fp32 (`-c 1`) | ~154.8 | 25.1 | fp16 is worth **+65% decode** (F10) |
| t8 fp16 | 170.7 | 31.5 | A55s raise pp but drag tg below t4 |
| t8 fp32 | 159.2 | 29.1 | |
| decode wall speed (incl. sampling) | — | 40.9 | `llm_bench` prints it separately |

Memory: peak RSS **930 MB** for the 449 MB model (2.1× model size; vs mistral.rs 4×,
llama.cpp mmap-based). Measured with `/usr/bin/time -v` on `llm_bench` (text-only model).

## 3. llama.cpp baselines (inherited, for reference)

From [`../ventunoq-llamacpp/BENCHMARKS.md`](../ventunoq-llamacpp/BENCHMARKS.md) §1/§3:

| Model | pinned-CPU pp512 / tg128 | NPU (Hexagon runner) pp512 / tg128 |
|---|---|---|
| Qwen3.5-0.8B-Q4_0 | 148.7 / 31.9 | 1366 / 43.0 |
| Qwen3.5-2B-Q4_0 | 69.8 / 16.1 | 863 / 24.3 |
| LFM2.5-2.6B-Q4_K_M | 40.1 / 13.1 | 679 / 17.7 |
| LFM3-1.7B-Q4_K_M | 49.0 / 11.7 | ✗ runner predates lfm3 (parent R1) |

**MNN does not change the NPU picture**: its Hexagon backend is Android+QNN (F13). The
interesting cross-engine question left is quality — M1 below.

## 4. Planned (fill from ROADMAP items; keep this file numbers-only)

### 4.1 Quality parity (M1) — MNN int4 vs llama.cpp Q4_0 / Q4_K_M

Method: `scripts/quality_matrix_mnn.sh` → `scripts/quality_eval_mnn.py` → `llm_eval` driver
(greedy sampler, fresh KV per item, `taskset -c 0-3`, engine defaults t4/fp16). Same
`suites/n/prompts/stride()` as the sibling `quality_matrix.sh` on identical `bench/` items,
so `scripts/quality_parity.py` joins MNN vs llama.cpp item-by-item. Deviations from the
sibling harness: all six MC suites score gen-style (MNN exposes no logprobs → walk/raw-walk
impossible); stop strings emulated by truncation; ppl via MNN's `ppl_eval` (its JSON
`score` is the PPL — keep ppl out of parity globs).

| Model | MNN int4 GSM8K | llama.cpp Q4_0 | llama.cpp Q4_K_M | ppl (gatsby) MNN vs Q4_0 |
|---|---|---|---|---|
| Qwen3.5-0.8B | **34.0%** (n=100) | ⏳ (parent harness) | ⏳ | **31.08** vs ⏳ |
| Qwen3.5-2B | **60.0%** (n=100) | 22.0% (N=50, parent) | ⏳ | **21.25** vs ⏳ |
| Qwen3.5-4B | **91.0%** (n=100) | ⏳ | ⏳ | **16.63** vs ⏳ |

Full MNN row, Qwen3.5-0.8B int4 (`QWEN35-08B-INT4`, greedy no-think, 69 min for 10 suites):
gsm8k 34.0 · humaneval 16.0 (n=50) · arc 33.0 · hellaswag 24.0 · winogrande 48.0 ·
openbookqa 32.0 · commonsenseqa 26.0 · mmlu 20.2 (n=198) · **niah 5/5** (depths 0.1–0.9,
~11k ctx) · ppl 31.08 (full gatsby). Generative suites are healthy; the MC block sits at
or below chance (hellaswag 24% < 25%, mmlu 20.2% < 25%) — the known small-model
gen-letter bias, not obviously an MNN effect; the llama.cpp rows on the same items decide.

#### q_all ladder (2026-11-01 → 11-03), % correct, greedy, same items/stride as above

| Row | GSM8K | HumanEval | ARC | HellaSwag | WinoGrande | OBQA | CSQA | MMLU | NIAH | ppl ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.8B int4 | 34.0 | 16.0 | 33.0 | 24.0 | 48.0 | 32.0 | 26.0 | 20.2 | 5/5 | 31.08 |
| 2B int4 | 60.0 | 34.0 | 69.0 | 57.0 | 59.0 | 68.0 | 51.0 | 40.4 | 5/5 | 21.25 |
| **4B int4** | **91.0** | **58.0** | **87.0** | **80.0** | 61.0 | **88.0** | **80.0** | **66.2** | 5/5 | **16.63** |
| 0.8B base | 1.0 | 0.0 | 63.0 | 32.0 | 53.0 | 66.0 | 53.0 | 51.0 | 2/5 | 25.88 |
| 2B base | 41.0 | 2.0 | 79.0 | 47.0 | 57.0 | 76.0 | 63.0 | 55.0 | 0/5 | 19.19 |
| 0.8B DIST (R1) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 47.80 |
| 2B DIST (R1) | 55.0 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 33.71 |
| 4B DIST (R1) | 65.0 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 24.10 |
| 2B DIST **R2** | 60.0 | 38.0 | 71.0 | 52.0 | 57.0 | 68.0 | 57.0 | 51.0 | **5/5** | 33.71 |
| 4B DIST **R2** | 72.0 | **74.0** | **92.0** | **68.0** | **68.0** | **93.0** | 77.0 | **79.3** | 5/5 | 24.10 |
| LFM3 int4 **R2** | 86.0 | 38.0 | 80.0 | 41.0 | 66.0 | 72.0 | 53.0 | 69.2 | 5/5 | 39.65 |
| LFM3 int4+HQQ **R2** | 78.0 | 56.0 | 79.0 | 40.0 | 63.0 | 69.0 | 48.0 | 67.2 | 5/5 | 41.08 |
| LFM3 int8 **R2** | 86.0 | 72.0 | 81.0 | 46.0 | 68.0 | 70.0 | 55.0 | 77.3 | 5/5 | 33.03 |
| LFM3-VL-2B int4 **R2** | 72.0 | 40.0 | 71.0 | 53.0 | 68.0 | 76.0 | 61.0 | 55.0 | **5/5** | 151.77 |
| MiniCPM5-2B int4 **R2** | 76.0 | 52.0 | 75.0 | 50.0 | 51.0 | 50.0 | 47.0 | 58.1 | 5/5 | 31.93 |

- LFM3-1.7B rows (2026-11-06/07, M7): exported on the UM890 (`scripts/lfm3_um890.sh`), scored
  via `q_lfm3_*_r2` — same R2 protocol/budgets as the DIST2 rows. Read: at 1.7B it beats 2B
  DIST2 on 7/10 tasks (gsm8k +26, MMLU +18, ARC +9) and loses HellaSwag −11 and ppl — reasoning-
  shaped strength, commonsense/literary weakness. HQQ is **net negative** here (worse 7/10; its
  HumanEval +18 doesn't redeem gsm8k −8) — plain int4 stays the recipe. int8 is the quality tier
  (HumanEval 0.38→0.72, MMLU +8, ppl at 2B parity) for 2× size and −42% decode (12.4 vs 21.4 tg).

- LFM3-VL-2B row (2026-11-08, M7 VL): exported on the UM890 (`scripts/lfm3vl_um890.sh`, int4 b64 /
  visual int8), scored via `q_lfm3vl_r2` — same R2 protocol. 8-task mean 62.0 vs 63.2 for the
  text-only 1.7B int4: parity overall, different shape — commonsense up (HellaSwag +12, CSQA +8,
  OBQA +4) vs reasoning/knowledge down (gsm8k −14, ARC −9, MMLU −14.2). ppl 151.8 is a large
  literary-domain regression, but an Austen continuation probe is fluent and on-style, and task
  scores are healthy — read as a vision-mix checkpoint property on 1925 prose, not an export
  artifact (the LFM3 family was already the ppl laggard: 39.7 vs Qwen2B 21.3).

- MiniCPM5-2B row (2026-11-08, M10): stock llama arch, zero export-side changes
  (`scripts/minicpm5_um890.sh`, int4 b64; the real HF chat_template.jinja passes minja — no
  override needed, unlike lfm3). Scored via `q_minicpm5_r2` (thinking-style: plain-protocol
  smoke scored MC 0.0 on 4/6 suites via 16-tok gen-miss, R2 fixes extraction — audit: 0–1%
  gen-miss left). 8-task mean 57.4 ≈ 2B DIST2's 56.8; reasoning-shaped: best int4 HumanEval
  (52), gsm8k 76, MMLU 58.1, healthy ppl 31.9 — but the commonsense block is genuinely weak
  (WinoGrande 51 ≈ coin flip, OBQA 50, CSQA 47; LFM3-1.7B +15/+22/+6). Speed is the other
  weakness: 57/18 pp/tg vs Qwen3.5-2B's 76/19 — 42 layers, 130K untied vocab at 2.49B params.

- ✗ = harness artifact, not a model score: R1 distilled rows died on the reasoning-style
  output (F17 — 16-tok MC budget, shot-prefix replay, prose-as-code, 24-tok niah). The R2
  protocol (`q_dist2b2`/`q_dist4b2`, zero-shot MC + brief-answer instruction, budgets
  512/1536/1024/512) reruns those rows; R1 JSONs stay for provenance.
- 9B int4 excluded from the table: export unusable as fetched (F15 — missing
  `embeddings_bf16.bin` → `!!!` garbage, ppl 248,300); 9B DIST repo is an empty placeholder
  (F16). Both parked (also the two slowest rows).
- ppl is perplexity (not accuracy) — full gatsby, comparable within the column only.
- † hellaswag needed its own budget: at 512 tok the reasoners ramble past the cap
  without ever writing "Answer: X" (2B: 41/100 unparsed). Re-scored at 1024 tok
  (`q_dist_fix`, `--hellaswag-tokens`): 2B 36.0→52.0 (14 unparsed), 4B 57.0→68.0 (1).
- humaneval DIST rows re-scored after F19b (prompt+completion gluing produced bogus
  IndentationError on restated functions): 2B 10.0→38.0 (failures now mostly real
  AssertionErrors), 4B 78.0→74.0 (standalone execution exposes 3 NameErrors that the
  prompt gluing had papered over).
- Read (final): the distillation pays off only at 4B — 4B DIST beats 4B int4 on
  humaneval +16, MMLU +13.1, WinoGrande +7, ARC/OBQA +5, but loses GSM8K −19,
  HellaSwag −12, CSQA −3; 2B DIST ≈ 2B int4 (best: MMLU +10.6, humaneval +4; worst:
  HellaSwag −5) and never approaches 4B int4. ppl (24–34 vs 17–21 same size) reflects
  the reasoning-style vocabulary shift, not knowledge loss.

### 4.2 Long context (M5) — pp curve + RSS growth per KV token vs llama.cpp's ~8.9 KB/tok

| Model | pp2k | pp8k | pp16k | pp32k | KV RSS growth |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

### 4.3 Sustained + energy (M6) — 10-min loop + INA232 (hwmon2), J/token = W ÷ t/s

| Engine/config | avg W (pp) | avg W (tg) | J/tok pp | J/tok tg | throttle Δ over 10 min |
|---|---|---|---|---|---|
| MNN fp16 t4 pinned | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| llama.cpp Q4_0 t4 pinned | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| llama.cpp NPU (parent repo runs it) | — | — | — | — | — |
