# Failures — llama.cpp CPU+NPU depth

Numbering continues the family: [`../ventunoq-llamacpp/FAILURES.md`](../ventunoq-llamacpp/FAILURES.md)
owns **F1–F8**, [`../ventunoq-mnn/FAILURES.md`](../ventunoq-mnn/FAILURES.md) owns F9–F19;
this repo starts at **F20**. Legend (parent convention): 🔒 blocked outside our control ·
🛠️ actionable by us · 📝 gotcha (solved, don't re-learn it the hard way).

## Inherited failures — all still binding here

| # | One-liner | Effect on this repo |
|---|---|---|
| F1 🔒 | QNN/QAIRT needs the `smmu_proxy` kernel service; `6.8.0-1084-qcom` has none | NPU work stays exclusively on the llama.cpp Hexagon runner (FastRPC/FBCL) |
| F2 🔒🛠️ | Turnip Vulkan lacks `StorageBuffer8BitAccess`; quantized/tiled matmuls fault | GPU out of scope here (it's neither CPU nor NPU) |
| F3 🔒 | 32-bit CDSP VA window ⇒ per-buffer mapping ceiling | **Re-derived 2026-10-30 (L5)**: fingerprint `HTP0 buffer mapping failed : domain_id 3 size <N> fd <n> error 0x00000001` (core dump). NOT a flat ~1.75 GiB weight budget: Q8_0-LFM3 fits 1.94 GB; the failing single buffer ranges 134 MB (K2-4B ngl30) → 898 MB (K2-7B ngl24); non-monotonic in `-ngl` (4B-Q4_0 fails at 24, works at 30); build-dependent (20261020's known-good `-ngl 24` crashes on b11996). Practical rule: ≥6B has no usable `-ngl` (BENCHMARKS §4). Raising it is R4/vendor work. |
| F4 📝 | Partial offload ping-pong: decode collapses below CPU in dead zones | **Confirmed live 2026-10-30 (L5)**: 4B-Q4_0 `-ngl 18` = **2.6** t/s vs CPU 7.7; K2-4B `-ngl 18` = 3.5 vs 7.9. The usable window is exactly ONE `-ngl` depth per model (4B→30, K2-4B→24); quant shifts the depth, not the collapse (BENCHMARKS §4). |
| F5 ✅ | Runner `llamacpp/20261020` (ggml 0.27.0) predates the `lfm3` arch | **Resolved 2026-10-30**: the public `llamacpp/20261029` asset (ggml 0.28.0) runs lfm3 on HTP0 — LFM3-1.7B pp512 **1033 t/s** (`results/bench_LFM3-17B-Q4KM_npu_ngl100.csv`) |
| F6 📝 | Env gotchas (unversioned `libcdsprpc.so` symlink, HF stalls, …) | `scripts/env.sh` encodes the fixes; HF pull recipe = `curl -L -C - --retry 3 --speed-limit 200000 --speed-time 30` |
| F7 📝 | Harness/data artifacts (reduced-N cells, prompt-echo rambles) | inherited CSVs in BENCHMARKS §1–2 carry the same caveats |
| F8 📝 | QCS8300 lacks FMLAL/FMLSL **vector** despite HWCAP `asimdhp` | build any llama.cpp variant with default `-march` only; never assume `asimdhp` ⇒ FMLAL |

## F20+ — this repo's own

| # | One-liner | Detail / workaround |
|---|---|---|
| F20 📝 | `-ctk q4_0 -ctv q4_0` + `--device HTP0` **crashes** (core dump) | `ggml-backend.cpp:1089: pre-allocated tensor (cache_k_l2 (view)) in a buffer (HTP0) that cannot run the operation (SET_ROWS)` — the KV-write op isn't supported on HTP buffers for q4_0 cache views. q8_0 KV on HTP works. CPU path fine (but pp8192 −38%: 54.2→33.6). Log: `results/bench_LFM3-17B-Q4KM-kvq4_npu_ngl100.err`. Candidate upstream issue. |

No other F20+ entries yet — that means the depth program hasn't been allowed to fail much yet.
Each new failure gets: symptom verbatim, minimal repro, root cause as far as evidence
goes, and workaround or 🔒.
