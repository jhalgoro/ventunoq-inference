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
| F3 🔒 | 32-bit CDSP VA window ⇒ ~1.75 GiB NPU-side budget | L5 sweeps map the boundary; raising it is R4/vendor work |
| F4 📝 | Partial offload ping-pong: K2-4B `-ngl 18` = 2.6 t/s vs CPU 7.9 | L5 re-tests the rule's edges (quant dependence, per-layer split) |
| F5 🛠️ | Runner `llamacpp/20261020` (ggml 0.27.0) predates the `lfm3` arch | **L1 is exactly this fix** — rebuild via Arduino's GitHub workflow |
| F6 📝 | Env gotchas (unversioned `libcdsprpc.so` symlink, HF stalls, …) | `scripts/env.sh` encodes the fixes; HF pull recipe = `curl -L -C - --retry 3 --speed-limit 200000 --speed-time 30` |
| F7 📝 | Harness/data artifacts (reduced-N cells, prompt-echo rambles) | inherited CSVs in BENCHMARKS §1–2 carry the same caveats |
| F8 📝 | QCS8300 lacks FMLAL/FMLSL **vector** despite HWCAP `asimdhp` | build any llama.cpp variant with default `-march` only; never assume `asimdhp` ⇒ FMLAL |

No F20+ entries yet — that means the depth program hasn't been allowed to fail yet.
Each new failure gets: symptom verbatim, minimal repro, root cause as far as evidence
goes, and workaround or 🔒.
