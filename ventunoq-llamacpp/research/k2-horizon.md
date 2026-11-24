# K2-Horizon (IFM / K2 Lab) — scoping notes (2026-10-28)

Requested as an addition to the benchmark matrix.

## Family (HF org `IFM`, Apache-2.0, arch `k2_horizon` / `K2HorizonForCausalLM`)
| Model | Notes | On this board |
|---|---|---|
| K2-Horizon-0.9B (+GGUF) | below our lineup range | skipped |
| K2-Horizon-3.7B-GGUF (files named "4B") | Q4_K_M **3.16 GB** | **tested — full cycle** |
| K2-Horizon-7B-GGUF | Q4_K_M **5.59 GB** | **tested — reduced scope** (swap-dominated like Qwen 9B) |
| K2-Horizon-32B / -32B-FP8 | ≥18 GB at Q4 | doesn't fit disk policy |
| K2-Horizon-MoVA-36B-A4B-GGUF | MoE, Q4_K_M 22.4 GB | breaks 50%-disk policy; 7.6 GB RAM anyway |
| K2-Horizon-375B-A23B | server-class | no |
| K2-Horizon-{7B,0.9B}-Uno | **not Arduino-related**: LoRA adapters for a *diffusion* pathway (paper 2609.04010, "Uno" diffusion-LLM project, adapter-only repo). Requires custom diffusion code; not AR-runnable in llama.cpp | skipped |

## Compatibility
- Prebuilt llama.cpp v0.8.0 (b11972) **already ships the arch**: `llama_model_k2_horizon`
  symbols (incl. MoE `build_routed_value`) present in `libllama.so` — no rebuild needed.
- Official GGUFs are public (no auth): `IFM/K2-Horizon-3.7B-GGUF/resolve/main/K2-Horizon-4B-Q4_K_M.gguf`,
  `IFM/K2-Horizon-7B-GGUF/resolve/main/K2-Horizon-7B-Q4_K_M.gguf`.
- Family released 2026-09-01, updated 2026-10-14 — recent enough that only newer
  llama.cpp builds support it; our Feb-2026-era prebuilt already does.

## Harness
`scripts/eval_k2.sh` — waits for `results/eval_done`, then per model:
llama-bench + GSM8K (N=50 full / 20 reduced) + HumanEval (N=20 / 10), deletes the
GGUF afterwards. Outputs: `bench_k2h_{4B,7B}_main.csv`, `gsm8k_k2h_*.log`,
`humaneval_k2h_*.log`, `k2h{4B,7B}_status`, final marker `results/k2_done`.
