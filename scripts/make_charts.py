#!/usr/bin/env python3
"""Render README charts for ventunoq-llamacpp as PNGs.
Data: BENCHMARKS.md §1 (CPU matrix), §3 (NPU), §4 (quality). All measured on-board."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os

OUT = os.path.expanduser("~/Code/ventunoq-llamacpp/docs/assets/charts")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "x", "grid.alpha": 0.25, "grid.linestyle": ":",
    "figure.facecolor": "white",
})

# ---------------------------------------------------------------- 1. CPU throughput
# model, pp512, pp8192, tg128, company
cpu = [
    ("LFM3-1.7B QAD-Q4_0", 87.4, 62.6, 10.2, "Liquid AI"),
    ("Qwen3.5-2B",         49.1, 41.3,  7.9, "Qwen"),
    ("LFM3-1.7B Q4_K_M",   49.0, 43.3, 11.7, "Liquid AI"),
    ("LFM2.5-2.6B",        32.0, 28.0,  7.3, "Liquid AI"),
    ("Qwen3.5-4B",         19.8, 16.9,  4.2, "Qwen"),
    ("K2-Horizon-4B",      19.8, 11.5,  4.3, "K2 Lab"),
    ("Qwen3.5-9B",         11.6, 10.2,  2.7, "Qwen"),
    ("K2-Horizon-7B",      11.2,  8.3,  3.0, "K2 Lab"),
]
names = [r[0] for r in cpu][::-1]
pp512 = [r[1] for r in cpu][::-1]; pp8192 = [r[2] for r in cpu][::-1]; tg128 = [r[3] for r in cpu][::-1]
y = np.arange(len(names)); h = 0.26
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.barh(y + h, pp512, height=h, color="#2563eb", label="pp512 (prompt eval t/s)")
ax.barh(y,     pp8192, height=h, color="#8fb0f7", label="pp8192 (prompt eval t/s)")
ax.barh(y - h, tg128, height=h, color="#f59e0b", label="tg128 (generation t/s)")
for yy, v in zip(y + h, pp512): ax.text(v + 1, yy, f"{v:.1f}", va="center", fontsize=9)
for yy, v in zip(y, pp8192):    ax.text(v + 1, yy, f"{v:.1f}", va="center", fontsize=9)
for yy, v in zip(y - h, tg128): ax.text(v + 1, yy, f"{v:.1f}", va="center", fontsize=9)
ax.set_yticks(y, names)
ax.set_xlim(0, 100)
ax.set_xlabel("tokens / second  (llama-bench, -ngl 0, Q4, llama.cpp v0.8.0)")
ax.set_title("CPU throughput — Arduino Ventuno Q (QCS8300), 4×A78 pinned where relevant", pad=12)
ax.legend(loc="lower right", frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/cpu-throughput.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- 2. quality — one chart per test
q = [  # model, gsm8k, humaneval, reduced?, company
    ("Qwen3.5-9B",    85.0,  80.0, True,  "Qwen"),
    ("K2-Horizon-7B", 75.0, 100.0, True,  "K2 Lab"),
    ("Qwen3.5-4B",    72.0,  85.0, False, "Qwen"),
    ("LFM3-1.7B",     60.0,  65.0, False, "Liquid AI"),
    ("K2-Horizon-4B", 60.0,  75.0, False, "K2 Lab"),
    ("Qwen3.5-2B",    22.0,  40.0, False, "Qwen"),
]
for idx, title, col in [(1, "GSM8K — math word problems (strict last-number match)", "#0f766e"),
                        (2, "HumanEval — code generation (sandboxed execution)", "#b91c1c")]:
    rows = sorted(q, key=lambda r: r[idx])          # ascending → best on top of barh
    names = [r[0] for r in rows]; vals = [r[idx] for r in rows]; red = [r[3] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.0))
    y = np.arange(len(names))
    ax.barh(y, vals, height=0.6, color=col)
    for yy, v, s in zip(y, vals, red):
        ax.text(v + 1.5, yy, f"{v:.0f}%" + ("*" if s else ""), va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(y, names)
    ax.set_xlim(0, 112)
    ax.set_xlabel("% correct   (* = reduced sample: N=20 GSM8K / 10 tasks HumanEval; others N=50 / 20)")
    ax.set_title(title + " — Ventuno Q, CPU, Q4_K_M", pad=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/quality-{'gsm8k' if idx == 1 else 'humaneval'}.png", dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------- 3. NPU
# model, npu_pp512, cpu_pp512(pinned same binary), pp8192 speedup, tg128 speedup
npu = [
    ("LFM2.5-2.6B Q4_K_M",       679,  40.1, 21.1, 1.35),
    ("Qwen3.5-2B Q4_0",          863,  69.8, 14.9, 1.51),
    ("Qwen3.5-0.8B Q4_0",       1366, 148.7, 13.6, 1.35),
    ("Qwen3.5-4B Q4_0 (-ngl 24)", 70.8, None, 3.0, 2.4),
    ("K2-Horizon-4B (-ngl 18)",   None, None, 1.9, 0.33),
]
names = [r[0] for r in npu][::-1]
pp_sp = [r[3] for r in npu][::-1]; tg_sp = [r[4] for r in npu][::-1]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
y = np.arange(len(names)); h = 0.34
a1.barh(y + h/2, pp_sp, height=h, color="#2563eb")
a1.barh(y - h/2, tg_sp, height=h, color=["#dc2626" if v < 1 else "#f59e0b" for v in tg_sp])
for yy, v in zip(y + h/2, pp_sp): a1.text(v + 0.3, yy, f"{v:g}×", va="center", fontsize=9.5, fontweight="bold")
for yy, v in zip(y - h/2, tg_sp):
    a1.text(v + 0.3, yy, f"{v:g}×" + ("  ← regression" if v < 1 else ""), va="center", fontsize=9.5,
            fontweight="bold", color="#dc2626" if v < 1 else "black")
a1.axvline(1, color="k", lw=1, ls="--", alpha=0.6)
a1.set_yticks(y, names)
a1.set_xlim(0, 24); a1.set_xlabel("speed-up vs same binary, CPU pinned to big cores (1× line = parity)")
a1.set_title("NPU (Hexagon V75) speed-up", pad=10)
a1.legend([Patch(color="#2563eb"), Patch(color="#f59e0b"), Patch(color="#dc2626")],
          ["prefill pp8192", "decode tg128", "decode < 1× (regression)"], loc="lower right", frameon=False, fontsize=9)

n3 = [r[0] for r in npu[:3]]  # full-offload trio, bottom→top
npu_pp = [679, 863, 1366]; cpu_pp = [40.1, 69.8, 148.7]
y2 = np.arange(3)
a2.barh(y2 + h/2, npu_pp, height=h, color="#7c3aed", label="NPU pp512")
a2.barh(y2 - h/2, cpu_pp, height=h, color="#9ca3af", label="CPU pp512 (pinned)")
for yy, v in zip(y2 + h/2, npu_pp): a2.text(v + 20, yy, f"{v:g}", va="center", fontsize=9.5, fontweight="bold")
for yy, v in zip(y2 - h/2, cpu_pp): a2.text(v + 20, yy, f"{v:g}", va="center", fontsize=9)
a2.set_yticks(y2, n3)
a2.set_xlim(0, 1600); a2.set_xlabel("prompt-processing tokens / second")
a2.set_title("Prefill, absolute — `-ngl 100`", pad=10)
a2.legend(loc="lower right", frameon=False, fontsize=9)
fig.suptitle("Hexagon NPU vs CPU — Arduino Ventuno Q  (~1.75 GiB NPU-side budget; partial offload marked in labels)",
             y=1.04, fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(f"{OUT}/npu-speedup.png", dpi=150); plt.close(fig)

print("wrote:", *sorted(os.listdir(OUT)), sep="\n  ")
