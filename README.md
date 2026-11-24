# ventunoq-inference

Umbrella repo for the ventunoq on-device inference experiments, migrated from
`arduino@localhost:/home/arduino/Code` with full commit history preserved
(each project merged in via `git subtree`, so original authors, dates and
messages are intact under their subdirectories).

| Directory | What it is |
|---|---|
| `ventunoq-llamacpp/` | llama.cpp line of work on the board |
| `ventunoq-llamacpp-cpu+npu/` | CPU+NPU offload experiments (Gemma, etc.) |
| `ventunoq-mnn/` | MNN line of work (bench harness, conversions, results) |
| `ventunoq-mnn-tandem/` | Tandem design notes (DESIGN.md; upstream MNN clone excluded) |

Model weights and other large artifacts (`*.gguf`, `*.mnn.weight`, exported
model dirs) are transient by policy and were not migrated.
