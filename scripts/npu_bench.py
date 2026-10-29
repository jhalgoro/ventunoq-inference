#!/usr/bin/env python3
"""NPU (Hexagon HTP V75) inference benchmark via QAIRT on-device convert+compile.

Runs the same ONNX test models as the CPU reference (onnxruntime) and reports
inference latency on the NPU. Usage: python3 scripts/npu_bench.py [model.onnx ...]
"""
import importlib.metadata  # noqa: F401  (qairt/__init__.py bug workaround)
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SDK = ROOT / "npu/qairt-sdk/qairt/2.51.0.260929"
os.environ.setdefault("QAIRT_SDK_ROOT", str(SDK))
os.environ["LD_LIBRARY_PATH"] = f"{ROOT}/npu/compat:{SDK}/lib/aarch64-ubuntu-gcc9.4:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["ADSP_LIBRARY_PATH"] = f"{SDK}/lib/hexagon-v75/unsigned;/vendor/dsp/cdsp;/usr/lib/rfsa/adsp"
os.environ["CDSP_LIBRARY_PATH"] = os.environ["ADSP_LIBRARY_PATH"]
os.environ["GDSP0_LIBRARY_PATH"] = os.environ["ADSP_LIBRARY_PATH"]

import numpy as np
import onnx
from qairt.api.converter._convert import convert
from qairt.api.compiler._compile import compile as qcompile

models = sys.argv[1:] or [str(ROOT / "npu/test_convnet.onnx")]
out = {}
for mp in models:
    name = Path(mp).stem
    g = onnx.load(mp).graph
    iname = g.input[0].name
    ishape = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    print(f"== {name}: convert ONNX->DLC (HTP)...", flush=True)
    m = convert(mp, backend="HTP")
    print(f"== {name}: AOT compile for HTP...", flush=True)
    cm = qcompile(m, backend="HTP")
    x = {"input": np.random.randn(*ishape).astype(np.float32)}
    run = lambda x: cm(x, backend="HTP")  # noqa: E731  (ModelBase.__call__)
    print(f"== {name}: warmup...", flush=True)
    for _ in range(3):
        run(x)
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        run(x)
    dt = (time.perf_counter() - t0) / n
    out[name] = {"ms_per_infer": round(dt * 1000, 3), "inf_per_s": round(1 / dt, 2),
                 "backend": "HTP-V75", "iters": n}
    print(f"== {name}: NPU {dt*1000:.1f} ms/infer ({1/dt:.1f} inf/s)", flush=True)

dst = ROOT / "results" / "npu_bench.json"
prev = json.loads(dst.read_text()) if dst.exists() else {}
prev.update(out)
dst.write_text(json.dumps(prev, indent=1))
print("wrote", dst)
