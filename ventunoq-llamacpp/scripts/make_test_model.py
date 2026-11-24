#!/usr/bin/env python3
"""Generate small ONNX test models for NPU (Hexagon HTP) vs CPU throughput comparison.

Models (chosen to load HTP compute, not FastRPC transfer):
  convnet  : 8x [Conv 3x3, 64ch, stride1] + Relu @ 224x224x3  (MAC-heavy CNN)
  matmul   : 4x [MatMul 1024x1024] + Relu @ 1x1024             (dense GEMM)
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

def convnet(path, layers=8, ch=64, size=224):
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, size, size])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, ch, size, size])
    nodes, inits = [], []
    cur = inp.name
    in_ch = 3
    for i in range(layers):
        w = np.random.randn(ch, in_ch, 3, 3).astype(np.float32) * 0.05
        b = np.zeros(ch, dtype=np.float32)
        inits += [numpy_helper.from_array(w, f"w{i}"), numpy_helper.from_array(b, f"b{i}")]
        nxt = f"c{i}"
        nodes.append(helper.make_node("Conv", [cur, f"w{i}", f"b{i}"], [nxt],
                                      kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
        nodes.append(helper.make_node("Relu", [nxt], [nxt + "_r"]))
        cur = nxt + "_r"
        in_ch = ch
    nodes.append(helper.make_node("Identity", [cur], [out.name]))
    g = helper.make_graph(nodes, "convnet", [inp], [out], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10  # onnxruntime 1.23 supports max IR version 11
    onnx.save(onnx.shape_inference.infer_shapes(m), path)

def matmul(path, layers=4, n=1024):
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, n])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, n])
    nodes, inits = [], []
    cur = inp.name
    for i in range(layers):
        w = (np.random.randn(n, n) * 0.02).astype(np.float32)
        inits.append(numpy_helper.from_array(w, f"w{i}"))
        nodes.append(helper.make_node("MatMul", [cur, f"w{i}"], [f"m{i}"]))
        nodes.append(helper.make_node("Relu", [f"m{i}"], [f"m{i}_r"]))
        cur = f"m{i}_r"
    nodes.append(helper.make_node("Identity", [cur], [out.name]))
    g = helper.make_graph(nodes, "matmul", [inp], [out], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10  # onnxruntime 1.23 supports max IR version 11
    onnx.save(onnx.shape_inference.infer_shapes(m), path)

if __name__ == "__main__":
    convnet("npu/test_convnet.onnx")
    matmul("npu/test_matmul.onnx")
    print("written npu/test_convnet.onnx npu/test_matmul.onnx")
