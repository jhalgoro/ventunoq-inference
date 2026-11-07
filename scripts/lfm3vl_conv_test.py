#!/usr/bin/env python3
"""lfm3vl_conv_test.py — numerical gate for the grid-free gather rewrite of
Lfm3VlVisionConv2d (MNN utils/vision.py Lfm3VlVisionConv), run on the UM890.

Checks the exported torch module against an inline HF-equivalent reference
(merge-block -> raster, depthwise 3x3 zero-pad conv, back) on C++
Omni::VisionProcessor::Qwen::gridPositions ordering, for single and
multi-frame grids; then round-trips the module through ONNX/onnxruntime.

Usage: python3 lfm3vl_conv_test.py [export_dir]
  export_dir defaults to $LFM3_WORK/MNN/transformers/llm/export
"""
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT_DIR = (sys.argv[1] if len(sys.argv) > 1
              else os.environ.get('LFM3_WORK', os.path.join(REPO_ROOT, 'lfm3-work'))
              + '/MNN/transformers/llm/export')
sys.path.insert(0, EXPORT_DIR)

import torch
import torch.nn.functional as F

from utils.vision import Lfm3VlVisionConv

MERGE = 2
C = 7
KERNEL = 3
torch.manual_seed(0)


def grid_positions(t, h, w, m=MERGE):
    # C++ Omni::VisionProcessor::Qwen::gridPositions == HF get_vision_position_ids
    pos = torch.zeros(2, t * h * w, dtype=torch.int32)
    for f in range(t):
        for y in range(h):
            for x in range(w):
                i = f * h * w + (y // m * (w // m) + x // m) * m * m + y % m * m + x % m
                pos[0, i] = y
                pos[1, i] = x
    return pos


def reference_conv(module, hidden_states, pos, t, h, w):
    # HF Lfm3VlVisionConv2d.forward: in_proj chunk, gate, raster depthwise conv, out_proj
    value, scale, key = module.in_proj(hidden_states).chunk(3, dim=-1)
    gated = value * key
    weight = module.spatial_conv.weight  # (C, 1, k, k)
    out = torch.zeros_like(gated)
    for f in range(t):
        frame = gated[f * h * w:(f + 1) * h * w]
        # merge-block order -> raster grid
        raster = torch.zeros(h, w, C, dtype=gated.dtype)
        for y in range(h):
            for x in range(w):
                i = (y // MERGE * (w // MERGE) + x // MERGE) * MERGE * MERGE + y % MERGE * MERGE + x % MERGE
                raster[y, x] = frame[i]
        conv = F.conv2d(raster.permute(2, 0, 1).unsqueeze(0), weight,
                        padding=KERNEL // 2, groups=C)
        back = torch.zeros_like(frame)
        conv_raster = conv.squeeze(0).permute(1, 2, 0).reshape(h * w, C)
        for y in range(h):
            for x in range(w):
                i = (y // MERGE * (w // MERGE) + x // MERGE) * MERGE * MERGE + y % MERGE * MERGE + x % MERGE
                back[i] = conv_raster[y * w + x]
        out[f * h * w:(f + 1) * h * w] = back
    return module.out_proj(scale * out)


def make_pair():
    conv = types.SimpleNamespace(
        in_proj=torch.nn.Linear(C, 3 * C, bias=False),
        spatial_conv=torch.nn.Conv2d(C, C, KERNEL, padding=KERNEL // 2, groups=C, bias=False),
        out_proj=torch.nn.Linear(C, C, bias=False),
    )
    return conv, Lfm3VlVisionConv(conv, MERGE)


def main():
    print(f'lfm3vl_conv_test: export dir {EXPORT_DIR}')
    failures = 0
    # --- torch-level equivalence on C++ grid ordering -------------------------
    for (t, h, w) in [(1, 6, 10), (2, 8, 8), (3, 4, 6), (1, 2, 2), (1, 14, 30)]:
        conv, rewrite = make_pair()
        x = torch.randn(t * h * w, C)
        pos = grid_positions(t, h, w)
        want = reference_conv(conv, x, pos, t, h, w)
        got = rewrite(x, pos.float())
        err = (want - got).abs().max().item()
        ok = err < 1e-5
        failures += not ok
        print(f'  torch t={t} grid={h}x{w}: max abs diff {err:.3e} {"OK" if ok else "FAIL"}')

    # --- ONNX round-trip of the rewrite ---------------------------------------
    # export through MNN's own onnx_export wrapper (dynamo=False, opset 15) so the
    # gate exercises the exact production export path, not the dynamo exporter
    onnx_path = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'lfm3vl_conv.onnx')
    from utils.torch_utils import onnx_export
    conv, rewrite = make_pair()
    x = torch.randn(2 * 8 * 8, C)
    pos = grid_positions(2, 8, 8)
    want = rewrite(x, pos.float())
    onnx_export(rewrite, (x, pos), onnx_path,
                input_names=['hidden_states', 'position_ids'],
                output_names=['output'],
                dynamic_axes={'hidden_states': {0: 'size'}, 'position_ids': {1: 'size'}})
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    got = torch.from_numpy(sess.run(None, {'hidden_states': x.numpy(),
                                           'position_ids': pos.numpy()})[0])
    err = (want - got).abs().max().item()
    ok = err < 1e-4
    failures += not ok
    print(f'  onnx  t=2 grid=8x8: max abs diff {err:.3e} {"OK" if ok else "FAIL"}')

    print('lfm3vl_conv_test:', 'PASS' if failures == 0 else f'FAIL ({failures})')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
