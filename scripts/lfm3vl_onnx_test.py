#!/usr/bin/env python3
"""lfm3vl_onnx_test.py — verify exported visual.onnx against HF ground truth.

Feeds the same normalized image two ways on the UM890:
  reference: HF Lfm3Vl get_image_features(processor pixel_values, image_grid_thw)
  under test: onnxruntime on the exported visual.onnx, fed exactly what the C++
  qwen2_vl patch processor emits — merge-block patches (HF pixel_values share
  that layout), Qwen::gridPositions position_ids, per-frame block mask.

Usage: python3 lfm3vl_onnx_test.py <model_dir> <onnx_dir_or_file> [image.jpg]
  <onnx_dir_or_file>: visual.onnx, or a dir holding it (also checks onnx/,
  where llmexport writes intermediates). No image: synthetic deterministic one.
"""
import os
import sys

import numpy as np
import torch

MODEL_DIR, MNN_DIR = sys.argv[1], sys.argv[2]
IMAGE = sys.argv[3] if len(sys.argv) > 3 else None
if MNN_DIR.endswith('.onnx'):
    ONNX_PATH = MNN_DIR
else:
    ONNX_PATH = next((p for p in (os.path.join(MNN_DIR, 'visual.onnx'),
                                  os.path.join(MNN_DIR, 'onnx', 'visual.onnx'))
                      if os.path.exists(p)), None)
    if ONNX_PATH is None:
        sys.exit(f'no visual.onnx under {MNN_DIR} (llmexport --export onnx first)')
MERGE = 2


def load_image():
    from PIL import Image
    if IMAGE:
        return Image.open(IMAGE).convert('RGB')
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(368, 544, 3), dtype=np.uint8)
    # smooth a bit so bicubic resize has structure, then quantize to blocks
    img = Image.fromarray(arr).resize((136, 92)).resize((544, 368), Image.BICUBIC)
    return img


def grid_positions(t, h, w, m=MERGE):
    # C++ Omni::VisionProcessor::Qwen::gridPositions == HF get_vision_position_ids
    pos = np.zeros((2, t * h * w), dtype=np.int32)
    for f in range(t):
        for y in range(h):
            for x in range(w):
                i = f * h * w + (y // m * (w // m) + x // m) * m * m + y % m * m + x % m
                pos[0, i] = y
                pos[1, i] = x
    return pos


def main():
    from transformers import AutoImageProcessor, AutoModelForImageTextToText
    from onnxruntime import InferenceSession

    image = load_image()
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    enc = processor(images=image, return_tensors='pt')
    pixel_values, grid_thw = enc.pixel_values.float(), enc.image_grid_thw
    print(f'image {image.size}, pixel_values {tuple(pixel_values.shape)}, grid {grid_thw.tolist()}')

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, trust_remote_code=True, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        ref = model.model.get_image_features(pixel_values, grid_thw).last_hidden_state
    print(f'HF image_embeds {tuple(ref.shape)}')

    # C++ qwen2_vl processor contract: merged patches in merge-block order
    # (identical to HF pixel_values), (2, N) grid positions, per-frame mask.
    n = int(grid_thw.prod(-1).sum())
    assert pixel_values.shape[0] == n, (pixel_values.shape, n)
    position_ids = np.concatenate([grid_positions(*(int(v) for v in g)) for g in grid_thw], axis=1)
    frames = np.concatenate([np.repeat(i, int(g[0] * g[1] * g[2]))
                             for i, g in enumerate(grid_thw)])
    mask = np.zeros((1, n, n), dtype=np.float32)
    mask[frames[:, None] != frames[None, :]] = torch.finfo(torch.float32).min

    sess = InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    got = torch.from_numpy(sess.run(None, {
        'patches': pixel_values.numpy(),
        'position_ids': position_ids,
        'attention_mask': mask,
    })[0])
    got = got.squeeze(1) if got.dim() == 3 else got
    print(f'ONNX image_embeds {tuple(got.shape)}')

    assert got.shape == ref.shape, f'shape mismatch {tuple(got.shape)} vs {tuple(ref.shape)}'
    diff = (ref - got).abs()
    rel = diff.max().item() / max(ref.abs().max().item(), 1e-6)
    print(f'max abs diff {diff.max().item():.4e}, max rel {rel:.4e}, mean abs {diff.mean().item():.4e}')
    ok = rel < 2e-2
    print('lfm3vl_onnx_test:', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
