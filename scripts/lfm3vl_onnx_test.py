#!/usr/bin/env python3
"""lfm3vl_onnx_test.py — verify exported visual.onnx against HF ground truth.

Feeds the same normalized image two ways on the UM890:
  reference: HF Lfm3Vl get_image_features(processor pixel_values, image_grid_thw)
             -> pooler_output (multi_modal_projector applied, 4:1 merged)
  under test: onnxruntime on the exported visual.onnx, fed exactly what the C++
  Qwen processor feeds with input_layout 'pixels': raw normalized frames
  (temporal, 3, H, W) rebuilt by inverting the HF processor patchify
  permutation, Qwen::gridPositions position_ids, per-frame block mask.

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
        # pooler_output = multi_modal_projector(vision_tower) per image — the same
        # post-merge features our visual.onnx emits (last_hidden_state is the
        # pre-projector, pre-merge vision tower output).
        out = model.model.get_image_features(pixel_values, grid_thw)
    ref = torch.cat(out.pooler_output, dim=0).float()
    print(f'HF image_embeds {tuple(ref.shape)}')

    # C++ contract (input_layout 'pixels', omni.cpp Patch::prepare/Qwen::patchInputs):
    # raw normalized frames (temporal_patch_size, 3, gh*p, gw*p) + Qwen::gridPositions
    # ids + per-frame block mask; the merge-block patchify lives INSIDE visual.onnx
    # (Vision.export_inputs). Rebuild the frames by inverting the HF processor's
    # patchify permutation (f, tp, c, bh, m, p, bw, m2, p2) -> rows
    # (bh, bw, m, m2, c, tp, p, p2), so the gate also verifies the in-graph flatten.
    assert len(grid_thw) == 1, 'single image only'
    vc = model.config.vision_config
    c, tp = 3, int(vc.temporal_patch_size)
    p, m = int(vc.patch_size), int(vc.spatial_merge_size)
    gt, gh, gw = (int(v) for v in grid_thw[0])
    n = gt * gh * gw
    assert pixel_values.shape == (n, c * tp * p * p), tuple(pixel_values.shape)
    patched = pixel_values.float().reshape(gt, gh // m, gw // m, m, m, c, tp, p, p)
    frames_raw = patched.permute(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(gt * tp, c, gh * p, gw * p)
    # self-check: forward flatten of the rebuilt frames must reproduce HF pixel_values
    re_patched = frames_raw.reshape(gt, tp, c, gh // m, m, p, gw // m, m, p) \
        .permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(n, c * tp * p * p)
    assert torch.equal(re_patched, patched.reshape(n, c * tp * p * p)), 'permutation inverse mismatch'

    position_ids = np.concatenate([grid_positions(*(int(v) for v in g)) for g in grid_thw], axis=1)
    frames = np.concatenate([np.repeat(i, int(g[0] * g[1] * g[2]))
                             for i, g in enumerate(grid_thw)])
    mask = np.zeros((1, n, n), dtype=np.float32)
    mask[0, frames[:, None] != frames[None, :]] = torch.finfo(torch.float32).min

    sess = InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    got = torch.from_numpy(sess.run(None, {
        'pixel_values': frames_raw.numpy(),
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
