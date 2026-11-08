#!/usr/bin/env python3
"""lfm3vl_hf_ppl.py — HF-side twin of MNN's ppl_eval, for conversion-fidelity review.

Computes teacher-forced perplexity of the ORIGINAL HF checkpoint (torch CPU) on the
same text file MNN's ppl_eval scored (bench/gatsby.txt), mirroring ppl_eval's
conventions exactly (MNN/transformers/llm/engine/tools/ppl_eval.cpp):

  * raw text tokenization, NO chat template (engine uses use_template:false)
  * sliding window: stride = 512, contextLength = 768 by default; if the exported
    llm_config.json sets `chunk`/`chunk_limits`, those are mirrored instead
  * per-chunk loss = mean CE over (trgLen-1) tokens with the 1-token overlap when
    prevEnd != 0; final ppl = exp(mean of per-chunk losses)  (unweighted mean,
    exactly like ppl_eval's lossSum/lossNumber)

Decisive read for LFM3-VL-2B (MNN int4 ppl 151.77 on gatsby):
  HF ppl ~= 140-155  -> checkpoint property on literary prose, export is faithful
  HF ppl ~=  40      -> export/quantization lost ~110 ppl -> investigate the port
  in between         -> partial artifact; compare llm_logits_diff int4-vs-int8 KL

Run on the UM890 (venv has torch CPU + transformers with the custom code):
  scripts/lfm3vl_um890.sh hfppl
or directly:
  venv/bin/python scripts/lfm3vl_hf_ppl.py <MODEL_DIR> bench/gatsby.txt \
      [--mnn-config <llm_config.json>] [--max-length N] [--fp32] [--no-specials]
"""
import argparse
import json
import math
import os
import sys
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("text_file")
    ap.add_argument("--mnn-config", default=None,
                    help="exported llm_config.json to mirror chunk/chunk_limits "
                         "(default: sibling <MODEL_DIR>-MNN/llm_config.json if present)")
    ap.add_argument("--max-length", type=int, default=-1, help="token cap (ppl_eval maxLength)")
    ap.add_argument("--fp32", action="store_true", help="load fp32 instead of bf16")
    ap.add_argument("--no-specials", action="store_true",
                    help="tokenize without special tokens (default follows tokenizer config, "
                         "which is what MNN's tokenizer_encode does)")
    ap.add_argument("--print-loss", action="store_true", help="per-chunk loss, like -DPRINT_LOSS")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    # registered via auto_map: AutoModelForImageTextToText -> modeling_lfm3_vl
    try:
        from transformers import AutoModelForImageTextToText as AutoVL
    except ImportError:  # older transformers
        from transformers import AutoModelForVision2Seq as AutoVL

    torch.set_grad_enabled(False)
    nthreads = int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count() or 8)
    torch.set_num_threads(nthreads)

    # --- mirror ppl_eval windowing -------------------------------------------------
    stride, context_length = 512, 768  # ppl_eval defaults: stride=512, ctx=stride+stride/2
    mc = args.mnn_config
    if mc and not os.path.isfile(mc):
        print(f"[hf-ppl] --mnn-config {mc} not found; using window defaults")
        mc = None
    if mc is None:
        guess = args.model_dir.rstrip("/") + "-MNN/llm_config.json"
        mc = guess if os.path.isfile(guess) else None
    if mc:
        cfg = json.load(open(mc))
        if cfg.get("chunk_limits"):
            context_length = int(cfg["chunk_limits"][0])
            stride = (context_length // 3) * 2
        elif cfg.get("chunk"):
            context_length = int(cfg["chunk"])
            stride = (context_length // 3) * 2
        print(f"[hf-ppl] window from {mc}: contextLength={context_length} stride={stride}")
    else:
        print(f"[hf-ppl] window defaults (no llm_config.json): contextLength={context_length} "
              f"stride={stride}")

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    text = open(args.text_file).read()
    ids = tok(text, add_special_tokens=not args.no_specials)["input_ids"]
    seq_len = min(len(ids), args.max_length) if args.max_length > 0 else len(ids)
    print(f"[hf-ppl] tokens={len(ids)} scored={seq_len} (ppl_eval scores the same span)")
    print(f"[hf-ppl] first ids: {ids[:8]}")

    dtype = torch.float32 if args.fp32 else torch.bfloat16
    print(f"[hf-ppl] loading {args.model_dir} ({dtype}, CPU, {nthreads} threads) ...")
    t0 = time.time()
    try:
        model = AutoVL.from_pretrained(args.model_dir, trust_remote_code=True, dtype=dtype)
    except TypeError:  # transformers < 5 uses torch_dtype=
        model = AutoVL.from_pretrained(args.model_dir, trust_remote_code=True, torch_dtype=dtype)
    model.eval()
    print(f"[hf-ppl] loaded in {time.time() - t0:.1f} s")

    # --- teacher-forced loop, line-for-line equivalent of ppl_eval's ----------------
    prev_end = 0
    loss_sum, loss_number = 0.0, 0
    t0 = time.time()
    for begin in range(0, seq_len, stride):
        end = min(begin + context_length, seq_len)
        chunk = torch.tensor([ids[begin:end]], dtype=torch.long)
        # text-only forward: the VL wrapper runs the vision tower only when
        # pixel_values are given; logits_to_keep=0 keeps every position (as ppl_eval needs)
        logits = model(input_ids=chunk, use_cache=False).logits[0].float()  # [T, V]
        trg_len = end - prev_end + (1 if prev_end != 0 else 0)
        start = chunk.shape[1] - trg_len                  # index into the chunk
        # logits row i predicts token i+1; rows [start, start+trg_len-1) score
        # targets ids[start+1 : start+trg_len] — identical slice to ppl_eval.
        sel = logits[start:start + trg_len - 1]
        tgt = torch.tensor(ids[begin + start + 1:begin + start + trg_len], dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(sel, tgt).item()
        loss_sum += loss
        loss_number += 1
        prev_end = end
        if args.print_loss:
            print(f"Compute: {begin}/{seq_len}, loss={loss:.6f}", flush=True)
        if end == seq_len:
            break
        if loss_number % 10 == 0:
            done = end / seq_len * 100
            rate = (time.time() - t0) / loss_number
            print(f"[hf-ppl] {done:5.1f}% chunks={loss_number} running_ppl="
                  f"{math.exp(loss_sum / loss_number):8.2f} ({rate:.1f} s/chunk)", flush=True)

    ppl = math.exp(loss_sum / loss_number)
    print(f"[hf-ppl] HF Perplexity: {ppl:.4f}  ({loss_number} chunks, "
          f"{time.time() - t0:.0f} s)")
    print("[hf-ppl] compare with: MNN ppl_eval on the same file "
          "(results/quality_*_mnn_ppl.json details[0].ppl)")


if __name__ == "__main__":
    sys.exit(main())
