#!/usr/bin/env python3
"""Fetch MC-suite subsamples via the HF datasets-server JSON API (no pyarrow).

Port of ../ventunoq-llamacpp-cpu+npu/scripts/fetch_bench.py — SPECS are byte-identical
on purpose: MNN runs then score on the *same items* as the llama.cpp quality runs, so
scripts/quality_parity.py can join two engines item-by-item (M1).
Deterministic: fixed (offset,length) chunks, source order preserved. Committed
to bench/ so every later run scores on the identical items. GSM8K (test.jsonl)
and HumanEval.jsonl are copied from the parent repo bench/ instead.
"""
import json, os, shutil, sys, time, urllib.request

API = ("https://datasets-server.huggingface.co/rows?dataset={ds}"
       "&config={cfg}&split={split}&offset={off}&length={len}")
BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench")
PARENT_BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "ventunoq-llamacpp", "bench")

# name -> (dataset, config, split, [(offset,length), ...])
SPECS = {
    "arc_challenge":  ("allenai/ai2_arc", "ARC-Challenge", "test", [(0, 100), (500, 100)]),
    "hellaswag":      ("Rowan/hellaswag", "default", "validation", [(0, 100), (3300, 100), (6600, 100)]),
    "winogrande":     ("allenai/winogrande", "winogrande_xl", "validation", [(0, 100), (600, 100)]),
    "openbookqa":     ("allenai/openbookqa", "main", "test", [(0, 100), (400, 100)]),
    "commonsenseqa":  ("tau/commonsense_qa", "default", "validation", [(0, 100), (600, 100)]),
    # MMLU-lite: 3 subjects x first 66 test items each (knowledge, 4-option)
    "mmlu_bio":       ("cais/mmlu", "college_biology", "test", [(0, 66)]),
    "mmlu_geo":       ("cais/mmlu", "high_school_geography", "test", [(0, 66)]),
    "mmlu_math":      ("cais/mmlu", "college_mathematics", "test", [(0, 66)]),
}

def fetch_rows(ds, cfg, split, chunks):
    rows = []
    for off, ln in chunks:
        url = API.format(ds=ds, cfg=cfg, split=split, off=off, len=ln)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    d = json.load(r)
                rows += [x["row"] for x in d["rows"]]
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(5)
    return rows

def main():
    os.makedirs(BENCH, exist_ok=True)
    for name, (ds, cfg, split, chunks) in SPECS.items():
        out = os.path.join(BENCH, f"{name}.jsonl")
        rows = fetch_rows(ds, cfg, split, chunks)
        meta = {"_src": f"{ds} cfg={cfg} split={split} chunks={chunks}"}
        with open(out, "w") as f:
            f.write(json.dumps(meta) + "\n")
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} rows -> {out}")
    # local copies from parent repo (provenance: ../ventunoq-llamacpp/bench)
    for fn in ("test.jsonl", "HumanEval.jsonl", "gatsby.txt"):
        shutil.copy(os.path.join(PARENT_BENCH, fn), os.path.join(BENCH, fn))
        print(f"copied {fn}")

if __name__ == "__main__":
    sys.exit(main())
