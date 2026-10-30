#!/usr/bin/env python3
"""Make a taobao-mnn multimodal export text-only (F9).

Usage: scripts/text_only.py models/Qwen3.5-0.8B-MNN [dest-dir]
Hardlink-copies the dir (fallback: full copy), removes visual.* files, sets
is_visual=false in llm_config.json, drops the mllm block from config.json.
"""
import json, os, shutil, subprocess, sys


def main(src: str, dst: str | None) -> None:
    dst = dst or src.rstrip("/") + "-text"
    if os.path.exists(dst):
        sys.exit(f"{dst} already exists")
    try:  # hardlink copy: instant, weights not duplicated on disk
        subprocess.run(["cp", "-al", src, dst], check=True)
    except subprocess.CalledProcessError:
        shutil.copytree(src, dst)

    for f in ("visual.mnn", "visual.mnn.weight"):
        p = os.path.join(dst, f)
        if os.path.exists(p):
            os.remove(p)

    lc = os.path.join(dst, "llm_config.json")
    cfg = json.load(open(lc))
    cfg["is_visual"] = False
    json.dump(cfg, open(lc, "w"), indent=2)

    c = os.path.join(dst, "config.json")
    cfg = json.load(open(c))
    cfg.pop("mllm", None)
    json.dump(cfg, open(c, "w"), indent=2)
    print(f"text-only copy: {dst}")


if __name__ == "__main__":
    if not 2 <= len(sys.argv) <= 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
