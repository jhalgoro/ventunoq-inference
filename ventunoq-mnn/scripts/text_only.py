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
            os.remove(p)  # unlink only — the source keeps its own link

    # F18: write configs via tmp+rename so dst gets a NEW inode. open(path,"w")
    # truncates the cp -al hardlink — the source copy mutates too, then a later
    # dl_model.sh `curl -C -` resumes the shrunken file and appends garbage.
    def rewrite(path, mutate):
        cfg = json.load(open(path))
        mutate(cfg)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(cfg, fh, indent=2)
        os.replace(tmp, path)

    rewrite(os.path.join(dst, "llm_config.json"), lambda c: c.__setitem__("is_visual", False))
    rewrite(os.path.join(dst, "config.json"), lambda c: c.pop("mllm", None))
    print(f"text-only copy: {dst}")


if __name__ == "__main__":
    if not 2 <= len(sys.argv) <= 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
