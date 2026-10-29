#!/usr/bin/env python3
"""HumanEval pass@1 for LFM3 via llama-server /completion (greedy), sandboxed exec."""
import json, subprocess, sys, tempfile, os, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8181
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
MODEL = sys.argv[3] if len(sys.argv) > 3 else "Q4_K_M"
OUT = f"results/humaneval_{MODEL}.json"

def ask(prompt, max_tokens=400):
    body = json.dumps({"prompt": prompt, "temperature": 0, "n_predict": max_tokens,
                       "stop": ["\n# ", "\nif __name__"]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r).get("content", "")

def run_check(task, completion):
    src = task["prompt"] + completion + "\n" + task["test"] + f"\ncheck({task['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=20)
        return r.returncode == 0, r.stderr.decode()[-300:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(path)

tasks = [json.loads(l) for i, l in enumerate(open("bench/HumanEval.jsonl")) if i < N]
results, passed = [], 0
t0 = time.time()
for i, t in enumerate(tasks):
    out = ask(t["prompt"])
    # some models re-emit the signature in raw completion mode; strip it if so
    first = out.lstrip("\n").split("\n", 1)
    if first[0].startswith("def ") and first[0].strip() in t["prompt"]:
        out = first[1] if len(first) > 1 else ""
    # truncate at hallucinated test code / extra top-level definitions
    cut = len(out)
    for marker in ("def check(", "\ncheck(", "\nassert ", "\ndef "):
        pos = out.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    out = out[:cut]
    ok, err = run_check(t, out)
    passed += ok
    results.append({"task": t["task_id"], "ok": ok, "err": "" if ok else err})
    print(f"[{i+1}/{len(tasks)}] {t['task_id']}: {'PASS' if ok else 'FAIL ' + err[:80]}", flush=True)

summary = {"model": MODEL, "dataset": "HumanEval", "n": len(tasks),
           "pass_at_1": round(passed / len(tasks), 4), "passed": passed,
           "seconds": round(time.time() - t0, 1), "details": results}
json.dump(summary, open(OUT, "w"), indent=1)
print(f"\nHumanEval pass@1: {passed}/{len(tasks)} -> {OUT}")
