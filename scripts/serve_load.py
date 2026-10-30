#!/usr/bin/env python3
"""L7 — concurrency load generator for llama-server (streaming /completion).

N parallel clients, each sends `rounds` identical ~pp512 prompts, 128 tokens gen,
stream=true so we can time first token. Emits results/serve_<tag>.json:
per-client tg, aggregate tg, TTFT p50/p95, KV tokens seen.

Usage: serve_load.py <port> <tag> <clients> [rounds]
"""
import json, statistics, sys, threading, time, urllib.request

PORT, TAG, CLIENTS = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
ROUNDS = int(sys.argv[4]) if len(sys.argv) > 4 else 2
PROMPT = ("You are a helpful assistant. Context: " +
          "The quick brown fox jumps over the lazy dog near the riverbank. " * 45)
LOCK = threading.Lock()
rows = []


def one_request():
    body = json.dumps({"prompt": PROMPT, "n_predict": 128, "temperature": 0,
                       "stream": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    tok = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            try:
                d = json.loads(raw[6:])
            except json.JSONDecodeError:
                continue
            if ttft is not None or not d.get("token"):
                if d.get("tokens_predicted"):
                    tok = d["tokens_predicted"]
                continue
            ttft = time.monotonic() - t0
            tok = d.get("tokens_predicted", 1)
    total = time.monotonic() - t0
    with LOCK:
        rows.append(dict(ttft=ttft, total=total, tok=tok))


def main():
    t0 = time.monotonic()
    for _ in range(ROUNDS):
        ths = [threading.Thread(target=one_request) for _ in range(CLIENTS)]
        [t.start() for t in ths]
        [t.join() for t in ths]
    wall = time.monotonic() - t0                 # global elapsed (all rounds)
    gen = sum(r["tok"] for r in rows)
    ttfts = sorted(r["ttft"] for r in rows)
    mean_req = sum(r["total"] for r in rows) / len(rows)
    out = dict(
        tag=TAG, clients=CLIENTS, rounds=ROUNDS, n_req=len(rows),
        per_client_tg=round(128 / mean_req, 2),   # incl. each req's own pp share
        aggregate_tg=round(gen / wall, 2),
        ttft_p50=round(ttfts[len(ttfts) // 2], 2),
        ttft_p95=round(ttfts[min(int(len(ttfts) * 0.95), len(ttfts) - 1)], 2),
        tok_total=gen, wall_s=round(wall, 1),
    )
    # aggregate via mean per-client × clients is exact for equal-length gens
    fn = f"results/serve_{TAG}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(json.dumps(out))


main()
