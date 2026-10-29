#!/usr/bin/env python3
"""GSM8K eval for LFM3 via llama-server /completion endpoint (4-shot CoT, greedy)."""
import json, re, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8181
N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
MODEL = sys.argv[3] if len(sys.argv) > 3 else "Q4_K_M"
OUT = f"results/gsm8k_{MODEL}.json"

SHOTS = [
    ("Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?\nA:",
     "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\nThe answer is 72"),
    ("Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nA:",
     "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\nThe answer is 10"),
    ("Q: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nA:",
     "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\nThe answer is 5"),
    ("Q: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?\nA:",
     "He writes each friend 3*2=<<3*2=6>>6 pages a week.\nSo he writes 6*2=<<6*2=12>>12 pages every week.\nThat means he writes 12*52=<<12*52=624>>624 pages a year.\nThe answer is 624"),
]

def ask(prompt, max_tokens=512):
    body = json.dumps({"prompt": prompt, "temperature": 0, "n_predict": max_tokens,
                       "stop": ["\nQ:"]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)

def extract_answer(text):
    m = re.findall(r"The answer is\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if not m:
        m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if not m:
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        return nums[-1].replace(",", "") if nums else None
    return m[-1].replace(",", "")

def gold(answer):
    return answer.split("####")[-1].strip().replace(",", "")

def norm(x):
    try:
        f = float(x)
        return f"{f:g}"
    except (TypeError, ValueError):
        return str(x)

probs = [json.loads(l) for i, l in enumerate(open("bench/test.jsonl")) if i < N]
results, correct = [], 0
t0 = time.time()
for i, p in enumerate(probs):
    ctx = "\n\n".join(f"{q} {a}" for q, a in SHOTS)
    prompt = ctx + "\n\n" + f"Q: {p['question']}\nA:"
    r = ask(prompt)
    out = r.get("content", "")
    got, g = extract_answer(out), gold(p["answer"])
    ok = norm(got) == norm(g)
    correct += ok
    results.append({"idx": i, "ok": ok, "got": got, "gold": g,
                    "tokens": r.get("tokens_predicted"), "out_tail": out[-160:]})
    print(f"[{i+1}/{len(probs)}] {'OK ' if ok else 'FAIL'} got={got} gold={g}", flush=True)

acc = correct / len(probs)
summary = {"model": MODEL, "dataset": "GSM8K-test", "n": len(probs),
           "accuracy": round(acc, 4), "correct": correct,
           "seconds": round(time.time() - t0, 1), "details": results}
json.dump(summary, open(OUT, "w"), indent=1)
print(f"\nGSM8K acc: {correct}/{len(probs)} = {acc:.1%}  -> {OUT}")
