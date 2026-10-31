#!/usr/bin/env python3
"""Parity report between two quality runs (usually npu vs cpu), item-by-item.

  quality_parity.py results/quality_TAG_npu_*.json results/quality_TAG_cpu_*.json
"""
import glob, json, sys

def load(patterns):
    out = {}
    for pat in patterns:
        for f in glob.glob(pat):
            d = json.load(open(f))
            if "score" in d:
                out[d["task"]] = d
    return out

a, b = load(sys.argv[1:2]), load(sys.argv[2:])
if not a or not b:
    sys.exit("no matching scored JSONs on one side")
ta, tb = next(iter(a.values())), next(iter(b.values()))
print(f"A = {ta['tag']}/{ta['engine']}   B = {tb['tag']}/{tb['engine']}\n")
print(f"{'task':<14} {'n':>4} {'A acc':>7} {'B acc':>7} {'Δ':>7} {'item agree':>10}")
tot_n = tot_ag = 0
for task in sorted(set(a) & set(b)):
    da, db = {d["id"]: d["ok"] for d in a[task]["details"]}, \
             {d["id"]: d["ok"] for d in b[task]["details"]}
    ids = sorted(set(da) & set(db))
    if not ids:
        continue
    aa = sum(da[i] for i in ids) / len(ids)
    ab = sum(db[i] for i in ids) / len(ids)
    ag = sum(da[i] == db[i] for i in ids) / len(ids)
    tot_n += len(ids); tot_ag += sum(da[i] == db[i] for i in ids)
    print(f"{task:<14} {len(ids):>4} {aa:>7.1%} {ab:>7.1%} {aa-ab:>+7.1%} {ag:>9.1%}")
print(f"\noverall per-item agreement: {tot_ag}/{tot_n} = {tot_ag/tot_n:.1%}")
print("acceptance (F7 rule): |Δacc| within one sampling band AND agreement >= ~95% "
      "=> engines quality-equivalent")
