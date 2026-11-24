#!/usr/bin/env bash
# L4 energy — avg W + J/token per engine, aligned power_probe.py windows.
# One condition = probe(hz=5) → dedicated llama-bench run → stop probe → parse W + t/s.
# Usage: scripts/energy.sh <gguf> <cpu|npu|npu2> <tag> [extra flags...]
#   cpu = $HEX2 -ngl 0 (same binary as NPU rows — scope: CPU+NPU arch only)
#   npu = $HEX (0.27) | npu2 = $HEX2 (0.28, lfm3)
# Emits results/energy_<tag>.json (W, t/s, J/tok for pp and tg separately).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

GGUF=${1:?usage: energy.sh <gguf> <engine> <tag> [extra]}; ENGINE=${2:?cpu|npu|npu2}; TAG=${3:?}
shift 3; EXTRA="${*:-}"
OUT=results/energy_$TAG.json

case "$ENGINE" in
  cpu)  BIN=$HEX2/bin/llama-bench; NGL=0; BASE="-ngl 0 $CPU_PIN"
        export LD_LIBRARY_PATH=/tmp/opencode/libfix:$HEX2/lib ADSP_LIBRARY_PATH=$HEX2/lib ;;
  npu)  BIN=$HEX/bin/llama-bench;  NGL=100; BASE="--device HTP0 -ngl 100 -t 4 --cpu-mask 0x0f -b 1024 -ub 512 -fa on"
        export LD_LIBRARY_PATH=/tmp/opencode/libfix:$HEX/lib ADSP_LIBRARY_PATH=$HEX/lib ;;
  npu2) BIN=$HEX2/bin/llama-bench; NGL=100; BASE="--device HTP0 -ngl 100 -t 4 --cpu-mask 0x0f -b 1024 -ub 512 -fa on"
        export LD_LIBRARY_PATH=/tmp/opencode/libfix:$HEX2/lib ADSP_LIBRARY_PATH=$HEX2/lib ;;
esac
[ "$ENGINE" = cpu ] || export GGML_HEXAGON_OPBATCH=2048 GGML_HEXAGON_MBUF=256,256

run_cond() { # $1=name $2=bench args ("pp512"|"tg128")
  local name=$1 args=$2
  scripts/power_probe.py ${TAG}_$name 5 > /dev/null 2>&1 &
  local probe=$!
  sleep 2                                   # idle baseline inside the probe window
  local log=results/energy_${TAG}_${name}.csv
  timeout 3600 $BIN -m "$GGUF" $BASE $args $EXTRA -o csv > "$log" 2>&1
  sleep 1; kill -TERM $probe; wait $probe 2>/dev/null || true
}
run_cond pp "-p 512 -n 0 -r 5"
run_cond tg "-p 0 -n 128 -r 5"

python3 - "$TAG" <<'EOF'
import csv, json, sys
from datetime import datetime, timedelta, timezone
tag = sys.argv[1]
out = {}
for cond in ("pp", "tg"):
    fn = f"results/energy_{tag}_{cond}.csv"
    lines = [l for l in open(fn) if l.startswith("build_commit") or l.startswith('"')]
    r = list(csv.DictReader(lines))[-1]
    # window = measurement only: bench ends at test_time, ran reps x avg_ns
    end = datetime.strptime(r["test_time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    dur = int(r["avg_ns"]) * 5 / 1e9          # -r 5
    start = end - timedelta(seconds=dur + 2)  # +2 s pipeline settle
    pw = []
    for x in csv.reader(open(f"results/power_{tag}_{cond}.csv")):
        if x[0] == "ts":
            continue
        t = datetime.fromisoformat(x[0])
        if start <= t <= end + timedelta(seconds=1):
            pw.append(float(x[1]))
    ts = float(r["avg_ts"])
    avgw = sum(pw) / max(len(pw), 1)
    out[cond] = dict(t_s=ts, avg_W=round(avgw, 2), J_per_tok=round(avgw / ts, 3),
                     samples=len(pw), window_s=round(dur))
json.dump(out, open(f"results/energy_{tag}.json", "w"), indent=1)
print(json.dumps(out))
EOF
