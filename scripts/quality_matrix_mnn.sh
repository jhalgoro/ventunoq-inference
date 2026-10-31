#!/usr/bin/env bash
# M1 quality matrix for MNN — shape of ../ventunoq-llamacpp-cpu+npu/scripts/quality_matrix.sh,
# but there is no server (F14): every task runs quality_eval_mnn.py, which batches prompts
# through the llm_eval driver (one model load per suite). Greedy everywhere.
# Resume-safe: a task is skipped if its JSON already has "score"; llm_eval additionally
# skips items whose answer file exists, so a killed run continues where it died.
# Disk policy: model downloaded at stage start, deleted at stage end, both in DISK_LOG.md.
# Usage: scripts/quality_matrix_mnn.sh [stage ...]   (default: q_qwen08)
# Smoke variant (plumbing check, ~15 min): scripts/quality_matrix_mnn.sh q_qwen08_smoke
set -u
cd "$(dirname "$0")/.."
MNN_SRC=${MNN_SRC:-$PWD/MNN}
export MNN_SRC

LOG=results/DISK_LOG.md
QLOG=results/quality_status.log
# same task:n list as the sibling matrix; ppl:0 = full gatsby.txt
TASKS="gsm8k:100 humaneval:50 arc:100 hellaswag:100 winogrande:100 openbookqa:100 commonsenseqa:100 mmlu:198 niah:5 ppl:0"
TASKS_SMOKE="gsm8k:5 humaneval:3 arc:5 hellaswag:5 winogrande:5 openbookqa:5 commonsenseqa:5 mmlu:6 niah:2 ppl:1"

log() { echo "$(date -u +'%FT%TZ') $*" | tee -a $QLOG; }

disklog() { # $1=what $2=file-or-dir
  echo "" >> $LOG
  echo "### $(date -u +'%F %R UTC') — $1: \`$2\`" >> $LOG
  echo '```' >> $LOG
  [ -e "$2" ] && { sha256sum "$2"/* 2>/dev/null | grep -v tokenizer.txt || true; du -sh "$2" 2>/dev/null; }
  df -h / | tail -1 >> $LOG
  echo '```' >> $LOG
}

q_model() { # $1 = taobao-mnn model name; leaves text-only dir at models/$1-text
  scripts/dl_model.sh "$1" || { log "FAIL download $1"; return 1; }
  [ -d "models/$1-text" ] || python3 scripts/text_only.py "models/$1" "models/$1-text"
}

del_model() { # $1 = model name (both dirs) — delete-after-suite policy
  disklog "DELETE (quality suite done)" "models/$1"
  disklog "DELETE (quality suite done)" "models/$1-text"
  rm -rf "models/$1" "models/$1-text"
  rm -rf /tmp/opencode/mnn_tmp /tmp/opencode/qbatch
}

run_tasks() { # $1=config.json $2=tag $3="task:n ..."
  local cfg=$1 tag=$2 spec task n out
  for spec in $3; do
    task=${spec%%:*}; n=${spec##*:}
    out="results/quality_${tag}_mnn_${task}.json"
    if grep -q '"score"' "$out" 2>/dev/null; then log "skip $tag/mnn/$task (have score)"; continue; fi
    python3 scripts/quality_eval_mnn.py "$cfg" "$task" "$out" --tag "$tag" --engine mnn --n "$n" \
      > "results/quality_${tag}_mnn_${task}.log" 2>&1
    if grep -q '"score"' "$out" 2>/dev/null; then
      log "done $tag/mnn/$task $(python3 -c "import json;d=json.load(open('$out'));print(f\"{d['score']:.3f} n={d['n']} {d['seconds']}s\")")"
    else
      log "FAILED $tag/mnn/$task (see ${out%.json}.log)"
    fi
  done
}

# ---------------- stages ----------------
q_qwen08() { # M1 main row: Qwen3.5-0.8B MNN int4, chat mode
  TAG=QWEN35-08B-INT4
  q_model Qwen3.5-0.8B-MNN || return
  run_tasks "models/Qwen3.5-0.8B-MNN-text/config.json" "$TAG" "$TASKS"
  del_model Qwen3.5-0.8B-MNN
}

q_qwen08_smoke() { # plumbing check: every suite at tiny n, same disk policy
  TAG=QWEN35-08B-INT4-SMOKE
  q_model Qwen3.5-0.8B-MNN || return
  run_tasks "models/Qwen3.5-0.8B-MNN-text/config.json" "$TAG" "$TASKS_SMOKE"
  del_model Qwen3.5-0.8B-MNN
}

for S in ${@:-q_qwen08}; do
  log "=== stage $S"
  $S
done
log "=== quality run done"
