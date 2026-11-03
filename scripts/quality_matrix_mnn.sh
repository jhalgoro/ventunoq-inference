#!/usr/bin/env bash
# M1 quality matrix for MNN — shape of ../ventunoq-llamacpp-cpu+npu/scripts/quality_matrix.sh,
# but there is no server (F14): every task runs quality_eval_mnn.py, which batches prompts
# through the llm_eval driver (one model load per suite). Greedy everywhere.
# Resume-safe: a task is skipped if its JSON already has "score"; llm_eval additionally
# skips items whose answer file exists, so a killed run continues where it died.
# Disk policy: model downloaded at stage start, deleted at stage end, both in DISK_LOG.md.
# Usage: scripts/quality_matrix_mnn.sh [stage ...]   (default: q_qwen08)
# Stages: q_qwen08 q_qwen08_smoke q_qwen2b q_qwen4b q_qwen9b q_qwen08base q_qwen2bbase
#         q_dist08 q_dist2b q_dist4b q_dist9b q_all   (see ladder below for tags)
# Smoke variant (plumbing check, ~15 min): scripts/quality_matrix_mnn.sh q_qwen08_smoke
# One-shot full ladder, detached from any SSH/session (survives disconnect):
#   cd ~/Code/ventunoq-mnn && setsid nohup scripts/quality_matrix_mnn.sh q_all \
#     >> results/quality_all.out 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
MNN_SRC=${MNN_SRC:-$PWD/MNN}
export MNN_SRC

LOG=results/DISK_LOG.md
QLOG=results/quality_status.log
EXTRA=()                                # extra quality_eval_mnn.py flags for run_tasks (R2)
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

run_tasks() { # $1=config.json $2=tag $3="task:n ..." $4=mode (chat|base, default chat)
  local cfg=$1 tag=$2 spec task n out mode=${4:-chat}
  for spec in $3; do
    task=${spec%%:*}; n=${spec##*:}
    out="results/quality_${tag}_mnn_${task}.json"
    if grep -q '"score"' "$out" 2>/dev/null; then log "skip $tag/mnn/$task (have score)"; continue; fi
    python3 scripts/quality_eval_mnn.py "$cfg" "$task" "$out" --tag "$tag" --engine mnn --n "$n" \
      --mode "$mode" "${EXTRA[@]}" > "results/quality_${tag}_mnn_${task}.log" 2>&1
    if grep -q '"score"' "$out" 2>/dev/null; then
      log "done $tag/mnn/$task $(python3 -c "import json;d=json.load(open('$out'));print(f\"{d['score']:.3f} n={d['n']} {d['seconds']}s\")")"
    else
      log "FAILED $tag/mnn/$task (see ${out%.json}.log)"
    fi
  done
}

# ---------------- stages ----------------
q_stage() { # $1=model name $2=tag $3=mode $4="task:n ..." — download, run, delete
  q_model "$1" || return
  run_tasks "models/$1-text/config.json" "$2" "$4" "$3"
  del_model "$1"
}

q_qwen08() { # M1 main row: Qwen3.5-0.8B MNN int4, chat mode
  q_stage Qwen3.5-0.8B-MNN QWEN35-08B-INT4 chat "$TASKS"
}

q_qwen08_smoke() { # plumbing check: every suite at tiny n, same disk policy
  q_stage Qwen3.5-0.8B-MNN QWEN35-08B-INT4-SMOKE chat "$TASKS_SMOKE"
}

# ---- q_all ladder: every board-viable taobao-mnn Qwen3.5 export, cheapest first.
# 27B/35B-A3B/122B excluded (RAM); Qwen3.5-2B-Dflash is a spec-decode draft, not a
# quality target. Est. total ~45 h at the 0.8B run's throughput (69 min for 0.8B).
q_qwen2b()      { q_stage Qwen3.5-2B-MNN   QWEN35-2B-INT4 chat "$TASKS"; }
q_qwen4b()      { q_stage Qwen3.5-4B-MNN   QWEN35-4B-INT4 chat "$TASKS"; }
q_qwen9b()      { q_stage Qwen3.5-9B-MNN   QWEN35-9B-INT4 chat "$TASKS"; }
# base variants: same weights without the chat template -> isolates template damage
# from quant damage on the flat MC block (harness --mode base, use_template:false)
q_qwen08base()  { q_stage Qwen3.5-0.8B-Base-MNN QWEN35-08B-INT4-BASE base "$TASKS"; }
q_qwen2bbase()  { q_stage Qwen3.5-2B-Base-MNN   QWEN35-2B-INT4-BASE base "$TASKS"; }
# reasoning-distilled variants: is the gsm8k-34%/flat-MC profile quant or recipe?
q_dist08()      { q_stage Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled-MNN QWEN35-08B-DIST chat "$TASKS"; }
q_dist2b()      { q_stage Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN   QWEN35-2B-DIST  chat "$TASKS"; }
q_dist4b()      { q_stage Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN   QWEN35-4B-DIST  chat "$TASKS"; }
q_dist9b()      { q_stage Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-MNN   QWEN35-9B-DIST  chat "$TASKS"; }
# R2 reruns of the distilled rows: R1 scored 0 on MC+humaneval+niah because reasoning-
# style output needs more than the sibling-pinned budgets, and the 3-shot MC prefix
# makes these models re-solve the shots (F17). R2 protocol: zero-shot MC + explicit
# "end with Answer: X" cue + lifted budgets; parsing is reasoning-safe. Deviation from
# the sibling harness is deliberate and tagged (-DIST2), plain-model rows unchanged.
DIST2_INSTR="Think briefly (at most 3 sentences; do NOT analyze the options one by one), then end your reply with a line 'Answer: X' where X is the option letter."
q_dist2b2() {
  EXTRA=(--mc-tokens 512 --he-tokens 1024 --gsm-tokens 1024 --niah-tokens 128
         --no-mc-shots --mc-instruction "$DIST2_INSTR")
  q_stage Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN QWEN35-2B-DIST2 chat "$TASKS"
}
q_dist4b2() {
  EXTRA=(--mc-tokens 512 --he-tokens 1024 --gsm-tokens 1024 --niah-tokens 128
         --no-mc-shots --mc-instruction "$DIST2_INSTR")
  q_stage Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN QWEN35-4B-DIST2 chat "$TASKS"
}
q_all2() { for s in q_dist2b2 q_dist4b2; do log "--- ladder step $s"; $s; done; }

q_all() { # one-shot ladder; resume-safe per suite, so re-run after any interruption
  for s in q_qwen2b q_qwen08base q_qwen2bbase q_qwen4b q_dist08 q_dist2b q_dist4b q_qwen9b q_dist9b; do
    log "--- ladder step $s"
    $s
  done
}

for S in ${@:-q_qwen08}; do
  log "=== stage $S"
  $S
done
log "=== quality run done"
