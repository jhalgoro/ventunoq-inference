#!/usr/bin/env bash
# M1 quality matrix for MNN — shape of ../ventunoq-llamacpp-cpu+npu/scripts/quality_matrix.sh,
# but there is no server (F14): every task runs quality_eval_mnn.py, which batches prompts
# through the llm_eval driver (one model load per suite). Greedy everywhere.
# Resume-safe: a task is skipped if its JSON already has "score"; llm_eval additionally
# skips items whose answer file exists, so a killed run continues where it died.
# Disk policy: model downloaded at stage start, deleted at stage end, both in DISK_LOG.md.
# Usage: scripts/quality_matrix_mnn.sh [stage ...]   (default: q_qwen08)
# Stages: q_qwen08 q_qwen08_smoke q_qwen2b q_qwen4b q_qwen9b q_qwen08base q_qwen2bbase
#         q_dist08 q_dist2b q_dist4b q_dist9b q_all
#         q_lfm3_smoke q_lfm3_r2 q_lfm3_hqq_r2 q_lfm3_int8_r2 q_lfm3_all   (M7, R2 protocol)
#         q_lfm3vl_smoke q_lfm3vl_r2 q_lfm3vl_all   (M7 LFM3-VL-2B, R2 protocol)
#         q_minicpm5_smoke q_minicpm5_r2 q_minicpm5_hqq_r2 q_minicpm5_int8_r2 q_minicpm5_all (M9)
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
  # F18: the sha/du block must go to $LOG too (was leaking to stdout)
  { [ -e "$2" ] && { sha256sum "$2"/* 2>/dev/null | grep -v tokenizer.txt || true; du -sh "$2" 2>/dev/null; }; } >> $LOG
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
  EXTRA=(--mc-tokens 512 --he-tokens 1536 --gsm-tokens 1024 --niah-tokens 512
         --no-mc-shots --mc-instruction "$DIST2_INSTR")
  q_stage Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN QWEN35-2B-DIST2 chat "$TASKS"
}
q_dist4b2() {
  EXTRA=(--mc-tokens 512 --he-tokens 1536 --gsm-tokens 1024 --niah-tokens 512
         --no-mc-shots --mc-instruction "$DIST2_INSTR")
  q_stage Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN QWEN35-4B-DIST2 chat "$TASKS"
}
q_all2() { for s in q_dist2b2 q_dist4b2; do log "--- ladder step $s"; $s; done; }

q_dist_fix() { # F19 fixups: rerun the suites the R2 ladder broke (2B humaneval open-fence
  # extraction, 2B mmlu Unicode crash) or under-budgeted (hellaswag @512: 41-57% of
  # reasoner outputs never reach "Answer: X"). Resume-safe — delete the affected JSONs
  # first (2B: humaneval/hellaswag/mmlu; 4B: hellaswag); scored suites skip. hellaswag
  # gets its own 1024-tok budget; mmlu stays at 512 for parity with the scored 4B row.
  EXTRA=(--mc-tokens 512 --hellaswag-tokens 1024 --he-tokens 1536 --gsm-tokens 1024
         --niah-tokens 512 --no-mc-shots --mc-instruction "$DIST2_INSTR")
  q_stage Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-MNN QWEN35-2B-DIST2 chat "$TASKS"
  q_stage Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-MNN QWEN35-4B-DIST2 chat "$TASKS"
}

# ---- M7 LFM3-1.7B rows: exported on the UM890 (scripts/lfm3_um890.sh export/export8/
# exporthq, then push) and copied to lfm3-models/ on this board — .mnn is cross-arch.
# No download/delete here: models are small and worth keeping for reruns. Scored under
# the R2 protocol (see q_lfm3_r2) — comparable with the QWEN35-*-DIST2 rows.
lfm3_stage() { # $1=exported dir $2=tag [$3=tasks]
  local cfg="$1/config.json"
  if [ ! -f "$cfg" ]; then
    log "MISSING $cfg — export on the UM890 and run scripts/lfm3_um890.sh push there first"
    return 1
  fi
  run_tasks "$cfg" "$2" "${3:-$TASKS}" chat
}
q_lfm3()       { lfm3_stage lfm3-models/LFM3-1.7B-MNN           LFM3-1.7B-INT4; }
q_lfm3_hqq()   { lfm3_stage lfm3-models/LFM3-1.7B-MNN-int4hqq   LFM3-1.7B-INT4-HQQ; }
q_lfm3_int8()  { lfm3_stage lfm3-models/LFM3-1.7B-MNN-int8      LFM3-1.7B-INT8; }
q_lfm3_smoke() { lfm3_stage lfm3-models/LFM3-1.7B-MNN           LFM3-1.7B-INT4-SMOKE "$TASKS_SMOKE"; }
# LFM3 is thinking-style even rendered non-thinking: MC answers open with a reasoning
# preamble ("The question asks...") and never reach a letter inside the sibling 16-tok
# MC budget — smoke scored 0.00 on arc/hellaswag/openbookqa/mmlu (F17 verbatim). Score
# under the R2 protocol (zero-shot MC + "Answer: X" cue + lifted budgets) so the rows
# are comparable with the QWEN35-*-DIST2 ladder.
lfm3_r2() { # $1=exported dir $2=tag
  EXTRA=(--mc-tokens 512 --he-tokens 1536 --gsm-tokens 1024 --niah-tokens 512
         --no-mc-shots --mc-instruction "$DIST2_INSTR")
  lfm3_stage "$1" "$2"
}
q_lfm3_r2()      { lfm3_r2 lfm3-models/LFM3-1.7B-MNN          LFM3-1.7B-INT4-R2; }
q_lfm3_hqq_r2()  { lfm3_r2 lfm3-models/LFM3-1.7B-MNN-int4hqq  LFM3-1.7B-INT4HQQ-R2; }
q_lfm3_int8_r2() { lfm3_r2 lfm3-models/LFM3-1.7B-MNN-int8     LFM3-1.7B-INT8-R2; }
q_lfm3_all()     { for s in q_lfm3_r2 q_lfm3_hqq_r2 q_lfm3_int8_r2; do log "--- ladder step $s"; $s; done; }

# ---- M7 LFM3-VL-2B rows: text tower scored under the same R2 protocol as the LFM3 rows
# (thinking-style template, same MC preamble behavior — q_lfm3_smoke precedent). Exported
# on the UM890 (scripts/lfm3vl_um890.sh export, int4 b64 / visual int8) and pushed to
# lfm3-models/LFM3-VL-2B-MNN. The vision module loads at init but is unused in chat mode;
# board binaries must carry MNN_BUILD_OPENCV+MNN_IMGCODECS (scripts/build_mnn.sh).
q_lfm3vl_smoke() { lfm3_stage lfm3-models/LFM3-VL-2B-MNN LFM3-VL-2B-INT4-SMOKE "$TASKS_SMOKE"; }
q_lfm3vl_r2()    { lfm3_r2 lfm3-models/LFM3-VL-2B-MNN LFM3-VL-2B-INT4-R2; }
q_lfm3vl_all()   { log "--- ladder step q_lfm3vl_r2"; q_lfm3vl_r2; }

# ---- M9 MiniCPM5-2B rows: stock llama arch, exported on the UM890 (scripts/minicpm5_um890.sh
# export/export8/exporthq, then push) into minicpm5-models/ on this board. Thinking-style
# like LFM3 (template has reasoning_content): run q_minicpm5_smoke first — if MC answers carry
# a reasoning preamble (F17), the R2 protocol below is the comparable row.
mc5_stage() { # $1=exported dir $2=tag [$3=tasks]
  local cfg="$1/config.json"
  if [ ! -f "$cfg" ]; then
    log "MISSING $cfg — export on the UM890 and run scripts/minicpm5_um890.sh push there first"
    return 1
  fi
  run_tasks "$cfg" "$2" "${3:-$TASKS}" chat
}
q_minicpm5_smoke() { mc5_stage minicpm5-models/MiniCPM5-2B-MNN MINICPM5-2B-SMOKE "$TASKS_SMOKE"; }
mc5_r2() { # $1=exported dir $2=tag — R2 protocol, same budgets as lfm3_r2/DIST2
  EXTRA=(--mc-tokens 512 --he-tokens 1536 --gsm-tokens 1024 --niah-tokens 512
         --no-mc-shots --mc-instruction "$DIST2_INSTR")
  mc5_stage "$1" "$2"
}
q_minicpm5_r2()      { mc5_r2 minicpm5-models/MiniCPM5-2B-MNN          MINICPM5-2B-INT4-R2; }
q_minicpm5_hqq_r2()  { mc5_r2 minicpm5-models/MiniCPM5-2B-MNN-int4hqq  MINICPM5-2B-INT4HQQ-R2; }
q_minicpm5_int8_r2() { mc5_r2 minicpm5-models/MiniCPM5-2B-MNN-int8     MINICPM5-2B-INT8-R2; }
# `all` = the default int4 row only; int8/hqq ladder steps exist but stay opt-in
# (~13 extra board-hours; LFM3 already showed HQQ net negative — decide after the int4 row).
q_minicpm5_all()     { log "--- ladder step q_minicpm5_r2"; q_minicpm5_r2; }

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
