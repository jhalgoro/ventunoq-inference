#!/usr/bin/env python3
"""MNN quality eval (M1) — port of ../ventunoq-llamacpp-cpu+npu/scripts/quality_eval.py.

llm_demo has no HTTP API and its batch mode is one-line prompts with separator-less
streaming (F14), so generation goes through our llm_eval driver (scripts/llm_eval.cpp,
built by scripts/build_mnn.sh): one prompt file per item, greedy sampling, fresh KV
per item, answers read back from batch_dir/outputs. Same corpora, same prompts, same
stride() item selection and same scorers as the llama.cpp harness, so
scripts/quality_parity.py joins the two engines item-by-item.

Differences from the llama.cpp harness (also noted in BENCHMARKS §4.1):
  * all MC suites score in the "gen" style (3-shot greedy letter + first-mention
    fallback) — MNN exposes no logprobs, so the walk/raw-walk styles are impossible.
  * server-side stop strings are emulated by truncating the answer at the same markers.
  * the ppl leg uses MNN's ppl_eval on bench/gatsby.txt; its JSON "score" is the
    perplexity, NOT an accuracy — keep ppl JSONs out of quality_parity.py's globs.

  quality_eval_mnn.py CONFIG TASK OUT --tag TAG --engine mnn [--n N] [--mode chat|base]

TASKs: gsm8k humaneval arc hellaswag winogrande openbookqa commonsenseqa mmlu niah ppl
"""
import argparse, json, os, re, shutil, subprocess, sys, textwrap, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "bench")
MNN_BUILD = os.path.join(os.environ.get("MNN_SRC", os.path.join(ROOT, "MNN")), "build")
SANDBOX = "/tmp/opencode/sandbox"
SCRATCH = "/tmp/opencode/qbatch"
TMP = "/tmp/opencode/mnn_tmp"          # llm_eval cwd: weight-mmap tmp cache lives here

CONFIG = "?"                            # model config.json (argv)
MODE = "chat"                           # chat = engine applies the model's chat template
NOTHINK = False                         # template has enable_thinking -> disable it
MC_TOK, HE_TOK, GSM_TOK = 16, 512, 700  # sibling-pinned budgets; --*-tokens flags lift
HS_TOK = 16  # hellaswag-only override (--hellaswag-tokens, F19; 0 => follow MC_TOK)
NIAH_TOK = 24                           # them for reasoning-style models (F17 R2 runs)
SYSTEM = "You are a helpful assistant answering benchmark questions."  # sibling default
BATCH = "?"                             # per-run scratch root for prompt/output files
ENGINE = "mnn"

MNN = None                              # MnnRunner, set in main()

class MnnRunner:
    """Drives llm_eval once per suite (one model load per batch). Resume-safe:
    llm_eval skips items whose output file already exists."""

    def __init__(self, config, mode, nothink):
        self.config, self.mode, self.nothink = config, mode, nothink

    def run(self, prompts, max_tokens, name, system=None):
        batch = os.path.join(BATCH, name)
        pdir = os.path.join(batch, "prompts")
        odir = os.path.join(batch, "outputs")
        os.makedirs(pdir, exist_ok=True)
        os.makedirs(TMP, exist_ok=True)
        names = [f"{i:05d}.txt" for i in range(len(prompts))]
        for name_, text in zip(names, prompts):
            with open(os.path.join(pdir, name_), "w") as f:
                f.write(text)
        outs = [os.path.join(odir, n) for n in names]
        if not all(os.path.exists(o) for o in outs):
            cmd = ["taskset", "-c", "0-3", os.path.join(MNN_BUILD, "llm_eval"),
                   self.config, batch, "--max-tokens", str(max_tokens)]
            if self.mode == "base":
                cmd.append("--base")
            if self.nothink:
                cmd.append("--nothink")
            if system and self.mode == "chat":
                cmd += ["--system", system]
            r = subprocess.run(cmd, cwd=TMP)   # progress flows on stderr into our log
            if r.returncode != 0:
                sys.exit(f"llm_eval failed rc={r.returncode} (batch {batch})")
        missing = [o for o in outs if not os.path.exists(o)]
        if missing:
            sys.exit(f"llm_eval finished but {len(missing)} outputs missing, e.g. {missing[0]}")
        # errors="replace": llm_eval can end an output mid-multibyte-char when generation
        # stops at the token limit (2B-DIST2 mmlu crashed reading such a file, R2 ladder).
        return [open(o, encoding="utf-8", errors="replace").read() for o in outs]

def cut(text, stops):
    """Emulate server-side stop strings: keep the text before the earliest marker."""
    pos = len(text)
    for s in stops:
        i = text.find(s)
        if i >= 0:
            pos = min(pos, i)
    return text[:pos]

def strip_think(text):
    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.S)
    return text.replace("<think></think>", "").strip()

# ---------------- shared corpus helpers (identical to the sibling harness) ----------------

LETTERS = "ABCDEFG"

def rows(path):
    out = []
    for line in open(os.path.join(BENCH, path)):
        j = json.loads(line)
        if not line.startswith('{"_src"'):
            out.append(j)
    return out

def stride(items, n):
    if n >= len(items):
        return items
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]

# ---------------- MC: gen style only (no logprobs in MNN) ----------------
# 3-shot lettered prefix for generative MC answering; letters spread B,B,C to
# break the position prior. Same shots as the sibling harness.
MC_SHOTS = (
    "Question: Which of the following is a living thing?\nA. rock\nB. tree\nC. water\nD. air\nAnswer: B\n\n"
    "Question: What is the primary role of the roots of a plant?\nA. to attract pollinators\n"
    "B. to absorb water and nutrients\nC. to make seeds\nD. to catch sunlight\nAnswer: B\n\n"
    "Question: Which material is a good conductor of electricity?\nA. rubber\nB. plastic\nC. copper\nD. wood\nAnswer: C\n\n")

MC_INSTR = ""                           # --mc-instruction; appended before "Answer:" (R2)
MC_SHOTS_ON = True                      # --no-mc-shots (R2): reasoning models re-solve
                                        # the shots instead of the real question (F17)

def mc_prompt(base, instr):
    """R2 (F17): reasoning models need an explicit 'end with Answer: X' instruction;
    it goes between the lettered options and the final 'Answer:' cue. Empty instr =
    sibling-identical prompt."""
    if not instr:
        return base
    head = base[:-len("Answer:")] if base.endswith("Answer:") else base
    return head + instr.strip() + "\nAnswer:"

def gen_letter(out, choices, reason=False):
    """Greedy short generation after a lettered prompt; parse the answer letter,
    else first-mention of choice text in what was generated. (Sibling gen_letter
    minus the server call: the batch output is already truncated at the stops.)
    Explicit-answer patterns win first: reasoning-style models (F17) enumerate
    every option letter before concluding, so "first bare letter" would grab a
    spurious one from the analysis. Short "B"-style outputs are unaffected.
    reason=True (F17 R2): the legacy first-letter/choice-text heuristics are
    poison on analysis prose (enumeration lists A..D in order, then any choice
    text may be quoted); trust only explicit answers and a letter at the tail."""
    out = strip_think(out)
    m = re.findall(r"answer(?:\s+is)?\s*[:=\-]?\s*\(?\s*([A-E])\b", out, re.I)
    if m:
        return LETTERS.index(m[-1].upper()), out[:24], "gen-letter"
    if not reason:
        m = re.search(r"\b([A-E])\b", out)
        if m:
            return LETTERS.index(m.group(1)), out[:24], "gen-letter"
    low = (out[-200:] if reason else out).lower()
    hits = [(low.find(c.strip()[:12].lower()), i) for i, c in enumerate(choices)]
    hits = [(p, i) for p, i in hits if p >= 0]
    if hits and not reason:
        return min(hits)[1], out[:24], "gen-match"
    if hits:
        m = re.search(r"\b([A-E])\b\s*[.)]?\s*$", out.strip()[-40:])
        if m:
            return LETTERS.index(m.group(1)), out[-24:], "gen-letter"
        return min(hits)[1], out[-24:], "gen-match"
    return None, out[:24], "gen-miss"

# ---------------- generation suites ----------------

GSM_SHOTS = [
    ("Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?\nA:",
     "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\nThe answer is 72"),
    ("Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nA:",
     "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\nThe answer is 10"),
    ("Q: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nA:",
     "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\nThe answer is 5"),
    ("Q: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?\nA:",
     "He writes each friend 3*2=<<3*2=6>>6 pages a week.\nSo he writes 6*2=<<6*2=12>>12 pages every week.\nThat means he writes 12*52=<<12*52=624>>624 pages a year.\nThe answer is 624"),
]

def extract_answer(text):
    m = re.findall(r"The answer is\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if not m:
        m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if not m:
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        return nums[-1].replace(",", "") if nums else None
    return m[-1].replace(",", "")

def norm(x):
    try:
        return f"{float(x):g}"
    except (TypeError, ValueError):
        return str(x)

def run_gsm8k(n):
    ctx = "\n\n".join(f"{q} {a}" for q, a in GSM_SHOTS)
    items = stride(rows("test.jsonl"), n)
    prompts = [ctx + f"\n\n{p['question']}\n"
               "Solve step by step. End your reply with a line 'The answer is N'."
               for p in items]
    outs = MNN.run(prompts, GSM_TOK, "gsm8k", SYSTEM)
    det = []
    for i, (p, out) in enumerate(zip(items, outs)):
        out = cut(out, ["\nQ:"])
        got = extract_answer(out)
        gold = p["answer"].split("####")[-1].strip().replace(",", "")
        det.append({"id": i, "ok": norm(got) == norm(gold), "got": got, "gold": gold,
                    "tail": out[-120:]})
        print(f"[gsm8k {i+1}] {'OK ' if det[-1]['ok'] else 'FAIL'} got={got} gold={gold}", flush=True)
    return det

def _sandbox_limits():
    import resource
    def lim():
        # RLIMIT_AS kept generous (CPython startup + imports); CPU time is the
        # real guard against runaway generated code.
        for what, (soft, hard) in ((resource.RLIMIT_CPU, (10, 12)),
                                   (resource.RLIMIT_AS, (6 << 30, 6 << 30)),
                                   (resource.RLIMIT_FSIZE, (0, 0))):
            try:
                resource.setrlimit(what, (soft, hard))
            except ValueError:
                pass
    return lim

def extract_code(out, entry):
    """Reasoning-style models (F17) wrap code in prose and/or fences; the sibling
    harness assumes bare code, and cut()-ing at the first ``` would erase fenced
    answers entirely. Candidates best-first: LAST fenced block containing the entry
    point — reasoners show a sketch first and the final answer last, and the R2
    ladder showed the closed-fence regex misses a fence left OPEN when generation
    hits the token cap (31/50 IndentationErrors: rfind then landed on a prose
    re-statement of the signature) — else text from the last definition. Return the
    first candidate that compiles (dedented variant included); bare code passes
    through unchanged."""
    cands = []
    parts = out.split("```")  # odd indices are inside fences, incl. an open last one
    fenced = [p for p in parts[1::2] if f"def {entry}(" in p]
    if fenced:
        cands.append(re.sub(r"^\s*python\s*\n", "", fenced[-1], count=1))
    i = out.rfind(f"def {entry}(")
    if i < 0:
        i = out.rfind("def ")
    if i >= 0:
        cands.append(out[i:])
    for c in cands:
        for cand in (c, textwrap.dedent(c)):
            try:
                compile(cand, "<extract>", "exec")
                return cand
            except SyntaxError:
                pass
    return cands[0] if cands else ""

def run_humaneval(n):
    os.makedirs(SANDBOX, exist_ok=True)
    tasks = stride(rows("HumanEval.jsonl"), n)
    prompts = [("Complete the following Python function. Return ONLY the code — "
                "the complete function including its signature, no explanations, "
                "no markdown fences.\n\n" + t["prompt"]) for t in tasks]
    outs = MNN.run(prompts, HE_TOK, "humaneval", SYSTEM)
    det = []
    for i, (t, out) in enumerate(zip(tasks, outs)):
        out = extract_code(out, t["entry_point"])
        out = cut(out, ["\nif __name__", "```", "\nprint(", "\nclass "])
        out = out.strip("\n")
        if out:  # chat model restated the signature? drop the restated def line —
            m = re.match(r"[ \t]*def\s+" + re.escape(t["entry_point"]) + r"\s*\(.*?\)\s*(->[^:\n]*)?:[ \t]*\n",
                         out, re.S)   # regex, not find("):"): annotated ") -> bool:" has no "):"
            if m:
                out = out[m.end():]
        # bench/HumanEval.jsonl tests define check(candidate) (no *_tests wrapper)
        script = t["prompt"] + out + "\n" + t["test"] + f"\ncheck({t['entry_point']})\n"
        path = os.path.join(SANDBOX, f"he_{i}.py")
        open(path, "w").write(script)
        try:
            r = subprocess.run([sys.executable, path], capture_output=True, text=True,
                               timeout=20, preexec_fn=_sandbox_limits(),
                               cwd=SANDBOX, env={"PATH": "/usr/bin:/bin", "HOME": SANDBOX})
            ok, err = r.returncode == 0, (r.stderr or "")[-200:]
        except subprocess.TimeoutExpired:
            ok, err = False, "timeout"
        det.append({"id": t["task_id"], "ok": ok, "err": err if not ok else "",
                    "comp_tail": out[-120:]})
        print(f"[humaneval {i+1}] {t['task_id']} {'PASS' if ok else 'FAIL'} {err[:80]}", flush=True)
        os.remove(path)
    return det

# ---------------- MC suites ----------------
# Prompts identical to the sibling harness; only the LETTERED prompt is needed
# now that every suite scores in gen style.

def fmt_arc(r):
    ch = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(r["choices"]["text"]))
    return (f"Question: {r['question']}\n{ch}\nAnswer:",
            f"Question: {r['question']}\nAnswer:",
            r["choices"]["text"], r["choices"]["label"].index(r["answerKey"]))

def fmt_hellaswag(r):
    ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
    ends = [e.strip() for e in r["endings"]]
    ch = "\n".join(f"{LETTERS[i]}. {e}" for i, e in enumerate(ends))
    return (f"{ctx}\nWhat happens next?\n{ch}\nAnswer:", ctx, ends, int(r["label"]))

def fmt_winogrande(r):
    pre, _, post = r["sentence"].partition("_")
    opts = [r["option1"], r["option2"]]
    ch = f"A. {opts[0]}\nB. {opts[1]}"
    stem = f"Fill in the blank: {pre.strip()} _____ {post.strip()}"
    return (f"{stem}\n{ch}\nAnswer:", f"{stem}\nAnswer:", opts, int(r["answer"]) - 1)

def fmt_obqa(r):
    ch = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(r["choices"]["text"]))
    return (f"Question: {r['question_stem']}\n{ch}\nAnswer:",
            f"Question: {r['question_stem']}\nAnswer:",
            r["choices"]["text"], "ABCDE".index(r["answerKey"]))

def fmt_csqa(r):
    opts = [r["choices"]["text"][i] for i in range(5)]
    ch = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(opts))
    return (f"Question: {r['question']}\n{ch}\nAnswer:",
            f"Question: {r['question']}\nAnswer:", opts, "ABCDE".index(r["answerKey"]))

SUBJ = {"mmlu_bio": "college biology", "mmlu_geo": "high school geography", "mmlu_math": "college mathematics"}

def fmt_mmlu(r):
    ch = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(r["choices"]))
    subj = SUBJ.get(r.get("subject"), r.get("subject", "general"))
    head = f"The following is a multiple choice question about {subj}.\n\nQuestion: {r['question']}"
    return (f"{head}\n{ch}\nAnswer:", f"{head}\nAnswer:", r["choices"], int(r["answer"]))

def run_mc(task, n):
    fn = {"arc": fmt_arc, "hellaswag": fmt_hellaswag, "winogrande": fmt_winogrande,
          "openbookqa": fmt_obqa, "commonsenseqa": fmt_csqa, "mmlu": fmt_mmlu}[task]
    src = {"arc": "arc_challenge", "hellaswag": "hellaswag", "winogrande": "winogrande",
           "openbookqa": "openbookqa", "commonsenseqa": "commonsenseqa", "mmlu": None}[task]
    if src:
        items = rows(f"{src}.jsonl")
    else:
        items = [r for s in SUBJ for r in rows(f"{s}.jsonl")]
    items = stride(items, n)
    prompts = [MC_SHOTS + mc_prompt(fn(r)[0], MC_INSTR) for r in items] \
        if MC_SHOTS_ON else [mc_prompt(fn(r)[0], MC_INSTR) for r in items]
    tok = HS_TOK if task == "hellaswag" else MC_TOK  # F19: reasoners ramble on hellaswag
    outs = MNN.run(prompts, tok, task, SYSTEM)
    det = []
    for i, (r, out) in enumerate(zip(items, outs)):
        _, _, choices, gold = fn(r)
        # the "\n\n" cut fits short answers but amputates reasoning at its first
        # blank line (F17) — reasoning mode reads the full generation instead
        parsed = out if tok > 16 else cut(out, ["\n\n", "\nQ"])
        pred, tail, method = gen_letter(parsed, choices, reason=tok > 16)
        det.append({"id": i, "ok": pred == gold, "pred": pred, "gold": gold,
                    "scores": [method == "gen-letter"], "method": str(method)})
        print(f"[{task} {i+1}] {'OK ' if det[-1]['ok'] else 'FAIL'} pred={pred} "
              f"gold={gold} ({method})", flush=True)
    return det

# ---------------- NIAH ----------------

def run_niah(n):
    filler = "\n".join(p["question"] for p in rows("test.jsonl"))
    det, rng = [], 20261031
    depths = [0.1, 0.3, 0.5, 0.7, 0.9][:n]
    prompts, codes = [], []
    for i, d in enumerate(depths):
        rng = (rng * 1103515245 + 12345) % (1 << 31)
        code = f"{rng % 90000 + 10000}"
        needle = f"The secret passcode for vault {i + 1} is {code}."
        target = 11000
        hay = filler[: int(target * 4.2)]
        cut_i = int(len(hay) * d)
        cut_i = hay.find("\n", cut_i) + 1
        ctx = hay[:cut_i] + "\n" + needle + "\n" + hay[cut_i:]
        ask = (f"Question: What is the secret passcode for vault {i + 1}? "
               "Answer with only the number.")
        if NIAH_TOK > 24:  # F17: reasoners otherwise enumerate the whole haystack
            ask += (" Find the passcode directly; do NOT list or enumerate the "
                    "context. Think briefly, then end with only the 5-digit number.")
        prompts.append(f"{ctx}\n\n{ask}")
        codes.append(code)
    outs = MNN.run(prompts, NIAH_TOK, "niah", SYSTEM)
    for i, (d, code, out) in enumerate(zip(depths, codes, outs)):
        if NIAH_TOK > 24:  # reasoning mode (F17): no cut — preamble precedes the number;
            nums5 = re.findall(r"\b\d{5}\b", out)   # codes are 5-digit, take the last
            got = nums5[-1] if nums5 else (re.findall(r"\d+", out) or [None])[-1]
            out = out[-60:]
        else:
            out = cut(out, ["\n", "."])
            nums = re.findall(r"\d+", out)
            got = nums[0] if nums else None
        det.append({"id": i, "depth": d, "code": code, "got": got,
                    "ok": got == code, "out": out[:60]})
        print(f"[niah {i+1}] depth={d} {'OK' if det[-1]['ok'] else 'MISS'} got={got} want={code}", flush=True)
    return det

# ---------------- PPL (ppl_eval, not llm_eval) ----------------

def run_ppl(n):
    batch = os.path.join(BATCH, "ppl")
    os.makedirs(batch, exist_ok=True)
    shutil.copy(os.path.join(BENCH, "gatsby.txt"), os.path.join(batch, "prompt.txt"))
    os.makedirs(TMP, exist_ok=True)
    cmd = ["taskset", "-c", "0-3", os.path.join(MNN_BUILD, "ppl_eval"), CONFIG, batch]
    if n > 0:
        cmd.append(str(n * 1000))        # --n in k-tokens for smoke runs, full text at n=0
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=TMP)
    m = re.search(r"Perplexity:\s*([0-9.]+)", r.stdout + r.stderr)
    if not m:
        sys.exit(f"ppl_eval produced no Perplexity line (rc={r.returncode}):\n"
                 f"{(r.stdout + r.stderr)[-400:]}")
    ppl = float(m.group(1))
    det = [{"id": 0, "ok": True, "ppl": ppl, "max_tokens": n * 1000 if n > 0 else "full"}]
    print(f"[ppl] gatsby perplexity = {ppl}", flush=True)
    return det

# ---------------- main ----------------

def detect_thinking(config):
    """True if the model's exported template speaks enable_thinking (Qwen3-style):
    then llm_eval gets --nothink, the sibling harness' '<think></think>' equivalent."""
    d = os.path.dirname(config)
    for f in ("tokenizer.txt", "llm_config.json", "config.json"):
        p = os.path.join(d, f)
        try:
            if "enable_thinking" in open(p, errors="ignore").read(4 << 20):
                return True
        except OSError:
            pass
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="model dir config.json")
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--tag", default="?")
    ap.add_argument("--engine", default="mnn")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--mode", choices=("chat", "base"), default="chat",
                    help="chat = engine applies the model's chat template "
                         "(llm_eval default); base = use_template:false raw prompts")
    ap.add_argument("--mc-tokens", type=int, default=16,
                    help="generation budget per MC item (F17: reasoning models need more)")
    ap.add_argument("--mc-instruction", default="",
                    help="instruction appended to MC prompts before the Answer: cue "
                         "(F17 R2: e.g. \"End your reply with a line 'Answer: X'.\")")
    ap.add_argument("--no-mc-shots", action="store_true",
                    help="drop the 3-shot prefix from MC prompts (F17 R2: reasoning "
                         "models continue the shot block instead of answering)")
    ap.add_argument("--he-tokens", type=int, default=512, help="humaneval budget")
    ap.add_argument("--hellaswag-tokens", type=int, default=0,
                    help="override MC budget for hellaswag only (0 = follow --mc-tokens; F19)")
    ap.add_argument("--gsm-tokens", type=int, default=700, help="gsm8k budget")
    ap.add_argument("--niah-tokens", type=int, default=24, help="niah budget")
    a = ap.parse_args()
    global CONFIG, MODE, NOTHINK, BATCH, ENGINE, MNN, MC_TOK, HE_TOK, GSM_TOK, MC_INSTR, MC_SHOTS_ON, NIAH_TOK, HS_TOK
    CONFIG, MODE, ENGINE = a.config, a.mode, a.engine
    MC_TOK, HE_TOK, GSM_TOK, NIAH_TOK = a.mc_tokens, a.he_tokens, a.gsm_tokens, a.niah_tokens
    HS_TOK = a.hellaswag_tokens or a.mc_tokens
    MC_INSTR = a.mc_instruction
    MC_SHOTS_ON = not a.no_mc_shots
    CONFIG = os.path.abspath(CONFIG)
    NOTHINK = detect_thinking(CONFIG)
    BATCH = os.path.join(SCRATCH, f"{a.tag}_{a.mode}")
    MNN = MnnRunner(CONFIG, MODE, NOTHINK)
    print(f"config={CONFIG} mode={MODE} nothink={NOTHINK} batch={BATCH}", flush=True)
    if a.task == "humaneval":
        a.n = min(a.n, 164)
    runners = {"gsm8k": run_gsm8k, "humaneval": run_humaneval, "niah": run_niah, "ppl": run_ppl,
               "arc": run_mc, "hellaswag": run_mc, "winogrande": run_mc,
               "openbookqa": run_mc, "commonsenseqa": run_mc, "mmlu": run_mc}
    if a.task not in runners:
        sys.exit(f"unknown task {a.task}; have {' '.join(sorted(runners))}")
    t0 = time.time()
    fn = runners[a.task]
    det = fn(a.n) if a.task not in ("arc", "hellaswag", "winogrande", "openbookqa",
                                    "commonsenseqa", "mmlu") else fn(a.task, a.n)
    score = sum(d["ok"] for d in det) / len(det)
    json.dump({"tag": a.tag, "engine": a.engine, "task": a.task, "n": len(det),
               "score": round(score, 4), "seconds": round(time.time() - t0, 1),
               "details": det}, open(a.out, "w"), indent=1)
    print(f"\n{a.task} {a.tag}/{a.engine}: {sum(d['ok'] for d in det)}/{len(det)} = {score:.1%} -> {a.out}")

if __name__ == "__main__":
    main()
