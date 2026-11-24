#!/usr/bin/env python3
"""Judge the experiment 05 probes in batches of 6 independent items through Brando's plan router (plan-rr).
Run on skampere1; foreground per batch; resumable (batches already in the results file are skipped).
Usage: python3 judge_batches.py --probes ~/at3450/probes.jsonl --config A --lanes vals-codex --out ~/at3450
       (--lanes = the only router lanes allowed; every other lane is passed to plan-rr --skip)"""
import argparse, collections, json, os, random, re, subprocess, sys, time
from pathlib import Path

ALL_LANES = "vals-codex,vals-claude-opus,stanford-codex,stanford-claude,pi:stanford-stair-bm,sundial,pi:koyejolab-openai-bm,native:cursor,native:grok,pi:vals-openai-bm,personal-codex,personal-claude,pi:allen-google-veribench".split(",")
PROMPT_DIR = Path("/lfs/skampere1/0/brando9/cert-judge_e39/experiments/39_label_free_proxy_index_train_val_test_over_judge_families/prompts")
RUBRIC_FILE = PROMPT_DIR / "P4_CHECKLIST.txt"   # default; --rubric P8_STRICT_B selects the other one
K, SEED = 6, 20261124

def make_batches(rows):
    rng = random.Random(SEED)
    by_task = collections.defaultdict(list)
    for r in rows: by_task[r["task"]].append(r["probe_id"])
    for v in by_task.values(): rng.shuffle(v)
    batches = []
    while any(by_task.values()):
        tasks = [t for t, v in by_task.items() if v]
        rng.shuffle(tasks); tasks.sort(key=lambda t: -len(by_task[t]))
        pick = tasks[:K]
        batches.append([by_task[t].pop() for t in pick])
    return batches

def build_prompt(batch, byid):
    rub = RUBRIC_FILE.read_text().split("REFERENCE IMPLEMENTATION:")[0].strip()
    head = (f"You are an expert in Lean 4 formal verification. You will judge {len(batch)} independent items. Judge each item on its own and ignore the other items. "
            "Do not use tools and do not read files; answer directly.\n\nFor each item apply this rubric:\n" + rub + "\n")
    body = ""
    for i, pid in enumerate(batch, 1):
        r = byid[pid]
        body += f"\n===== ITEM {i} =====\nREFERENCE IMPLEMENTATION:\n{r['reference']}\n\nGENERATED THEOREM:\n{r['candidate']}\n"
    tail = ('\n===== END OF ITEMS =====\nReturn JSON only (no markdown), a list with one object per item, in order: '
            '[{"item": 1, "score": <float 0.0-1.0>}, ...]')
    return head + body + tail

def parse_text(stdout):
    texts = []
    for line in stdout.splitlines():
        try: ev = json.loads(line)
        except Exception: continue
        if isinstance(ev, dict):
            it = ev.get("item")
            if isinstance(it, dict) and it.get("type") == "agent_message": texts.append(it.get("text", ""))
            if ev.get("type") == "result" and isinstance(ev.get("result"), str): texts.append(ev["result"])
            if ev.get("type") == "assistant":
                for c in (ev.get("message", {}) or {}).get("content", []) or []:
                    if isinstance(c, dict) and c.get("type") == "text": texts.append(c.get("text", ""))
    return texts

def parse_scores(texts, n):
    for t in reversed(texts):
        m = re.search(r"\[\s*\{.*\}\s*\]", t, re.S)
        if not m: continue
        try: arr = json.loads(m.group(0))
        except Exception: continue
        d = {}
        for o in arr:
            try:
                i = int(o["item"]); s = float(o["score"])
                if 1 <= i <= n and 0.0 <= s <= 1.0: d[i] = s
            except Exception: pass
        if d: return d
    return {}

def call(prompt_path, ws, skip):
    p = subprocess.run(["plan-rr", "--experiment", "--prompt-file", str(prompt_path), "--workspace", str(ws), "--skip", skip],
                       capture_output=True, text=True, timeout=900)
    ran = re.search(r"plan-rr: ran (\S+)", p.stdout + p.stderr)
    receipt = None
    for line in p.stdout.splitlines():
        if line.startswith("{"):
            try: receipt = json.loads(line)
            except Exception: pass
    stdout = ""
    if receipt and receipt.get("logs"):
        f = Path(receipt["logs"]) / "stdout"
        if f.exists(): stdout = f.read_text()
    lane = ran.group(1) if ran else (receipt or {}).get("plan")
    return p.returncode, lane, receipt, stdout, (p.stdout + p.stderr)[-400:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--lanes", required=True); ap.add_argument("--out", required=True); ap.add_argument("--max-batches", type=int, default=10**9); ap.add_argument("--rubric", default="P4_CHECKLIST")
    a = ap.parse_args()
    global RUBRIC_FILE; RUBRIC_FILE = PROMPT_DIR / f"{a.rubric}.txt"
    out = Path(a.out).expanduser(); ws = out / "ws"; ws.mkdir(parents=True, exist_ok=True)
    os.environ["PATH"] = "/dfs/scratch0/brando9/bin:" + os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"]
    rows = [json.loads(l) for l in open(Path(a.probes).expanduser())]
    byid = {r["probe_id"]: r for r in rows}
    batches = make_batches(rows)
    json.dump(batches, open(out / "batches.json", "w"))
    allowed = set(a.lanes.split(","))
    skip = ",".join(l for l in ALL_LANES if l not in allowed)
    resf = out / f"scores_{a.config}.jsonl"
    done = set()
    if resf.exists():
        for l in open(resf): done.add(json.loads(l)["batch"])
    n_run = 0
    for bi, batch in enumerate(batches):
        if bi in done: continue
        if n_run >= a.max_batches: break
        pp = out / f"prompt_{a.config}_{bi:03d}.md"
        pp.write_text(build_prompt(batch, byid))
        scores, meta = {}, None
        for attempt in (1, 2):
            for wait_try in range(60):          # the lane is shared with Brando's other agents: wait for it, do not fall to another lane
                rc, lane, receipt, stdout, tail = call(pp, ws, skip)
                if lane is not None: break
                print("no route yet, waiting 30 s:", rc, tail[-160:].replace("\n", " "), flush=True); time.sleep(30)
            if lane is None:
                print("STOP no route after 30 min", flush=True); sys.exit(3)
            scores = parse_scores(parse_text(stdout), len(batch))
            meta = dict(lane=lane, model=(receipt or {}).get("model"), tokens=(receipt or {}).get("tokens"), attempt=attempt, rc=rc)
            if len(scores) == len(batch): break
        rec = dict(batch=bi, config=a.config, probe_ids=batch, scores={batch[i-1]: s for i, s in scores.items()}, **meta)
        with open(resf, "a") as f: f.write(json.dumps(rec) + "\n")
        print(bi, meta["lane"], meta["model"], f"{len(scores)}/{len(batch)}", flush=True)
        n_run += 1
        pp.unlink()

main()
