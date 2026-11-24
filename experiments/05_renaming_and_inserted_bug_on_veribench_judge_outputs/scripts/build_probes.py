#!/usr/bin/env python3
"""Build the experiment 05 probes (renamed golds, bug golds, identity, criterion items) with Lean validation.
Run on skampere1 (needs the Lean project and the cert-judge experiment 39 private probes). Writes a private
probes.jsonl (contains Lean source: never committed) and a public probes_public.json (counts, opaque ids, hashes).
Usage: python3 build_probes.py --out ~/at3450"""
import argparse, hashlib, json, re, sys
from pathlib import Path

E39 = Path("/lfs/skampere1/0/brando9/cert-judge_e39/experiments/39_label_free_proxy_index_train_val_test_over_judge_families")
VB = Path("/lfs/skampere1/0/brando9/veribench_e39/experiments/100_karpathy_recipe_tc_judge_search_human_train_val_test")
sys.path[:0] = [str(E39 / "scripts"), str(E39.parents[1] / "py_src")]
import probes as P                                    # noqa: E402
from cert_judge.probe_validation import LeanCompiler, mask_lean_noncode   # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
man = json.load(open(E39 / "results/probe_manifest.json"))
lc = LeanCompiler(man["lean"]["lake_dir"], man["lean"]["lake_bin"], timeout_seconds=300)
items = [json.loads(l) for l in open(VB / "inputs/items.jsonl") if l.strip()]
gold = {}
for r in items:
    gold[r["task"]] = (VB / r["ref_ann_path"]).read_text()
valid_tasks = sorted(r["task"] for r in man["rows"] if r["kind"] == "ident" and r["valid"])
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
rows = []
def add(kind, task, item, level, text, note, valid=True):
    rows.append(dict(probe_id=f"{kind}__{task}__{level}" + (f"__{item}" if item else ""), kind=kind, task=task, item=item,
                     level=level, candidate=text, candidate_sha256=sha(text), note=note, valid=valid))

for t in valid_tasks:
    g = gold[t]
    add("ID1", t, None, 0, g, "gold as candidate, first judging")
    add("ID2", t, None, 0, g, "gold as candidate, second judging")
# binder rename
alpha_status = {}
for t in valid_tasks:
    art = P.alpha(gold[t], validate=lambda c: lc.compile(c).ok)
    if art is not None:
        add("RN_b", t, None, 1, art.code, art.note)
    alpha_status[t] = art is not None
# declaration rename
DEF = re.compile(r"(?m)^(?:@\[[^\]\n]*\][ \t]*)?(?:private[ \t]+|protected[ \t]+|partial[ \t]+|noncomputable[ \t]+)*def[ \t]+([A-Za-z_][A-Za-z0-9_']*)")
def rename_defs(text, names, fresh):
    masked = mask_lean_noncode(text); code = text
    spans = []
    for n in names:
        for m in re.finditer(rf"(?<![A-Za-z0-9_'.]){re.escape(n)}(?![A-Za-z0-9_'])", masked):
            spans.append((m.start(), m.end(), fresh[n]))
    for s, e, f in sorted(spans, reverse=True):
        code = code[:s] + f + code[e:]
    return code
dstatus = {}
for t in valid_tasks:
    g = gold[t]; masked = mask_lean_noncode(g)
    names = list(dict.fromkeys(m.group(1) for m in DEF.finditer(masked)))
    names = [n for n in names if n != "_" ]
    done = None
    for k in (len(names), 1):
        if k == 0: break
        sel = names[:k]; fresh = {n: f"ptiFn{i+1}" for i, n in enumerate(sel)}
        if any(re.search(rf"\bptiFn{i+1}\b", g) for i in range(k)): break
        code = rename_defs(g, sel, fresh)
        if code != g and lc.compile(code).ok:
            done = (code, f"renamed {k} def(s): " + ", ".join(f"{n}->{fresh[n]}" for n in sel)); break
        if k == 1: break
    if done: add("RN_d", t, None, 1, done[0], done[1])
    dstatus[t] = bool(done)
# bug ladder (validated in experiment 39)
for r in man["rows"]:
    if r["kind"] == "bm_gold" and r["valid"] and r["task"] in valid_tasks:
        txt = Path(r["path"]).read_text()
        assert sha(txt) == r["sha256"], r["probe_id"]
        add("BUG", r["task"], None, r["level"], txt, r["note"])
# criterion items
for it in items:
    txt = (VB / it["candidate_path"]).read_text()
    rows.append(dict(probe_id=f"CRIT__{it['item']}", kind="CRIT", task=it["task"], item=it["item"], level=0, candidate=txt,
                     candidate_sha256=sha(txt), note="rated candidate", valid=True,
                     target_expert=float(it["target_expert"]), target_all5=float(it["target_all5"]), split=it["split"]))
import legacy                                           # noqa: E402
for r in rows:
    r["reference"] = legacy.last_theorem_block(gold[r["task"]])
with open(out / "probes.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
import collections
cnt = collections.Counter(r["kind"] for r in rows)
pub = dict(counts=dict(cnt), valid_tasks=len(valid_tasks), rn_b_tasks=sum(alpha_status.values()), rn_d_tasks=sum(dstatus.values()),
           probes=[dict(probe_id=r["probe_id"], kind=r["kind"], task=r["task"], level=r["level"], sha256=r["candidate_sha256"]) for r in rows])
json.dump(pub, open(out / "probes_public.json", "w"), indent=1)
print(dict(cnt), "valid golds", len(valid_tasks), "RN_b", sum(alpha_status.values()), "RN_d", sum(dstatus.values()))
