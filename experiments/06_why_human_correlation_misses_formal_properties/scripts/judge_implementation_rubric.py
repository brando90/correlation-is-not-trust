#!/usr/bin/env python3
"""Arm I of experiment 06 (2026-11-26). Not run by the CPU analysis.

Calls judge LLM Claude Opus 5.5 through Brando's plan router, on the
implementation rubric frozen in ../PREREG.md. Refuses to spend a plan
call unless --confirm-plan-spend is set. Refuses agy and Llama lanes.
Refuses to write prompts inside this git repository.

Dry-run (no router, no files written):

    python3 judge_implementation_rubric.py --dry-run --probes probes_min.jsonl

Real run, on skampere1, only after the PREREG is on main:

    python3 experiments/06_why_human_correlation_misses_formal_properties/scripts/judge_implementation_rubric.py \
      --probes ~/at3450/probes.jsonl \
      --lanes stanford-claude \
      --out ~/at3450/exp06_arm_I \
      --confirm-plan-spend
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

# Frozen in PREREG.md between RUBRIC_I_BEGIN and RUBRIC_I_END. Do not edit without an amendment.
RUBRIC_I = """Score the candidate Lean file as an implementation of the reference theorem, on a scale from 0.0 to 1.0.

The reference is the theorem statement. The candidate is the whole file, including definitions and proofs.

Give 1.0 only if the definitions denote the function the theorem is about and the theorem is the reference statement. Identifier names may differ when every occurrence is renamed consistently.

If a definition changes an operator, a constant, a comparison, or a bound so that the denoted function can differ on some input, the score must be at most 0.4, even when the theorem statement text matches the reference.

A consistent renaming of binders or of file-local definition names, with the same operators, must receive the same score as the unmodified file.

Ignore comments and whitespace. Do not give credit for a matching theorem statement when the definition implements a different function."""

PROPERTY_KINDS = ("ID1", "ID2", "RN_b", "RN_d", "BUG")
BANNED_LANE_TOKENS = ("agy", "llama")
ALL_LANES = (
    "vals-codex,vals-claude-opus,stanford-codex,stanford-claude,pi:stanford-stair-bm,"
    "sundial,pi:koyejolab-openai-bm,native:cursor,native:grok,pi:vals-openai-bm,"
    "personal-codex,personal-claude,pi:allen-google-veribench"
).split(",")
K = 6
SEED = 20261124
REPO_ROOT = Path(__file__).resolve().parents[3]


def validate_lanes(lanes: list[str]) -> list[str]:
    cleaned = [lane.strip() for lane in lanes]
    if not cleaned or any(not lane for lane in cleaned):
        raise SystemExit("refusing empty lane list")
    for lane in cleaned:
        low = lane.lower()
        for banned in BANNED_LANE_TOKENS:
            if banned in low:
                raise SystemExit(f"refusing banned lane {lane}")
    return cleaned


def assert_out_outside_repo(out: Path) -> None:
    resolved = out.resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise SystemExit(f"refusing to write prompts inside the git repo: {resolved}")


def keep_property_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("kind") in PROPERTY_KINDS]


def make_batches(rows: list[dict]) -> list[list[str]]:
    """Same batch rule as experiment 05: one probe per task per call, seed 20261124."""
    rng = random.Random(SEED)
    by_task = collections.defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row["probe_id"])
    for probes in by_task.values():
        rng.shuffle(probes)
    batches = []
    while any(by_task.values()):
        tasks = [task for task, probes in by_task.items() if probes]
        rng.shuffle(tasks)
        tasks.sort(key=lambda task: -len(by_task[task]))
        pick = tasks[:K]
        batches.append([by_task[task].pop() for task in pick])
    return batches


def build_prompt(batch: list[str], byid: dict) -> str:
    head = (
        f"You are an expert in Lean 4 formal verification. You will judge {len(batch)} independent items. "
        "Judge each item on its own and ignore the other items. "
        "Do not use tools and do not read files; answer directly.\n\n"
        "For each item apply this rubric:\n" + RUBRIC_I + "\n"
    )
    body = ""
    for i, pid in enumerate(batch, 1):
        row = byid[pid]
        body += (
            f"\n===== ITEM {i} =====\nREFERENCE IMPLEMENTATION:\n{row['reference']}\n\n"
            f"GENERATED THEOREM:\n{row['candidate']}\n"
        )
    tail = (
        "\n===== END OF ITEMS =====\nReturn JSON only (no markdown), a list with one object per item, in order: "
        '[{"item": 1, "score": <float 0.0-1.0>}, ...]'
    )
    return head + body + tail


def dry_run_report(rows: list[dict]) -> dict:
    kept = keep_property_rows(rows)
    byid = {row["probe_id"]: row for row in kept}
    batches = make_batches(kept) if kept else []
    prompt = build_prompt(batches[0], byid) if batches else ""
    return {
        "n_in": len(rows),
        "n_kept": len(kept),
        "kinds_kept": dict(collections.Counter(row["kind"] for row in kept)),
        "kinds_dropped": dict(collections.Counter(row.get("kind", "") for row in rows if row.get("kind") not in PROPERTY_KINDS)),
        "n_batches": len(batches),
        "batch_sizes": [len(batch) for batch in batches],
        "batch_tasks": [[byid[pid]["task"] for pid in batch] for batch in batches],
        "rubric_sha256": hashlib.sha256(RUBRIC_I.encode("utf-8")).hexdigest(),
        "prompt_contains_rubric": "score must be at most 0.4" in prompt,
        "prompt_contains_item_header": "GENERATED THEOREM" in prompt,
        "router_called": False,
    }


def parse_text(stdout: str) -> list[str]:
    texts = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            texts.append(item.get("text", ""))
        if event.get("type") == "result" and isinstance(event.get("result"), str):
            texts.append(event["result"])
        if event.get("type") == "assistant":
            for chunk in (event.get("message") or {}).get("content") or []:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    texts.append(chunk.get("text", ""))
    return texts


def parse_scores(texts: list[str], n_items: int) -> dict[int, float]:
    for text in reversed(texts):
        match = re.search(r"\[\s*\{.*\}\s*\]", text, re.S)
        if not match:
            continue
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        found = {}
        for obj in payload:
            try:
                index = int(obj["item"])
                score = float(obj["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if 1 <= index <= n_items and 0.0 <= score <= 1.0:
                found[index] = score
        if found:
            return found
    return {}


def token_counts(receipt) -> dict:
    raw = (receipt or {}).get("tokens") or {}
    out = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out[str(key)] = int(value)
    return out


def call_router(prompt_path: Path, workspace: Path, skip: str):
    proc = subprocess.run(
        ["plan-rr", "--experiment", "--prompt-file", str(prompt_path), "--workspace", str(workspace), "--skip", skip],
        capture_output=True,
        text=True,
        timeout=900,
    )
    combined = proc.stdout + proc.stderr
    ran = re.search(r"plan-rr: ran (\S+)", combined)
    receipt = None
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            try:
                receipt = json.loads(line)
            except json.JSONDecodeError:
                pass
    stdout = ""
    if receipt and receipt.get("logs"):
        stdout_path = Path(receipt["logs"]) / "stdout"
        if stdout_path.exists():
            stdout = stdout_path.read_text(encoding="utf-8")
    lane = ran.group(1) if ran else (receipt or {}).get("plan")
    model = (receipt or {}).get("model")
    return proc.returncode, lane, model, token_counts(receipt), stdout, combined[-400:]


def load_probes(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_scores_csv(path: Path, byid: dict, scores: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["probe_id", "kind", "task", "level", "score"])
        for pid in sorted(scores):
            row = byid[pid]
            writer.writerow([pid, row["kind"], row["task"], row["level"], scores[pid]])


def main():
    parser = argparse.ArgumentParser(description="Experiment 06 Arm I. Does not call the router without --confirm-plan-spend.")
    parser.add_argument("--probes", required=True)
    parser.add_argument("--lanes", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-plan-spend", action="store_true")
    parser.add_argument("--max-batches", type=int, default=10**9)
    args = parser.parse_args()
    rows = load_probes(Path(args.probes).expanduser())
    if args.dry_run:
        print(json.dumps(dry_run_report(rows), indent=2))
        return
    lanes = validate_lanes(args.lanes.split(","))
    if not args.confirm_plan_spend:
        raise SystemExit("refusing to call the plan router without --confirm-plan-spend")
    if not args.out:
        raise SystemExit("refusing to call the plan router without --out")
    out = Path(args.out).expanduser()
    assert_out_outside_repo(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ws").mkdir(parents=True, exist_ok=True)
    kept = keep_property_rows(rows)
    byid = {row["probe_id"]: row for row in kept}
    if len(byid) != len(kept):
        raise SystemExit("duplicate probe_id in property rows")
    batches = make_batches(kept)
    allowed = set(lanes)
    skip = ",".join(lane for lane in ALL_LANES if lane not in allowed)
    resf = out / "scores_I.jsonl"
    done = set()
    if resf.exists():
        for line in resf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["batch"])
    n_run = 0
    for bi, batch in enumerate(batches):
        if bi in done:
            continue
        if n_run >= args.max_batches:
            break
        prompt_path = out / f"prompt_I_{bi:03d}.md"
        prompt_path.write_text(build_prompt(batch, byid), encoding="utf-8")
        scores: dict[int, float] = {}
        meta = None
        for attempt in (1, 2):
            lane = None
            for _wait in range(60):
                rc, lane, model, tokens, stdout, tail = call_router(prompt_path, out / "ws", skip)
                if lane is not None:
                    break
                print("no route yet, waiting 30 s", file=sys.stderr)
                time.sleep(30)
            if lane is None:
                raise SystemExit("STOP no route after the wait loop")
            if any(banned in str(lane).lower() for banned in BANNED_LANE_TOKENS):
                raise SystemExit(f"refusing banned lane {lane}")
            if model and "llama" in str(model).lower():
                raise SystemExit(f"refusing banned model {model}")
            scores = parse_scores(parse_text(stdout), len(batch))
            meta = {"lane": lane, "model": model, "tokens": tokens, "attempt": attempt, "rc": rc}
            if len(scores) == len(batch):
                break
        record = {
            "batch": bi,
            "config": "I",
            "rubric_sha256": hashlib.sha256(RUBRIC_I.encode("utf-8")).hexdigest(),
            "probe_ids": batch,
            "scores": {batch[i - 1]: score for i, score in scores.items()},
            **meta,
        }
        with resf.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(bi, meta["lane"], meta["model"], f"{len(scores)}/{len(batch)}", flush=True)
        n_run += 1
        prompt_path.unlink()
    # Public-safe CSV: ids, kinds, scores. No Lean source.
    scored = {}
    if resf.exists():
        for line in resf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                scored.update(rec["scores"])
    write_scores_csv(out / "scores_I.csv", byid, scored)


if __name__ == "__main__":
    main()
