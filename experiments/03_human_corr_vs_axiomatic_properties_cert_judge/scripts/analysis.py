#!/usr/bin/env python3
"""Reanalyze cached cert-judge E39/E42 judge scores and Lean 4 human labels (2026-11-24).

Zero model calls. Outputs aggregate tables, alignment verification, H1/H2 verdicts,
and matplotlib figures (with grid lines) to results/ without writing raw human labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DEFAULT_E39_DIR = Path(
    "/Users/brandomiranda/cert-judge/experiments/"
    "39_label_free_proxy_index_train_val_test_over_judge_families/results"
)
DEFAULT_E42_DIR = Path(
    "/Users/brandomiranda/cert-judge/experiments/"
    "42_semantic_weakening_overconstraint_vacuity_probes_on_e39_judges/results"
)
DEFAULT_HUMAN_DIR = Path("/Users/brandomiranda/cert-judge/experiments/human_labels")
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "results"

SEED = 20261124
BOOT_DRAWS = 2000
PERM_DRAWS = 5000
MIN_CRITERION_COVERAGE = 0.90
MIN_COMPONENT_COVERAGE = 0.80
SCORED_STATUSES = ("measured", "deterministic")
TOL = 1e-12

FIXED_POOL_FILES = {
    "P1_correct": "participant1_data_correct_prompt.tsv",
    "P2": "participant2_data_pass_1.tsv",
    "P3": "participant3_data_pass_1.tsv",
    "P4": "participant4_data_pass_1.tsv",
    "P5": "participant5_data_pass_1.tsv",
}
WITHIN_HUMAN_FILES = {
    "P1_correct": "participant1_data_correct_prompt.tsv",
    "P1_old_pass1": "old_prompt/participant1_data_pass_1.tsv",
    "P1_old_pass2": "old_prompt/participant1_data_pass_2.tsv",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def spearman_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    val = float(stats.spearmanr(xs, ys).statistic)
    return val if math.isfinite(val) else None


def pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    val = float(stats.pearsonr(xs, ys).statistic)
    return val if math.isfinite(val) else None


def cluster_mean_ci(
    pairs: list[tuple[str, float]], draws: int = BOOT_DRAWS, seed: int = SEED
) -> dict[str, Any]:
    """Mean and 95% percentile task-cluster bootstrap CI."""
    if not pairs:
        return {"value": None, "ci_lo": None, "ci_hi": None, "n": 0, "n_tasks": 0}
    groups: dict[str, list[float]] = {}
    for task, val in pairs:
        groups.setdefault(task, []).append(float(val))
    keys = sorted(groups)
    point = sum(v for _, v in pairs) / len(pairs)
    rng = random.Random(seed)
    boot = []
    for _ in range(draws):
        sampled = [v for k in rng.choices(keys, k=len(keys)) for v in groups[k]]
        boot.append(sum(sampled) / len(sampled))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "value": float(point),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": len(pairs),
        "n_tasks": len(keys),
    }


def unit_corr_ci_perm(
    xs: list[float], ys: list[float], boot_draws: int = BOOT_DRAWS, perm_draws: int = PERM_DRAWS, seed: int = SEED
) -> dict[str, Any]:
    """Spearman rho across configurations with unit bootstrap 95% CI and two-sided permutation p-val."""
    n = len(xs)
    obs = spearman_corr(xs, ys)
    if obs is None or n < 3:
        return {"n": n, "spearman": None, "ci_lo": None, "ci_hi": None, "p_val": None, "valid_draws": 0}
    rng_b = random.Random(seed)
    boot = []
    for _ in range(boot_draws):
        idx = rng_b.choices(range(n), k=n)
        val = spearman_corr([xs[i] for i in idx], [ys[i] for i in idx])
        if val is not None:
            boot.append(val)
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (None, None))
    rng_p = random.Random(seed)
    ys_perm = list(ys)
    ge_count = 0
    valid_perm = 0
    for _ in range(perm_draws):
        rng_p.shuffle(ys_perm)
        val = spearman_corr(xs, ys_perm)
        if val is not None:
            valid_perm += 1
            if abs(val) >= abs(obs) - TOL:
                ge_count += 1
    p_val = (ge_count + 1) / (valid_perm + 1) if valid_perm > 0 else None
    return {
        "n": n,
        "spearman": float(obs),
        "ci_lo": float(lo) if lo is not None else None,
        "ci_hi": float(hi) if hi is not None else None,
        "p_val": float(p_val) if p_val is not None else None,
        "valid_draws": len(boot),
    }


def aggregate_harvest_scores(
    harvest: dict[str, Any], min_repeats_by_kind: dict[str, int], default_min_repeats: int = 2
) -> tuple[dict[tuple[str, str, int], float], dict[tuple[str, str, int], list[float]]]:
    """Group harvest cells into scored repeats and medians by (kind, key, level)."""
    raw_by_cell: dict[tuple[str, str, int], list[float]] = {}
    for c in harvest["cells"]:
        if c.get("status") in SCORED_STATUSES and c.get("score") is not None:
            key_str = c.get("key") if c.get("key") is not None else c["task"]
            raw_by_cell.setdefault((c["kind"], str(key_str), int(c["level"])), []).append(float(c["score"]))
    medians: dict[tuple[str, str, int], float] = {}
    for cell_id, vals in raw_by_cell.items():
        req = min_repeats_by_kind.get(cell_id[0], default_min_repeats)
        if len(vals) >= req:
            medians[cell_id] = float(statistics.median(vals))
    return medians, raw_by_cell


def summarize_paired_violation(
    rows: list[dict[str, Any]],
    scores: dict[tuple[str, str, int], float],
    base_lookup: Any,
    probe_lookup: Any,
    mode: str,
    min_coverage: float = MIN_COMPONENT_COVERAGE,
) -> dict[str, Any]:
    """Compute violation rate (`>=` or `abs_diff`) and task-cluster bootstrap 95% CI."""
    declared = len(rows)
    viol_pairs: list[tuple[str, float]] = []
    strict_pairs: list[tuple[str, float]] = []
    tie_pairs: list[tuple[str, float]] = []
    abs_gap_pairs: list[tuple[str, float]] = []
    raw_probe_pairs: list[tuple[str, float]] = []

    for r in rows:
        base_score = scores.get(base_lookup(r))
        probe_score = scores.get(probe_lookup(r))
        if base_score is None or probe_score is None:
            continue
        task = str(r["task"])
        raw_probe_pairs.append((task, probe_score))
        gap = abs(probe_score - base_score)
        abs_gap_pairs.append((task, gap))
        if mode == "ge_base":
            viol = 0.0 if probe_score < base_score else 1.0
            strict = 1.0 if probe_score > base_score + TOL else 0.0
            tie = 1.0 if (not (probe_score < base_score) and probe_score <= base_score + TOL) else 0.0
            viol_pairs.append((task, viol))
            strict_pairs.append((task, strict))
            tie_pairs.append((task, tie))
        elif mode == "abs_diff":
            viol = 1.0 if gap > TOL else 0.0
            viol_pairs.append((task, viol))
        else:
            raise ValueError(f"Unknown comparison mode: {mode}")

    coverage = len(viol_pairs) / declared if declared else 0.0
    computable = coverage + 1e-9 >= min_coverage and declared > 0
    ci = cluster_mean_ci(viol_pairs) if viol_pairs else cluster_mean_ci([])
    out: dict[str, Any] = {
        "value": ci["value"] if computable else None,
        "raw_value": ci["value"],
        "ci_lo": ci["ci_lo"] if computable else None,
        "ci_hi": ci["ci_hi"] if computable else None,
        "n_scored": len(viol_pairs),
        "n_declared_valid": declared,
        "n_tasks": ci["n_tasks"],
        "coverage": round(coverage, 6),
        "computable": computable,
    }
    if mode == "ge_base":
        strict_ci = cluster_mean_ci(strict_pairs)
        tie_ci = cluster_mean_ci(tie_pairs)
        out["strict_gt"] = strict_ci["value"] if computable else None
        out["tie_eq"] = tie_ci["value"] if computable else None
    elif mode == "abs_diff":
        gap_ci = cluster_mean_ci(abs_gap_pairs)
        out["mean_abs_gap"] = gap_ci["value"] if computable else None
        out["gap_ci_lo"] = gap_ci["ci_lo"] if computable else None
        out["gap_ci_hi"] = gap_ci["ci_hi"] if computable else None
    out["mean_probe_score"] = cluster_mean_ci(raw_probe_pairs)["value"] if raw_probe_pairs else None
    return out


def summarize_repeat_inconsistency(
    rows: list[dict[str, Any]], raw_by_cell: dict[tuple[str, str, int], list[float]], min_repeats: int = 2
) -> dict[str, Any]:
    """Compute repeat inconsistency rate (`max - min > 1e-12`), range, and pop SD across items."""
    declared = len(rows)
    inc_pairs: list[tuple[str, float]] = []
    range_pairs: list[tuple[str, float]] = []
    sd_pairs: list[tuple[str, float]] = []
    for r in rows:
        key_str = str(r.get("key") if r.get("key") is not None else r["task"])
        reps = raw_by_cell.get((r["kind"], key_str, int(r["level"])), [])
        if len(reps) < min_repeats:
            continue
        task = str(r["task"])
        rng_val = max(reps) - min(reps)
        mean_val = sum(reps) / len(reps)
        sd_val = math.sqrt(sum((v - mean_val) ** 2 for v in reps) / len(reps))
        inc_pairs.append((task, 1.0 if rng_val > TOL else 0.0))
        range_pairs.append((task, rng_val))
        sd_pairs.append((task, sd_val))
    coverage = len(inc_pairs) / declared if declared else 0.0
    computable = coverage + 1e-9 >= MIN_COMPONENT_COVERAGE and declared > 0
    ci = cluster_mean_ci(inc_pairs)
    rng_ci = cluster_mean_ci(range_pairs)
    sd_ci = cluster_mean_ci(sd_pairs)
    return {
        "value": ci["value"] if computable else None,
        "ci_lo": ci["ci_lo"] if computable else None,
        "ci_hi": ci["ci_hi"] if computable else None,
        "mean_range": rng_ci["value"] if computable else None,
        "range_ci_lo": rng_ci["ci_lo"] if computable else None,
        "range_ci_hi": rng_ci["ci_hi"] if computable else None,
        "mean_sd_pop": sd_ci["value"] if computable else None,
        "n_scored": len(inc_pairs),
        "n_declared_valid": declared,
        "n_tasks": ci["n_tasks"],
        "coverage": round(coverage, 6),
        "computable": computable,
    }


def analyze_judges(e39_dir: Path, e42_dir: Path) -> dict[str, Any]:
    panel = read_json(e39_dir / "panel.json")
    units_39 = read_json(e39_dir / "units_train_val.json")["units"] + read_json(e39_dir / "units_test.json")["units"]
    u39_by_cfg = {u["config"]: u for u in units_39}

    pm39 = [r for r in read_json(e39_dir / "probe_manifest.json")["rows"] if r["valid"] is True]
    rows39_by_kind: dict[str, list[dict[str, Any]]] = {}
    for r in pm39:
        rows39_by_kind.setdefault(r["kind"], []).append(r)

    pm42 = [r for r in read_json(e42_dir / "semantic_probe_manifest.json")["rows"] if r["valid"] is True]
    rows42_by_fam: dict[str, list[dict[str, Any]]] = {}
    for r in pm42:
        rows42_by_fam.setdefault(r["family"], []).append(r)

    screen42 = read_json(e42_dir / "semantic_screen.json")
    s42_by_cfg = {u["config"]: u for u in screen42["units"]}
    if set(u39_by_cfg) != set(s42_by_cfg):
        raise RuntimeError("Mismatch in unit configs between E39 and E42.")

    min_repeats_39 = {"ident": 2, "bm_gold": 2, "crit": 2, "ms_gold": 1, "blank": 1, "alpha": 1}
    min_repeats_42 = {k: 2 for k in ("ident", "weak", "vac_true", "vac_hyp", "overc", "fmt_comments", "fmt_layout")}

    max_align_diff = 0.0
    align_checks = 0
    records = []

    for cfg in sorted(u39_by_cfg):
        u39 = u39_by_cfg[cfg]
        u42 = s42_by_cfg[cfg]
        fname = cfg.replace("|", "__") + ".json"
        h39 = read_json(e39_dir / "harvest" / fname)
        h42 = read_json(e42_dir / "harvest" / fname)

        scores39, raw39 = aggregate_harvest_scores(h39, min_repeats_39, default_min_repeats=2)
        scores42, _ = aggregate_harvest_scores(h42, min_repeats_42, default_min_repeats=2)

        family = u39["family"]
        model_info = panel["models"].get(family, {})
        model_declared = model_info.get("selectors", [cfg])[0] if u39["kind"] == "llm" else cfg
        model_served = u42.get("selector") or h39.get("selector") or model_declared
        prompt = cfg.split("|", 1)[1] if "|" in cfg else ""

        crit_held = u39["criteria"]["heldout"]
        crit_all = u39["criteria"]["all_items"]

        # E39 properties
        v_bug = summarize_paired_violation(
            rows39_by_kind["bm_gold"],
            scores39,
            base_lookup=lambda r: ("ident", str(r["task"]), 0),
            probe_lookup=lambda r: ("bm_gold", str(r["task"]), int(r["level"])),
            mode="ge_base",
        )
        v_ms = summarize_paired_violation(
            rows39_by_kind["ms_gold"],
            scores39,
            base_lookup=lambda r: ("ident", str(r["task"]), 0),
            probe_lookup=lambda r: ("ms_gold", str(r["task"]), int(r["level"])),
            mode="ge_base",
        )
        # Pair blank only against tasks with valid ident (24/25 tasks).
        valid_ident_tasks = {str(r["task"]) for r in rows39_by_kind["ident"]}
        blank_paired_rows = [r for r in rows39_by_kind["blank"] if str(r["task"]) in valid_ident_tasks]
        v_blank = summarize_paired_violation(
            blank_paired_rows,
            scores39,
            base_lookup=lambda r: ("ident", str(r["task"]), 0),
            probe_lookup=lambda r: ("blank", str(r["task"]), 0),
            mode="ge_base",
        )
        blank_vals_all = [
            scores39[("blank", str(r["task"]), 0)]
            for r in rows39_by_kind["blank"]
            if ("blank", str(r["task"]), 0) in scores39
        ]
        v_blank["mean_all_valid_blank"] = sum(blank_vals_all) / len(blank_vals_all) if blank_vals_all else None

        v_rep = summarize_repeat_inconsistency(rows39_by_kind["crit"], raw39, min_repeats=2)
        v_rep_ident = summarize_repeat_inconsistency(rows39_by_kind["ident"], raw39, min_repeats=2)
        v_alpha = summarize_paired_violation(
            rows39_by_kind["alpha"],
            scores39,
            base_lookup=lambda r: ("crit", str(r["item"]), 0),
            probe_lookup=lambda r: ("alpha", str(r["item"]), int(r["level"])),
            mode="abs_diff",
            min_coverage=0.0,
        )

        # E42 properties
        v_weak = summarize_paired_violation(
            rows42_by_fam["weakening"],
            scores42,
            base_lookup=lambda r: ("ident", str(r["task"]), 1),
            probe_lookup=lambda r: (r["kind"], str(r["task"]), int(r["level"])),
            mode="ge_base",
        )
        v_vac = summarize_paired_violation(
            rows42_by_fam["vacuity"],
            scores42,
            base_lookup=lambda r: ("ident", str(r["task"]), 1),
            probe_lookup=lambda r: (r["kind"], str(r["task"]), int(r["level"])),
            mode="ge_base",
        )
        v_overc = summarize_paired_violation(
            rows42_by_fam["overconstraint"],
            scores42,
            base_lookup=lambda r: ("ident", str(r["task"]), 1),
            probe_lookup=lambda r: (r["kind"], str(r["task"]), int(r["level"])),
            mode="ge_base",
        )
        v_fmt = summarize_paired_violation(
            rows42_by_fam["formatting"],
            scores42,
            base_lookup=lambda r: ("ident", str(r["task"]), 1),
            probe_lookup=lambda r: (r["kind"], str(r["task"]), int(r["level"])),
            mode="abs_diff",
        )

        # Strict alignment verification against E42/results/semantic_screen.json
        for fam, prop in (("weakening", v_weak), ("vacuity", v_vac), ("overconstraint", v_overc)):
            m42 = u42["metrics"][fam]
            if prop["computable"] != m42["computable"]:
                raise RuntimeError(f"Computable mismatch for {cfg} {fam}: {prop['computable']} vs {m42['computable']}")
            if prop["computable"]:
                recomputed_acc = 1.0 - prop["value"]
                diff = abs(recomputed_acc - m42["value"])
                max_align_diff = max(max_align_diff, diff)
                align_checks += 1
                if diff > 1e-9 or prop["n_scored"] != m42["n"]:
                    raise RuntimeError(f"Alignment mismatch for {cfg} {fam}: diff={diff}")
        m42_fmt = u42["metrics"]["formatting"]
        if v_fmt["computable"] != m42_fmt["computable"]:
            raise RuntimeError(f"Computable mismatch for {cfg} formatting")
        if v_fmt["computable"]:
            diff = abs(v_fmt["mean_abs_gap"] - m42_fmt["value"])
            max_align_diff = max(max_align_diff, diff)
            align_checks += 1
            if diff > 1e-9 or v_fmt["n_scored"] != m42_fmt["n"]:
                raise RuntimeError(f"Alignment mismatch for {cfg} formatting: diff={diff}")

        records.append(
            {
                "config": cfg,
                "kind": u39["kind"],
                "harness": u39["harness"],
                "family": family,
                "prompt": prompt,
                "split": u39["split"],
                "model_declared": model_declared,
                "model_served": model_served,
                "rho_heldout": crit_held["spearman"],
                "rho_heldout_ci_lo": crit_held["ci"]["lo"] if crit_held["ci"] else None,
                "rho_heldout_ci_hi": crit_held["ci"]["hi"] if crit_held["ci"] else None,
                "rho_heldout_pearson": crit_held["pearson"],
                "rho_heldout_kendall": crit_held["kendall"],
                "rho_heldout_coverage": crit_held["coverage"],
                "rho_heldout_n_items": crit_held["n_items"],
                "rho_heldout_n_declared": crit_held["n_declared"],
                "rho_heldout_status": crit_held["status"],
                "rho_all_items": crit_all["spearman"],
                "rho_all_items_ci_lo": crit_all["ci"]["lo"] if crit_all["ci"] else None,
                "rho_all_items_ci_hi": crit_all["ci"]["hi"] if crit_all["ci"] else None,
                "V_bug": v_bug,
                "V_ms": v_ms,
                "V_weak": v_weak,
                "V_vac": v_vac,
                "V_overc": v_overc,
                "V_fmt": v_fmt,
                "V_rep": v_rep,
                "V_rep_ident": v_rep_ident,
                "V_blank": v_blank,
                "V_alpha_desc": v_alpha,
            }
        )

    # Cross-judge associations across real LLM/agent configurations (n <= 56)
    llm_units = [r for r in records if r["kind"] == "llm" and r["rho_heldout"] is not None]
    cross_judge = {}
    prop_extractors = {
        "V_bug": lambda r: r["V_bug"]["value"],
        "V_ms": lambda r: r["V_ms"]["value"],
        "V_weak": lambda r: r["V_weak"]["value"],
        "V_vac": lambda r: r["V_vac"]["value"],
        "V_overc": lambda r: r["V_overc"]["value"],
        "V_fmt": lambda r: r["V_fmt"]["value"],
        "fmt_mean_gap": lambda r: r["V_fmt"]["mean_abs_gap"] if r["V_fmt"]["computable"] else None,
        "V_rep": lambda r: r["V_rep"]["value"],
        "V_blank": lambda r: r["V_blank"]["value"],
    }
    for pname, fn in prop_extractors.items():
        paired = [(r["rho_heldout"], fn(r)) for r in llm_units if fn(r) is not None]
        cross_judge[pname] = unit_corr_ci_perm([a for a, _ in paired], [b for _, b in paired])

    # Evaluate preregistered H1 rule
    high_corr_llms = [
        r for r in llm_units if r["rho_heldout_coverage"] + 1e-9 >= MIN_CRITERION_COVERAGE and r["rho_heldout"] >= 0.70
    ]
    failing_high_corr = []
    failing_high_corr_no_vbug = []
    for r in high_corr_llms:
        triggers = []
        for pname in ("V_bug", "V_weak", "V_rep"):
            p = r[pname]
            if p["computable"] and p["value"] >= 0.20 and p["ci_lo"] > 0.05:
                triggers.append(pname)
        if triggers:
            failing_high_corr.append({"config": r["config"], "rho_heldout": r["rho_heldout"], "triggers": triggers})
        no_vbug_triggers = [t for t in triggers if t != "V_bug"]
        if no_vbug_triggers:
            failing_high_corr_no_vbug.append(
                {"config": r["config"], "rho_heldout": r["rho_heldout"], "triggers": no_vbug_triggers}
            )

    all_clean = len(high_corr_llms) > 0
    for r in high_corr_llms:
        for pname, thresh in (("V_bug", 0.10), ("V_ms", 0.10), ("V_weak", 0.10), ("V_vac", 0.10), ("V_rep", 0.10)):
            p = r[pname]
            if not p["computable"] or p["value"] >= thresh:
                all_clean = False
        if not r["V_fmt"]["computable"] or r["V_fmt"]["mean_abs_gap"] > 0.05:
            all_clean = False

    h1_verdict = "SUPPORTED" if failing_high_corr else ("FALSIFIED" if all_clean else "MIXED")

    return {
        "alignment_check": {"checks": align_checks, "max_abs_diff": max_align_diff, "passed": max_align_diff <= 1e-9},
        "n_units": len(records),
        "n_llm_units": len([r for r in records if r["kind"] == "llm"]),
        "n_control_units": len([r for r in records if r["kind"] == "control"]),
        "n_high_corr_llms_ge_0_70": len(high_corr_llms),
        "failing_high_corr_llms": failing_high_corr,
        "n_failing_high_corr_excluding_v_bug": len(failing_high_corr_no_vbug),
        "failing_high_corr_excluding_v_bug": failing_high_corr_no_vbug,
        "h1_verdict": h1_verdict,
        "cross_judge_spearman": cross_judge,
        "units": records,
    }


def load_tsv_ratings(path: Path) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            raw = (row.get("human_score_0_5") or "").strip()
            if raw == "" or raw.upper() == "NA" or raw.lower() == "nan":
                continue
            out[(row["gold_file"].strip(), row["candidate_file"].strip())] = float(raw)
    return out


def compute_pair_stats_from_keys(
    keys: list[tuple[str, str]], map_a: dict[tuple[str, str], float], map_b: dict[tuple[str, str], float]
) -> dict[str, Any]:
    xs, ys, tasks = [], [], []
    for k in keys:
        if k in map_a and k in map_b:
            xs.append(map_a[k])
            ys.append(map_b[k])
            tasks.append(k[0])
    if not xs:
        return {"n": 0, "n_tasks": 0}
    diffs = [b - a for a, b in zip(xs, ys)]
    abs_diffs = [abs(d) for d in diffs]
    return {
        "n": len(xs),
        "n_tasks": len(set(tasks)),
        "spearman": spearman_corr(xs, ys),
        "pearson": pearson_corr(xs, ys),
        "exact_agree": sum(1 for d in abs_diffs if d <= TOL) / len(abs_diffs),
        "disagree_ge_1": sum(1 for d in abs_diffs if d >= 1.0 - TOL) / len(abs_diffs),
        "disagree_ge_2": sum(1 for d in abs_diffs if d >= 2.0 - TOL) / len(abs_diffs),
        "mad": sum(abs_diffs) / len(abs_diffs),
        "mean_signed_diff": sum(diffs) / len(diffs),
    }


def bootstrap_pair_metrics(
    all_items: list[tuple[str, str]],
    map_a: dict[tuple[str, str], float],
    map_b: dict[tuple[str, str], float],
    draws: int = BOOT_DRAWS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compute pair metrics with 95% task-cluster bootstrap intervals over observed paired tasks."""
    paired_keys = [k for k in all_items if k in map_a and k in map_b]
    point = compute_pair_stats_from_keys(paired_keys, map_a, map_b)
    by_task: dict[str, list[tuple[str, str]]] = {}
    for k in paired_keys:
        by_task.setdefault(k[0], []).append(k)
    task_names = sorted(by_task)
    rng = random.Random(seed)
    metrics_boot: dict[str, list[float]] = {
        m: [] for m in ("spearman", "pearson", "exact_agree", "disagree_ge_1", "disagree_ge_2", "mad", "mean_signed_diff")
    }
    if task_names:
        for _ in range(draws):
            sample_keys = [it for t in rng.choices(task_names, k=len(task_names)) for it in by_task[t]]
            st = compute_pair_stats_from_keys(sample_keys, map_a, map_b)
            for m in metrics_boot:
                val = st.get(m)
                if val is not None:
                    metrics_boot[m].append(val)
    res: dict[str, Any] = {"n": point["n"], "n_tasks": point["n_tasks"]}
    for m, vals in metrics_boot.items():
        lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (None, None))
        res[m] = {
            "value": point.get(m),
            "ci_lo": float(lo) if lo is not None else None,
            "ci_hi": float(hi) if hi is not None else None,
        }
    return res


def compute_item_spread_stats(
    all_items: list[tuple[str, str]],
    raters: dict[str, dict[tuple[str, str], float]],
    min_raters: int = 2,
    draws: int = BOOT_DRAWS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compute per-item cross-rater range and SD distributions with task-cluster CIs."""
    by_task: dict[str, list[tuple[str, str]]] = {}
    item_vals: dict[tuple[str, str], list[float]] = {}
    for k in all_items:
        vals = [raters[r][k] for r in raters if k in raters[r]]
        if len(vals) >= min_raters:
            item_vals[k] = vals
            by_task.setdefault(k[0], []).append(k)

    def summarize_keys(keys: list[tuple[str, str]]) -> dict[str, float] | None:
        if not keys:
            return None
        ranges = [max(item_vals[k]) - min(item_vals[k]) for k in keys]
        sds = []
        for k in keys:
            vs = item_vals[k]
            m = sum(vs) / len(vs)
            sds.append(math.sqrt(sum((v - m) ** 2 for v in vs) / len(vs)))
        n = len(keys)
        return {
            "mean_range": sum(ranges) / n,
            "mean_sd_pop": sum(sds) / n,
            "range_eq_0": sum(1 for r in ranges if r <= TOL) / n,
            "range_ge_1": sum(1 for r in ranges if r >= 1.0 - TOL) / n,
            "range_ge_2": sum(1 for r in ranges if r >= 2.0 - TOL) / n,
            "range_ge_3": sum(1 for r in ranges if r >= 3.0 - TOL) / n,
        }

    eligible_keys = [k for k in all_items if k in item_vals]
    point = summarize_keys(eligible_keys)
    assert point is not None
    hist_counts = {str(d): sum(1 for k in eligible_keys if int(round(max(item_vals[k]) - min(item_vals[k]))) == d) for d in range(6)}

    task_names = sorted(by_task)
    rng = random.Random(seed)
    boot: dict[str, list[float]] = {m: [] for m in point}
    for _ in range(draws):
        sample_keys = [it for t in rng.choices(task_names, k=len(task_names)) for it in by_task[t]]
        st = summarize_keys(sample_keys)
        if st is None:
            continue
        for m in boot:
            boot[m].append(st[m])

    out: dict[str, Any] = {
        "n_items": len(eligible_keys),
        "n_tasks": len(task_names),
        "min_raters_required": min_raters,
        "range_histogram_counts": hist_counts,
    }
    for m, vals in boot.items():
        lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (None, None))
        out[m] = {
            "value": float(point[m]),
            "ci_lo": float(lo) if lo is not None else None,
            "ci_hi": float(hi) if hi is not None else None,
        }
    return out


def analyze_humans(human_dir: Path) -> dict[str, Any]:
    pool = {name: load_tsv_ratings(human_dir / rel) for name, rel in FIXED_POOL_FILES.items()}
    within = {name: load_tsv_ratings(human_dir / rel) for name, rel in WITHIN_HUMAN_FILES.items()}
    all_items = sorted({k for m in (*pool.values(), *within.values()) for k in m})
    pool_items = [k for k in all_items if sum(1 for r in pool if k in pool[r]) >= 2]
    pool_by_task: dict[str, list[tuple[str, str]]] = {}
    for k in pool_items:
        pool_by_task.setdefault(k[0], []).append(k)
    pool_task_names = sorted(pool_by_task)
    all_task_names = sorted({k[0] for k in all_items})

    # 1. All 10 unordered pairs in fixed-rubric 5-rater pool
    rater_names = list(pool.keys())
    pair_results = {}
    pair_tuples = [(rater_names[i], rater_names[j]) for i in range(len(rater_names)) for j in range(i + 1, len(rater_names))]
    for ra, rb in pair_tuples:
        pair_results[f"{ra}__vs__{rb}"] = bootstrap_pair_metrics(pool_items, pool[ra], pool[rb])

    # Task-cluster bootstrap for mean across the 10 pairs (resampling observed 25 pool tasks)
    def mean_over_pairs(keys: list[tuple[str, str]]) -> dict[str, float | None]:
        per = [compute_pair_stats_from_keys(keys, pool[ra], pool[rb]) for ra, rb in pair_tuples]
        out_m: dict[str, float | None] = {}
        for m in ("spearman", "pearson", "exact_agree", "disagree_ge_1", "disagree_ge_2", "mad"):
            vals = [p[m] for p in per if p.get(m) is not None]
            out_m[m] = (sum(vals) / len(vals)) if vals else None
        return out_m

    point_10 = mean_over_pairs(pool_items)
    rng = random.Random(SEED)
    boot_10: dict[str, list[float]] = {m: [] for m in point_10}
    for _ in range(BOOT_DRAWS):
        sample_keys = [it for t in rng.choices(pool_task_names, k=len(pool_task_names)) for it in pool_by_task[t]]
        st = mean_over_pairs(sample_keys)
        for m in boot_10:
            if st[m] is not None:
                boot_10[m].append(st[m])
    summary_10_pairs = {}
    for m, vals in boot_10.items():
        lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (None, None))
        summary_10_pairs[m] = {
            "value": float(point_10[m]) if point_10[m] is not None else None,
            "ci_lo": float(lo) if lo is not None else None,
            "ci_hi": float(hi) if hi is not None else None,
        }

    # 2. Per-item spread across all >=2 rater items + sensitivity subsets (all 5 raters; 4 full raters)
    spread_all = compute_item_spread_stats(pool_items, pool, min_raters=2)
    spread_complete5 = compute_item_spread_stats(pool_items, pool, min_raters=5)
    pool4 = {k: v for k, v in pool.items() if k != "P3"}
    spread_4raters = compute_item_spread_stats(pool_items, pool4, min_raters=4)

    # 3. Leave-one-rater-out (LORO)
    loro = {}
    for r in rater_names:
        rest_mean_map = {}
        for k in pool_items:
            if k not in pool[r]:
                continue
            others = [pool[o][k] for o in rater_names if o != r and k in pool[o]]
            if others:
                rest_mean_map[k] = sum(others) / len(others)
        loro[r] = bootstrap_pair_metrics(pool_items, pool[r], rest_mean_map)

    def loro_mean(keys: list[tuple[str, str]]) -> dict[str, float | None]:
        sp_vals, mae_vals = [], []
        for r in rater_names:
            rmap = {}
            for k in keys:
                if k in pool[r]:
                    others = [pool[o][k] for o in rater_names if o != r and k in pool[o]]
                    if others:
                        rmap[k] = sum(others) / len(others)
            st = compute_pair_stats_from_keys(keys, pool[r], rmap)
            if st.get("spearman") is not None:
                sp_vals.append(st["spearman"])
            if st.get("mad") is not None:
                mae_vals.append(st["mad"])
        return {
            "spearman": (sum(sp_vals) / len(sp_vals)) if sp_vals else None,
            "mae": (sum(mae_vals) / len(mae_vals)) if mae_vals else None,
        }

    loro_point = loro_mean(pool_items)
    rng_l = random.Random(SEED)
    loro_boot: dict[str, list[float]] = {"spearman": [], "mae": []}
    for _ in range(BOOT_DRAWS):
        sample_keys = [it for t in rng_l.choices(pool_task_names, k=len(pool_task_names)) for it in pool_by_task[t]]
        st = loro_mean(sample_keys)
        for m in ("spearman", "mae"):
            if st[m] is not None:
                loro_boot[m].append(st[m])
    loro_summary = {}
    for m in ("spearman", "mae"):
        vals = loro_boot[m]
        lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (None, None))
        loro_summary[m] = {
            "value": float(loro_point[m]) if loro_point[m] is not None else None,
            "ci_lo": float(lo) if lo is not None else None,
            "ci_hi": float(hi) if hi is not None else None,
        }

    # 4. Within-human rubric / strictness shifts
    within_shifts = {
        "P1_old_pass1__vs__P1_old_pass2": bootstrap_pair_metrics(
            all_items, within["P1_old_pass1"], within["P1_old_pass2"]
        ),
        "P1_correct__vs__P1_old_pass1": bootstrap_pair_metrics(
            all_items, within["P1_correct"], within["P1_old_pass1"]
        ),
        "P1_correct__vs__P1_old_pass2": bootstrap_pair_metrics(
            all_items, within["P1_correct"], within["P1_old_pass2"]
        ),
    }

    # Evaluate preregistered H2 rule
    exact_10 = summary_10_pairs["exact_agree"]
    rge2 = spread_all["range_ge_2"]
    shift_exacts = [v["exact_agree"] for v in within_shifts.values()]

    supported_point = (
        (exact_10["value"] is not None and exact_10["value"] < 0.50)
        or (rge2["value"] is not None and rge2["value"] >= 0.50)
        or any(s["value"] is not None and s["value"] < 0.50 for s in shift_exacts)
    )
    supported_ci = (
        (exact_10["ci_hi"] is not None and exact_10["ci_hi"] < 0.50)
        or (rge2["ci_lo"] is not None and rge2["ci_lo"] >= 0.50)
        or any(s["ci_hi"] is not None and s["ci_hi"] < 0.50 for s in shift_exacts)
    )
    falsified = (
        exact_10["value"] >= 0.85
        and rge2["value"] <= 0.10
        and all(s["value"] >= 0.85 for s in shift_exacts)
    )
    h2_verdict = "SUPPORTED" if (supported_point and supported_ci) else ("FALSIFIED" if falsified else "MIXED")

    return {
        "rater_counts": {k: len(v) for k, v in pool.items()},
        "within_counts": {k: len(v) for k, v in within.items()},
        "n_unique_items": len(all_items),
        "n_unique_tasks": len(all_task_names),
        "n_pool_items": len(pool_items),
        "n_pool_tasks": len(pool_task_names),
        "fixed_pool_10_pairs_summary": summary_10_pairs,
        "fixed_pool_pairs": pair_results,
        "item_spread_all_ge2": spread_all,
        "item_spread_complete5": spread_complete5,
        "item_spread_4_full_raters": spread_4raters,
        "leave_one_rater_out": loro,
        "leave_one_rater_out_mean": loro_summary,
        "within_human_shifts": within_shifts,
        "h2_verdict": h2_verdict,
        "h2_supported_point": supported_point,
        "h2_supported_ci": supported_ci,
    }


def write_judge_csv(out_path: Path, units: list[dict[str, Any]]) -> None:
    fields = [
        "config",
        "kind",
        "harness",
        "family",
        "prompt",
        "split",
        "model_declared",
        "model_served",
        "rho_heldout",
        "rho_heldout_ci_lo",
        "rho_heldout_ci_hi",
        "rho_heldout_coverage",
        "rho_heldout_n_items",
        "rho_heldout_n_declared",
        "rho_all_items",
        "V_bug",
        "V_bug_ci_lo",
        "V_bug_ci_hi",
        "V_ms",
        "V_ms_ci_lo",
        "V_ms_ci_hi",
        "V_weak",
        "V_weak_ci_lo",
        "V_weak_ci_hi",
        "V_vac",
        "V_vac_ci_lo",
        "V_vac_ci_hi",
        "V_overc",
        "V_overc_ci_lo",
        "V_overc_ci_hi",
        "V_fmt",
        "fmt_mean_gap",
        "fmt_mean_gap_ci_lo",
        "fmt_mean_gap_ci_hi",
        "V_rep",
        "V_rep_ci_lo",
        "V_rep_ci_hi",
        "V_rep_mean_range",
        "V_blank",
        "V_alpha_desc_n3",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in units:
            w.writerow(
                {
                    "config": r["config"],
                    "kind": r["kind"],
                    "harness": r["harness"],
                    "family": r["family"],
                    "prompt": r["prompt"],
                    "split": r["split"],
                    "model_declared": r["model_declared"],
                    "model_served": r["model_served"],
                    "rho_heldout": r["rho_heldout"],
                    "rho_heldout_ci_lo": r["rho_heldout_ci_lo"],
                    "rho_heldout_ci_hi": r["rho_heldout_ci_hi"],
                    "rho_heldout_coverage": r["rho_heldout_coverage"],
                    "rho_heldout_n_items": r["rho_heldout_n_items"],
                    "rho_heldout_n_declared": r["rho_heldout_n_declared"],
                    "rho_all_items": r["rho_all_items"],
                    "V_bug": r["V_bug"]["value"],
                    "V_bug_ci_lo": r["V_bug"]["ci_lo"],
                    "V_bug_ci_hi": r["V_bug"]["ci_hi"],
                    "V_ms": r["V_ms"]["value"],
                    "V_ms_ci_lo": r["V_ms"]["ci_lo"],
                    "V_ms_ci_hi": r["V_ms"]["ci_hi"],
                    "V_weak": r["V_weak"]["value"],
                    "V_weak_ci_lo": r["V_weak"]["ci_lo"],
                    "V_weak_ci_hi": r["V_weak"]["ci_hi"],
                    "V_vac": r["V_vac"]["value"],
                    "V_vac_ci_lo": r["V_vac"]["ci_lo"],
                    "V_vac_ci_hi": r["V_vac"]["ci_hi"],
                    "V_overc": r["V_overc"]["value"],
                    "V_overc_ci_lo": r["V_overc"]["ci_lo"],
                    "V_overc_ci_hi": r["V_overc"]["ci_hi"],
                    "V_fmt": r["V_fmt"]["value"],
                    "fmt_mean_gap": r["V_fmt"]["mean_abs_gap"],
                    "fmt_mean_gap_ci_lo": r["V_fmt"]["gap_ci_lo"],
                    "fmt_mean_gap_ci_hi": r["V_fmt"]["gap_ci_hi"],
                    "V_rep": r["V_rep"]["value"],
                    "V_rep_ci_lo": r["V_rep"]["ci_lo"],
                    "V_rep_ci_hi": r["V_rep"]["ci_hi"],
                    "V_rep_mean_range": r["V_rep"]["mean_range"],
                    "V_blank": r["V_blank"]["value"],
                    "V_alpha_desc_n3": r["V_alpha_desc"]["raw_value"],
                }
            )


def make_figures(out_dir: Path, jres: dict[str, Any], hres: dict[str, Any]) -> None:
    units = jres["units"]

    # Figure 1: Held-out human correlation vs 4 axiomatic violation rates
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.2), dpi=160)
    panels = [
        ("V_bug", "Bug-insertion violation rate\n(E39 bm_gold: score >= ident; lower = better)"),
        ("V_weak", "Semantic-weakening violation rate\n(E42 weak: score >= ident; lower = better)"),
        ("V_ms", "Missing-spec violation rate\n(E39 ms_gold: score >= ident; lower = better)"),
        ("V_rep", "Repeat inconsistency rate\n(E39 crit 3 repeats: range > 0; lower = better)"),
    ]
    styles = {
        "vb103_judge": {"color": "#1f77b4", "marker": "o", "label": "Single-call judge LLM (vb103_judge)"},
        "terminus2": {"color": "#d62728", "marker": "s", "label": "Judge agent (terminus2)"},
        "none": {"color": "#555555", "marker": "^", "label": "Deterministic control (none)"},
    }
    for ax, (prop, ylabel) in zip(axes.flat, panels):
        ax.grid(True, linestyle="-", alpha=0.35)
        ax.axvline(0.70, color="#2ca02c", linestyle="--", linewidth=1.1, alpha=0.75, label="rho >= 0.70 threshold" if prop == "V_bug" else None)
        ax.axhline(0.20, color="#ff7f0e", linestyle=":", linewidth=1.1, alpha=0.85, label="violation >= 0.20 threshold" if prop == "V_bug" else None)
        for harness, st in styles.items():
            pts = [
                r
                for r in units
                if r["harness"] == harness and r["rho_heldout"] is not None and r[prop]["value"] is not None
            ]
            if not pts:
                continue
            xs = [r["rho_heldout"] for r in pts]
            ys = [r[prop]["value"] for r in pts]
            ax.scatter(xs, ys, c=st["color"], marker=st["marker"], s=58, alpha=0.88, edgecolors="black", linewidths=0.5, label=st["label"] if prop == "V_bug" else None, zorder=3)
        # Annotate worst-violating judge among rho_heldout >= 0.70
        hi_pts = [r for r in units if r["kind"] == "llm" and r["rho_heldout"] is not None and r["rho_heldout"] >= 0.70 and r[prop]["value"] is not None]
        if hi_pts:
            worst = max(hi_pts, key=lambda r: r[prop]["value"])
            y_offset = -14 if worst[prop]["value"] > 0.78 else 8
            ax.annotate(
                f"{worst['config']} ({worst[prop]['value']:.2f})",
                xy=(worst["rho_heldout"], worst[prop]["value"]),
                xytext=(-10, y_offset),
                textcoords="offset points",
                fontsize=8.5,
                ha="right",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#999999", alpha=0.9),
            )
        ax.set_xlabel("Held-out human rank correlation (Spearman rho, 42-45 scored / 45 items; higher = closer to human)", fontsize=9.2)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_ylim(-0.04, 1.06)
        ax.set_xlim(-0.45, 0.98)
        ax.set_title(f"{prop} vs. held-out human Spearman rho", fontsize=11, fontweight="bold")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.955), frameon=True, fontsize=9)
    fig.suptitle(
        "Exp 03 (2026-11-24): Lean 4 judges with high human correlation still fail axiomatic checks",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.905], h_pad=2.0, w_pad=2.2)
    fig.savefig(out_dir / "fig1_human_corr_vs_property_violations.png", dpi=160)
    plt.close(fig)

    # Figure 2: H2 human inconsistency on Lean 4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=160)

    ax1.grid(True, linestyle="-", alpha=0.35)
    labels_p, vals_p, lo_p, hi_p, cols_p = [], [], [], [], []
    for k, st in hres["fixed_pool_pairs"].items():
        short = k.replace("P1_correct", "P1").replace("__vs__", "-")
        labels_p.append(short)
        v = st["exact_agree"]["value"]
        vals_p.append(v)
        lo_p.append(v - st["exact_agree"]["ci_lo"])
        hi_p.append(st["exact_agree"]["ci_hi"] - v)
        cols_p.append("#1f77b4")
    shift_map = {
        "P1_old_pass1__vs__P1_old_pass2": "P1 old p1-p2",
        "P1_correct__vs__P1_old_pass1": "P1 corr-old p1",
        "P1_correct__vs__P1_old_pass2": "P1 corr-old p2",
    }
    for k, short in shift_map.items():
        st = hres["within_human_shifts"][k]
        labels_p.append(short)
        v = st["exact_agree"]["value"]
        vals_p.append(v)
        lo_p.append(v - st["exact_agree"]["ci_lo"])
        hi_p.append(st["exact_agree"]["ci_hi"] - v)
        cols_p.append("#d62728")

    x = np.arange(len(labels_p))
    ax1.bar(x, vals_p, yerr=[lo_p, hi_p], capsize=3, color=cols_p, alpha=0.88, edgecolor="black", linewidth=0.6, zorder=3)
    ax1.axhline(0.50, color="#ff7f0e", linestyle="--", linewidth=1.2, label="H2 support cutoff (< 0.50)")
    ax1.axhline(0.85, color="#2ca02c", linestyle=":", linewidth=1.2, label="H2 falsification cutoff (>= 0.85)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_p, rotation=40, ha="right", fontsize=8.5)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Exact score agreement P(|a - b| == 0) on 0-5 scale\n(higher = more consistent)", fontsize=9.5)
    ax1.set_xlabel("Human rater pairs (blue: 5-rater pool) and within-P1 shifts (red)", fontsize=9.5)
    ax1.set_title("Human exact agreement across raters and rubric shifts", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.5)

    ax2.grid(True, linestyle="-", alpha=0.35)
    hist = hres["item_spread_all_ge2"]["range_histogram_counts"]
    bins = [0, 1, 2, 3, 4, 5]
    counts = [hist[str(b)] for b in bins]
    bar_cols = ["#2ca02c" if b == 0 else ("#ffbf00" if b == 1 else "#d62728") for b in bins]
    ax2.bar(bins, counts, color=bar_cols, alpha=0.88, edgecolor="black", linewidth=0.7, zorder=3)
    total_items = sum(counts)
    for b, c in zip(bins, counts):
        ax2.text(b, c + 0.6, f"{c}\n({100 * c / total_items:.0f}%)", ha="center", va="bottom", fontsize=8.5)
    ax2.set_xticks(bins)
    ax2.set_ylim(0, max(counts) + 6)
    ax2.set_xlabel("Per-item cross-rater score range (max - min on 0-5 scale; 0 = full agreement)", fontsize=9.5)
    ax2.set_ylabel("Number of rated Lean 4 items (out of 75 items, >= 2 raters)", fontsize=9.5)
    ax2.set_title("Cross-rater score spread across 5 fixed-rubric raters", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Exp 03 (2026-11-24): Lean 4 human raters show low exact agreement and wide score spreads",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "fig2_human_inconsistency_lean4.png", dpi=160)
    plt.close(fig)


def run_analysis(e39_dir: Path, e42_dir: Path, human_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jres = analyze_judges(e39_dir, e42_dir)
    hres = analyze_humans(human_dir)
    write_judge_csv(out_dir / "judge_table.csv", jres["units"])
    write_json(out_dir / "human_consistency.json", hres)
    summary = {
        "date": "2026-11-24",
        "experiment": "03_human_corr_vs_axiomatic_properties_cert_judge",
        "h1_verdict": jres["h1_verdict"],
        "h2_verdict": hres["h2_verdict"],
        "alignment_check": jres["alignment_check"],
        "n_units": jres["n_units"],
        "n_llm_units": jres["n_llm_units"],
        "n_control_units": jres["n_control_units"],
        "n_high_corr_llms_ge_0_70": jres["n_high_corr_llms_ge_0_70"],
        "failing_high_corr_llms": jres["failing_high_corr_llms"],
        "n_failing_high_corr_excluding_v_bug": jres["n_failing_high_corr_excluding_v_bug"],
        "failing_high_corr_excluding_v_bug": jres["failing_high_corr_excluding_v_bug"],
        "cross_judge_spearman": jres["cross_judge_spearman"],
        "human_summary": {
            "fixed_pool_10_pairs_summary": hres["fixed_pool_10_pairs_summary"],
            "item_spread_all_ge2": hres["item_spread_all_ge2"],
            "item_spread_complete5": hres["item_spread_complete5"],
            "item_spread_4_full_raters": hres["item_spread_4_full_raters"],
            "leave_one_rater_out_mean": hres["leave_one_rater_out_mean"],
            "within_human_shifts": hres["within_human_shifts"],
        },
        "units": jres["units"],
    }
    write_json(out_dir / "summary.json", summary)
    make_figures(out_dir, jres, hres)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze cert-judge E39/E42 and Lean 4 human ratings.")
    parser.add_argument("--e39-dir", type=Path, default=DEFAULT_E39_DIR)
    parser.add_argument("--e42-dir", type=Path, default=DEFAULT_E42_DIR)
    parser.add_argument("--human-dir", type=Path, default=DEFAULT_HUMAN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_analysis(args.e39_dir, args.e42_dir, args.human_dir, args.out_dir)
    print(
        json.dumps(
            {
                "h1_verdict": summary["h1_verdict"],
                "h2_verdict": summary["h2_verdict"],
                "alignment_check": summary["alignment_check"],
                "n_high_corr_llms_ge_0_70": summary["n_high_corr_llms_ge_0_70"],
                "n_failing_high_corr_llms": len(summary["failing_high_corr_llms"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
