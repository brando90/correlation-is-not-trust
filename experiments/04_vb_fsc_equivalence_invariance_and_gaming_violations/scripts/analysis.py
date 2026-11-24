#!/usr/bin/env python3
"""Reanalyze cached VeriBench FSC judge scores for equivalence invariance,
reference-order permutation invariance, and adversarial gaming monotonicity (2026-11-24).

Zero new model calls ($0.00). Reads cached CSV/JSON aggregates from sibling repos,
verifies exact numeric alignment against source summaries, computes 2,000-draw
task-cluster bootstrap 95% intervals and exact permutation/sign-test p-values,
and emits aggregate CSV/JSON tables plus matplotlib figures with grid lines.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DEFAULT_JRA_DIR = Path(
    "/Users/brandomiranda/judge-reference-anchoring/experiments/"
    "01_veribench_fsc_reference_anchoring_open_and_frontier_llm_vs_harness_start_17Nov2026/results"
)
DEFAULT_E23_DIR = Path(
    "/Users/brandomiranda/trustworthy-theorem-judges/experiments/"
    "23_do_gamed_or_off_format_lean_files_score_lower_on_veribench_vb256_start_18Nov2026"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "results"

SEED = 20261124
BOOT_DRAWS = 2000
PERM_DRAWS = 5000
TOL = 1e-12

CONDITION_CATEGORY = {
    "faithful_gold_copy": "baseline",
    "sorry_proofs": "proof_gaming",
    "false_axiom_proofs": "proof_gaming",
    "weakened_props": "statement_gaming",
    "vacuous_true_props": "statement_gaming",
    "dropped_most_theorems": "statement_gaming",
    "theorem_spam": "statement_gaming",
    "shadowed_names": "statement_gaming",
    "shadowed_perm_sorted_pairwise": "statement_gaming",
    "off_format_missing_sections": "statement_gaming",
    "empty_file": "degenerate_control",
    "weak_prompt_natural": "natural_control",
    "historical_natural_agent": "natural_control",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _parse_bounded_float(raw: Any, lo: float, hi: float, label: str) -> float:
    val = float(raw)
    if not math.isfinite(val) or val < lo - TOL or val > hi + TOL:
        raise ValueError(f"Invalid {label}={raw!r}: expected finite value in [{lo}, {hi}]")
    return val


def cluster_mean_ci(
    pairs: list[tuple[str, float]], draws: int = BOOT_DRAWS, seed: int = SEED
) -> dict[str, Any]:
    """Mean and 95% percentile task-cluster bootstrap CI over observed tasks."""
    if not pairs:
        return {"value": None, "ci_lo": None, "ci_hi": None, "n": 0, "n_tasks": 0}
    groups: dict[str, list[float]] = {}
    for task, val in pairs:
        fval = float(val)
        if not math.isfinite(fval):
            raise ValueError(f"Non-finite value in cluster_mean_ci for task={task!r}: {val!r}")
        groups.setdefault(task, []).append(fval)
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


def paired_signflip_pvalue(
    diffs: list[float], draws: int = PERM_DRAWS, seed: int = SEED
) -> float | None:
    """Two-sided paired sign-flip p-value (exact 2^n for n <= 12, else Monte Carlo)."""
    n = len(diffs)
    if n == 0:
        return None
    for d in diffs:
        if not math.isfinite(float(d)):
            raise ValueError(f"Non-finite difference in paired_signflip_pvalue: {d!r}")
    obs = abs(sum(diffs) / n)
    if obs <= TOL:
        return 1.0
    if n <= 12:
        total = 1 << n
        ge = 0
        for mask in range(total):
            s = sum(diffs[i] if ((mask >> i) & 1) else -diffs[i] for i in range(n))
            if abs(s / n) >= obs - TOL:
                ge += 1
        return float(ge / total)
    rng = random.Random(seed)
    ge = 0
    for _ in range(draws):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(s / n) >= obs - TOL:
            ge += 1
    return float((ge + 1) / (draws + 1))


def exact_sign_test_p(diffs: list[float]) -> float | None:
    """Two-sided exact binomial sign test on non-zero differences."""
    if not diffs:
        return None
    for d in diffs:
        if not math.isfinite(d):
            raise ValueError(f"Non-finite difference in exact_sign_test_p: {d!r}")
    pos = sum(1 for d in diffs if d > TOL)
    neg = sum(1 for d in diffs if d < -TOL)
    total = pos + neg
    if total == 0:
        return 1.0
    return float(stats.binomtest(pos, total, p=0.5, alternative="two-sided").pvalue)


def _valid_jra_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for r in rows:
        if r.get("error", "").strip():
            continue
        raw = r.get("fsc_norm", "").strip()
        if not raw:
            continue
        _parse_bounded_float(raw, 0.0, 1.0, "fsc_norm")
        rep = int(r["repeat_idx"])
        if rep < 0:
            raise ValueError(f"Negative repeat_idx: {rep}")
        out.append(r)
    return out


def analyze_jra(jra_dir: Path) -> dict[str, Any]:
    ab_rows = _valid_jra_rows(read_csv(jra_dir / "exp_a_b_per_item_scores.csv"))
    bcli_rows = _valid_jra_rows(read_csv(jra_dir / "exp_b_cli_agent_scores.csv"))
    ec_rows = read_csv(jra_dir / "exp_c_veribench_impact_per_task.csv")
    sm = read_json(jra_dir / "summary_metrics.json")

    all_ab = ab_rows + bcli_rows
    meta_by_cfg: dict[tuple[str, str], dict[str, str]] = {}
    rep0_by_cfg: dict[tuple[str, str], dict[tuple[str, str], dict[str, str]]] = {}
    repeats_by_cfg: dict[tuple[str, str], dict[tuple[str, str], list[float]]] = {}

    seen_obs_keys: set[tuple[str, str, str, str, int]] = set()
    efforts_by_cfg: dict[tuple[str, str], set[str]] = {}
    for r in all_ab:
        cfg_key = (r["judge_id"], r["method"])
        rep_idx = int(r["repeat_idx"])
        obs_key = (r["judge_id"], r["method"], r["task_id"], r["condition"], rep_idx)
        if obs_key in seen_obs_keys:
            raise ValueError(f"Duplicate JRA observation key: {obs_key}")
        seen_obs_keys.add(obs_key)
        prev = meta_by_cfg.get(cfg_key)
        if prev is not None:
            for k in ("tier", "harness", "model"):
                if prev[k] != r[k]:
                    raise ValueError(f"Heterogeneous {k} in {cfg_key}: {prev[k]!r} vs {r[k]!r}")
        efforts_by_cfg.setdefault(cfg_key, set()).add(r["effort"])
        eff_set = efforts_by_cfg[cfg_key]
        eff_label = next(iter(eff_set)) if len(eff_set) == 1 else f"mixed ({'/'.join(sorted(eff_set))})"
        meta_by_cfg[cfg_key] = {
            "judge_id": r["judge_id"],
            "tier": r["tier"],
            "harness": r["harness"],
            "model": r["model"],
            "effort": eff_label,
            "method": r["method"],
        }
        item_key = (r["task_id"], r["condition"])
        fsc_val = _parse_bounded_float(r["fsc_norm"], 0.0, 1.0, "fsc_norm")
        repeats_by_cfg.setdefault(cfg_key, {}).setdefault(item_key, []).append(fsc_val)
        if rep_idx == 0:
            rep0_by_cfg.setdefault(cfg_key, {})[item_key] = r

    # Alignment check A1 against exp_ab_matrix in summary_metrics.json
    def _acc_diff(diff: float, ctx: str) -> None:
        nonlocal max_align_diff, align_checks
        if not math.isfinite(diff):
            raise RuntimeError(f"Non-finite alignment difference in {ctx}: {diff!r}")
        max_align_diff = max(max_align_diff, diff)
        align_checks += 1

    max_align_diff = 0.0
    align_checks = 0
    sm_ab = sm["exp_ab_matrix"]["configurations"]
    for sm_key, sm_cfg in sm_ab.items():
        cfg_key = (sm_cfg["judge_id"], sm_cfg["method"])
        imap = rep0_by_cfg.get(cfg_key, {})
        t1_match = [float(v["fsc_norm"]) for (_, cond), v in imap.items() if cond == "match_refactored" and v["truth"] == "T1"]
        t1_diff = [float(v["fsc_norm"]) for (_, cond), v in imap.items() if cond == "diff_gold" and v["truth"] == "T1"]
        fc_cond = "blind" if sm_cfg["method"] == "M2_blind_no_ref" else "non_covering"
        t0_fc = [float(v["fsc_norm"]) for (_, cond), v in imap.items() if cond == fc_cond and v["truth"] == "T0"]
        if "cm" in sm_cfg and sm_cfg["cm"]["mean"] is not None:
            if not t1_match or not math.isfinite(float(sm_cfg["cm"]["mean"])):
                raise RuntimeError(f"Invalid cm alignment data for {sm_key}")
            _acc_diff(abs((sum(t1_match) / len(t1_match)) - float(sm_cfg["cm"]["mean"])), f"{sm_key} cm")
        if "cd" in sm_cfg and sm_cfg["cd"]["mean"] is not None:
            if not t1_diff or not math.isfinite(float(sm_cfg["cd"]["mean"])):
                raise RuntimeError(f"Invalid cd alignment data for {sm_key}")
            _acc_diff(abs((sum(t1_diff) / len(t1_diff)) - float(sm_cfg["cd"]["mean"])), f"{sm_key} cd")
        if "A_G" in sm_cfg and sm_cfg["A_G"]["mean"] is not None:
            t1_tasks = sorted({t for (t, _) in imap if imap.get((t, "match_refactored")) and imap.get((t, "diff_gold"))})
            if not t1_tasks or not math.isfinite(float(sm_cfg["A_G"]["mean"])):
                raise RuntimeError(f"Invalid A_G alignment data for {sm_key}")
            ag_val = sum(float(imap[(t, "match_refactored")]["fsc_norm"]) - float(imap[(t, "diff_gold")]["fsc_norm"]) for t in t1_tasks) / len(t1_tasks)
            _acc_diff(abs(ag_val - float(sm_cfg["A_G"]["mean"])), f"{sm_key} A_G")
        if "fc" in sm_cfg and sm_cfg["fc"]["mean"] is not None:
            if not t0_fc or not math.isfinite(float(sm_cfg["fc"]["mean"])):
                raise RuntimeError(f"Invalid fc alignment data for {sm_key}")
            _acc_diff(abs((sum(t0_fc) / len(t0_fc)) - float(sm_cfg["fc"]["mean"])), f"{sm_key} fc")

    # Compute rich axiomatic metrics for all 31 Part-A configurations (28 exp_a_b + 3 exp_b_cli)
    part_a_records = []
    for cfg_key in sorted(meta_by_cfg, key=lambda x: (meta_by_cfg[x]["tier"], meta_by_cfg[x]["method"], x[0])):
        meta = meta_by_cfg[cfg_key]
        imap = rep0_by_cfg.get(cfg_key, {})
        all_tasks_t1 = sorted({t for (t, _), r in imap.items() if r["truth"] == "T1"})
        all_tasks_t0 = sorted({t for (t, _), r in imap.items() if r["truth"] == "T0"})

        # Equivalence invariance: match_refactored vs diff_gold on paired T1 tasks
        eq_tasks = [t for t in all_tasks_t1 if (t, "match_refactored") in imap and (t, "diff_gold") in imap]
        ag_pairs = [(t, float(imap[(t, "match_refactored")]["fsc_norm"]) - float(imap[(t, "diff_gold")]["fsc_norm"])) for t in eq_tasks]
        veq_pairs = [(t, 1.0 if abs(d) > TOL else 0.0) for t, d in ag_pairs]
        abs_eq_pairs = [(t, abs(d)) for t, d in ag_pairs]
        flip_ge9_pairs = [
            (t, 1.0 if int(imap[(t, "match_refactored")]["harsh_credit_ge9"]) != int(imap[(t, "diff_gold")]["harsh_credit_ge9"]) else 0.0)
            for t in eq_tasks
        ]
        flip_eq10_pairs = [
            (t, 1.0 if int(imap[(t, "match_refactored")]["harsh_credit_eq10"]) != int(imap[(t, "diff_gold")]["harsh_credit_eq10"]) else 0.0)
            for t in eq_tasks
        ]
        cm_pairs = [(t, float(imap[(t, "match_refactored")]["fsc_norm"])) for t in eq_tasks]
        cd_pairs = [(t, float(imap[(t, "diff_gold")]["fsc_norm"])) for t in eq_tasks]

        ag_stat = cluster_mean_ci(ag_pairs)
        ag_stat["p_val"] = paired_signflip_pvalue([d for _, d in ag_pairs]) if ag_pairs else None

        # Same-effort sensitivity for equivalence, renaming, and order pairs
        eq_same_tasks = [t for t in eq_tasks if imap[(t, "match_refactored")]["effort"] == imap[(t, "diff_gold")]["effort"]]
        n_mismatched_equiv = len(eq_tasks) - len(eq_same_tasks)
        ag_same_pairs = [(t, float(imap[(t, "match_refactored")]["fsc_norm"]) - float(imap[(t, "diff_gold")]["fsc_norm"])) for t in eq_same_tasks]
        veq_same_pairs = [(t, 1.0 if abs(d) > TOL else 0.0) for t, d in ag_same_pairs]
        ag_same_stat = cluster_mean_ci(ag_same_pairs)
        ag_same_stat["p_val"] = paired_signflip_pvalue([d for _, d in ag_same_pairs]) if ag_same_pairs else None

        # Exact vs refactored renaming: match_exact vs match_refactored on paired T1 tasks
        ren_tasks = [t for t in all_tasks_t1 if (t, "match_exact") in imap and (t, "match_refactored") in imap]
        n_mismatched_rename = sum(
            1 for t in ren_tasks if imap[(t, "match_exact")]["effort"] != imap[(t, "match_refactored")]["effort"]
        )
        vren_pairs = [
            (t, 1.0 if abs(float(imap[(t, "match_exact")]["fsc_norm"]) - float(imap[(t, "match_refactored")]["fsc_norm"])) > TOL else 0.0)
            for t in ren_tasks
        ]
        ren_shift_pairs = [
            (t, abs(float(imap[(t, "match_exact")]["fsc_norm"]) - float(imap[(t, "match_refactored")]["fsc_norm"])))
            for t in ren_tasks
        ]

        # Reference-order permutation invariance (M4_dual_ref: dual_ab vs dual_ba)
        ord_tasks = [t for t in all_tasks_t1 if (t, "dual_ab") in imap and (t, "dual_ba") in imap]
        n_mismatched_order = sum(
            1 for t in ord_tasks if imap[(t, "dual_ab")]["effort"] != imap[(t, "dual_ba")]["effort"]
        )
        ord_diff_pairs = [(t, float(imap[(t, "dual_ab")]["fsc_norm"]) - float(imap[(t, "dual_ba")]["fsc_norm"])) for t in ord_tasks]
        vord_pairs = [(t, 1.0 if abs(d) > TOL else 0.0) for t, d in ord_diff_pairs]
        ord_abs_pairs = [(t, abs(d)) for t, d in ord_diff_pairs]
        ord_bias_stat = cluster_mean_ci(ord_diff_pairs)
        ord_bias_stat["p_val"] = paired_signflip_pvalue([d for _, d in ord_diff_pairs]) if ord_diff_pairs else None

        # Blind T1 score (M2_blind_no_ref)
        blind_tasks = [t for t in all_tasks_t1 if (t, "blind") in imap]
        blind_pairs = [(t, float(imap[(t, "blind")]["fsc_norm"])) for t in blind_tasks]

        # False credit on T0 (non_covering or blind T0)
        t0_cond = "blind" if meta["method"] == "M2_blind_no_ref" else "non_covering"
        fc_tasks = [t for t in all_tasks_t0 if (t, t0_cond) in imap]
        fc_pairs = [(t, float(imap[(t, t0_cond)]["fsc_norm"])) for t in fc_tasks]
        fc_ge9_pairs = [(t, float(imap[(t, t0_cond)]["harsh_credit_ge9"])) for t in fc_tasks]

        # 3-repeat inconsistency across repeat_idx {0, 1, 2} where >= 2 repeats exist
        rep_items = [(t, cond, vals) for (t, cond), vals in sorted(repeats_by_cfg.get(cfg_key, {}).items()) if len(vals) >= 2]
        vrep_pairs = [(t, 1.0 if (max(vals) - min(vals)) > TOL else 0.0) for t, _, vals in rep_items]
        rep_rng_pairs = [(t, max(vals) - min(vals)) for t, _, vals in rep_items]

        part_a_records.append(
            {
                **meta,
                "config_id": f"{meta['judge_id']}::{meta['method']}",
                "cm": cluster_mean_ci(cm_pairs),
                "cd": cluster_mean_ci(cd_pairs),
                "blind_t1": cluster_mean_ci(blind_pairs),
                "A_G": ag_stat,
                "V_equiv": cluster_mean_ci(veq_pairs),
                "V_equiv_same_effort": cluster_mean_ci(veq_same_pairs),
                "A_G_same_effort": ag_same_stat,
                "n_mismatched_equiv_effort": n_mismatched_equiv,
                "n_mismatched_rename_effort": n_mismatched_rename,
                "n_mismatched_order_effort": n_mismatched_order,
                "mean_abs_equiv_shift": cluster_mean_ci(abs_eq_pairs),
                "harsh_flip_ge9": cluster_mean_ci(flip_ge9_pairs),
                "harsh_flip_eq10": cluster_mean_ci(flip_eq10_pairs),
                "V_rename_exact": cluster_mean_ci(vren_pairs),
                "mean_abs_rename_shift": cluster_mean_ci(ren_shift_pairs),
                "V_order": cluster_mean_ci(vord_pairs),
                "mean_abs_order_shift": cluster_mean_ci(ord_abs_pairs),
                "order_bias_ab_minus_ba": ord_bias_stat,
                "fc": cluster_mean_ci(fc_pairs),
                "fc_harsh_ge9": cluster_mean_ci(fc_ge9_pairs),
                "V_rep_ab": cluster_mean_ci(vrep_pairs),
                "rep_mean_range": cluster_mean_ci(rep_rng_pairs),
            }
        )

    # Part A2: Real VeriBench Exp C (10 tasks x 5 solvers = 50 pairs across 5 judge x method configs)
    ec_by_cfg: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in ec_rows:
        ec_by_cfg.setdefault((r["judge_id"], r["method"]), []).append(r)

    sm_ec = sm["exp_c_veribench_impact"]["by_judge_method"]
    exp_c_records = []
    expected_solvers_ec = {r["solver_id"] for r in ec_rows}
    if not expected_solvers_ec:
        raise RuntimeError("No solver_id entries found in exp_c_veribench_impact_per_task.csv")
    for cfg_key in sorted(ec_by_cfg):
        rows = ec_by_cfg[cfg_key]
        seen_ec: set[tuple[str, str]] = set()
        all_valid = []
        for r in rows:
            pk = (r["task_id"], r["solver_id"])
            if pk in seen_ec:
                raise ValueError(f"Duplicate exp_c (task_id, solver_id) for {cfg_key}: {pk}")
            seen_ec.add(pk)
            if r["fsc_R_A"].strip():
                _parse_bounded_float(r["fsc_R_A"], 0.0, 1.0, "fsc_R_A")
            if r["fsc_R_B"].strip():
                _parse_bounded_float(r["fsc_R_B"], 0.0, 1.0, "fsc_R_B")
            if r["fsc_R_A"].strip() and r["fsc_R_B"].strip():
                all_valid.append(r)
        by_task_valid: dict[str, list[dict[str, str]]] = {}
        for r in all_valid:
            by_task_valid.setdefault(r["task_id"], []).append(r)
        complete_tasks = sorted(
            t
            for t, trs in by_task_valid.items()
            if {r["solver_id"] for r in trs} == expected_solvers_ec
        )
        valid_rows = [r for t in complete_tasks for r in by_task_valid[t]]
        cfg_id = f"{cfg_key[0]}::{cfg_key[1]}"
        meta = meta_by_cfg.get(
            cfg_key,
            {"judge_id": cfg_key[0], "method": cfg_key[1], "harness": "unknown", "model": cfg_key[0], "tier": "unknown", "effort": "unknown"},
        )
        swap_pairs = [(r["task_id"], 1.0 if abs(float(r["fsc_R_A"]) - float(r["fsc_R_B"])) > TOL else 0.0) for r in valid_rows]
        abs_pairs = [(r["task_id"], abs(float(r["fsc_R_A"]) - float(r["fsc_R_B"]))) for r in valid_rows]
        signed_pairs = [(r["task_id"], float(r["fsc_R_A"]) - float(r["fsc_R_B"])) for r in valid_rows]
        flip_nodt_eq1 = [
            (r["task_id"], 1.0 if int(r["harsh_nodt_R_A"]) != int(r["harsh_nodt_R_B"]) else 0.0)
            for r in valid_rows
            if r["harsh_nodt_R_A"].strip() and r["harsh_nodt_R_B"].strip()
        ]
        flip_nodt_ge09 = [
            (r["task_id"], 1.0 if int(r["harsh_nodt_ge09_R_A"]) != int(r["harsh_nodt_ge09_R_B"]) else 0.0)
            for r in valid_rows
            if r["harsh_nodt_ge09_R_A"].strip() and r["harsh_nodt_ge09_R_B"].strip()
        ]
        flip_dt_eq1 = [
            (r["task_id"], 1.0 if int(r["harsh_dt_R_A"]) != int(r["harsh_dt_R_B"]) else 0.0)
            for r in valid_rows
            if r["harsh_dt_R_A"].strip() and r["harsh_dt_R_B"].strip()
        ]

        abs_stat = cluster_mean_ci(abs_pairs)
        signed_stat = cluster_mean_ci(signed_pairs)
        by_t: dict[str, list[float]] = {}
        for t, v in signed_pairs:
            by_t.setdefault(t, []).append(v)
        task_signed = [sum(vs) / len(vs) for _, vs in sorted(by_t.items())]
        signed_stat["p_val"] = paired_signflip_pvalue(task_signed) if task_signed else None

        if cfg_id in sm_ec and sm_ec[cfg_id]["mean_abs_fsc_shift"]["mean"] is not None:
            ref_shift = sm_ec[cfg_id]["mean_abs_fsc_shift"]["mean"]
            if not abs_pairs or not math.isfinite(float(ref_shift)):
                raise RuntimeError(f"Invalid exp_c alignment data for {cfg_id}")
            _acc_diff(abs(abs_stat["value"] - float(ref_shift)), f"{cfg_id} exp_c mean_abs_fsc_shift")

        all_valid_swap = [(r["task_id"], 1.0 if abs(float(r["fsc_R_A"]) - float(r["fsc_R_B"])) > TOL else 0.0) for r in all_valid]
        exp_c_records.append(
            {
                **meta,
                "config_id": cfg_id,
                "n_pairs": len(valid_rows),
                "n_all_valid_pairs": len(all_valid),
                "V_ref_swap": cluster_mean_ci(swap_pairs),
                "V_ref_swap_all_valid": cluster_mean_ci(all_valid_swap),
                "mean_abs_fsc_shift": abs_stat,
                "signed_shift_RA_minus_RB": signed_stat,
                "harsh_flip_nodt_eq1": cluster_mean_ci(flip_nodt_eq1),
                "harsh_flip_nodt_ge09": cluster_mean_ci(flip_nodt_ge09),
                "harsh_flip_dt_eq1": cluster_mean_ci(flip_dt_eq1),
            }
        )

    if align_checks == 0 or not math.isfinite(max_align_diff) or max_align_diff > 1e-9:
        raise RuntimeError(f"JRA alignment check failed: checks={align_checks}, max_abs_diff={max_align_diff}")

    # Evaluate preregistered H1-equiv verdict
    frontier_m1 = [r for r in part_a_records if r["tier"] == "frontier" and r["method"] == "M1_deployed_tc_v1"]
    m4_configs = [r for r in part_a_records if r["method"] == "M4_dual_ref" and r["V_order"]["value"] is not None]
    expc_m1_frontier = [r for r in exp_c_records if r["judge_id"] in ("claude-sonnet-4-6", "gpt-5.4") and r["method"] == "M1_deployed_tc_v1"]

    equiv_triggers = []
    for r in frontier_m1:
        st = r["V_equiv"]
        if st["value"] is not None and st["value"] >= 0.25 and st["ci_lo"] > 0.05:
            equiv_triggers.append({"config_id": r["config_id"], "metric": "V_equiv", "value": st["value"], "ci_lo": st["ci_lo"]})
    for r in expc_m1_frontier:
        st = r["V_ref_swap"]
        if st["value"] is not None and st["value"] >= 0.25 and st["ci_lo"] > 0.05:
            equiv_triggers.append({"config_id": r["config_id"], "metric": "V_ref_swap", "value": st["value"], "ci_lo": st["ci_lo"]})
    for r in m4_configs:
        st = r["V_order"]
        if st["value"] is not None and st["value"] >= 0.15 and st["ci_lo"] > 0.02:
            equiv_triggers.append({"config_id": r["config_id"], "metric": "V_order", "value": st["value"], "ci_lo": st["ci_lo"]})

    has_equiv_evidence = bool(frontier_m1 and expc_m1_frontier and m4_configs)
    falsified_equiv = has_equiv_evidence and (
        all(r["V_equiv"]["value"] is not None and r["V_equiv"]["value"] < 0.10 for r in frontier_m1)
        and all(r["V_ref_swap"]["value"] is not None and r["V_ref_swap"]["value"] < 0.10 for r in expc_m1_frontier)
        and all(r["V_order"]["value"] is not None and r["V_order"]["value"] < 0.05 for r in m4_configs)
    )
    h1_equiv_verdict = (
        "SUPPORTED"
        if equiv_triggers
        else ("INCONCLUSIVE" if not has_equiv_evidence else ("FALSIFIED" if falsified_equiv else "MIXED"))
    )

    return {
        "alignment_jra": {"checks": align_checks, "max_abs_diff": max_align_diff, "passed": max_align_diff <= 1e-9},
        "h1_equiv_verdict": h1_equiv_verdict,
        "equiv_triggers": equiv_triggers,
        "part_a_configs": part_a_records,
        "part_a_exp_c": exp_c_records,
    }


def _score_e23_dataset(
    dataset_name: str,
    judge_rows: list[dict[str, str]],
    gate_rows: list[dict[str, str]],
    pipeline_cols: list[tuple[str, str | None]],
) -> list[dict[str, Any]]:
    gate_map: dict[tuple[str, str], dict[str, str]] = {}
    for r in gate_rows:
        gk = (r["task"], r["condition"])
        if gk in gate_map:
            raise ValueError(f"Duplicate gate row for {dataset_name}: {gk}")
        gate_map[gk] = r
    by_cond_task: dict[tuple[str, str], dict[str, float]] = {}
    conditions = sorted({r["condition"] for r in judge_rows})
    tasks = sorted({r["task"] for r in judge_rows})

    for jr in judge_rows:
        key = (jr["task"], jr["condition"])
        if key in by_cond_task:
            raise ValueError(f"Duplicate judge row for {dataset_name}: {key}")
        gr = gate_map.get(key)
        if gr is None:
            raise RuntimeError(f"Missing gate row for {dataset_name} {key}")
        raw_score = _parse_bounded_float(jr["judge_score"], 0.0, 5.0, f"{dataset_name} judge_score")
        scores = {}
        for pname, col in pipeline_cols:
            if col is None:
                scores[pname] = raw_score
            else:
                gval = gr[col].strip()
                if gval not in ("0", "1"):
                    raise ValueError(f"Invalid gate flag {col}={gval!r} for {dataset_name} {key}")
                scores[pname] = raw_score * float(int(gval))
        by_cond_task[key] = scores

    out = []
    for cond in conditions:
        cat = CONDITION_CATEGORY.get(cond, "other")
        for pname, col in pipeline_cols:
            score_pairs = [(t, by_cond_task[(t, cond)][pname]) for t in tasks if (t, cond) in by_cond_task]
            gate_pass_count = (
                len(score_pairs)
                if col is None
                else sum(int(gate_map[(t, cond)][col]) for t, _ in score_pairs)
            )
            mean_score_st = cluster_mean_ci(score_pairs)
            hc_pairs = [(t, 1.0 if v >= 3.0 - TOL else 0.0) for t, v in score_pairs]
            hc_st = cluster_mean_ci(hc_pairs)

            if cond == "faithful_gold_copy":
                out.append(
                    {
                        "dataset": dataset_name,
                        "condition": cond,
                        "category": cat,
                        "pipeline": pname,
                        "n_tasks": len(score_pairs),
                        "gate_pass_count": gate_pass_count,
                        "mean_score": mean_score_st,
                        "mean_drop": {"value": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n": len(score_pairs), "n_tasks": len(score_pairs)},
                        "sign_p_val": 1.0,
                        "V_mono": {"value": None, "ci_lo": None, "ci_hi": None, "n": len(score_pairs), "n_tasks": len(score_pairs)},
                        "strict_gt": {"value": None, "ci_lo": None, "ci_hi": None, "n": len(score_pairs), "n_tasks": len(score_pairs)},
                        "tie_eq": {"value": None, "ci_lo": None, "ci_hi": None, "n": len(score_pairs), "n_tasks": len(score_pairs)},
                        "high_credit_ge3": hc_st,
                    }
                )
                continue

            drop_pairs, mono_pairs, gt_pairs, eq_pairs = [], [], [], []
            for t in tasks:
                if (t, cond) not in by_cond_task:
                    continue
                if (t, "faithful_gold_copy") not in by_cond_task:
                    raise ValueError(f"Missing faithful_gold_copy baseline for {dataset_name} task={t!r}, cond={cond!r}")
                sc = by_cond_task[(t, cond)][pname]
                sf = by_cond_task[(t, "faithful_gold_copy")][pname]
                drop_pairs.append((t, sf - sc))
                mono_pairs.append((t, 1.0 if sc >= sf - TOL else 0.0))
                gt_pairs.append((t, 1.0 if sc > sf + TOL else 0.0))
                eq_pairs.append((t, 1.0 if abs(sc - sf) <= TOL else 0.0))

            out.append(
                {
                    "dataset": dataset_name,
                    "condition": cond,
                    "category": cat,
                    "pipeline": pname,
                    "n_tasks": len(drop_pairs),
                    "gate_pass_count": gate_pass_count,
                    "mean_score": mean_score_st,
                    "mean_drop": cluster_mean_ci(drop_pairs),
                    "sign_p_val": exact_sign_test_p([d for _, d in drop_pairs]),
                    "V_mono": cluster_mean_ci(mono_pairs),
                    "strict_gt": cluster_mean_ci(gt_pairs),
                    "tie_eq": cluster_mean_ci(eq_pairs),
                    "high_credit_ge3": hc_st,
                }
            )
    return out


def analyze_e23(e23_dir: Path) -> dict[str, Any]:
    v1_dir = e23_dir / "expt_v1_tinyveribench256/results"
    v2_dir = e23_dir / "expt_v2_vb256/results"

    rec_v1_s30 = _score_e23_dataset(
        "tiny256_sample30",
        read_csv(v1_dir / "judge_tiny256_sample30.csv"),
        read_csv(v1_dir / "gate_tiny256_sample30.csv"),
        [("judge_only", None), ("gated", "passes_gate"), ("gated_format", "passes_gate_and_format")],
    )
    rec_v1_perm = _score_e23_dataset(
        "tiny256_perm_sorted_pairwise",
        read_csv(v1_dir / "judge_tiny256_perm_sorted_pairwise.csv"),
        read_csv(v1_dir / "gate_tiny256_perm_sorted_pairwise.csv"),
        [("judge_only", None), ("gated", "passes_gate"), ("gated_format", "passes_gate_and_format")],
    )
    rec_v2_s30 = _score_e23_dataset(
        "vb256_sample30",
        read_csv(v2_dir / "judge_vb256_sample30.csv"),
        read_csv(v2_dir / "gate_vb256.csv"),
        [("judge_only", None), ("gated_kernel", "passes_gate_kernel"), ("full_pipeline", "passes_all_checks")],
    )

    # Alignment check B against summary_judge_30.csv (33 rows, 2dp CSV values)
    s30_ref = {(r["condition"], r["score"]): r for r in read_csv(v1_dir / "summary_judge_30.csv")}
    max_e23_diff = 0.0
    e23_checks = 0
    for r in rec_v1_s30:
        ref = s30_ref.get((r["condition"], r["pipeline"]))
        if ref is None:
            raise RuntimeError(f"Missing summary_judge_30 row for {(r['condition'], r['pipeline'])}")
        d_mean = abs(r["mean_score"]["value"] - float(ref["mean"]))
        d_drop = abs(r["mean_drop"]["value"] - float(ref["mean_drop"]))
        if not (math.isfinite(d_mean) and math.isfinite(d_drop)):
            raise RuntimeError(f"Non-finite E23 alignment diff for {(r['condition'], r['pipeline'])}")
        max_e23_diff = max(max_e23_diff, d_mean, d_drop)
        e23_checks += 2
    if e23_checks == 0 or not math.isfinite(max_e23_diff) or max_e23_diff > 0.0051:
        raise RuntimeError(f"E23 alignment check failed: checks={e23_checks}, max_abs_diff={max_e23_diff}")

    all_e23 = rec_v1_s30 + rec_v1_perm + rec_v2_s30

    # Evaluate preregistered H1-game rule (exact 7 conditions listed in PREREG.md)
    prereg_game_conditions = (
        "sorry_proofs",
        "false_axiom_proofs",
        "shadowed_perm_sorted_pairwise",
        "dropped_most_theorems",
        "theorem_spam",
        "weakened_props",
        "vacuous_true_props",
    )
    judge_only_gaming = [
        r for r in all_e23 if r["pipeline"] == "judge_only" and r["condition"] in prereg_game_conditions
    ]
    game_triggers = [
        {
            "dataset": r["dataset"],
            "condition": r["condition"],
            "category": r["category"],
            "V_mono": r["V_mono"]["value"],
            "ci_lo": r["V_mono"]["ci_lo"],
            "high_credit_ge3": r["high_credit_ge3"]["value"],
        }
        for r in judge_only_gaming
        if r["V_mono"]["value"] is not None and r["V_mono"]["value"] >= 0.20 and r["V_mono"]["ci_lo"] > 0.05
    ]

    # Check gated reduction on gated proof/format attacks (sorry_proofs, false_axiom_proofs, off_format_missing_sections)
    gated_proof_fmt = [
        r
        for r in all_e23
        if r["condition"] in ("sorry_proofs", "false_axiom_proofs", "off_format_missing_sections")
        and (
            (r["dataset"].startswith("tiny256") and r["pipeline"] == "gated_format")
            or (r["dataset"] == "vb256_sample30" and r["pipeline"] == "full_pipeline")
        )
    ]
    if not judge_only_gaming or not gated_proof_fmt:
        raise RuntimeError("Missing required H1-game evaluation subsets")
    gated_drops_below_05 = all(r["V_mono"]["value"] is not None and r["V_mono"]["value"] < 0.05 for r in gated_proof_fmt)
    falsified_game = all(
        r["V_mono"]["value"] < 0.05 and r["high_credit_ge3"]["value"] < 0.05 for r in judge_only_gaming
    )
    h1_game_verdict = (
        "SUPPORTED" if (game_triggers and gated_drops_below_05) else ("FALSIFIED" if falsified_game else "MIXED")
    )

    return {
        "alignment_e23": {"checks": e23_checks, "max_abs_diff": max_e23_diff, "passed": max_e23_diff <= 0.0051},
        "h1_game_verdict": h1_game_verdict,
        "game_triggers": game_triggers,
        "gated_proof_fmt_all_below_0_05": gated_drops_below_05,
        "records": all_e23,
    }


def write_csvs(out_dir: Path, jra_res: dict[str, Any], e23_res: dict[str, Any]) -> None:
    def _v(st: dict[str, Any], k: str) -> Any:
        return st.get(k) if st else None

    fields_a = [
        "config_id", "tier", "harness", "model", "effort", "method",
        "n_t1_equiv", "n_t1_equiv_same_effort", "n_mismatched_effort_pairs",
        "cm", "cd", "A_G", "A_G_ci_lo", "A_G_ci_hi", "A_G_p_val",
        "V_equiv", "V_equiv_ci_lo", "V_equiv_ci_hi",
        "V_equiv_same_effort", "V_equiv_same_effort_ci_lo", "V_equiv_same_effort_ci_hi",
        "mean_abs_equiv_shift",
        "harsh_flip_ge9", "harsh_flip_eq10",
        "n_rename_exact", "V_rename_exact", "V_rename_exact_ci_lo", "V_rename_exact_ci_hi",
        "n_order", "V_order", "V_order_ci_lo", "V_order_ci_hi", "order_bias_ab_minus_ba", "order_bias_p_val",
        "n_t0", "fc", "fc_ci_lo", "fc_ci_hi", "fc_harsh_ge9",
        "n_rep_items", "V_rep_ab", "V_rep_ab_ci_lo", "V_rep_ab_ci_hi", "rep_mean_range",
    ]
    with (out_dir / "equiv_invariance_table.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields_a, lineterminator="\n")
        w.writeheader()
        for r in jra_res["part_a_configs"]:
            w.writerow(
                {
                    "config_id": r["config_id"],
                    "tier": r["tier"],
                    "harness": r["harness"],
                    "model": r["model"],
                    "effort": r["effort"],
                    "method": r["method"],
                    "n_t1_equiv": r["V_equiv"]["n"],
                    "n_t1_equiv_same_effort": r["V_equiv_same_effort"]["n"],
                    "n_mismatched_effort_pairs": r["n_mismatched_equiv_effort"],
                    "cm": _v(r["cm"], "value"),
                    "cd": _v(r["cd"], "value"),
                    "A_G": _v(r["A_G"], "value"),
                    "A_G_ci_lo": _v(r["A_G"], "ci_lo"),
                    "A_G_ci_hi": _v(r["A_G"], "ci_hi"),
                    "A_G_p_val": r["A_G"].get("p_val"),
                    "V_equiv": _v(r["V_equiv"], "value"),
                    "V_equiv_ci_lo": _v(r["V_equiv"], "ci_lo"),
                    "V_equiv_ci_hi": _v(r["V_equiv"], "ci_hi"),
                    "V_equiv_same_effort": _v(r["V_equiv_same_effort"], "value"),
                    "V_equiv_same_effort_ci_lo": _v(r["V_equiv_same_effort"], "ci_lo"),
                    "V_equiv_same_effort_ci_hi": _v(r["V_equiv_same_effort"], "ci_hi"),
                    "mean_abs_equiv_shift": _v(r["mean_abs_equiv_shift"], "value"),
                    "harsh_flip_ge9": _v(r["harsh_flip_ge9"], "value"),
                    "harsh_flip_eq10": _v(r["harsh_flip_eq10"], "value"),
                    "n_rename_exact": r["V_rename_exact"]["n"],
                    "V_rename_exact": _v(r["V_rename_exact"], "value"),
                    "V_rename_exact_ci_lo": _v(r["V_rename_exact"], "ci_lo"),
                    "V_rename_exact_ci_hi": _v(r["V_rename_exact"], "ci_hi"),
                    "n_order": r["V_order"]["n"],
                    "V_order": _v(r["V_order"], "value"),
                    "V_order_ci_lo": _v(r["V_order"], "ci_lo"),
                    "V_order_ci_hi": _v(r["V_order"], "ci_hi"),
                    "order_bias_ab_minus_ba": _v(r["order_bias_ab_minus_ba"], "value"),
                    "order_bias_p_val": r["order_bias_ab_minus_ba"].get("p_val"),
                    "n_t0": r["fc"]["n"],
                    "fc": _v(r["fc"], "value"),
                    "fc_ci_lo": _v(r["fc"], "ci_lo"),
                    "fc_ci_hi": _v(r["fc"], "ci_hi"),
                    "fc_harsh_ge9": _v(r["fc_harsh_ge9"], "value"),
                    "n_rep_items": r["V_rep_ab"]["n"],
                    "V_rep_ab": _v(r["V_rep_ab"], "value"),
                    "V_rep_ab_ci_lo": _v(r["V_rep_ab"], "ci_lo"),
                    "V_rep_ab_ci_hi": _v(r["V_rep_ab"], "ci_hi"),
                    "rep_mean_range": _v(r["rep_mean_range"], "value"),
                }
            )

    fields_c = [
        "config_id", "harness", "model", "method", "n_pairs", "n_tasks",
        "V_ref_swap", "V_ref_swap_ci_lo", "V_ref_swap_ci_hi",
        "mean_abs_fsc_shift", "mean_abs_fsc_shift_ci_lo", "mean_abs_fsc_shift_ci_hi",
        "signed_shift_RA_minus_RB", "signed_shift_p_val",
        "harsh_flip_nodt_eq1", "harsh_flip_nodt_eq1_ci_lo", "harsh_flip_nodt_eq1_ci_hi",
        "harsh_flip_nodt_ge09", "harsh_flip_dt_eq1",
    ]
    with (out_dir / "exp_c_ref_swap_table.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields_c, lineterminator="\n")
        w.writeheader()
        for r in jra_res["part_a_exp_c"]:
            w.writerow(
                {
                    "config_id": r["config_id"],
                    "harness": r["harness"],
                    "model": r["model"],
                    "method": r["method"],
                    "n_pairs": r["n_pairs"],
                    "n_tasks": r["V_ref_swap"]["n_tasks"],
                    "V_ref_swap": r["V_ref_swap"]["value"],
                    "V_ref_swap_ci_lo": r["V_ref_swap"]["ci_lo"],
                    "V_ref_swap_ci_hi": r["V_ref_swap"]["ci_hi"],
                    "mean_abs_fsc_shift": r["mean_abs_fsc_shift"]["value"],
                    "mean_abs_fsc_shift_ci_lo": r["mean_abs_fsc_shift"]["ci_lo"],
                    "mean_abs_fsc_shift_ci_hi": r["mean_abs_fsc_shift"]["ci_hi"],
                    "signed_shift_RA_minus_RB": r["signed_shift_RA_minus_RB"]["value"],
                    "signed_shift_p_val": r["signed_shift_RA_minus_RB"]["p_val"],
                    "harsh_flip_nodt_eq1": r["harsh_flip_nodt_eq1"]["value"],
                    "harsh_flip_nodt_eq1_ci_lo": r["harsh_flip_nodt_eq1"]["ci_lo"],
                    "harsh_flip_nodt_eq1_ci_hi": r["harsh_flip_nodt_eq1"]["ci_hi"],
                    "harsh_flip_nodt_ge09": r["harsh_flip_nodt_ge09"]["value"],
                    "harsh_flip_dt_eq1": r["harsh_flip_dt_eq1"]["value"],
                }
            )

    fields_g = [
        "dataset", "condition", "category", "pipeline", "n_tasks", "gate_pass_count",
        "mean_score", "mean_score_ci_lo", "mean_score_ci_hi",
        "mean_drop", "mean_drop_ci_lo", "mean_drop_ci_hi", "sign_p_val",
        "V_mono", "V_mono_ci_lo", "V_mono_ci_hi", "strict_gt", "tie_eq",
        "high_credit_ge3", "high_credit_ge3_ci_lo", "high_credit_ge3_ci_hi",
    ]
    with (out_dir / "gaming_monotonicity_table.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields_g, lineterminator="\n")
        w.writeheader()
        for r in e23_res["records"]:
            w.writerow(
                {
                    "dataset": r["dataset"],
                    "condition": r["condition"],
                    "category": r["category"],
                    "pipeline": r["pipeline"],
                    "n_tasks": r["n_tasks"],
                    "gate_pass_count": r["gate_pass_count"],
                    "mean_score": r["mean_score"]["value"],
                    "mean_score_ci_lo": r["mean_score"]["ci_lo"],
                    "mean_score_ci_hi": r["mean_score"]["ci_hi"],
                    "mean_drop": r["mean_drop"]["value"],
                    "mean_drop_ci_lo": r["mean_drop"]["ci_lo"],
                    "mean_drop_ci_hi": r["mean_drop"]["ci_hi"],
                    "sign_p_val": r["sign_p_val"],
                    "V_mono": r["V_mono"]["value"],
                    "V_mono_ci_lo": r["V_mono"]["ci_lo"],
                    "V_mono_ci_hi": r["V_mono"]["ci_hi"],
                    "strict_gt": r["strict_gt"]["value"],
                    "tie_eq": r["tie_eq"]["value"],
                    "high_credit_ge3": r["high_credit_ge3"]["value"],
                    "high_credit_ge3_ci_lo": r["high_credit_ge3"]["ci_lo"],
                    "high_credit_ge3_ci_hi": r["high_credit_ge3"]["ci_hi"],
                }
            )


def make_figures(out_dir: Path, jra_res: dict[str, Any], e23_res: dict[str, Any]) -> None:
    # Figure 1: Equivalence & Order Invariance + Real VeriBench Exp C swaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.0), dpi=160)

    ax1.grid(True, linestyle="-", alpha=0.35)
    # Focus Panel A on M1 (deployed), M3 (prompt), M5 (Lean-kernel harness), M6 (CLI agent) across models
    order_cfgs = [
        ("claude-sonnet-4-6::M1_deployed_tc_v1", "Sonnet-4.6 M1\n(single-call)"),
        ("claude-sonnet-4-6::M3_prompt_nonexcl_cot", "Sonnet-4.6 M3\n(prompt CoT)"),
        ("claude-sonnet-4-6::M5_harness_ref_hidden_lean", "Sonnet-4.6 M5\n(Lean harness)"),
        ("claude-sonnet-4-6-agent::M6_harness_cli_agent", "Sonnet-4.6 M6\n(CLI agent)"),
        ("gpt-5.4::M1_deployed_tc_v1", "GPT-5.4 M1\n(single-call)"),
        ("gpt-5.4::M5_harness_ref_hidden_lean", "GPT-5.4 M5\n(Lean harness)"),
        ("pi-frontier-gpt54::M6_harness_cli_agent", "GPT-5.4 M6\n(Pi CLI agent)"),
        ("leanstral::M1_deployed_tc_v1", "Leanstral M1\n(single-call)"),
        ("leanstral::M5_harness_ref_hidden_lean", "Leanstral M5\n(Lean harness)"),
        ("ministral-14b::M1_deployed_tc_v1", "Ministral-14B M1\n(single-call)"),
        ("ministral-14b::M5_harness_ref_hidden_lean", "Ministral-14B M5\n(Lean harness)"),
        ("qwen3.5-9b::M1_deployed_tc_v1", "Qwen3.5-9B M1\n(single-call)"),
    ]
    by_id = {r["config_id"]: r for r in jra_res["part_a_configs"]}
    labels1, vals1, lo1, hi1, cols1 = [], [], [], [], []
    color_map = {
        "M1_deployed_tc_v1": "#d62728",
        "M3_prompt_nonexcl_cot": "#ff7f0e",
        "M5_harness_ref_hidden_lean": "#1f77b4",
        "M6_harness_cli_agent": "#2ca02c",
    }
    for cid, lab in order_cfgs:
        r = by_id[cid]
        st = r["V_equiv"]
        v = st["value"]
        labels1.append(lab)
        vals1.append(v)
        lo1.append(v - st["ci_lo"])
        hi1.append(st["ci_hi"] - v)
        cols1.append(color_map[r["method"]])

    x1 = np.arange(len(labels1))
    ax1.bar(x1, vals1, yerr=[lo1, hi1], capsize=3, color=cols1, alpha=0.88, edgecolor="black", linewidth=0.6, zorder=3)
    ax1.axhline(0.25, color="#8c564b", linestyle="--", linewidth=1.2, label="H1-equiv frontier cutoff (>= 0.25)")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(labels1, rotation=38, ha="right", fontsize=8)
    ax1.set_ylim(0.0, 1.14)
    ax1.set_ylabel("Reference-equivalence violation rate V_equiv\n(T1 pairs with match_refactored != diff_gold; lower = better)", fontsize=9.2)
    ax1.set_xlabel("VB FSC judge configuration (red: M1; orange: prompt M3; blue/green: M5/M6)", fontsize=9.2)
    ax1.set_title("Equivalence invariance failures on VB FSC T1 pairs", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.5)

    # Panel B: Real VeriBench Exp C (7-10 tasks x 5 solvers)
    ax2.grid(True, linestyle="-", alpha=0.35)
    ec_list = [r for r in jra_res["part_a_exp_c"] if r["V_ref_swap"]["value"] is not None]
    labels2 = [f"{r['config_id'].replace('::', chr(10))}\n(n={r['n_pairs']})" for r in ec_list]
    v_sw = [r["V_ref_swap"]["value"] for r in ec_list]
    lo_sw = [r["V_ref_swap"]["value"] - r["V_ref_swap"]["ci_lo"] for r in ec_list]
    hi_sw = [r["V_ref_swap"]["ci_hi"] - r["V_ref_swap"]["value"] for r in ec_list]
    v_fl = [r["harsh_flip_nodt_eq1"]["value"] for r in ec_list]
    lo_fl = [r["harsh_flip_nodt_eq1"]["value"] - r["harsh_flip_nodt_eq1"]["ci_lo"] for r in ec_list]
    hi_fl = [r["harsh_flip_nodt_eq1"]["ci_hi"] - r["harsh_flip_nodt_eq1"]["value"] for r in ec_list]

    x2 = np.arange(len(labels2))
    w = 0.36
    ax2.bar(x2 - w / 2, v_sw, yerr=[lo_sw, hi_sw], width=w, capsize=3, color="#1f77b4", alpha=0.88, edgecolor="black", linewidth=0.6, label="Score shift |fsc_R_A - fsc_R_B| > 0", zorder=3)
    ax2.bar(x2 + w / 2, v_fl, yerr=[lo_fl, hi_fl], width=w, capsize=3, color="#d62728", alpha=0.88, edgecolor="black", linewidth=0.6, label="Harsh pass flip (R_A != R_B, no DT)", zorder=3)
    ax2.axhline(0.25, color="#8c564b", linestyle="--", linewidth=1.2, label="H1-equiv swap cutoff (>= 0.25)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2, rotation=22, ha="right", fontsize=8.5)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("Real VeriBench violation / flip rate\n(35-50 pairs, 7-10 tasks x 5 solver agents; lower = better)", fontsize=9.2)
    ax2.set_xlabel("VB FSC judge x method (real solver-agent outputs, R_A vs. R_B)", fontsize=9.2)
    ax2.set_title("Real VeriBench R_A vs. R_B score shifts & harsh flips", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        "Exp 04 Part A (11-24-2026): VB FSC judge LLMs fail semantic-equivalence invariance",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "fig1_vb_fsc_equivalence_and_order_invariance.png", dpi=160)
    plt.close(fig)

    # Figure 2: Adversarial gaming monotonicity on TinyVeriBench-256 (judge_only vs gated_format)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2), dpi=160)
    e23_map = {(r["dataset"], r["condition"], r["pipeline"]): r for r in e23_res["records"]}
    conds_plot = [
        ("tiny256_sample30", "sorry_proofs", "sorry_proofs\n(proof)"),
        ("tiny256_sample30", "false_axiom_proofs", "false_axiom\n(proof)"),
        ("tiny256_sample30", "off_format_missing_sections", "off_format\n(format)"),
        ("tiny256_perm_sorted_pairwise", "shadowed_perm_sorted_pairwise", "shadow_perm\n(stmt, n=12)"),
        ("tiny256_sample30", "dropped_most_theorems", "dropped_thms\n(stmt)"),
        ("tiny256_sample30", "theorem_spam", "theorem_spam\n(stmt)"),
        ("tiny256_sample30", "vacuous_true_props", "vacuous_true\n(stmt)"),
        ("tiny256_sample30", "weakened_props", "weakened_props\n(stmt)"),
        ("tiny256_sample30", "weak_prompt_natural", "weak_prompt\n(natural)"),
    ]

    x = np.arange(len(conds_plot))
    w = 0.27
    jo_v, jo_lo, jo_hi = [], [], []
    gf_v, gf_lo, gf_hi = [], [], []
    hc_v, hc_lo, hc_hi = [], [], []
    labs = []
    for ds, cond, lab in conds_plot:
        r_jo = e23_map[(ds, cond, "judge_only")]
        r_gf = e23_map[(ds, cond, "gated_format")]
        labs.append(lab)
        jo_v.append(r_jo["V_mono"]["value"])
        jo_lo.append(r_jo["V_mono"]["value"] - r_jo["V_mono"]["ci_lo"])
        jo_hi.append(r_jo["V_mono"]["ci_hi"] - r_jo["V_mono"]["value"])
        gf_v.append(r_gf["V_mono"]["value"])
        gf_lo.append(r_gf["V_mono"]["value"] - r_gf["V_mono"]["ci_lo"])
        gf_hi.append(r_gf["V_mono"]["ci_hi"] - r_gf["V_mono"]["value"])
        hc_v.append(r_jo["high_credit_ge3"]["value"])
        hc_lo.append(r_jo["high_credit_ge3"]["value"] - r_jo["high_credit_ge3"]["ci_lo"])
        hc_hi.append(r_jo["high_credit_ge3"]["ci_hi"] - r_jo["high_credit_ge3"]["value"])

    ax1.grid(True, linestyle="-", alpha=0.35)
    ax1.bar(x - w, jo_v, yerr=[jo_lo, jo_hi], width=w, capsize=2.5, color="#d62728", alpha=0.88, edgecolor="black", linewidth=0.5, label="judge_only V_mono (score >= faithful)", zorder=3)
    ax1.bar(x, hc_v, yerr=[hc_lo, hc_hi], width=w, capsize=2.5, color="#ff7f0e", alpha=0.88, edgecolor="black", linewidth=0.5, label="judge_only P(score >= 3/5)", zorder=3)
    ax1.bar(x + w, gf_v, yerr=[gf_lo, gf_hi], width=w, capsize=2.5, color="#1f77b4", alpha=0.88, edgecolor="black", linewidth=0.5, label="gated_format V_mono", zorder=3)
    ax1.axhline(0.20, color="#8c564b", linestyle="--", linewidth=1.2, label="H1-game violation cutoff (>= 0.20)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labs, rotation=32, ha="right", fontsize=8.2)
    ax1.set_ylim(0.0, 1.24)
    ax1.set_ylabel("Violation / high-credit rate vs. faithful_gold_copy\n(TinyVeriBench-256 sample30 / perm12; lower = better)", fontsize=9.2)
    ax1.set_xlabel("Adversarial / control condition (Claude Opus 5.5 via claude -p)", fontsize=9.2)
    ax1.set_title("Monotonicity violations & >=3/5 credit leakage", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)

    ax2.grid(True, linestyle="-", alpha=0.35)
    d_jo = [e23_map[(ds, cond, "judge_only")]["mean_drop"]["value"] for ds, cond, _ in conds_plot]
    d_jo_lo = [e23_map[(ds, cond, "judge_only")]["mean_drop"]["value"] - e23_map[(ds, cond, "judge_only")]["mean_drop"]["ci_lo"] for ds, cond, _ in conds_plot]
    d_jo_hi = [e23_map[(ds, cond, "judge_only")]["mean_drop"]["ci_hi"] - e23_map[(ds, cond, "judge_only")]["mean_drop"]["value"] for ds, cond, _ in conds_plot]
    d_gf = [e23_map[(ds, cond, "gated_format")]["mean_drop"]["value"] for ds, cond, _ in conds_plot]
    d_gf_lo = [e23_map[(ds, cond, "gated_format")]["mean_drop"]["value"] - e23_map[(ds, cond, "gated_format")]["mean_drop"]["ci_lo"] for ds, cond, _ in conds_plot]
    d_gf_hi = [e23_map[(ds, cond, "gated_format")]["mean_drop"]["ci_hi"] - e23_map[(ds, cond, "gated_format")]["mean_drop"]["value"] for ds, cond, _ in conds_plot]

    w2 = 0.36
    ax2.bar(x - w2 / 2, d_jo, yerr=[d_jo_lo, d_jo_hi], width=w2, capsize=3, color="#d62728", alpha=0.88, edgecolor="black", linewidth=0.5, label="judge_only mean drop", zorder=3)
    ax2.bar(x + w2 / 2, d_gf, yerr=[d_gf_lo, d_gf_hi], width=w2, capsize=3, color="#1f77b4", alpha=0.88, edgecolor="black", linewidth=0.5, label="gated_format mean drop", zorder=3)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labs, rotation=32, ha="right", fontsize=8.2)
    ax2.set_ylim(-0.3, 6.2)
    ax2.set_ylabel("Mean score drop (faithful - condition, 0-5 scale)\n(higher = stronger penalty against gamed/flawed file)", fontsize=9.2)
    ax2.set_xlabel("Adversarial / control condition (Claude Opus 5.5 via claude -p)", fontsize=9.2)
    ax2.set_title("Score penalty with vs. without Lean/format gating", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        "Exp 04 Part B (11-24-2026): Statement-only VB FSC judge LLM needs compile/source-scan/format gating (TinyVeriBench-256)",
        fontsize=11.2,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "fig2_vb_fsc_gaming_monotonicity.png", dpi=160)
    plt.close(fig)


def run_analysis(jra_dir: Path, e23_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jra_res = analyze_jra(jra_dir)
    e23_res = analyze_e23(e23_dir)
    write_csvs(out_dir, jra_res, e23_res)
    summary = {
        "date": "2026-11-24",
        "experiment": "04_vb_fsc_equivalence_invariance_and_gaming_violations",
        "h1_equiv_verdict": jra_res["h1_equiv_verdict"],
        "h1_game_verdict": e23_res["h1_game_verdict"],
        "alignment_jra": jra_res["alignment_jra"],
        "alignment_e23": e23_res["alignment_e23"],
        "equiv_triggers": jra_res["equiv_triggers"],
        "game_triggers": e23_res["game_triggers"],
        "gated_proof_fmt_all_below_0_05": e23_res["gated_proof_fmt_all_below_0_05"],
        "part_a_configs": jra_res["part_a_configs"],
        "part_a_exp_c": jra_res["part_a_exp_c"],
        "part_b_records": e23_res["records"],
    }
    write_json(out_dir / "summary.json", summary)
    make_figures(out_dir, jra_res, e23_res)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze VB FSC equivalence invariance and gaming monotonicity.")
    parser.add_argument("--jra-dir", type=Path, default=DEFAULT_JRA_DIR)
    parser.add_argument("--e23-dir", type=Path, default=DEFAULT_E23_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_analysis(args.jra_dir, args.e23_dir, args.out_dir)
    print(
        json.dumps(
            {
                "h1_equiv_verdict": summary["h1_equiv_verdict"],
                "h1_game_verdict": summary["h1_game_verdict"],
                "alignment_jra": summary["alignment_jra"],
                "alignment_e23": summary["alignment_e23"],
                "n_equiv_triggers": len(summary["equiv_triggers"]),
                "n_game_triggers": len(summary["game_triggers"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
