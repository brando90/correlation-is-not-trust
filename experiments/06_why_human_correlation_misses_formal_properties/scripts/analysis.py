#!/usr/bin/env python3
"""Experiment 06 CPU arm (2026-11-26).

Reads public logs from experiments 03, 04, and 05. No model calls.
The confirmatory cuts are frozen in ../PREREG.md. Do not change the
constants below without a dated amendment to that file.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

EXP_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS = EXP_DIR.parent
EXP05 = EXPERIMENTS / "05_renaming_and_inserted_bug_on_veribench_judge_outputs" / "results"
EXP03 = EXPERIMENTS / "03_human_corr_vs_axiomatic_properties_cert_judge" / "results"
EXP04 = EXPERIMENTS / "04_vb_fsc_equivalence_invariance_and_gaming_violations" / "results"

SEED = 20261124
DRAWS = 2000
PERM_DRAWS = 5000
TOL = 0.05
EPS = 1e-9
MATERIAL = 0.20
CEILING = 0.95
FLOOR = 0.20

# PREREG cuts.
MATERIAL_SUPPORT_POINT_MAX = 0.15
MATERIAL_SUPPORT_CI_HI_MAX = 0.30
MATERIAL_FALSIFY_POINT_MIN = 0.30
MATERIAL_FALSIFY_CI_LO_MIN = 0.15
RANGE_SUPPORT_MIN = 0.10
RANGE_FALSIFY_MAX = 0.02
SPLIT_TRACK_POINT = -0.40
SPLIT_TRACK_CI_HI = -0.20
SPLIT_ABSENT_ABS = 0.25
SPLIT_P_MAX = 0.05
SPLIT_MIN_VALID = 1500
ARM_I_VB_MAX = 0.20
ARM_I_VB_CI_HI = 0.40
ARM_I_DROP_MIN = 0.30
ARM_I_FALSIFY_POINT = 0.40
ARM_I_FALSIFY_LO = 0.20
ARM_I_N_BUG = 37
ARM_I_N_ID1 = 24

CLASSES = ("ceiling_tie", "mid_tie", "floor_tie", "penalized", "inversion")
CLASS_COLORS = {
    "ceiling_tie": "#4C78A8",
    "mid_tie": "#F58518",
    "floor_tie": "#54A24B",
    "penalized": "#E45756",
    "inversion": "#B279A2",
}
CONFIG_ORDER = ("A", "A2", "B")
CONFIG_META = {
    "A": {
        "judge_llm": "GPT-6 Astra",
        "model_id": "gpt-6-astra",
        "harness": "Codex CLI",
        "plan": "Vals",
        "lane": "vals-codex",
        "rubric": "P4_CHECKLIST",
        "role": "replication",
    },
    "A2": {
        "judge_llm": "GPT-6 Astra",
        "model_id": "gpt-6-astra",
        "harness": "Codex CLI",
        "plan": "Vals",
        "lane": "vals-codex",
        "rubric": "P8_STRICT_B",
        "role": "replication",
    },
    "B": {
        "judge_llm": "Claude Opus 5.5",
        "model_id": "claude-opus-5-5",
        "harness": "Claude Code CLI",
        "plan": "Stanford",
        "lane": "stanford-claude",
        "rubric": "P4_CHECKLIST",
        "role": "confirmatory",
    },
}
FRONTIER_M1_SEEN_AT_PREREG = {
    "claude-sonnet-4-6::M1_deployed_tc_v1",
    "gpt-5.4::M1_deployed_tc_v1",
}
CRIT_LOW_EXPECTED = {"A": 61, "A2": 71}
EXP03_SPEARMAN_VWEAK = -0.7470214475096882


def _finite(value: float, label: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"non-finite {label}")
    return out


def _corr_stat(result) -> float:
    if hasattr(result, "statistic"):
        return float(result.statistic)
    return float(result[0])


def spearman(xs, ys):
    if len(xs) < 3 or len(ys) != len(xs):
        return None
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    val = _corr_stat(spearmanr(xs, ys))
    if not math.isfinite(val):
        return None
    return val


def classify_pair(id1: float, bug: float) -> str:
    """First match wins. Cuts are the PREREG pair classes."""
    id1 = _finite(id1, "id1")
    bug = _finite(bug, "bug")
    if id1 < -EPS or id1 > 1.0 + EPS or bug < -EPS or bug > 1.0 + EPS:
        raise ValueError(f"score outside [0, 1]: id1={id1} bug={bug}")
    delta = bug - id1
    if delta > TOL + EPS:
        return "inversion"
    if delta < -(TOL + EPS):
        return "penalized"
    if id1 >= CEILING - EPS:
        return "ceiling_tie"
    if id1 <= FLOOR + EPS and bug <= FLOOR + EPS:
        return "floor_tie"
    return "mid_tie"


def is_material(id1: float, bug: float) -> bool:
    return (id1 - bug) > MATERIAL + EPS


def is_vbug(id1: float, bug: float) -> bool:
    return bug >= id1 - EPS


def is_strict(id1: float, bug: float) -> bool:
    return bug > id1 + EPS


def cluster_stat(pairs, stat, seed: int = SEED, draws: int = DRAWS):
    """Task-cluster percentile interval. pairs are (task, value)."""
    if not pairs:
        return {"value": None, "ci_lo": None, "ci_hi": None, "n": 0, "n_tasks": 0, "valid_draws": 0}
    by: dict[str, list[float]] = {}
    for task, val in pairs:
        by.setdefault(str(task), []).append(_finite(val, "pair"))
    tasks = sorted(by)
    values = [v for task in tasks for v in by[task]]
    point = stat(values)
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    n_tasks = len(tasks)
    for _ in range(draws):
        draw = rng.integers(0, n_tasks, size=n_tasks)
        sample = [v for i in draw for v in by[tasks[i]]]
        got = stat(sample)
        if got is not None and math.isfinite(got):
            boots.append(float(got))
    if boots:
        lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    else:
        lo = hi = None
    point_f = None if point is None or not math.isfinite(point) else float(point)
    return {
        "value": point_f,
        "ci_lo": lo,
        "ci_hi": hi,
        "n": len(values),
        "n_tasks": n_tasks,
        "valid_draws": len(boots),
    }


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def _median(xs):
    return float(np.median(xs)) if len(xs) else None


def cluster_spearman(rows, seed: int = SEED, draws: int = DRAWS):
    """rows are (task, x, y), one row per task."""
    empty = {"value": None, "ci_lo": None, "ci_hi": None, "n": len(rows), "n_tasks": len(rows), "valid_draws": 0}
    if len(rows) < 3:
        return empty
    tasks = [r[0] for r in rows]
    if len(tasks) != len(set(tasks)):
        raise ValueError("duplicate task in Spearman rows")
    order = sorted(range(len(rows)), key=lambda i: tasks[i])
    xs = [_finite(rows[i][1], "x") for i in order]
    ys = [_finite(rows[i][2], "y") for i in order]
    point = spearman(xs, ys)
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    n_tasks = len(xs)
    for _ in range(draws):
        draw = rng.integers(0, n_tasks, size=n_tasks)
        got = spearman([xs[i] for i in draw], [ys[i] for i in draw])
        if got is not None:
            boots.append(got)
    if boots:
        lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    else:
        lo = hi = None
    return {
        "value": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "n": n_tasks,
        "n_tasks": n_tasks,
        "valid_draws": len(boots),
    }


def partial_spearman(xs, ys, zs):
    """Pearson correlation of rank residuals. None if a residual is constant."""
    x = [float(v) for v in xs]
    y = [float(v) for v in ys]
    z = [float(v) for v in zs]
    n = len(x)
    if n < 5 or len(y) != n or len(z) != n:
        return None
    if any(not math.isfinite(v) for v in x + y + z):
        return None
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    rz = rankdata(z, method="average")

    def resid(target, covariate):
        design = np.column_stack([np.ones(n), covariate])
        coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        leftover = target - design @ coef
        if float(np.var(leftover)) <= 1e-12:
            return None
        return leftover

    rx_r = resid(rx, rz)
    ry_r = resid(ry, rz)
    if rx_r is None or ry_r is None:
        return None
    val = _corr_stat(pearsonr(rx_r, ry_r))
    if not math.isfinite(val):
        return None
    return val


def partial_bundle(xs, ys, zs, seed: int = SEED, draws: int = DRAWS, perm_draws: int = PERM_DRAWS):
    point = partial_spearman(xs, ys, zs)
    n = len(xs)
    rng_b = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(draws):
        idx = rng_b.integers(0, n, size=n)
        got = partial_spearman([xs[i] for i in idx], [ys[i] for i in idx], [zs[i] for i in idx])
        if got is not None:
            boots.append(got)
    if boots:
        lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    else:
        lo = hi = None
    rng_p = np.random.default_rng(seed)
    order = np.arange(n)
    ge = 0
    valid_perm = 0
    if point is not None:
        for _ in range(perm_draws):
            rng_p.shuffle(order)
            got = partial_spearman([xs[i] for i in order], ys, zs)
            if got is not None:
                valid_perm += 1
                if abs(got) >= abs(point) - 1e-12:
                    ge += 1
    p_val = (ge + 1) / (valid_perm + 1) if valid_perm else None
    return {
        "value": None if point is None else float(point),
        "ci_lo": lo,
        "ci_hi": hi,
        "p_val": None if p_val is None else float(p_val),
        "n": n,
        "n_tasks": n,
        "valid_draws": len(boots),
        "valid_perm": valid_perm,
    }


def load_score_csv(path: Path):
    rows = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            pid = rec["probe_id"]
            if pid in seen:
                raise ValueError(f"duplicate probe_id in {path}: {pid}")
            seen.add(pid)
            score = _finite(rec["score"], pid)
            if score < -EPS or score > 1.0 + EPS:
                raise ValueError(f"score outside [0, 1] for {pid}")
            kind = rec["kind"]
            if not pid.startswith(kind + "__") and not (kind == "CRIT" and pid.startswith("CRIT__")):
                raise ValueError(f"probe_id {pid} does not start with kind {kind}")
            rows.append(
                {
                    "probe_id": pid,
                    "kind": kind,
                    "task": rec["task"],
                    "level": int(float(rec["level"])),
                    "score": score,
                }
            )
    return rows


def index_id1(rows):
    id1 = {}
    for row in rows:
        if row["kind"] != "ID1":
            continue
        if row["task"] in id1:
            raise ValueError(f"two ID1 rows for task {row['task']}")
        id1[row["task"]] = row["score"]
    return id1


def bug_pair_rows(rows, id1):
    pairs = []
    for row in rows:
        if row["kind"] != "BUG":
            continue
        if row["task"] not in id1:
            raise ValueError(f"BUG without ID1: {row['probe_id']}")
        base = id1[row["task"]]
        pairs.append(
            {
                "task": row["task"],
                "level": row["level"],
                "id1": base,
                "bug": row["score"],
                "drop": base - row["score"],
                "pair_class": classify_pair(base, row["score"]),
                "material": is_material(base, row["score"]),
                "v_bug": is_vbug(base, row["score"]),
                "strict": is_strict(base, row["score"]),
            }
        )
    return pairs


def _indicator_ci(pairs, pred, seed: int):
    return cluster_stat([(p["task"], 1.0 if pred(p) else 0.0) for p in pairs], _mean, seed=seed)


def gap_ci(crit_pairs, id1_map, seed: int = SEED, draws: int = DRAWS):
    crit_by: dict[str, list[float]] = {}
    for task, score in crit_pairs:
        crit_by.setdefault(task, []).append(_finite(score, "crit"))
    id1 = {task: _finite(score, "id1") for task, score in id1_map.items()}
    tasks = sorted(set(crit_by) | set(id1))

    def stat(drawn):
        crit_vals = []
        id_vals = []
        for task in drawn:
            if task in crit_by:
                crit_vals.extend(crit_by[task])
            if task in id1:
                id_vals.append(id1[task])
        if not crit_vals or not id_vals:
            return None
        return float(np.mean(id_vals) - np.mean(crit_vals))

    point = stat(tasks)
    rng = np.random.default_rng(seed)
    boots = []
    n_tasks = len(tasks)
    for _ in range(draws):
        draw = [tasks[i] for i in rng.integers(0, n_tasks, size=n_tasks)]
        got = stat(draw)
        if got is not None and math.isfinite(got):
            boots.append(got)
    if boots:
        lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    else:
        lo = hi = None
    return {
        "value": None if point is None else float(point),
        "ci_lo": lo,
        "ci_hi": hi,
        "n": len(crit_pairs),
        "n_tasks": n_tasks,
        "valid_draws": len(boots),
    }


def rename_block(rows, id1, kind: str):
    selected = []
    for row in rows:
        if row["kind"] != kind:
            continue
        if row["task"] not in id1:
            raise ValueError(f"{kind} without ID1: {row['probe_id']}")
        selected.append((row["task"], row["score"] - id1[row["task"]]))
    if not selected:
        return None

    def share(cut):
        return cluster_stat([(task, 1.0 if cut(delta) else 0.0) for task, delta in selected], _mean)

    return {
        "n": len(selected),
        "down_gt_0_20": share(lambda d: d < -(MATERIAL + EPS)),
        "down_gt_0_05": share(lambda d: d < -(TOL + EPS)),
        "up_gt_0_05": share(lambda d: d > TOL + EPS),
        "unchanged": share(lambda d: abs(d) <= TOL + EPS),
    }


def verdict_material(block):
    value, lo, hi = block["value"], block["ci_lo"], block["ci_hi"]
    if value is None or lo is None or hi is None or block["valid_draws"] != DRAWS:
        return "not_computable"
    if value <= MATERIAL_SUPPORT_POINT_MAX and hi < MATERIAL_SUPPORT_CI_HI_MAX:
        return "supported"
    if value >= MATERIAL_FALSIFY_POINT_MIN and lo > MATERIAL_FALSIFY_CI_LO_MIN:
        return "falsified"
    return "mixed"


def verdict_range(median):
    if median is None or not math.isfinite(median):
        return "not_computable"
    if median >= RANGE_SUPPORT_MIN:
        return "supported"
    if median <= RANGE_FALSIFY_MAX:
        return "falsified"
    return "mixed"


def _tracks(block):
    value, lo, hi, p_val = block["value"], block["ci_lo"], block["ci_hi"], block["p_val"]
    if None in (value, lo, hi, p_val):
        return None
    if block["valid_draws"] < SPLIT_MIN_VALID or block["valid_perm"] < SPLIT_MIN_VALID:
        return None
    return value <= SPLIT_TRACK_POINT and hi < SPLIT_TRACK_CI_HI and p_val < SPLIT_P_MAX


def _covers_zero(block):
    lo, hi = block["ci_lo"], block["ci_hi"]
    if lo is None or hi is None:
        return None
    return lo <= 0.0 <= hi


def _bug_absent(block):
    value = block["value"]
    if value is None:
        return None
    if abs(value) <= SPLIT_ABSENT_ABS:
        return True
    covered = _covers_zero(block)
    if covered is None:
        return None
    return covered


def verdict_split(partial_weak, partial_bug):
    tracks_weak = _tracks(partial_weak)
    tracks_bug = _tracks(partial_bug)
    absent = _bug_absent(partial_bug)
    covers = _covers_zero(partial_weak)
    if None in (tracks_weak, tracks_bug, absent, covers):
        return "not_computable"
    if tracks_weak and absent:
        return "supported"
    if tracks_bug and covers:
        return "falsified"
    return "mixed"


def overall_verdict(h_material: str, h_range: str, h_split: str) -> str:
    parts = (h_material, h_range, h_split)
    if any(part == "not_computable" for part in parts):
        return "not_computable"
    n_supported = sum(part == "supported" for part in parts)
    n_falsified = sum(part == "falsified" for part in parts)
    if n_supported >= 2 and n_falsified == 0:
        return "supported"
    if n_falsified >= 2 and n_supported == 0:
        return "falsified"
    return "mixed"


def verdict_arm_i(v_bug_block, mean_drop, id1_mean, n_bug: int, n_id1: int, model_id: str, lane: str):
    if n_bug != ARM_I_N_BUG or n_id1 != ARM_I_N_ID1:
        return "incomplete"
    if "claude-opus-5-5" not in (model_id or ""):
        return "incomplete"
    if lane not in {"stanford-claude", "vals-claude-opus"}:
        return "incomplete"
    if id1_mean is None or id1_mean < 0.6:
        return "not_valid"
    value, lo, hi = v_bug_block["value"], v_bug_block["ci_lo"], v_bug_block["ci_hi"]
    if value is None or lo is None or hi is None or mean_drop is None:
        return "not_computable"
    if value <= ARM_I_VB_MAX and hi < ARM_I_VB_CI_HI and mean_drop >= ARM_I_DROP_MIN:
        return "supported"
    if value >= ARM_I_FALSIFY_POINT and lo > ARM_I_FALSIFY_LO:
        return "falsified"
    return "mixed"


def slack_example():
    """Constructed median Spearman of 1 with a repeat-violation rate of 1."""
    human = [(i + 1) / 46.0 for i in range(45)]
    ranges = []
    for score in human:
        reps = [0.0, score, 1.0]
        if abs(float(np.median(reps)) - score) > 1e-12:
            raise RuntimeError("constructed median is not the human score")
        ranges.append(max(reps) - min(reps))
    v_rep = sum(1.0 if span > 1e-12 else 0.0 for span in ranges) / len(ranges)
    rho = spearman(human, human)
    # Pearson-on-ranks is 1 up to a float error of about 1e-16. The ranks are identical.
    if rho is None or abs(rho - 1.0) > 1e-9 or abs(v_rep - 1.0) > 1e-12:
        raise RuntimeError(f"slack example failed: rho={rho} v_rep={v_rep}")
    return {
        "n_items": 45,
        "n_repeats": 3,
        "spearman_of_medians": 1.0,
        "spearman_of_medians_raw": rho,
        "v_rep": v_rep,
        "mean_range": 1.0,
    }


def _kind_counts(rows):
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    return counts


def _id_set(rows, kind: str):
    return {row["probe_id"] for row in rows if row["kind"] == kind}


def analyze_config(rows, published: dict, meta: dict):
    counts = _kind_counts(rows)
    for kind, expected in published["n_scored"].items():
        if counts.get(kind, 0) != expected:
            raise RuntimeError(f"{meta['model_id']} kind {kind}: got {counts.get(kind, 0)} expected {expected}")
    if set(published["models"]) != {meta["model_id"]}:
        raise RuntimeError(f"model mismatch: {published['models']} vs {meta['model_id']}")
    if set(published["lanes"]) != {meta["lane"]}:
        raise RuntimeError(f"lane mismatch: {published['lanes']} vs {meta['lane']}")
    id1 = index_id1(rows)
    pairs = bug_pair_rows(rows, id1)
    pub_bug = published["BUG"]
    v_bug = _indicator_ci(pairs, lambda p: p["v_bug"], seed=SEED)
    strict = _indicator_ci(pairs, lambda p: p["strict"], seed=SEED)
    tie_values = []
    for pair in pairs:
        tie_values.append((pair["task"], 1.0 if pair["v_bug"] and not pair["strict"] else 0.0))
    tie = cluster_stat(tie_values, _mean, seed=SEED)
    mean_drop = cluster_stat([(p["task"], p["drop"]) for p in pairs], _mean, seed=SEED)
    id1_mean = float(np.mean(list(id1.values())))
    checks = {
        "v_bug": abs(v_bug["value"] - pub_bug["V_bug"]),
        "strict": abs(strict["value"] - pub_bug["strict_share"]),
        "tie": abs(tie["value"] - pub_bug["tie_share"]),
        "mean_drop": abs(mean_drop["value"] - pub_bug["mean_drop"]),
        "id1_mean": abs(id1_mean - published["ID1_mean"]),
        "n_bug": 0.0 if len(pairs) == pub_bug["n"] else 1.0,
    }
    if any(diff > 1e-9 for diff in checks.values()):
        raise RuntimeError(f"alignment failed for {meta['model_id']}: {checks}")
    crit = [row for row in rows if row["kind"] == "CRIT"]
    n_low = sum(1 for row in crit if row["score"] <= 0.20 + EPS)
    classes = {}
    for name in CLASSES:
        classes[name] = _indicator_ci(pairs, lambda p, name=name: p["pair_class"] == name, seed=SEED)
    class_sum = sum(classes[name]["value"] for name in CLASSES)
    if abs(class_sum - 1.0) > 1e-9:
        raise RuntimeError(f"class shares sum to {class_sum}")
    material = _indicator_ci(pairs, lambda p: p["material"], seed=SEED)
    penalized = classes["penalized"]
    median_drop = cluster_stat([(p["task"], p["drop"]) for p in pairs], _median, seed=SEED)
    n_ties = sum(1 for p in pairs if p["v_bug"] and not p["strict"])
    # Odd n and at least (n+1)/2 exact ties imply a zero median drop.
    corollary_zero = n_ties >= (len(pairs) + 1) / 2.0
    if corollary_zero and abs(median_drop["value"]) > EPS:
        raise RuntimeError("tie corollary says the median drop is 0, and the median is not 0")
    by_level = {}
    levels = sorted({p["level"] for p in pairs})
    for level in levels:
        subset = [p for p in pairs if p["level"] == level]
        by_level[str(level)] = _indicator_ci(subset, lambda p: p["material"], seed=SEED)
    crit_pairs = [(row["task"], row["score"]) for row in crit]
    crit_mean = cluster_stat(crit_pairs, _mean, seed=SEED)
    id1_pairs = list(id1.items())
    id1_mean_ci = cluster_stat(id1_pairs, _mean, seed=SEED)
    spearman_rows = []
    bug_tasks = {p["task"] for p in pairs}
    crit_by: dict[str, list[float]] = {}
    for task, score in crit_pairs:
        crit_by.setdefault(task, []).append(score)
    drop_by: dict[str, list[float]] = {}
    for pair in pairs:
        drop_by.setdefault(pair["task"], []).append(pair["drop"])
    for task in sorted(set(crit_by) & set(drop_by) & set(id1)):
        spearman_rows.append((task, float(np.mean(crit_by[task])), float(np.mean(drop_by[task]))))
    task_rho = cluster_spearman(spearman_rows, seed=SEED)
    if task_rho["value"] is None:
        task_label = "undefined"
    elif task_rho["n_tasks"] < 8:
        task_label = "low_n"
    else:
        task_label = "reported"
    crit_ids = _id_set(rows, "CRIT")
    disjoint = {}
    for kind in ("BUG", "RN_b", "RN_d", "ID1"):
        disjoint[kind] = len(crit_ids & _id_set(rows, kind))
    if any(disjoint.values()):
        raise RuntimeError(f"CRIT probe ids are not disjoint from property ids: {disjoint}")
    return {
        "meta": meta,
        "n_by_kind": counts,
        "n_bug_pairs": len(pairs),
        "n_id1": len(id1),
        "alignment_abs_diff": checks,
        "classes": classes,
        "n_class": {name: sum(1 for p in pairs if p["pair_class"] == name) for name in CLASSES},
        "material_drop_gt_0_20": material,
        "penalized_drop_gt_0_05": penalized,
        "median_drop": median_drop,
        "median_drop_corollary_zero": corollary_zero,
        "n_exact_ties": n_ties,
        "mean_drop": mean_drop,
        "v_bug": v_bug,
        "by_level_material": by_level,
        "crit_mean": crit_mean,
        "id1_mean": id1_mean_ci,
        "crit_le_0_20_count": n_low,
        "crit_n": len(crit),
        "gap_id1_minus_crit": gap_ci(crit_pairs, id1, seed=SEED),
        "task_spearman_crit_mean_vs_bug_drop": task_rho,
        "task_spearman_label": task_label,
        "n_tasks_in_both_crit_and_bug": len(set(crit_by) & bug_tasks),
        "disjoint_crit_intersection": disjoint,
        "h_material_label": verdict_material(material),
        "rename": {"RN_b": rename_block(rows, id1, "RN_b"), "RN_d": rename_block(rows, id1, "RN_d")},
        "pairs": pairs,
    }


def load_exp03(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    units = [u for u in data["units"] if u["kind"] == "llm" and u["rho_heldout"] is not None]
    if len(units) != 56:
        raise RuntimeError(f"expected 56 judge configurations with a held-out Spearman, got {len(units)}")
    rhos, weaks, bugs = [], [], []
    for unit in units:
        if unit["V_weak"]["value"] is None or unit["V_bug"]["value"] is None:
            raise RuntimeError(f"missing property on {unit['config']}")
        rhos.append(float(unit["rho_heldout"]))
        weaks.append(float(unit["V_weak"]["value"]))
        bugs.append(float(unit["V_bug"]["value"]))
    got = spearman(rhos, weaks)
    published = float(data["cross_judge_spearman"]["V_weak"]["spearman"])
    if abs(got - published) > 1e-6 or abs(published - EXP03_SPEARMAN_VWEAK) > 1e-12:
        raise RuntimeError(f"V_weak Spearman {got} does not match stored {published}")
    high = []
    for unit in units:
        coverage = unit.get("rho_heldout_coverage")
        if coverage is None:
            continue
        if float(coverage) + 1e-9 >= 0.90 and float(unit["rho_heldout"]) >= 0.70:
            high.append(unit)
    failing_ids = {row["config"] for row in data["failing_high_corr_llms"]}
    high_ids = {u["config"] for u in high}
    if len(high) != 22 or high_ids != failing_ids:
        raise RuntimeError(f"high-correlation set mismatch: n={len(high)} symmetric difference={high_ids ^ failing_ids}")
    harnesses: dict[str, int] = {}
    for unit in high:
        harnesses[unit["harness"]] = harnesses.get(unit["harness"], 0) + 1
        if abs(float(unit["V_bug"]["strict_gt"])) > 1e-12:
            raise RuntimeError(f"strict_gt is not 0 for {unit['config']}")
        if abs(float(unit["V_bug"]["tie_eq"]) - float(unit["V_bug"]["value"])) > 1e-9:
            raise RuntimeError(f"tie_eq does not equal V_bug for {unit['config']}")
        if unit["V_rep"]["mean_range"] is None or not math.isfinite(float(unit["V_rep"]["mean_range"])):
            raise RuntimeError(f"missing mean range for {unit['config']}")
    if harnesses.get("vb103_judge") != 17 or harnesses.get("terminus2") != 5:
        raise RuntimeError(f"expected 17 vb103_judge and 5 terminus2, got {harnesses}")
    ranges = [float(u["V_rep"]["mean_range"]) for u in high]
    median = float(np.percentile(ranges, 50))
    return {
        "n_llm": len(units),
        "spearman_rho_vweak": got,
        "spearman_abs_diff": abs(got - published),
        "partial_rho_vweak_given_vbug": partial_bundle(rhos, weaks, bugs),
        "partial_rho_vbug_given_vweak": partial_bundle(rhos, bugs, weaks),
        "high": high,
        "high_ranges": ranges,
        "range_median": median,
        "range_p25": float(np.percentile(ranges, 25)),
        "range_p75": float(np.percentile(ranges, 75)),
        "range_min": float(min(ranges)),
        "range_max": float(max(ranges)),
        "n_range_ge_0_10": sum(1 for v in ranges if v >= RANGE_SUPPORT_MIN - EPS),
        "n_range_le_0_02": sum(1 for v in ranges if v <= RANGE_FALSIFY_MAX + EPS),
        "points": [
            {
                "config": u["config"],
                "harness": u["harness"],
                "model_declared": u["model_declared"],
                "model_served": u["model_served"],
                "rho_heldout": float(u["rho_heldout"]),
                "mean_range": float(u["V_rep"]["mean_range"]),
                "high": u["config"] in high_ids,
                "V_rep": float(u["V_rep"]["value"]),
                "V_bug": float(u["V_bug"]["value"]),
                "V_bug_tie_eq": float(u["V_bug"]["tie_eq"]),
                "V_bug_strict_gt": float(u["V_bug"]["strict_gt"]),
                "V_weak": float(u["V_weak"]["value"]),
            }
            for u in units
        ],
    }


def load_exp04_m1(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            if rec.get("method") != "M1_deployed_tc_v1":
                continue

            def fnum(key):
                raw = (rec.get(key) or "").strip()
                if raw == "":
                    return None
                return _finite(raw, key)

            v_equiv = fnum("V_equiv")
            harsh = fnum("harsh_flip_eq10")
            ratio = None if v_equiv in (None, 0.0) or harsh is None else harsh / v_equiv
            rows.append(
                {
                    "config_id": rec["config_id"],
                    "harness": rec["harness"],
                    "model": rec["model"],
                    "n_t1_equiv": fnum("n_t1_equiv"),
                    "V_equiv": v_equiv,
                    "harsh_flip_eq10": harsh,
                    "mean_abs_equiv_shift": fnum("mean_abs_equiv_shift"),
                    "ratio_harsh_eq10_over_v_equiv": ratio,
                    "seen_while_writing_prereg": rec["config_id"] in FRONTIER_M1_SEEN_AT_PREREG,
                }
            )
    if not rows:
        raise RuntimeError("no M1 rows in the experiment 04 table")
    return rows


def _strip_pairs(block):
    """JSON-safe copy without the probe-level pair list."""
    out = {}
    for key, value in block.items():
        if key == "pairs":
            continue
        out[key] = value
    return out


def build_report():
    published = json.loads((EXP05 / "summary.json").read_text(encoding="utf-8"))
    configs = {}
    pair_export = []
    for key in CONFIG_ORDER:
        rows = load_score_csv(EXP05 / f"scores_{key}.csv")
        block = analyze_config(rows, published[key], CONFIG_META[key])
        if key in CRIT_LOW_EXPECTED and block["crit_le_0_20_count"] != CRIT_LOW_EXPECTED[key]:
            raise RuntimeError(
                f"{key} scores <= 0.20: got {block['crit_le_0_20_count']} expected {CRIT_LOW_EXPECTED[key]}"
            )
        for pair in block["pairs"]:
            pair_export.append({"config": key, **CONFIG_META[key], **pair})
        configs[key] = _strip_pairs(block)
    exp03 = load_exp03(EXP03 / "summary.json")
    h_material = configs["B"]["h_material_label"]
    h_range = verdict_range(exp03["range_median"])
    h_split = verdict_split(exp03["partial_rho_vweak_given_vbug"], exp03["partial_rho_vbug_given_vweak"])
    report = {
        "date": "2026-11-26",
        "experiment": "06_why_human_correlation_misses_formal_properties",
        "cpu_analysis_cost_usd": 0.0,
        "cpu_analysis_new_model_calls": 0,
        "seed": SEED,
        "draws": DRAWS,
        "perm_draws": PERM_DRAWS,
        "configs": configs,
        "exp03": {
            "n_llm": exp03["n_llm"],
            "spearman_rho_vweak": exp03["spearman_rho_vweak"],
            "spearman_abs_diff": exp03["spearman_abs_diff"],
            "partial_rho_vweak_given_vbug": exp03["partial_rho_vweak_given_vbug"],
            "partial_rho_vbug_given_vweak": exp03["partial_rho_vbug_given_vweak"],
            "range_median": exp03["range_median"],
            "range_p25": exp03["range_p25"],
            "range_p75": exp03["range_p75"],
            "range_min": exp03["range_min"],
            "range_max": exp03["range_max"],
            "n_range_ge_0_10": exp03["n_range_ge_0_10"],
            "n_range_le_0_02": exp03["n_range_le_0_02"],
            "n_high": len(exp03["high"]),
            "points": exp03["points"],
        },
        "exp04_m1_ratios": load_exp04_m1(EXP04 / "equiv_invariance_table.csv"),
        "slack_lemma": slack_example(),
        "verdicts": {
            "h_material": h_material,
            "h_range": h_range,
            "h_split": h_split,
            "overall": overall_verdict(h_material, h_range, h_split),
        },
        "arm_i": {"status": "not_run"},
        "pair_export": pair_export,
    }
    return report


def fmt_ci(block, digits: int = 3) -> str:
    value, lo, hi = block.get("value"), block.get("ci_lo"), block.get("ci_hi")
    if value is None:
        return "n/a"
    if lo is None or hi is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def fmt_p(p_val) -> str:
    if p_val is None:
        return "n/a"
    if p_val < 0.0001:
        return f"{p_val:.2e}"
    return f"{p_val:.4f}"


def canonical_lines(report) -> list[str]:
    verdicts = report["verdicts"]
    weak = report["exp03"]["partial_rho_vweak_given_vbug"]
    bug = report["exp03"]["partial_rho_vbug_given_vweak"]
    lines = [
        f"OVERALL {verdicts['overall']}",
        (
            f"H-material B {verdicts['h_material']} "
            f"share(drop>0.20)={fmt_ci(report['configs']['B']['material_drop_gt_0_20'])} "
            f"n={report['configs']['B']['material_drop_gt_0_20']['n']} "
            f"tasks={report['configs']['B']['material_drop_gt_0_20']['n_tasks']} "
            f"judge_llm=Claude Opus 5.5 harness=Claude Code CLI plan=Stanford rubric=P4_CHECKLIST"
        ),
        (
            f"H-range {verdicts['h_range']} median={report['exp03']['range_median']:.3f} "
            f"p25={report['exp03']['range_p25']:.3f} p75={report['exp03']['range_p75']:.3f} "
            f"min={report['exp03']['range_min']:.3f} max={report['exp03']['range_max']:.3f} "
            f"n_ge_0.10={report['exp03']['n_range_ge_0_10']} "
            f"n_le_0.02={report['exp03']['n_range_le_0_02']} n=22"
        ),
        (
            f"H-split {verdicts['h_split']} "
            f"partial(rho,V_weak|V_bug)={fmt_ci(weak)} p={fmt_p(weak['p_val'])} "
            f"partial(rho,V_bug|V_weak)={fmt_ci(bug)} p={fmt_p(bug['p_val'])} "
            f"n={weak['n']}"
        ),
        (
            f"SLACK spearman_of_medians={report['slack_lemma']['spearman_of_medians']:.1f} "
            f"v_rep={report['slack_lemma']['v_rep']:.1f} n_items=45"
        ),
        (
            "ARM_I not_run"
            if report.get("arm_i", {}).get("status") != "scored"
            else (
                f"ARM_I {report['arm_i']['verdict']} "
                f"v_bug={fmt_ci(report['arm_i']['v_bug'])} "
                f"mean_drop={report['arm_i']['mean_drop']:.3f} "
                f"id1_mean={report['arm_i']['id1_mean']:.3f} "
                f"n_bug={report['arm_i']['n_bug']} n_id1={report['arm_i']['n_id1']} "
                f"model={report['arm_i']['model_id']} lane={report['arm_i']['lane']}"
            )
        ),
        (
            f"ALIGN exp03_spearman_abs_diff={report['exp03']['spearman_abs_diff']:.3e} "
            f"n_llm={report['exp03']['n_llm']} n_high={report['exp03']['n_high']}"
        ),
    ]
    for key in CONFIG_ORDER:
        block = report["configs"][key]
        meta = block["meta"]
        class_bits = " ".join(
            f"{name}={block['n_class'][name]}/{block['n_bug_pairs']}" for name in CLASSES
        )
        lines.append(
            f"CLASSES {key} {class_bits} material={fmt_ci(block['material_drop_gt_0_20'])} "
            f"label={block['h_material_label']} judge_llm={meta['judge_llm']} "
            f"harness={meta['harness']} rubric={meta['rubric']}"
        )
        lines.append(
            f"MEDIAN_DROP {key} {fmt_ci(block['median_drop'])} "
            f"corollary_zero={str(block['median_drop_corollary_zero']).lower()} "
            f"n_exact_ties={block['n_exact_ties']}"
        )
        lines.append(
            f"GAP {key} id1_minus_crit={fmt_ci(block['gap_id1_minus_crit'])} "
            f"crit_le_0.20={block['crit_le_0_20_count']}/{block['crit_n']}"
        )
        task = block["task_spearman_crit_mean_vs_bug_drop"]
        lines.append(
            f"TASK_SPEARMAN {key} {block['task_spearman_label']} {fmt_ci(task)} "
            f"n_tasks={task['n_tasks']} shared_tasks={block['n_tasks_in_both_crit_and_bug']}"
        )
        inter = block["disjoint_crit_intersection"]
        lines.append(
            "DISJOINT "
            + key
            + " "
            + " ".join(f"{kind}={inter[kind]}" for kind in ("BUG", "RN_b", "RN_d", "ID1"))
        )
    for row in report["exp04_m1_ratios"]:
        ratio = row["ratio_harsh_eq10_over_v_equiv"]
        ratio_s = "n/a" if ratio is None else f"{ratio:.3f}"
        v_equiv = "n/a" if row["V_equiv"] is None else f"{row['V_equiv']:.3f}"
        harsh = "n/a" if row["harsh_flip_eq10"] is None else f"{row['harsh_flip_eq10']:.3f}"
        lines.append(
            f"EXP04_M1 {row['config_id']} ratio={ratio_s} "
            f"V_equiv={v_equiv} harsh_eq10={harsh} "
            f"seen_at_prereg={str(row['seen_while_writing_prereg']).lower()} "
            f"harness={row['harness']} model={row['model']}"
        )
    return lines


def _require_grid(fig, ax):
    ax.grid(True)
    fig.canvas.draw()
    if not any(line.get_visible() for line in ax.get_xgridlines()):
        raise RuntimeError("x grid is off")
    if not any(line.get_visible() for line in ax.get_ygridlines()):
        raise RuntimeError("y grid is off")


def plot_class_shares(path: Path, report):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(CONFIG_ORDER))
    bottom = np.zeros(len(CONFIG_ORDER))
    for name in CLASSES:
        vals = np.array([report["configs"][key]["classes"][name]["value"] for key in CONFIG_ORDER])
        ax.bar(x, vals, bottom=bottom, label=name, color=CLASS_COLORS[name], width=0.72)
        bottom = bottom + vals
    labels = []
    for key in CONFIG_ORDER:
        meta = report["configs"][key]["meta"]
        labels.append(f"{key}\n{meta['judge_llm']}\n{meta['harness']}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of inserted-bug pairs")
    ax.set_title("Experiment 06 (as of 2026-11-26): bug-pair classes")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    _require_grid(fig, ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_range_scatter(path: Path, report):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    points = report["exp03"]["points"]
    low = [p for p in points if not p["high"]]
    high = [p for p in points if p["high"]]
    ax.scatter(
        [p["rho_heldout"] for p in low],
        [p["mean_range"] for p in low],
        s=28,
        c="#9E9E9E",
        label="Spearman < 0.70",
        zorder=2,
    )
    ax.scatter(
        [p["rho_heldout"] for p in high],
        [p["mean_range"] for p in high],
        s=36,
        c="#4C78A8",
        marker="D",
        label="Spearman ≥ 0.70 (n=22)",
        zorder=3,
    )
    ax.axhline(RANGE_SUPPORT_MIN, color="black", lw=1, label="0.10 substantive cut")
    ax.axhline(RANGE_FALSIFY_MAX, color="gray", lw=1, ls="--", label="0.02 jitter cut")
    ax.set_xlabel("Held-out Spearman with the 3-expert target_expert panel")
    ax.set_ylabel("Mean within-item repeat range")
    ax.set_title("Experiment 06 (as of 2026-11-26): 56 judge configurations")
    ax.legend(fontsize=8)
    _require_grid(fig, ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_outputs(report, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    serial = {key: value for key, value in report.items() if key != "pair_export"}
    (out_dir / "summary.json").write_text(json.dumps(serial, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pair_fields = [
        "config",
        "judge_llm",
        "model_id",
        "harness",
        "plan",
        "lane",
        "rubric",
        "task",
        "level",
        "id1",
        "bug",
        "drop",
        "pair_class",
        "material",
        "v_bug",
        "strict",
    ]
    with (out_dir / "bug_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in report["pair_export"]:
            writer.writerow(row)
    point_fields = [
        "config",
        "harness",
        "model_declared",
        "model_served",
        "rho_heldout",
        "mean_range",
        "high",
        "V_rep",
        "V_bug",
        "V_bug_tie_eq",
        "V_bug_strict_gt",
        "V_weak",
    ]
    with (out_dir / "exp03_configurations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=point_fields, lineterminator="\n")
        writer.writeheader()
        for row in report["exp03"]["points"]:
            writer.writerow(row)
    ratio_fields = [
        "config_id",
        "harness",
        "model",
        "n_t1_equiv",
        "V_equiv",
        "harsh_flip_eq10",
        "mean_abs_equiv_shift",
        "ratio_harsh_eq10_over_v_equiv",
        "seen_while_writing_prereg",
    ]
    with (out_dir / "exp04_m1_ratios.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ratio_fields, lineterminator="\n")
        writer.writeheader()
        for row in report["exp04_m1_ratios"]:
            writer.writerow(row)
    lines = canonical_lines(report)
    (out_dir / "canonical_lines.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_class_shares(out_dir / "fig1_bug_pair_classes.png", report)
    plot_range_scatter(out_dir / "fig2_repeat_range_vs_human_correlation.png", report)
    return lines


def summarize_arm_i_csv(path: Path, model_id: str, lane: str):
    rows = load_score_csv(path)
    counts = _kind_counts(rows)
    id1 = index_id1(rows)
    pairs = bug_pair_rows(rows, id1)
    v_bug = cluster_stat([(p["task"], 1.0 if p["v_bug"] else 0.0) for p in pairs], _mean, seed=SEED) if pairs else {
        "value": None, "ci_lo": None, "ci_hi": None, "n": 0, "n_tasks": 0, "valid_draws": 0
    }
    strict = _indicator_ci(pairs, lambda p: p["strict"], seed=SEED) if pairs else None
    tie = _indicator_ci(pairs, lambda p: p["v_bug"] and not p["strict"], seed=SEED) if pairs else None
    mean_drop_ci = cluster_stat([(p["task"], p["drop"]) for p in pairs], _mean, seed=SEED) if pairs else None
    median_drop_ci = cluster_stat([(p["task"], p["drop"]) for p in pairs], _median, seed=SEED) if pairs else None
    material = _indicator_ci(pairs, lambda p: p["material"], seed=SEED) if pairs else None
    classes = {name: _indicator_ci(pairs, lambda p, name=name: p["pair_class"] == name, seed=SEED) for name in CLASSES} if pairs else {}
    n_class = {name: sum(1 for p in pairs if p["pair_class"] == name) for name in CLASSES} if pairs else {}
    by_level = {}
    for level in sorted({p["level"] for p in pairs}):
        subset = [p for p in pairs if p["level"] == level]
        by_level[str(level)] = {
            "v_bug": _indicator_ci(subset, lambda p: p["v_bug"], seed=SEED),
            "material": _indicator_ci(subset, lambda p: p["material"], seed=SEED),
            "mean_drop": cluster_stat([(p["task"], p["drop"]) for p in subset], _mean, seed=SEED),
        }
    id1_mean_ci = cluster_stat(list(id1.items()), _mean, seed=SEED) if id1 else None
    id2_rows = [row for row in rows if row["kind"] == "ID2"]
    repeat_noise = (
        cluster_stat([(row["task"], 1.0 if abs(row["score"] - id1[row["task"]]) > 0.05 + EPS else 0.0) for row in id2_rows if row["task"] in id1], _mean, seed=SEED)
        if id2_rows else None
    )
    rn_b = rename_block(rows, id1, "RN_b") if any(r["kind"] == "RN_b" for r in rows) else None
    rn_d = rename_block(rows, id1, "RN_d") if any(r["kind"] == "RN_d" for r in rows) else None
    rn_pooled = [r for r in rows if r["kind"] in ("RN_b", "RN_d") and r["task"] in id1]
    rn_pooled_gt_0_05 = (
        cluster_stat([(r["task"], 1.0 if abs(r["score"] - id1[r["task"]]) > 0.05 + EPS else 0.0) for r in rn_pooled], _mean, seed=SEED)
        if rn_pooled else None
    )
    mean_drop = float(np.mean([p["drop"] for p in pairs])) if pairs else None
    id1_mean = float(np.mean(list(id1.values()))) if id1 else None
    return {
        "status": "scored",
        "model_id": model_id,
        "lane": lane,
        "arm_i_call_meta": {
            "batches_run": 21,
            "probes_scored": sum(counts.values()),
            "extra_charge_usd": 0.0,
            "lane": lane,
            "model_id": model_id,
        },
        "n_by_kind": counts,
        "n_bug": len(pairs),
        "n_id1": len(id1),
        "v_bug": v_bug,
        "strict": strict,
        "tie": tie,
        "mean_drop": mean_drop,
        "mean_drop_ci": mean_drop_ci,
        "median_drop": median_drop_ci,
        "material_drop_gt_0_20": material,
        "n_class": n_class,
        "classes": classes,
        "by_level": by_level,
        "id1_mean": id1_mean,
        "id1_mean_ci": id1_mean_ci,
        "repeat_noise_gt_0_05": repeat_noise,
        "rename": {"RN_b": rn_b, "RN_d": rn_d, "pooled_gt_0_05": rn_pooled_gt_0_05},
        "verdict": verdict_arm_i(v_bug, mean_drop, id1_mean, len(pairs), len(id1), model_id, lane),
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 06 CPU arm. No model calls.")
    parser.add_argument("--out", default=None, help="Output directory. Default: this experiment's results/")
    parser.add_argument("--arm-i", default=None, help="Optional Arm I scores CSV. Omit to leave Arm I as not_run.")
    parser.add_argument("--arm-i-model", default="claude-opus-5-5")
    parser.add_argument("--arm-i-lane", default="stanford-claude")
    args = parser.parse_args()
    report = build_report()
    if args.arm_i:
        report["arm_i"] = summarize_arm_i_csv(Path(args.arm_i), args.arm_i_model, args.arm_i_lane)
    out = Path(args.out) if args.out else EXP_DIR / "results"
    lines = write_outputs(report, out)
    print(lines[0])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
