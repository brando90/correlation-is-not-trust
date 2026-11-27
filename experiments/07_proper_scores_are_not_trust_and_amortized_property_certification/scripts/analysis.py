#!/usr/bin/env python3
"""Experiment 07: Proper scores vs. axiomatic trust, Idea 3 out-of-family certification, and H2 cross-regime ceiling.

Zero new model calls ($0.00). Strictly enforces <= 1e-9 numerical alignment with both:
- cert-judge Experiment 68 (`study_b_e39_proper_scores.json`)
- correlation-is-not-trust Experiment 03 (`summary.json`)
without copying any raw Lean source files or raw human rating rows into the repository.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

EXP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_DIR.parents[1]
DEFAULT_OUT_DIR = EXP_DIR / "results"

E39_DIR = Path("/Users/brandomiranda/cert-judge/experiments/39_label_free_proxy_index_train_val_test_over_judge_families/results")
E42_DIR = Path("/Users/brandomiranda/cert-judge/experiments/42_semantic_weakening_overconstraint_vacuity_probes_on_e39_judges/results")
E68_JSON = Path("/Users/brandomiranda/cert-judge/experiments/68_proper_scoring_expansion_e39_e63_and_vb_ratings_start_24Nov2026/results/study_b_e39_proper_scores.json")
EXP03_JSON = REPO_ROOT / "experiments/03_human_corr_vs_axiomatic_properties_cert_judge/results/summary.json"
ITEMS_JSONL = Path("/Users/brandomiranda/veribench/experiments/100_karpathy_recipe_tc_judge_search_human_train_val_test/inputs/items.jsonl")
ITEMS_SHA256_EXPECTED = "ad9ed552fc2ffac6328e6010bf6be82215dd74f1eddd791622d433126877a96c"

TTJ_E27_DIR = Path("/Users/brandomiranda/trustworthy-theorem-judges/experiments/27_build_a_judge_that_beats_the_constant_forecast_on_held_out_tasks_old_new_ratings_start_24Nov2026/data_11-24-2026")
OLD_RATINGS_CSV = TTJ_E27_DIR / "human_ratings_old_11-24-2026.csv"
NEW_RATINGS_CSV = TTJ_E27_DIR / "human_ratings_new_11-24-2026.csv"

SEED = 20261124
BOOT_DRAWS = 2000
PERM_DRAWS = 5000
PAIR_PERM_DRAWS = 2000
MIN_CRIT_COV = 0.90
MIN_COMP_COV = 0.80
SCORED_STATUSES = ("measured", "deterministic")
TOL = 1e-12
CE_FLOOR = 1e-12
ALPHA_SMOOTH = 0.5
CATEGORIES = np.arange(6, dtype=float)
RIDGE_LAMBDAS = (0.01, 0.1, 1.0, 10.0)
ATI_FEATURES = ("pass_ms", "pass_weak", "pass_vac", "pass_overc", "pass_fmt", "pass_rep")
GATE_CUTS = {
    "V_ms": 0.10,
    "V_weak": 0.15,
    "V_vac": 0.10,
    "V_overc": 0.25,
    "V_fmt": 0.10,
    "V_rep": 0.25,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_bounded_float(raw: Any, lo: float, hi: float, label: str) -> float:
    if isinstance(raw, bool):
        raise ValueError(f"Boolean not allowed for {label}: {raw!r}")
    val = float(raw)
    if not math.isfinite(val) or val < lo - TOL or val > hi + TOL:
        raise ValueError(f"Invalid {label}={raw!r}: expected finite in [{lo}, {hi}]")
    return val


def validate_probs(probs_q: Sequence[float]) -> np.ndarray:
    q = np.asarray(probs_q, dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)) or np.any(q < -1e-12):
        raise ValueError(f"Invalid probability vector: {probs_q!r}")
    q = np.clip(q, 0.0, None)
    s = float(q.sum())
    if s <= 0.0:
        raise ValueError("Probability vector must sum to > 0")
    return q / s


def rating_counts(ratings: Sequence[int | float]) -> np.ndarray:
    if not ratings:
        raise ValueError("Empty ratings list")
    c = np.zeros(6, dtype=float)
    for y in ratings:
        yf = float(y)
        yi = int(round(yf))
        if not math.isfinite(yf) or abs(yf - yi) > 1e-9 or yi < 0 or yi > 5:
            raise ValueError(f"Invalid 0..5 rating {y!r}")
        c[yi] += 1.0
    return c


def empirical_dist(ratings: Sequence[int | float]) -> np.ndarray:
    c = rating_counts(ratings)
    return c / float(c.sum())


def binned_repeats_to_probs(repeats_01: Sequence[float], alpha: float = ALPHA_SMOOTH) -> np.ndarray:
    if not repeats_01:
        raise ValueError("Need at least one repeat")
    c = np.zeros(6, dtype=float)
    for s in repeats_01:
        idx = min(5, max(0, math.floor(5.0 * float(s) + 0.5 + 1e-12)))
        c[idx] += 1.0
    q = c + float(alpha)
    return q / float(q.sum())


def rps_decomp(probs_q: Sequence[float], ratings: Sequence[int | float]) -> tuple[float, float, float]:
    q = validate_probs(probs_q)
    p = empirical_dist(ratings)
    Q = np.cumsum(q)[:5]
    F = np.cumsum(p)[:5]
    dist_err = float(np.mean((Q - F) ** 2))
    disag = float(np.mean(F * (1.0 - F)))
    return dist_err + disag, dist_err, disag


def ce_item(probs_q: Sequence[float], ratings: Sequence[int | float], floor: float = CE_FLOOR) -> float:
    q = validate_probs(probs_q)
    c = rating_counts(ratings)
    mask = c > 0
    return float(-np.sum(c[mask] * np.log(np.maximum(q[mask], floor))) / float(c.sum()))


def tvd_item(probs_q: Sequence[float], ratings: Sequence[int | float]) -> float:
    q = validate_probs(probs_q)
    p = empirical_dist(ratings)
    return float(0.5 * np.sum(np.abs(q - p)))


def tvd_dists(q1: Sequence[float], q2: Sequence[float]) -> float:
    a = validate_probs(q1)
    b = validate_probs(q2)
    return float(0.5 * np.sum(np.abs(a - b)))


def brier_item(probs_q: Sequence[float], ratings: Sequence[int | float]) -> float:
    q = validate_probs(probs_q)
    p = empirical_dist(ratings)
    return float(np.sum(q * q) - 2.0 * np.sum(p * q) + 1.0)


def icc_two_way_both(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    n = len(xa)
    if n < 3:
        return float("nan"), float("nan")
    s = xa + ya
    d = xa - ya
    ms = float(s.mean())
    md = float(d.mean())
    msr = 0.5 * float(np.sum((s - ms) ** 2)) / float(n - 1)
    msc = 0.5 * float(n) * (md * md)
    mse = 0.5 * float(np.sum((d - md) ** 2)) / float(n - 1)
    den_c = msr + mse
    den_a = den_c + 2.0 * (msc - mse) / float(n)
    icc_a = float((msr - mse) / den_a) if den_a > 0.0 else float("nan")
    icc_c = float((msr - mse) / den_c) if den_c > 0.0 else float("nan")
    return icc_a, icc_c


def spearman_corr(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    val = float(stats.spearmanr(xs, ys).statistic)
    return val if math.isfinite(val) else None


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    val = float(stats.pearsonr(xs, ys).statistic)
    return val if math.isfinite(val) else None


def _smooth_rows(p: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    p = np.clip(p, eps, None)
    return p / p.sum(axis=1, keepdims=True)


def fit_const(train_rating_lists: Sequence[Sequence[int | float]], alpha: float = ALPHA_SMOOTH) -> Callable[[Sequence[float]], np.ndarray]:
    c = np.sum([rating_counts(r) for r in train_rating_lists], axis=0)
    q = (c + float(alpha)) / float(c.sum() + 6.0 * float(alpha))
    return lambda scores: np.tile(q, (len(scores), 1))


def fit_ce_ordinal(
    train_scores: Sequence[float],
    train_rating_lists: Sequence[Sequence[int | float]],
    l2: float = 1e-3,
    eps: float = 1e-3,
) -> Callable[[Sequence[float]], np.ndarray]:
    s = np.asarray(train_scores, dtype=float)
    C = np.array([rating_counts(r) for r in train_rating_lists], dtype=float)

    def unpack(w: np.ndarray) -> tuple[np.ndarray, float]:
        th = np.cumsum(np.concatenate([[w[0]], np.exp(np.clip(w[1:5], -20.0, 20.0))]))
        return th, float(w[5])

    def nll(w: np.ndarray) -> float:
        th, b = unpack(w)
        z = np.clip(th[None, :] - b * s[:, None], -40.0, 40.0)
        cum = 1.0 / (1.0 + np.exp(-z))
        cum_full = np.concatenate([np.zeros((len(s), 1)), cum, np.ones((len(s), 1))], axis=1)
        p = np.clip(np.diff(cum_full, axis=1), 1e-9, 1.0)
        return float(-(C * np.log(p)).sum() / float(C.sum()) + l2 * b * b)

    marg = C.sum(axis=0) / float(C.sum())
    cm = np.clip(np.cumsum(marg)[:5], 1e-3, 1.0 - 1e-3)
    th0 = np.log(cm / (1.0 - cm))
    w0 = np.concatenate([[th0[0]], np.log(np.maximum(np.diff(th0), 1e-3)), [0.0]])
    opt = optimize.minimize(nll, w0, method="L-BFGS-B")
    th, b = unpack(opt.x)

    def predict(x: Sequence[float]) -> np.ndarray:
        xa = np.asarray(x, dtype=float)
        z = np.clip(th[None, :] - b * xa[:, None], -40.0, 40.0)
        cum = 1.0 / (1.0 + np.exp(-z))
        cum_full = np.concatenate([np.zeros((len(xa), 1)), cum, np.ones((len(xa), 1))], axis=1)
        return _smooth_rows(np.diff(cum_full, axis=1), eps=eps)

    return predict


def per_item_losses(probs_mat: np.ndarray, rating_lists: Sequence[Sequence[int | float]]) -> dict[str, np.ndarray]:
    n = len(rating_lists)
    if probs_mat.shape != (n, 6):
        raise ValueError(f"Expected ({n}, 6), got {probs_mat.shape}")
    out = {k: np.empty(n, dtype=float) for k in ("rps", "rps_dist_err", "rps_disag", "ce", "tvd", "brier", "ev_05", "hum_mean_05")}
    for i, ratings in enumerate(rating_lists):
        q = validate_probs(probs_mat[i])
        rps_val, dist_err, disag = rps_decomp(q, ratings)
        out["rps"][i] = rps_val
        out["rps_dist_err"][i] = dist_err
        out["rps_disag"][i] = disag
        out["ce"][i] = ce_item(q, ratings, floor=CE_FLOOR)
        out["tvd"][i] = tvd_item(q, ratings)
        out["brier"][i] = brier_item(q, ratings)
        out["ev_05"][i] = float(np.dot(CATEGORIES, q))
        out["hum_mean_05"][i] = float(np.mean(np.asarray(ratings, dtype=float)))
    return out


def cluster_mean_ci(pairs: Sequence[tuple[str, float]], draws: int = BOOT_DRAWS, seed: int = SEED) -> dict[str, Any]:
    if not pairs:
        return {"value": None, "ci_lo": None, "ci_hi": None, "n": 0, "n_tasks": 0, "finite_draws": 0}
    groups: dict[str, list[float]] = {}
    for task, val in pairs:
        fval = float(val)
        if not math.isfinite(fval):
            raise ValueError(f"Non-finite value in cluster_mean_ci for task={task!r}: {val!r}")
        groups.setdefault(str(task), []).append(fval)
    keys = sorted(groups)
    group_arrs = [np.asarray(groups[k], dtype=float) for k in keys]
    point = float(sum(v for _, v in pairs) / len(pairs))
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=float)
    n_keys = len(keys)
    for d_idx in range(draws):
        picks = rng.integers(0, n_keys, size=n_keys)
        boot[d_idx] = float(np.concatenate([group_arrs[int(k)] for k in picks]).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "value": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": len(pairs),
        "n_tasks": len(keys),
        "finite_draws": len(boot),
    }


def cluster_ratio_ci(
    triples: Sequence[tuple[str, float, float]],
    draws: int = BOOT_DRAWS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Task-cluster bootstrap CI for a ratio sum(num) / sum(den) (e.g. rating-weighted P(r == 5))."""
    if not triples:
        return {"value": None, "ci_lo": None, "ci_hi": None, "n": 0, "n_tasks": 0, "n_ratings": 0, "finite_draws": 0}
    groups: dict[str, tuple[float, float]] = {}
    tot_num = 0.0
    tot_den = 0.0
    for task, num, den in triples:
        fn, fd = float(num), float(den)
        if not (math.isfinite(fn) and math.isfinite(fd)) or fd < 0.0:
            raise ValueError(f"Invalid triple in cluster_ratio_ci for task={task!r}: {(num, den)!r}")
        tot_num += fn
        tot_den += fd
        prev_n, prev_d = groups.get(str(task), (0.0, 0.0))
        groups[str(task)] = (prev_n + fn, prev_d + fd)
    keys = sorted(groups)
    g_nums = np.array([groups[k][0] for k in keys], dtype=float)
    g_dens = np.array([groups[k][1] for k in keys], dtype=float)
    point = float(tot_num / tot_den)
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=float)
    n_keys = len(keys)
    for d_idx in range(draws):
        picks = rng.integers(0, n_keys, size=n_keys)
        boot[d_idx] = float(np.sum(g_nums[picks]) / np.sum(g_dens[picks]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "value": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": len(triples),
        "n_tasks": len(keys),
        "n_ratings": int(round(tot_den)),
        "finite_draws": len(boot),
    }


def paired_task_signflip_p(
    task_ids: Sequence[str],
    vals_a: Sequence[float],
    vals_b: Sequence[float],
    draws: int = PAIR_PERM_DRAWS,
    seed: int = SEED,
) -> dict[str, Any]:
    a = np.asarray(vals_a, dtype=float)
    b = np.asarray(vals_b, dtype=float)
    t_arr = np.asarray(task_ids)
    uniq = sorted(set(t_arr))
    n_total = float(len(a))
    if not uniq or n_total <= 0:
        raise ValueError("Empty task_ids in paired_task_signflip_p")
    # Item-count-weighted cluster sums so permutation test targets the exact item-weighted difference in cluster_mean_ci.
    cluster_sums = np.array([float(np.sum(a[t_arr == t] - b[t_arr == t])) for t in uniq], dtype=float)
    obs = float(np.sum(cluster_sums) / n_total)
    g = len(uniq)
    if g <= 12:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=g)), dtype=float)
        perm_means = (signs @ cluster_sums) / n_total
        p_val = float(np.mean(np.abs(perm_means) >= abs(obs) - 1e-12))
        method = f"exact_2^{g}"
        n_perm = 2 ** g
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(draws, g))
        perm_means = (signs @ cluster_sums) / n_total
        p_val = float((1 + np.sum(np.abs(perm_means) >= abs(obs) - 1e-12)) / (draws + 1))
        method = "mc_signflip"
        n_perm = draws
    return {"obs_diff": obs, "p_val": p_val, "method": method, "n_clusters": g, "n_perm": n_perm}


def leave_one_rater_out(rating_lists: Sequence[Sequence[int]], task_ids: Sequence[str], draws: int = BOOT_DRAWS, seed: int = SEED) -> dict[str, Any]:
    eligible = [i for i, r in enumerate(rating_lists) if len(r) >= 2]
    if not eligible:
        return {"n_items": 0}
    smooth_items = {k: [] for k in ("rps", "ce", "tvd", "brier", "mae_05")}
    emp_items = {k: [] for k in ("rps", "tvd", "brier", "mae_05")}
    for idx in eligible:
        rats = [int(v) for v in rating_lists[idx]]
        m = len(rats)
        s_acc = {k: 0.0 for k in smooth_items}
        e_acc = {k: 0.0 for k in emp_items}
        for j in range(m):
            others = rats[:j] + rats[j + 1 :]
            c_other = rating_counts(others)
            q_s = (c_other + ALPHA_SMOOTH) / float(c_other.sum() + 6.0 * ALPHA_SMOOTH)
            q_e = c_other / float(c_other.sum())
            y = rats[j]
            s_acc["rps"] += rps_decomp(q_s, [y])[0]
            s_acc["ce"] += ce_item(q_s, [y])
            s_acc["tvd"] += tvd_item(q_s, [y])
            s_acc["brier"] += brier_item(q_s, [y])
            s_acc["mae_05"] += abs(float(np.dot(CATEGORIES, q_s)) - y)
            e_acc["rps"] += rps_decomp(q_e, [y])[0]
            e_acc["tvd"] += tvd_item(q_e, [y])
            e_acc["brier"] += brier_item(q_e, [y])
            e_acc["mae_05"] += abs(float(np.dot(CATEGORIES, q_e)) - y)
        for k in smooth_items:
            smooth_items[k].append(s_acc[k] / float(m))
        for k in emp_items:
            emp_items[k].append(e_acc[k] / float(m))
    sub_tasks = [task_ids[i] for i in eligible]
    return {
        "n_items": len(eligible),
        "smoothed": {k: cluster_mean_ci(list(zip(sub_tasks, smooth_items[k])), draws=draws, seed=seed) for k in smooth_items},
        "empirical": {k: cluster_mean_ci(list(zip(sub_tasks, emp_items[k])), draws=draws, seed=seed) for k in emp_items},
    }


def unit_meta_corr(
    xs: Sequence[float],
    ys: Sequence[float],
    families: Sequence[str] | None = None,
    boot_draws: int = BOOT_DRAWS,
    perm_draws: int = PERM_DRAWS,
    seed: int = SEED,
) -> dict[str, Any]:
    n = len(xs)
    obs = spearman_corr(xs, ys)
    obs_pe = pearson_corr(xs, ys)
    if obs is None or n < 3:
        return {"n": n, "spearman": None, "pearson": obs_pe, "ci_lo": None, "ci_hi": None, "p_val": None, "within_family_p": None, "valid_draws": 0}
    xa = np.asarray(xs, dtype=float)
    ya = np.asarray(ys, dtype=float)
    rng_b = np.random.default_rng(seed)
    boot = []
    for _ in range(boot_draws):
        idx = rng_b.integers(0, n, size=n)
        v = spearman_corr(xa[idx], ya[idx])
        if v is not None:
            boot.append(v)
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (None, None))
    rx = stats.rankdata(xa)
    ry = stats.rankdata(ya)
    rx_d = rx - rx.mean()
    ry_d = ry - ry.mean()
    rx_c = rx_d / float(np.linalg.norm(rx_d))
    ry_c = ry_d / float(np.linalg.norm(ry_d))
    rng_p = np.random.default_rng(seed)
    if n <= 8:
        perms = np.array(list(itertools.permutations(range(n))), dtype=int)
        rs = ry_c[perms] @ rx_c
        p_val = float(np.mean(np.abs(rs) >= abs(obs) - 1e-12))
    else:
        base_mat = np.broadcast_to(ry_c, (perm_draws, n))
        perm_mat = rng_p.permuted(base_mat, axis=1)
        rs = perm_mat @ rx_c
        p_val = float((1 + np.sum(np.abs(rs) >= abs(obs) - 1e-12)) / (perm_draws + 1))

    within_p = None
    if families is not None:
        fam_arr = np.asarray(families)
        uniq_f = sorted(set(fam_arr))
        fam_idx = {f: np.where(fam_arr == f)[0] for f in uniq_f}
        if any(len(v) > 1 for v in fam_idx.values()):
            rng_w = np.random.default_rng(seed)
            perm_s = np.broadcast_to(ry_c, (perm_draws, n)).copy()
            for idxs in fam_idx.values():
                if len(idxs) > 1:
                    perm_s[:, idxs] = rng_w.permuted(perm_s[:, idxs], axis=1)
            rs_w = perm_s @ rx_c
            within_p = float((1 + np.sum(np.abs(rs_w) >= abs(obs) - 1e-12)) / (perm_draws + 1))

    return {
        "n": n,
        "spearman": float(obs),
        "pearson": float(obs_pe) if obs_pe is not None else None,
        "ci_lo": float(lo) if lo is not None else None,
        "ci_hi": float(hi) if hi is not None else None,
        "p_val": float(p_val),
        "within_family_p": within_p,
        "valid_draws": len(boot),
    }


def load_items_verified() -> list[dict[str, Any]]:
    raw = ITEMS_JSONL.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != ITEMS_SHA256_EXPECTED:
        raise RuntimeError(f"items.jsonl sha256 mismatch: {sha} != {ITEMS_SHA256_EXPECTED}")
    items = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    for it in items:
        it["all5_ratings"] = [int(round(float(it[p]))) for p in ("p1", "p2", "p3", "p4", "p5") if it[p] is not None]
        it["expert3_ratings"] = [int(round(float(it[p]))) for p in ("p1", "p2", "p3") if it[p] is not None]
    return items


def aggregate_harvest(
    harvest: dict[str, Any],
    min_repeats_by_kind: dict[str, int],
    default_min_repeats: int = 2,
) -> tuple[dict[tuple[str, str, int], float], dict[tuple[str, str, int], list[float]]]:
    raw_by_cell: dict[tuple[str, str, int], list[float]] = {}
    seen_repeats: set[tuple[str, str, int, Any]] = set()
    for idx, c in enumerate(harvest["cells"]):
        key_str = c.get("key") if c.get("key") is not None else c["task"]
        rep_id = c["repeat"] if "repeat" in c else idx
        rep_key = (c["kind"], str(key_str), int(c["level"]), rep_id)
        if rep_key in seen_repeats:
            raise ValueError(f"Duplicate harvest repeat key: {rep_key}")
        seen_repeats.add(rep_key)
        if c.get("status") in SCORED_STATUSES and c.get("score") is not None:
            s_val = parse_bounded_float(c["score"], 0.0, 1.0, "harvest score")
            raw_by_cell.setdefault((c["kind"], str(key_str), int(c["level"])), []).append(s_val)
    medians: dict[tuple[str, str, int], float] = {}
    for cid, vals in raw_by_cell.items():
        if len(vals) >= min_repeats_by_kind.get(cid[0], default_min_repeats):
            medians[cid] = float(statistics.median(vals))
    return medians, raw_by_cell


def eval_paired_axiom(
    rows: Sequence[dict[str, Any]],
    scores: dict[tuple[str, str, int], float],
    base_lookup: Callable[[dict[str, Any]], tuple[str, str, int]],
    probe_lookup: Callable[[dict[str, Any]], tuple[str, str, int]],
    mode: str,
    cal_fn: Callable[[Sequence[float]], np.ndarray] | None = None,
) -> dict[str, Any]:
    declared = len(rows)
    viol_pairs: list[tuple[str, float]] = []
    tvd_pairs: list[tuple[str, float]] = []
    for r in rows:
        b = scores.get(base_lookup(r))
        p = scores.get(probe_lookup(r))
        if b is None or p is None:
            continue
        task = str(r["task"])
        if cal_fn is not None:
            probs = cal_fn([b, p])
            b_eff = float(np.dot(CATEGORIES, probs[0])) / 5.0
            p_eff = float(np.dot(CATEGORIES, probs[1])) / 5.0
            tvd_pairs.append((task, tvd_dists(probs[0], probs[1])))
        else:
            b_eff, p_eff = b, p
        if mode == "ge_base":
            viol = 0.0 if p_eff < b_eff - (TOL if cal_fn is not None else 0.0) else 1.0
        elif mode == "abs_diff":
            viol = 1.0 if abs(p_eff - b_eff) > TOL else 0.0
        else:
            raise ValueError(f"Unknown mode {mode}")
        viol_pairs.append((task, viol))
    cov = len(viol_pairs) / declared if declared else 0.0
    computable = len(viol_pairs) > 0 and cov + 1e-9 >= MIN_COMP_COV
    ci = cluster_mean_ci(viol_pairs, draws=BOOT_DRAWS, seed=SEED)
    out = {
        "value": ci["value"] if computable else None,
        "ci_lo": ci["ci_lo"] if computable else None,
        "ci_hi": ci["ci_hi"] if computable else None,
        "n_scored": len(viol_pairs),
        "coverage": round(cov, 6),
        "computable": computable,
    }
    if tvd_pairs:
        tvd_ci = cluster_mean_ci(tvd_pairs, draws=BOOT_DRAWS, seed=SEED)
        out["mean_tvd"] = tvd_ci["value"] if computable else None
        out["mean_tvd_ci_lo"] = tvd_ci["ci_lo"] if computable else None
        out["mean_tvd_ci_hi"] = tvd_ci["ci_hi"] if computable else None
    return out


def eval_repeat_stats(
    rows: Sequence[dict[str, Any]],
    raw_scores: dict[tuple[str, str, int], list[float]],
    cal_fn: Callable[[Sequence[float]], np.ndarray] | None = None,
    min_repeats: int = 2,
) -> dict[str, Any]:
    declared = len(rows)
    v_rep_pairs: list[tuple[str, float]] = []
    raw_rng_pairs: list[tuple[str, float]] = []
    cal_ev_rng_pairs: list[tuple[str, float]] = []
    cal_tvd_pairs: list[tuple[str, float]] = []
    for r in rows:
        key = str(r.get("item") or r.get("key") or r["task"])
        vals = raw_scores.get((str(r["kind"]), key, int(r["level"])), [])
        if len(vals) < min_repeats:
            continue
        task = str(r["task"])
        span = max(vals) - min(vals)
        v_rep_pairs.append((task, 1.0 if span > TOL else 0.0))
        raw_rng_pairs.append((task, span))
        if cal_fn is not None:
            probs = cal_fn(vals)
            evs = probs @ CATEGORIES
            cal_ev_rng_pairs.append((task, float(np.max(evs) - np.min(evs))))
            pair_tvds = [tvd_dists(probs[i], probs[j]) for i in range(len(vals)) for j in range(i + 1, len(vals))]
            cal_tvd_pairs.append((task, float(np.mean(pair_tvds))))
    cov = len(v_rep_pairs) / declared if declared else 0.0
    computable = len(v_rep_pairs) > 0 and cov + 1e-9 >= MIN_COMP_COV
    v_ci = cluster_mean_ci(v_rep_pairs, draws=BOOT_DRAWS, seed=SEED)
    rng_ci = cluster_mean_ci(raw_rng_pairs, draws=BOOT_DRAWS, seed=SEED)
    out: dict[str, Any] = {
        "n_scored": len(v_rep_pairs),
        "coverage": round(cov, 6),
        "computable": computable,
        "v_rep": v_ci["value"] if computable else None,
        "v_rep_ci_lo": v_ci["ci_lo"] if computable else None,
        "v_rep_ci_hi": v_ci["ci_hi"] if computable else None,
        "raw_range": rng_ci["value"] if computable else None,
        "raw_range_ci_lo": rng_ci["ci_lo"] if computable else None,
        "raw_range_ci_hi": rng_ci["ci_hi"] if computable else None,
    }
    if cal_ev_rng_pairs:
        ev_ci = cluster_mean_ci(cal_ev_rng_pairs, draws=BOOT_DRAWS, seed=SEED)
        tvd_ci = cluster_mean_ci(cal_tvd_pairs, draws=BOOT_DRAWS, seed=SEED)
        out["cal_ev_range_05"] = ev_ci["value"] if computable else None
        out["cal_ev_range_05_ci_lo"] = ev_ci["ci_lo"] if computable else None
        out["cal_ev_range_05_ci_hi"] = ev_ci["ci_hi"] if computable else None
        out["cal_tvd"] = tvd_ci["value"] if computable else None
        out["cal_tvd_ci_lo"] = tvd_ci["ci_lo"] if computable else None
        out["cal_tvd_ci_hi"] = tvd_ci["ci_hi"] if computable else None
    return out


def fit_nonneg_ridge_ati(
    train_X: np.ndarray,
    train_y: np.ndarray,
    train_families: Sequence[str],
    lambdas: Sequence[float] = RIDGE_LAMBDAS,
) -> tuple[np.ndarray, float, float]:
    """Fit nonnegative Ridge (slopes >= 0 on centered X, inner LOMFO-CV for lambda)."""
    if len(train_X) < 3:
        raise ValueError("Need at least 3 training rows for ATI_cal_rps")

    def solve_one(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float, np.ndarray]:
        mu_x = X.mean(axis=0)
        mu_y = float(y.mean())
        Xc = X - mu_x
        yc = y - mu_y
        p = X.shape[1]
        A = np.vstack([Xc, math.sqrt(float(lam)) * np.eye(p)])
        b = np.concatenate([yc, np.zeros(p)])
        w, _ = optimize.nnls(A, b)
        return w, mu_y, mu_x

    uniq_f = sorted(set(train_families))
    best_lam = float(lambdas[0])
    best_cv = -float("inf")
    if len(uniq_f) >= 3:
        fam_arr = np.asarray(train_families)
        for lam in lambdas:
            oof = np.empty(len(train_y), dtype=float)
            for fam in uniq_f:
                mask = fam_arr == fam
                w, mu_y, mu_x = solve_one(train_X[~mask], train_y[~mask], lam)
                oof[mask] = mu_y + (train_X[mask] - mu_x) @ w
            sp = spearman_corr(oof, train_y)
            score = sp if sp is not None else -float("inf")
            if score > best_cv + 1e-12:
                best_cv = score
                best_lam = float(lam)

    w, mu_y, mu_x = solve_one(train_X, train_y, best_lam)
    return w, float(mu_y - float(mu_x @ w)), best_lam


def predict_nonneg_ridge(X: np.ndarray, w: np.ndarray, intercept: float) -> np.ndarray:
    return intercept + X @ w


def analyze_part_a_and_b(
    items: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    panel = read_json(E39_DIR / "panel.json")
    pm39 = read_json(E39_DIR / "probe_manifest.json")
    pm42 = read_json(E42_DIR / "semantic_probe_manifest.json")
    u39_all = read_json(E39_DIR / "units_train_val.json")["units"] + read_json(E39_DIR / "units_test.json")["units"]
    u39_by_cfg = {u["config"]: u for u in u39_all}
    s42_by_cfg = {u["config"]: u for u in read_json(E42_DIR / "semantic_screen.json")["units"]}

    e68 = read_json(E68_JSON)
    e68_by_cfg = {u["config"]: u for u in e68["units"]}
    exp03 = read_json(EXP03_JSON)
    exp03_by_cfg = {u["config"]: u for u in exp03["units"]}

    rows39_by_kind: dict[str, list[dict[str, Any]]] = {}
    for r in pm39["rows"]:
        if r.get("valid"):
            rows39_by_kind.setdefault(r["kind"], []).append(r)

    rows42_by_fam: dict[str, list[dict[str, Any]]] = {}
    for r in pm42["rows"]:
        if r.get("valid"):
            rows42_by_fam.setdefault(r["family"], []).append(r)

    train_items = sorted([it for it in items if it["split"] == "train"], key=lambda r: r["item"])
    heldout_items = sorted([it for it in items if it["split"] in ("val", "test")], key=lambda r: r["item"])
    ho_item_ids = {it["item"] for it in heldout_items}
    heldout_crit_rows = [r for r in rows39_by_kind["crit"] if str(r["item"]) in ho_item_ids]

    tr_all5 = [it["all5_ratings"] for it in train_items]
    ho_all5 = [it["all5_ratings"] for it in heldout_items]
    ho_exp3 = [it["expert3_ratings"] for it in heldout_items]
    ho_tasks = [it["task"] for it in heldout_items]

    const_probs_5 = fit_const(tr_all5)(np.zeros(len(heldout_items)))
    const_losses_5 = per_item_losses(const_probs_5, ho_all5)
    const_all5_ci = {
        k: cluster_mean_ci(list(zip(ho_tasks, const_losses_5[k])), draws=BOOT_DRAWS, seed=SEED)
        for k in ("rps", "ce", "tvd", "brier")
    }

    const_probs_3 = fit_const([it["expert3_ratings"] for it in train_items])(np.zeros(len(heldout_items)))
    const_losses_3 = per_item_losses(const_probs_3, ho_exp3)
    const_exp3_ci = {
        k: cluster_mean_ci(list(zip(ho_tasks, const_losses_3[k])), draws=BOOT_DRAWS, seed=SEED)
        for k in ("rps", "ce", "tvd", "brier")
    }

    loro_all5 = leave_one_rater_out(ho_all5, ho_tasks, draws=BOOT_DRAWS, seed=SEED)
    loro_exp3 = leave_one_rater_out(ho_exp3, ho_tasks, draws=BOOT_DRAWS, seed=SEED)

    # Alignment Gate A1 for baselines
    max_diff_a1 = 0.0
    checks_a1 = 0
    for key in ("rps", "ce", "tvd", "brier"):
        d5 = abs(float(np.mean(const_losses_5[key])) - float(e68["const_train_baseline_heldout_all5"][key]))
        d3 = abs(float(np.mean(const_losses_3[key])) - float(e68["const_train_baseline_heldout_exp3"][key]))
        dl5 = abs(loro_all5["smoothed"][key]["value"] - float(e68["human_loro_heldout_all5"]["smoothed"][key]))
        dl3 = abs(loro_exp3["smoothed"][key]["value"] - float(e68["human_loro_heldout_exp3"]["smoothed"][key]))
        for d in (d5, d3, dl5, dl3):
            if not math.isfinite(d) or d > 1e-9:
                raise RuntimeError(f"Alignment Gate A1 baseline mismatch ({key}): diff={d}")
            max_diff_a1 = max(max_diff_a1, d)
            checks_a1 += 1

    max_diff_a2 = 0.0
    checks_a2 = 0

    min_rep_39 = {"ident": 2, "bm_gold": 2, "crit": 2, "ms_gold": 1, "blank": 1, "alpha": 1}
    min_rep_42 = {k: 2 for k in ("ident", "weak", "vac_true", "vac_hyp", "overc", "fmt_comments", "fmt_layout")}

    records: list[dict[str, Any]] = []

    for cfg in sorted(u39_by_cfg):
        u39 = u39_by_cfg[cfg]
        u42 = s42_by_cfg[cfg]
        u68 = e68_by_cfg[cfg]
        u03 = exp03_by_cfg[cfg]
        fname = cfg.replace("|", "__") + ".json"
        h39 = read_json(E39_DIR / "harvest" / fname)
        h42 = read_json(E42_DIR / "harvest" / fname)

        scores39, raw39 = aggregate_harvest(h39, min_rep_39, default_min_repeats=2)
        scores42, _ = aggregate_harvest(h42, min_rep_42, default_min_repeats=2)

        family = u39["family"]
        model_info = panel["models"].get(family, {})
        model_declared = model_info.get("selectors", [cfg])[0] if u39["kind"] == "llm" else cfg
        model_served = u42.get("selector") or h39.get("selector") or model_declared
        base_model_family = model_declared if u39["kind"] == "llm" else "control"

        # Raw axiomatic property rates (must match Exp 03 within 1e-9)
        v_bug = eval_paired_axiom(rows39_by_kind["bm_gold"], scores39, lambda r: ("ident", str(r["task"]), 0), lambda r: ("bm_gold", str(r["task"]), int(r["level"])), "ge_base")
        v_ms = eval_paired_axiom(rows39_by_kind["ms_gold"], scores39, lambda r: ("ident", str(r["task"]), 0), lambda r: ("ms_gold", str(r["task"]), int(r["level"])), "ge_base")
        v_weak = eval_paired_axiom(rows42_by_fam["weakening"], scores42, lambda r: ("ident", str(r["task"]), 1), lambda r: (r["kind"], str(r["task"]), int(r["level"])), "ge_base")
        v_vac = eval_paired_axiom(rows42_by_fam["vacuity"], scores42, lambda r: ("ident", str(r["task"]), 1), lambda r: (r["kind"], str(r["task"]), int(r["level"])), "ge_base")
        v_overc = eval_paired_axiom(rows42_by_fam["overconstraint"], scores42, lambda r: ("ident", str(r["task"]), 1), lambda r: (r["kind"], str(r["task"]), int(r["level"])), "ge_base")
        v_fmt = eval_paired_axiom(rows42_by_fam["formatting"], scores42, lambda r: ("ident", str(r["task"]), 1), lambda r: (r["kind"], str(r["task"]), int(r["level"])), "abs_diff")
        rep_all = eval_repeat_stats(rows39_by_kind["crit"], raw39, cal_fn=None, min_repeats=2)

        # Alignment Gate A2 vs Exp 03
        for pname, pval in (
            ("V_bug", v_bug["value"]),
            ("V_ms", v_ms["value"]),
            ("V_weak", v_weak["value"]),
            ("V_vac", v_vac["value"]),
            ("V_overc", v_overc["value"]),
            ("V_fmt", v_fmt["value"]),
            ("V_rep", rep_all["v_rep"]),
        ):
            ref = u03[pname]["value"]
            if (pval is None) != (ref is None):
                raise RuntimeError(f"Alignment Gate A2 computability mismatch {cfg} {pname}")
            if pval is not None:
                diff = abs(float(pval) - float(ref))
                if not math.isfinite(diff) or diff > 1e-9:
                    raise RuntimeError(f"Alignment Gate A2 mismatch {cfg} {pname}: {diff}")
                max_diff_a2 = max(max_diff_a2, diff)
                checks_a2 += 1
        diff_rng = abs(float(rep_all["raw_range"]) - float(u03["V_rep"]["mean_range"]))
        if not math.isfinite(diff_rng) or diff_rng > 1e-9:
            raise RuntimeError(f"Alignment Gate A2 rep_mean_range mismatch {cfg}: {diff_rng}")
        max_diff_a2 = max(max_diff_a2, diff_rng)
        checks_a2 += 1

        # Proper-scoring calibration & evaluation (matching E68 exactly)
        crit_by_item = {key: vals for (kind, key, _lvl), vals in raw39.items() if kind == "crit" and len(vals) >= 1}
        tr_elig = [it for it in train_items if it["item"] in crit_by_item]
        ho_elig = [it for it in heldout_items if it["item"] in crit_by_item]
        tr_cov = len(tr_elig) / float(len(train_items))
        ho_cov = len(ho_elig) / float(len(heldout_items))

        proper: dict[str, Any] = {"train_cov": round(tr_cov, 4), "heldout_cov": round(ho_cov, 4)}
        cal_axioms: dict[str, Any] = {}
        rep_ho: dict[str, Any] = {}

        if ho_cov + 1e-9 >= MIN_CRIT_COV and tr_cov + 1e-9 >= MIN_CRIT_COV:
            raw_probs = np.vstack([binned_repeats_to_probs(crit_by_item[it["item"]], alpha=ALPHA_SMOOTH) for it in ho_elig])
            raw_med_05 = np.array([5.0 * float(np.median(crit_by_item[it["item"]])) for it in ho_elig], dtype=float)
            ho_r_all5 = [it["all5_ratings"] for it in ho_elig]
            ho_r_exp3 = [it["expert3_ratings"] for it in ho_elig]
            ho_t_elig = [it["task"] for it in ho_elig]

            raw_l5 = per_item_losses(raw_probs, ho_r_all5)
            raw_sp_exp3 = spearman_corr(raw_med_05, [float(np.mean(r)) for r in ho_r_exp3])
            raw_sp_all5 = spearman_corr(raw_med_05, [float(np.mean(r)) for r in ho_r_all5])

            tr_s = np.array([float(np.median(crit_by_item[it["item"]])) for it in tr_elig], dtype=float)
            ho_s = np.array([float(np.median(crit_by_item[it["item"]])) for it in ho_elig], dtype=float)
            tr_r_all5 = [it["all5_ratings"] for it in tr_elig]
            tr_r_exp3 = [it["expert3_ratings"] for it in tr_elig]

            ce_fn_5 = fit_ce_ordinal(tr_s, tr_r_all5)
            ce_probs_5 = ce_fn_5(ho_s)
            ce_l5 = per_item_losses(ce_probs_5, ho_r_all5)
            icc_a5, _ = icc_two_way_both(ce_l5["ev_05"], ce_l5["hum_mean_05"])

            ce_fn_3 = fit_ce_ordinal(tr_s, tr_r_exp3)
            ce_probs_3 = ce_fn_3(ho_s)
            ce_l3 = per_item_losses(ce_probs_3, ho_r_exp3)

            # Alignment Gate A1 vs E68
            m68 = u68["metrics"]
            for name, val in (
                ("raw_all5_rps", float(np.mean(raw_l5["rps"]))),
                ("ce_cal_all5_rps", float(np.mean(ce_l5["rps"]))),
                ("ce_cal_all5_ce", float(np.mean(ce_l5["ce"]))),
                ("ce_cal_all5_tvd", float(np.mean(ce_l5["tvd"]))),
                ("ce_cal_all5_icc_a1", icc_a5),
                ("raw_exp3_spearman", raw_sp_exp3),
                ("ce_cal_exp3_rps", float(np.mean(ce_l3["rps"]))),
            ):
                ref_val = m68.get(name)
                val_undef = val is None or not math.isfinite(float(val))
                ref_undef = ref_val is None or not math.isfinite(float(ref_val))
                if val_undef != ref_undef:
                    raise RuntimeError(f"Alignment Gate A1 definedness mismatch {cfg} {name}: {val} vs {ref_val}")
                if not val_undef:
                    diff = abs(float(val) - float(ref_val))
                    if not math.isfinite(diff) or diff > 1e-9:
                        raise RuntimeError(f"Alignment Gate A1 mismatch {cfg} {name}: {val} vs {ref_val} (diff={diff})")
                    max_diff_a1 = max(max_diff_a1, diff)
                checks_a1 += 1

            # Matched subset CONST_train baseline for paired contrasts
            ho_idx_map = {it["item"]: i for i, it in enumerate(heldout_items)}
            matched_idx = np.array([ho_idx_map[it["item"]] for it in ho_elig], dtype=int)

            def metric_bundle(losses: dict[str, np.ndarray], base_losses: dict[str, np.ndarray], ratings: list[list[int]]) -> dict[str, Any]:
                out_b: dict[str, Any] = {}
                for mkey in ("rps", "ce", "tvd", "brier"):
                    vals = losses[mkey]
                    base_vals = base_losses[mkey][matched_idx]
                    ci_v = cluster_mean_ci(list(zip(ho_t_elig, vals)), draws=BOOT_DRAWS, seed=SEED)
                    ci_d = cluster_mean_ci(list(zip(ho_t_elig, vals - base_vals)), draws=BOOT_DRAWS, seed=SEED)
                    p_d = paired_task_signflip_p(ho_t_elig, vals, base_vals, draws=PAIR_PERM_DRAWS, seed=SEED)
                    out_b[mkey] = ci_v
                    out_b[f"d_{mkey}_vs_const"] = {**ci_d, "p_val": p_d["p_val"]}
                ev = losses["ev_05"]
                hm = losses["hum_mean_05"]
                icc_a, _ = icc_two_way_both(ev, hm)
                out_b["icc_a1"] = icc_a
                out_b["mae_05"] = float(np.mean(np.abs(ev - hm)))
                out_b["bias_05"] = float(np.mean(ev - hm))
                out_b["spearman"] = spearman_corr(ev, hm)
                out_b["pearson"] = pearson_corr(ev, hm)
                return out_b

            proper["raw_all5_rps"] = float(np.mean(raw_l5["rps"]))
            proper["raw_all5_spearman"] = raw_sp_all5
            proper["raw_exp3_spearman"] = raw_sp_exp3
            proper["ce_cal_all5"] = metric_bundle(ce_l5, const_losses_5, ho_r_all5)
            proper["ce_cal_all5"]["n_heldout_items"] = len(ho_elig)
            if len(ho_elig) < len(heldout_items):
                loro_matched_s = leave_one_rater_out(ho_r_all5, ho_t_elig, draws=BOOT_DRAWS, seed=SEED)["smoothed"]
            else:
                loro_matched_s = loro_all5["smoothed"]
            proper["ce_cal_all5"]["matched_loro_rps"] = float(loro_matched_s["rps"]["value"])
            proper["ce_cal_all5"]["matched_loro_rps_ci"] = [float(loro_matched_s["rps"]["ci_lo"]), float(loro_matched_s["rps"]["ci_hi"])]
            proper["ce_cal_all5"]["matched_loro_ce"] = float(loro_matched_s["ce"]["value"])
            proper["ce_cal_all5"]["matched_loro_ce_ci"] = [float(loro_matched_s["ce"]["ci_lo"]), float(loro_matched_s["ce"]["ci_hi"])]
            proper["ce_cal_exp3"] = metric_bundle(ce_l3, const_losses_3, ho_r_exp3)

            # Calibrated-distribution axiomatic metrics (using ce_fn_5)
            cal_axioms["V_weak_cal"] = eval_paired_axiom(rows42_by_fam["weakening"], scores42, lambda r: ("ident", str(r["task"]), 1), lambda r: (r["kind"], str(r["task"]), int(r["level"])), "ge_base", cal_fn=ce_fn_5)
            cal_axioms["V_vac_cal"] = eval_paired_axiom(rows42_by_fam["vacuity"], scores42, lambda r: ("ident", str(r["task"]), 1), lambda r: (r["kind"], str(r["task"]), int(r["level"])), "ge_base", cal_fn=ce_fn_5)
            cal_axioms["V_overc_cal"] = eval_paired_axiom(rows42_by_fam["overconstraint"], scores42, lambda r: ("ident", str(r["task"]), 1), lambda r: (r["kind"], str(r["task"]), int(r["level"])), "ge_base", cal_fn=ce_fn_5)
            cal_axioms["V_fmt_cal"] = eval_paired_axiom(rows42_by_fam["formatting"], scores42, lambda r: ("ident", str(r["task"]), 1), lambda r: (r["kind"], str(r["task"]), int(r["level"])), "abs_diff", cal_fn=ce_fn_5)
            rep_ho = eval_repeat_stats(heldout_crit_rows, raw39, cal_fn=ce_fn_5, min_repeats=2)

        # Label-free ATI feature vector
        props_raw = {
            "V_ms": v_ms,
            "V_weak": v_weak,
            "V_vac": v_vac,
            "V_overc": v_overc,
            "V_fmt": v_fmt,
            "V_rep": {"value": rep_all["v_rep"], "ci_lo": rep_all["v_rep_ci_lo"], "ci_hi": rep_all["v_rep_ci_hi"], "raw_range": rep_all["raw_range"]},
            "V_bug": v_bug,
        }
        feat = {
            "pass_ms": 1.0 - float(v_ms["value"]),
            "pass_weak": 1.0 - float(v_weak["value"]),
            "pass_vac": 1.0 - float(v_vac["value"]),
            "pass_overc": 1.0 - float(v_overc["value"]),
            "pass_fmt": 1.0 - float(v_fmt["value"]),
            "pass_rep": 1.0 - float(rep_all["v_rep"]),
        }
        feat_vals = [feat[k] for k in ATI_FEATURES]
        ati_equal = float(np.mean(feat_vals))
        ati_min = float(np.min(feat_vals))
        ati_plus_bug = float(np.mean(feat_vals + [1.0 - float(v_bug["value"])]))
        gate_pass = all(float(props_raw[k]["value"]) < cut for k, cut in GATE_CUTS.items())

        # Qualification flags
        eligible = "ce_cal_all5" in proper
        beats_const_rps = bool(eligible and proper["ce_cal_all5"]["d_rps_vs_const"]["ci_hi"] < 0.0)
        beats_const_ce = bool(eligible and proper["ce_cal_all5"]["d_ce_vs_const"]["ci_hi"] < 0.0)
        beats_const_tvd = bool(eligible and proper["ce_cal_all5"]["d_tvd_vs_const"]["ci_hi"] < 0.0)
        beats_const_all3 = beats_const_rps and beats_const_ce and beats_const_tvd
        beats_loro_rps_ce = bool(
            beats_const_rps
            and proper["ce_cal_all5"]["rps"]["value"] <= proper["ce_cal_all5"]["matched_loro_rps"] + 1e-12
            and proper["ce_cal_all5"]["ce"]["value"] <= proper["ce_cal_all5"]["matched_loro_ce"] + 1e-12
        )
        beats_const_exp3_rps = bool(eligible and proper["ce_cal_exp3"]["d_rps_vs_const"]["ci_hi"] < 0.0)

        # Non-bug H1 trigger: PREREG H1-proper checks V_weak/V_rep >= 0.20 (ci_lo > 0.05); also track all 6 non-bug properties.
        prereg_h1_triggers = []
        all_nonbug_triggers = []
        for pname in ("V_weak", "V_rep", "V_overc", "V_vac", "V_ms", "V_fmt"):
            blk = props_raw[pname]
            if blk["value"] is not None and blk["value"] >= 0.20 - 1e-12 and blk["ci_lo"] > 0.05 + 1e-12:
                all_nonbug_triggers.append(pname)
                if pname in ("V_weak", "V_rep"):
                    prereg_h1_triggers.append(pname)

        records.append({
            "config": cfg,
            "kind": u39["kind"],
            "harness": u39["harness"],
            "family": family,
            "base_model_family": base_model_family,
            "split": u39["split"],
            "model_declared": model_declared,
            "model_served": model_served,
            "rho_heldout_exp3": (u39["criteria"].get("heldout") or {}).get("spearman"),
            "eligible": eligible,
            "beats_const_rps": beats_const_rps,
            "beats_const_ce": beats_const_ce,
            "beats_const_tvd": beats_const_tvd,
            "beats_const_all3": beats_const_all3,
            "beats_loro_rps_ce": beats_loro_rps_ce,
            "beats_const_exp3_rps": beats_const_exp3_rps,
            "proper": proper,
            "props_raw": props_raw,
            "rep_all": rep_all,
            "cal_axioms": cal_axioms,
            "rep_heldout": rep_ho,
            "prereg_h1_triggers": prereg_h1_triggers,
            "fails_prereg_h1": len(prereg_h1_triggers) > 0,
            "nonbug_triggers": all_nonbug_triggers,
            "fails_nonbug_h1": len(all_nonbug_triggers) > 0,
            "e39_indices": u39["indices"],
            "ati_features": feat,
            "ati_equal": ati_equal,
            "ati_min": ati_min,
            "ati_plus_bug": ati_plus_bug,
            "gate_all_pass": gate_pass,
        })

    # Part A summary & H1-proper verdict
    real_recs = [r for r in records if r["kind"] == "llm"]
    q_const_rps = sorted([r for r in real_recs if r["beats_const_rps"]], key=lambda r: r["proper"]["ce_cal_all5"]["rps"]["value"])
    q_const_all3 = [r for r in q_const_rps if r["beats_const_all3"]]
    q_loro_rps = [r for r in q_const_rps if r["beats_loro_rps_ce"]]
    q_const_exp3 = [r for r in real_recs if r["beats_const_exp3_rps"]]

    if not q_const_rps:
        h1_proper_verdict = "untestable"
    else:
        any_h1_fail = any(
            (r["props_raw"]["V_weak"]["value"] >= 0.20 - 1e-12 and r["props_raw"]["V_weak"]["ci_lo"] > 0.05 + 1e-12)
            or (r["props_raw"]["V_rep"]["value"] >= 0.20 - 1e-12 and r["props_raw"]["V_rep"]["ci_lo"] > 0.05 + 1e-12)
            for r in q_const_rps
        )
        all_clean = all(
            r["props_raw"]["V_weak"]["value"] < 0.10
            and r["props_raw"]["V_vac"]["value"] < 0.10
            and r["props_raw"]["V_ms"]["value"] < 0.10
            and r["props_raw"]["V_overc"]["value"] < 0.10
            and r["props_raw"]["V_fmt"]["value"] < 0.05
            and r["props_raw"]["V_rep"]["value"] < 0.10
            for r in q_const_rps
        )
        if any_h1_fail:
            h1_proper_verdict = "supported"
        elif all_clean:
            h1_proper_verdict = "falsified"
        else:
            h1_proper_verdict = "mixed"

    part_a_summary = {
        "n_real_configs": len(real_recs),
        "n_eligible": sum(1 for r in real_recs if r["eligible"]),
        "n_q_const_rps": len(q_const_rps),
        "n_q_const_rps_fail_prereg_h1": sum(1 for r in q_const_rps if r["fails_prereg_h1"]),
        "n_q_const_rps_fail_nonbug": sum(1 for r in q_const_rps if r["fails_nonbug_h1"]),
        "n_q_const_all3": len(q_const_all3),
        "n_q_const_all3_fail_prereg_h1": sum(1 for r in q_const_all3 if r["fails_prereg_h1"]),
        "n_q_const_all3_fail_nonbug": sum(1 for r in q_const_all3 if r["fails_nonbug_h1"]),
        "n_q_loro_rps": len(q_loro_rps),
        "n_q_loro_rps_fail_prereg_h1": sum(1 for r in q_loro_rps if r["fails_prereg_h1"]),
        "n_q_loro_rps_fail_nonbug": sum(1 for r in q_loro_rps if r["fails_nonbug_h1"]),
        "n_q_const_exp3_rps": len(q_const_exp3),
        "n_q_const_exp3_fail_prereg_h1": sum(1 for r in q_const_exp3 if r["fails_prereg_h1"]),
        "n_q_const_exp3_fail_nonbug": sum(1 for r in q_const_exp3 if r["fails_nonbug_h1"]),
        "q_const_rps_configs": [r["config"] for r in q_const_rps],
        "q_const_all3_configs": [r["config"] for r in q_const_all3],
        "q_loro_rps_configs": [r["config"] for r in q_loro_rps],
        "const_train_all5": const_all5_ci,
        "const_train_exp3": const_exp3_ci,
        "loro_all5": loro_all5,
        "loro_exp3": loro_exp3,
        "verdict_h1_proper": h1_proper_verdict,
    }

    # Part B: Idea 3 out-of-family certification
    def feature_mat(rows_list: Sequence[dict[str, Any]]) -> np.ndarray:
        return np.array([[r["ati_features"][k] for k in ATI_FEATURES] for r in rows_list], dtype=float)

    tv_real = [r for r in real_recs if r["split"] in ("train", "val") and r["eligible"]]
    te_real = [r for r in real_recs if r["split"] == "test" and r["eligible"]]
    te_unseen = [r for r in te_real if r["base_model_family"] != "anthropic/claude-sonnet-5-medium"]

    w_tv, b_tv, lam_tv = fit_nonneg_ridge_ati(
        feature_mat(tv_real),
        np.array([-r["proper"]["ce_cal_all5"]["rps"]["value"] for r in tv_real], dtype=float),
        [r["base_model_family"] for r in tv_real],
    )
    all_ati_cal = predict_nonneg_ridge(feature_mat(records), w_tv, b_tv)
    for r, val in zip(records, all_ati_cal):
        r["ati_cal_rps_tv"] = float(val)

    # 6-fold LOMFO-CV across base_model_family (zero weight leakage across terminus2 vs vb103_judge)
    base_fams = sorted({r["base_model_family"] for r in real_recs if r["eligible"]})
    lomfo_folds = []
    for hold_fam in base_fams:
        tr_f = [r for r in real_recs if r["eligible"] and r["base_model_family"] != hold_fam]
        te_f = [r for r in real_recs if r["eligible"] and r["base_model_family"] == hold_fam]
        w_f, b_f, lam_f = fit_nonneg_ridge_ati(
            feature_mat(tr_f),
            np.array([-r["proper"]["ce_cal_all5"]["rps"]["value"] for r in tr_f], dtype=float),
            [r["base_model_family"] for r in tr_f],
        )
        preds_f = predict_nonneg_ridge(feature_mat(te_f), w_f, b_f)
        for r, p_val in zip(te_f, preds_f):
            r["ati_cal_rps_lomfo_oof"] = float(p_val)
        best_rps_f = min(r["proper"]["ce_cal_all5"]["rps"]["value"] for r in te_f)
        fold_row: dict[str, Any] = {"heldout_family": hold_fam, "n_configs": len(te_f), "best_rps": best_rps_f, "ridge_lambda": lam_f}
        for idx_name, getter in (
            ("ATI_equal", lambda r: r["ati_equal"]),
            ("ATI_min", lambda r: r["ati_min"]),
            ("ATI_cal_rps", lambda r: r["ati_cal_rps_lomfo_oof"]),
            ("C_ms_gold", lambda r: r["e39_indices"].get("C_ms_gold")),
            ("1_minus_V_weak", lambda r: r["ati_features"]["pass_weak"]),
            ("TI_var_trainnorm", lambda r: r["e39_indices"].get("TI_var_trainnorm")),
        ):
            elig_f = [(r, getter(r)) for r in te_f if getter(r) is not None]
            if elig_f:
                xs_f = [v for _, v in elig_f]
                ys_f = [-r["proper"]["ce_cal_all5"]["rps"]["value"] for r, _ in elig_f]
                sp_f = spearman_corr(xs_f, ys_f)
                best_v = max(v for _, v in elig_f)
                tied_f = [r for r, v in elig_f if abs(v - best_v) <= 1e-12]
                reg_f = float(np.mean([r["proper"]["ce_cal_all5"]["rps"]["value"] for r in tied_f]) - best_rps_f)
            else:
                sp_f, reg_f = None, None
            fold_row[idx_name] = {"spearman": sp_f, "regret_rps": reg_f}
        lomfo_folds.append(fold_row)

    lomfo_fold_summary: dict[str, Any] = {}
    for idx_name in ("ATI_equal", "ATI_min", "ATI_cal_rps", "C_ms_gold", "1_minus_V_weak", "TI_var_trainnorm"):
        valid_folds = [f for f in lomfo_folds if f[idx_name]["regret_rps"] is not None]
        sps = [float(f[idx_name]["spearman"]) for f in valid_folds if f[idx_name]["spearman"] is not None]
        regs = [(f["heldout_family"], float(f[idx_name]["regret_rps"])) for f in valid_folds]
        max_fam, max_reg = max(regs, key=lambda kv: kv[1]) if regs else (None, None)
        lomfo_fold_summary[idx_name] = {
            "n_folds": len(valid_folds),
            "mean_fold_spearman": float(np.mean(sps)) if sps else None,
            "mean_fold_regret_rps": float(np.mean([rv for _, rv in regs])) if regs else None,
            "max_fold_regret_rps": max_reg,
            "max_fold_regret_family": max_fam,
            "per_fold_regret_rps": {f["heldout_family"]: f[idx_name]["regret_rps"] for f in lomfo_folds},
        }

    # Evaluate index table across splits (test n=18, test_unseen n=16, lomfo_oof n=56) and controls
    ctls = [r for r in records if r["kind"] == "control"]
    q_const_rps_tv = [r for r in q_const_rps if r["split"] in ("train", "val")]
    index_defs = [
        ("ATI_equal", lambda r: r["ati_equal"], lambda r: r["ati_equal"]),
        ("ATI_min", lambda r: r["ati_min"], lambda r: r["ati_min"]),
        ("ATI_cal_rps", lambda r: r["ati_cal_rps_tv"], lambda r: r["ati_cal_rps_lomfo_oof"]),
        ("ATI_plus_bug", lambda r: r["ati_plus_bug"], lambda r: r["ati_plus_bug"]),
        ("C_ms_gold", lambda r: r["e39_indices"].get("C_ms_gold"), lambda r: r["e39_indices"].get("C_ms_gold")),
        ("1_minus_V_weak", lambda r: r["ati_features"]["pass_weak"], lambda r: r["ati_features"]["pass_weak"]),
        ("1_minus_V_rep", lambda r: r["ati_features"]["pass_rep"], lambda r: r["ati_features"]["pass_rep"]),
        ("1_minus_V_fmt", lambda r: r["ati_features"]["pass_fmt"], lambda r: r["ati_features"]["pass_fmt"]),
        ("1_minus_V_bug", lambda r: 1.0 - float(r["props_raw"]["V_bug"]["value"]), lambda r: 1.0 - float(r["props_raw"]["V_bug"]["value"])),
        ("TI_var_trainnorm", lambda r: r["e39_indices"].get("TI_var_trainnorm"), lambda r: r["e39_indices"].get("TI_var_trainnorm")),
    ]

    part_b_rows: list[dict[str, Any]] = []
    for split_name, pool_rows, use_oof in (
        ("test", te_real, False),
        ("test_unseen_weights", te_unseen, False),
        ("lomfo_oof_all", [r for r in real_recs if r["eligible"]], True),
    ):
        best_rps_pool = min(r["proper"]["ce_cal_all5"]["rps"]["value"] for r in pool_rows)
        best_ce_pool = min(r["proper"]["ce_cal_all5"]["ce"]["value"] for r in pool_rows)
        best_tvd_pool = min(r["proper"]["ce_cal_all5"]["tvd"]["value"] for r in pool_rows)
        for idx_name, g_tv, g_oof in index_defs:
            getter = g_oof if use_oof else g_tv
            elig = [(r, getter(r)) for r in pool_rows if getter(r) is not None]
            xs = [float(v) for _, v in elig]
            fams = [r["base_model_family"] for r, _ in elig]
            # Pooled fitted LOMFO OOF predictions depend on other folds' labels, so fixed-predictor CI/p-vals are invalid.
            is_pooled_fitted = (split_name == "lomfo_oof_all" and idx_name == "ATI_cal_rps")

            def _assoc(ys: Sequence[float]) -> dict[str, Any]:
                if is_pooled_fitted:
                    sp = spearman_corr(xs, ys)
                    pe = pearson_corr(xs, ys)
                    return {"spearman": sp, "pearson": pe, "ci_lo": None, "ci_hi": None, "p_val": None, "within_family_p": None}
                return unit_meta_corr(xs, ys, fams)

            rps_assoc = _assoc([-r["proper"]["ce_cal_all5"]["rps"]["value"] for r, _ in elig])
            ce_assoc = _assoc([-r["proper"]["ce_cal_all5"]["ce"]["value"] for r, _ in elig])
            tvd_assoc = _assoc([-r["proper"]["ce_cal_all5"]["tvd"]["value"] for r, _ in elig])
            icc_assoc = _assoc([r["proper"]["ce_cal_all5"]["icc_a1"] for r, _ in elig])
            exp3_sp_assoc = _assoc([r["proper"]["raw_exp3_spearman"] for r, _ in elig])

            best_idx = max(v for _, v in elig)
            tied = [r for r, v in elig if abs(v - best_idx) <= 1e-12]
            reg_rps = float(np.mean([r["proper"]["ce_cal_all5"]["rps"]["value"] for r in tied]) - best_rps_pool)
            reg_ce = float(np.mean([r["proper"]["ce_cal_all5"]["ce"]["value"] for r in tied]) - best_ce_pool)
            reg_tvd = float(np.mean([r["proper"]["ce_cal_all5"]["tvd"]["value"] for r in tied]) - best_tvd_pool)

            # Keep preregistered full Q_const_rps cutoff, and add strictly leak-free train+val cutoff; evaluate both threshold & controls on matched g_tv scale.
            q_med_score = float(np.median([float(g_tv(r)) for r in q_const_rps if g_tv(r) is not None]))
            q_tv_med_score = float(np.median([float(g_tv(r)) for r in q_const_rps_tv if g_tv(r) is not None]))
            ctl_scores_tv = {c["config"]: (float(g_tv(c)) if g_tv(c) is not None else None) for c in ctls}
            ctl_rejected = sum(1 for v in ctl_scores_tv.values() if v is None or v < q_med_score - 1e-12)
            ctl_rejected_tv = sum(1 for v in ctl_scores_tv.values() if v is None or v < q_tv_med_score - 1e-12)
            part_b_rows.append({
                "split": split_name,
                "index": idx_name,
                "n": len(elig),
                "inference_mode": "descriptive_only_pooled_oof" if is_pooled_fitted else "valid_fixed_predictor",
                "rps_spearman": rps_assoc["spearman"],
                "rps_ci_lo": rps_assoc["ci_lo"],
                "rps_ci_hi": rps_assoc["ci_hi"],
                "rps_p": rps_assoc["p_val"],
                "rps_within_fam_p": rps_assoc["within_family_p"],
                "ce_spearman": ce_assoc["spearman"],
                "ce_ci_lo": ce_assoc["ci_lo"],
                "ce_ci_hi": ce_assoc["ci_hi"],
                "ce_p": ce_assoc["p_val"],
                "ce_within_fam_p": ce_assoc["within_family_p"],
                "tvd_spearman": tvd_assoc["spearman"],
                "tvd_ci_lo": tvd_assoc["ci_lo"],
                "tvd_ci_hi": tvd_assoc["ci_hi"],
                "tvd_p": tvd_assoc["p_val"],
                "tvd_within_fam_p": tvd_assoc["within_family_p"],
                "icc_spearman": icc_assoc["spearman"],
                "icc_ci_lo": icc_assoc["ci_lo"],
                "icc_ci_hi": icc_assoc["ci_hi"],
                "icc_p": icc_assoc["p_val"],
                "icc_within_fam_p": icc_assoc["within_family_p"],
                "exp3_rho_spearman": exp3_sp_assoc["spearman"],
                "exp3_rho_ci_lo": exp3_sp_assoc["ci_lo"],
                "exp3_rho_ci_hi": exp3_sp_assoc["ci_hi"],
                "exp3_rho_p": exp3_sp_assoc["p_val"],
                "exp3_rho_within_fam_p": exp3_sp_assoc["within_family_p"],
                "selected_configs": "|".join(r["config"] for r in tied),
                "regret_rps": reg_rps,
                "regret_ce": reg_ce,
                "regret_tvd": reg_tvd,
                "q_const_median_index": q_med_score,
                "ctl_rejected_of_4": ctl_rejected,
                "q_const_tv_median_index": q_tv_med_score,
                "ctl_rejected_trainval_q_of_4": ctl_rejected_tv,
                "ctl_scores": ctl_scores_tv,
            })

    gate_ctl_pass = sum(1 for c in ctls if c["gate_all_pass"])
    gate_real_pass = [r["config"] for r in real_recs if r["gate_all_pass"]]

    # H-Idea3 verdict: preserve literal PREREG rule (full Q_const_rps) AND explicitly compute strictly leak-free (train+val Q) rule & ATI_min extension.
    te_by_idx = {r["index"]: r for r in part_b_rows if r["split"] == "test"}

    def _idea3_rule(ctl_key: str, cands: tuple[str, ...]) -> str:
        supp = any(
            te_by_idx[k]["rps_spearman"] is not None
            and te_by_idx[k]["rps_spearman"] >= 0.50 - 1e-12
            and te_by_idx[k]["rps_p"] < 0.05
            and te_by_idx[k]["regret_rps"] <= 0.015 + 1e-12
            and te_by_idx[k][ctl_key] == 4
            and gate_ctl_pass == 0
            for k in cands
        )
        fals = (
            all(
                te_by_idx[k]["rps_spearman"] is None
                or te_by_idx[k]["rps_spearman"] < 0.30
                or te_by_idx[k]["rps_p"] >= 0.10
                for k in cands
            )
            or max(te_by_idx[k][ctl_key] for k in cands) < 4
            or gate_ctl_pass > 0
        )
        return "supported" if supp else ("falsified" if fals else "mixed")

    verdict_idea3 = _idea3_rule("ctl_rejected_of_4", ("ATI_equal", "ATI_cal_rps"))
    verdict_idea3_leak_free = _idea3_rule("ctl_rejected_trainval_q_of_4", ("ATI_equal", "ATI_cal_rps"))
    verdict_idea3_ext_min = _idea3_rule("ctl_rejected_trainval_q_of_4", ("ATI_min",))

    part_b_summary = {
        "ati_cal_weights": dict(zip(ATI_FEATURES, [float(x) for x in w_tv])),
        "ati_cal_intercept": float(b_tv),
        "ati_cal_lambda_tv": float(lam_tv),
        "gate_cuts": GATE_CUTS,
        "gate_ctl_pass_count": gate_ctl_pass,
        "gate_real_pass_configs": gate_real_pass,
        "lomfo_folds": lomfo_folds,
        "lomfo_fold_summary": lomfo_fold_summary,
        "verdict_h_idea3": verdict_idea3,
        "verdict_h_idea3_prereg_literal": verdict_idea3,
        "verdict_h_idea3_leak_free_prereg": verdict_idea3_leak_free,
        "verdict_h_idea3_leak_free_extension_ati_min": verdict_idea3_ext_min,
    }

    alignment = {
        "gate_a1_e68_checks": checks_a1,
        "gate_a1_e68_max_abs_diff": max_diff_a1,
        "gate_a2_exp03_checks": checks_a2,
        "gate_a2_exp03_max_abs_diff": max_diff_a2,
    }
    return part_a_summary, records, part_b_rows, {"part_b": part_b_summary, "alignment": alignment}


def analyze_part_c_human_regimes() -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for label, path in (("old_vb", OLD_RATINGS_CSV), ("new_vb", NEW_RATINGS_CSV)):
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        item_ratings: list[list[int]] = []
        tasks: list[str] = []
        seen_items: set[str] = set()
        for r in rows:
            it = r["item"]
            if it in seen_items:
                raise ValueError(f"Duplicate item in {path}: {it}")
            seen_items.add(it)
            rats = [int(x) for x in r["ratings_sorted_0to5_rater_identity_removed"].split()]
            if len(rats) < 2 or any(v < 0 or v > 5 for v in rats):
                raise ValueError(f"Unexpected ratings in {path} for {it}: {rats}")
            item_ratings.append(rats)
            tasks.append(str(r["task"]))

        # Individual rating distribution (prespecified rating-weighted via task-cluster ratio, plus item-weighted sensitivity)
        rat_p5_triples = [(t, float(sum(1 for v in rats if v == 5)), float(len(rats))) for t, rats in zip(tasks, item_ratings)]
        rat_p45_triples = [(t, float(sum(1 for v in rats if v >= 4)), float(len(rats))) for t, rats in zip(tasks, item_ratings)]
        p5_pairs = [(t, sum(1 for v in rats if v == 5) / len(rats)) for t, rats in zip(tasks, item_ratings)]
        p45_pairs = [(t, sum(1 for v in rats if v >= 4) / len(rats)) for t, rats in zip(tasks, item_ratings)]
        all_rats = [v for rats in item_ratings for v in rats]
        raw_p5 = sum(1 for v in all_rats if v == 5) / len(all_rats)
        raw_p45 = sum(1 for v in all_rats if v >= 4) / len(all_rats)

        # Item-level disagreement & pairwise exact agreement
        unanimous_pairs = []
        ge1_pairs = []
        ge2_pairs = []
        rng_pairs = []
        sd_pairs = []
        pw_exact_pairs = []
        disag_floor_pairs = []
        nonceil_ge1_pairs = []
        nonceil_ge2_pairs = []
        nonceil_rng_pairs = []
        nonceil_pw_pairs = []

        for t, rats in zip(tasks, item_ratings):
            span = max(rats) - min(rats)
            unanimous_pairs.append((t, 1.0 if span == 0 else 0.0))
            ge1_pairs.append((t, 1.0 if span >= 1 else 0.0))
            ge2_pairs.append((t, 1.0 if span >= 2 else 0.0))
            rng_pairs.append((t, float(span)))
            sd_pairs.append((t, float(np.std(rats, ddof=0))))
            m = len(rats)
            pw = [1.0 if rats[i] == rats[j] else 0.0 for i in range(m) for j in range(i + 1, m)]
            pw_mean = float(np.mean(pw))
            pw_exact_pairs.append((t, pw_mean))
            p_emp = empirical_dist(rats)
            F = np.cumsum(p_emp)[:5]
            disag_floor_pairs.append((t, float(np.mean(F * (1.0 - F)))))
            if min(rats) < 5:
                nonceil_ge1_pairs.append((t, 1.0 if span >= 1 else 0.0))
                nonceil_ge2_pairs.append((t, 1.0 if span >= 2 else 0.0))
                nonceil_rng_pairs.append((t, float(span)))
                nonceil_pw_pairs.append((t, pw_mean))

        loro = leave_one_rater_out(item_ratings, tasks, draws=BOOT_DRAWS, seed=SEED)
        datasets[label] = {
            "n_items": len(item_ratings),
            "n_tasks": len(set(tasks)),
            "n_ratings_total": len(all_rats),
            "mean_ratings_per_item": len(all_rats) / len(item_ratings),
            "raw_p_eq_5": raw_p5,
            "raw_p_ge_4": raw_p45,
            "rating_weighted_p_eq_5": cluster_ratio_ci(rat_p5_triples, draws=BOOT_DRAWS, seed=SEED),
            "rating_weighted_p_ge_4": cluster_ratio_ci(rat_p45_triples, draws=BOOT_DRAWS, seed=SEED),
            "item_weighted_p_eq_5": cluster_mean_ci(p5_pairs, draws=BOOT_DRAWS, seed=SEED),
            "item_weighted_p_ge_4": cluster_mean_ci(p45_pairs, draws=BOOT_DRAWS, seed=SEED),
            "unanimous_range_0": cluster_mean_ci(unanimous_pairs, draws=BOOT_DRAWS, seed=SEED),
            "range_ge_1": cluster_mean_ci(ge1_pairs, draws=BOOT_DRAWS, seed=SEED),
            "range_ge_2": cluster_mean_ci(ge2_pairs, draws=BOOT_DRAWS, seed=SEED),
            "mean_range": cluster_mean_ci(rng_pairs, draws=BOOT_DRAWS, seed=SEED),
            "mean_sd": cluster_mean_ci(sd_pairs, draws=BOOT_DRAWS, seed=SEED),
            "pairwise_exact_agree": cluster_mean_ci(pw_exact_pairs, draws=BOOT_DRAWS, seed=SEED),
            "irreducible_rps_floor": cluster_mean_ci(disag_floor_pairs, draws=BOOT_DRAWS, seed=SEED),
            "non_ceiling_n_items": len(nonceil_ge1_pairs),
            "non_ceiling_n_tasks": len({t for t, _ in nonceil_ge1_pairs}),
            "non_ceiling_range_ge_1": cluster_mean_ci(nonceil_ge1_pairs, draws=BOOT_DRAWS, seed=SEED),
            "non_ceiling_range_ge_2": cluster_mean_ci(nonceil_ge2_pairs, draws=BOOT_DRAWS, seed=SEED),
            "non_ceiling_mean_range": cluster_mean_ci(nonceil_rng_pairs, draws=BOOT_DRAWS, seed=SEED),
            "non_ceiling_pairwise_exact": cluster_mean_ci(nonceil_pw_pairs, draws=BOOT_DRAWS, seed=SEED),
            "loro_smoothed_rps": loro["smoothed"]["rps"],
            "loro_smoothed_ce": loro["smoothed"]["ce"],
            "loro_smoothed_tvd": loro["smoothed"]["tvd"],
            "loro_empirical_rps": loro["empirical"]["rps"],
            "loro_empirical_tvd": loro["empirical"]["tvd"],
            "rating_histogram_0_to_5": [all_rats.count(k) for k in range(6)],
        }

    new_d = datasets["new_vb"]
    p5_blk = new_d["rating_weighted_p_eq_5"]
    nc_blk = new_d["non_ceiling_range_ge_1"]
    if (
        p5_blk["value"] >= 0.65 - 1e-12
        and p5_blk["ci_lo"] > 0.50 + 1e-12
        and nc_blk["value"] >= 0.60 - 1e-12
        and nc_blk["ci_lo"] > 0.40 + 1e-12
    ):
        verdict_h2_new = "supported"
    elif p5_blk["value"] < 0.40 and new_d["unanimous_range_0"]["value"] >= 0.85:
        verdict_h2_new = "falsified"
    else:
        verdict_h2_new = "mixed"
    return {"datasets": datasets, "verdict_h2_new": verdict_h2_new}


def fmt_ci(blk: dict[str, Any] | None, digits: int = 3) -> str:
    if not blk or blk.get("value") is None:
        return "n/a"
    v, lo, hi = blk["value"], blk.get("ci_lo"), blk.get("ci_hi")
    if lo is None or hi is None:
        return f"{v:.{digits}f}"
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def canonical_lines(summary: dict[str, Any]) -> list[str]:
    pa = summary["part_a"]
    pb = summary["part_b"]
    pc = summary["part_c"]
    al = summary["alignment"]
    te_rows = {r["index"]: r for r in summary["part_b_table"] if r["split"] == "test"}
    oo_rows = {r["index"]: r for r in summary["part_b_table"] if r["split"] == "lomfo_oof_all"}
    old_d = pc["datasets"]["old_vb"]
    new_d = pc["datasets"]["new_vb"]
    lines = [
        f"VERDICT_H1_PROPER {pa['verdict_h1_proper']} q_const_rps={pa['n_q_const_rps']} q_const_rps_fail={pa['n_q_const_rps_fail_nonbug']} q_const_all3={pa['n_q_const_all3']} q_const_all3_fail={pa['n_q_const_all3_fail_nonbug']} q_loro={pa['n_q_loro_rps']} q_loro_fail={pa['n_q_loro_rps_fail_nonbug']}",
        f"VERDICT_H_IDEA3 prereg_literal={pb['verdict_h_idea3_prereg_literal']} leak_free_prereg={pb['verdict_h_idea3_leak_free_prereg']} extension_ati_min={pb['verdict_h_idea3_leak_free_extension_ati_min']} ati_equal_test_sp={te_rows['ATI_equal']['rps_spearman']:.3f} p={te_rows['ATI_equal']['rps_p']:.4f} reg_rps={te_rows['ATI_equal']['regret_rps']:.4f} ati_cal_test_sp={te_rows['ATI_cal_rps']['rps_spearman']:.3f} p={te_rows['ATI_cal_rps']['rps_p']:.4f} reg_rps={te_rows['ATI_cal_rps']['regret_rps']:.4f} ctl_full_q={te_rows['ATI_equal']['ctl_rejected_of_4']}/4 ctl_tv_q={te_rows['ATI_equal']['ctl_rejected_trainval_q_of_4']}/4 ctl_ati_min_tv_q={te_rows['ATI_min']['ctl_rejected_trainval_q_of_4']}/4 gate_ctl_pass={pb['gate_ctl_pass_count']}/4",
        f"LOMFO_OOF ati_equal_sp={oo_rows['ATI_equal']['rps_spearman']:.3f} p={oo_rows['ATI_equal']['rps_p']:.4f} ati_min_sp={oo_rows['ATI_min']['rps_spearman']:.3f} p={oo_rows['ATI_min']['rps_p']:.4f} ati_cal_sp={oo_rows['ATI_cal_rps']['rps_spearman']:.3f} (descriptive) cms_sp={oo_rows['C_ms_gold']['rps_spearman']:.3f} p={oo_rows['C_ms_gold']['rps_p']:.4f}",
        f"VERDICT_H2_NEW {pc['verdict_h2_new']} new_p5_rating={fmt_ci(new_d['rating_weighted_p_eq_5'])} new_nonceil_disag={fmt_ci(new_d['non_ceiling_range_ge_1'])} n_nonceil={new_d['non_ceiling_n_items']}/{new_d['n_items']} old_p5_rating={fmt_ci(old_d['rating_weighted_p_eq_5'])} old_nonceil_disag={fmt_ci(old_d['non_ceiling_range_ge_1'])}",
        f"ALIGN a1_checks={al['gate_a1_e68_checks']} a1_max_diff={al['gate_a1_e68_max_abs_diff']:.2e} a2_checks={al['gate_a2_exp03_checks']} a2_max_diff={al['gate_a2_exp03_max_abs_diff']:.2e}",
    ]
    return lines


def plot_fig1(path: Path, records: list[dict[str, Any]], pa: dict[str, Any]) -> None:
    real_elig = [r for r in records if r["kind"] == "llm" and r["eligible"]]
    ctls = [r for r in records if r["kind"] == "control" and r["eligible"]]
    q_loro = sorted([r for r in real_elig if r["beats_loro_rps_ce"]], key=lambda r: r["proper"]["ce_cal_all5"]["rps"]["value"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for grp, color, marker, label in (
        ([r for r in real_elig if not r["beats_const_rps"]], "#8c96a0", "o", "Real configs not beating CONST_train"),
        ([r for r in real_elig if r["beats_const_rps"] and not r["beats_loro_rps_ce"]], "#2b6cb0", "s", "Beats CONST_train RPS (95% CI < 0)"),
        (q_loro, "#d97706", "D", "Beats CONST_train & matches/beats LORO"),
        (ctls, "#dc2626", "^", "Deterministic controls"),
    ):
        if not grp:
            continue
        xs = [r["proper"]["ce_cal_all5"]["rps"]["value"] for r in grp]
        ys = [max(r["props_raw"]["V_weak"]["value"], r["props_raw"]["V_rep"]["value"]) for r in grp]
        ax1.scatter(xs, ys, c=color, marker=marker, s=55, alpha=0.88, label=f"{label} ({len(grp)})", edgecolors="black", linewidths=0.4)

    annot_offsets = {
        "qwen36|H2": (-5, -16),
        "qwen36|R4": (-5, 9),
        "t2_sonnet5|H2": (-10, -15),
        "gem31h|H2": (10, -9),
        "sol6h|H2": (10, 10),
        "opus55x|H2": (8, -6),
    }
    for r in q_loro[:6]:
        if r["config"] not in annot_offsets:
            continue
        x = r["proper"]["ce_cal_all5"]["rps"]["value"]
        y = max(r["props_raw"]["V_weak"]["value"], r["props_raw"]["V_rep"]["value"])
        dx, dy = annot_offsets[r["config"]]
        ax1.annotate(
            r["config"],
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            arrowprops={"arrowstyle": "-", "color": "#6b7280", "lw": 0.6},
        )

    ax1.axvline(pa["const_train_all5"]["rps"]["value"], color="#374151", linestyle="--", linewidth=1.2, label=f"CONST_train RPS ({pa['const_train_all5']['rps']['value']:.3f})")
    ax1.axvline(pa["loro_all5"]["smoothed"]["rps"]["value"], color="#059669", linestyle=":", linewidth=1.4, label=f"LORO smoothed RPS ({pa['loro_all5']['smoothed']['rps']['value']:.3f})")
    ax1.axhline(0.20, color="#b91c1c", linestyle="-.", linewidth=1.0, label="H1 violation threshold (0.20)")
    ax1.set_xlabel("CE-calibrated held-out RPS (all5, lower is better)")
    ax1.set_ylabel("Max non-bug violation: max(V_weak, V_rep)")
    ax1.set_title("Proper-Score Qualification vs. Axiomatic Violations")
    ax1.set_ylim(-0.06, 1.06)
    ax1.grid(True)
    ax1.legend(loc="upper left", fontsize=7.0, framealpha=0.92)

    top_show = sorted([r for r in real_elig if r["beats_const_rps"]], key=lambda r: r["proper"]["ce_cal_all5"]["rps"]["value"])[:10]
    y_pos = np.arange(len(top_show))
    h = 0.27
    ax2.barh(y_pos - h, [r["cal_axioms"]["V_weak_cal"]["value"] for r in top_show], height=h, color="#2563eb", label="Calibrated V_weak_cal")
    ax2.barh(y_pos, [r["props_raw"]["V_rep"]["value"] for r in top_show], height=h, color="#f59e0b", label="Raw V_rep")
    ax2.barh(y_pos + h, [r["rep_heldout"]["cal_tvd"] for r in top_show], height=h, color="#dc2626", label="Calibrated repeat TVD")
    ax2.axvline(0.20, color="#b91c1c", linestyle="-.", linewidth=1.0)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([r["config"] for r in top_show], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Violation rate / calibrated repeat TVD (lower is better)")
    ax2.set_title("Top-10 Proper-Score Configs Still Fail Calibrated Axioms")
    ax2.set_xlim(0.0, 0.85)
    ax2.grid(True)
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.92)

    fig.suptitle("Experiment 07 (11-26-2026): Proper Scores (RPS, CE) & TVD Distance Do Not Guarantee Axiomatic Trust", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fig2(path: Path, records: list[dict[str, Any]], part_b_rows: list[dict[str, Any]]) -> None:
    te_real = [r for r in records if r["kind"] == "llm" and r["split"] == "test" and r["eligible"]]
    ctls = [r for r in records if r["kind"] == "control" and r["eligible"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    ax1.scatter(
        [r["ati_equal"] for r in te_real],
        [r["proper"]["ce_cal_all5"]["rps"]["value"] for r in te_real],
        c="#2563eb", marker="o", s=65, label="Canonical test split (n=18; 16 unseen-weight)", edgecolors="black", linewidths=0.4,
    )
    ax1.scatter(
        [r["e39_indices"]["C_ms_gold"] for r in ctls if r["e39_indices"].get("C_ms_gold") is not None],
        [r["proper"]["ce_cal_all5"]["rps"]["value"] for r in ctls if r["e39_indices"].get("C_ms_gold") is not None],
        c="#f97316", marker="X", s=90, label="Controls scored by C_ms_gold (fooled=1.0)", edgecolors="black", linewidths=0.4,
    )
    ax1.scatter(
        [r["ati_equal"] for r in ctls],
        [r["proper"]["ce_cal_all5"]["rps"]["value"] for r in ctls],
        c="#dc2626", marker="^", s=85, label="Controls scored by ATI_equal (4/4 full Q, 3/4 train+val Q)", edgecolors="black", linewidths=0.4,
    )
    for r in ctls:
        dx = -28 if "random" in r["config"] else 5
        ax1.annotate(r["config"].replace("ctl|", ""), (r["ati_equal"], r["proper"]["ce_cal_all5"]["rps"]["value"]), xytext=(dx, 4), textcoords="offset points", fontsize=7.5, color="#991b1b")
    top_te_offsets = [(-74, -2), (-78, 9), (-74, -15)]
    for idx_t, r in enumerate(sorted(te_real, key=lambda z: z["proper"]["ce_cal_all5"]["rps"]["value"])[:3]):
        dx, dy = top_te_offsets[idx_t]
        ax1.annotate(
            r["config"],
            (r["ati_equal"], r["proper"]["ce_cal_all5"]["rps"]["value"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            color="#1e3a8a",
            arrowprops={"arrowstyle": "-", "color": "#64748b", "lw": 0.6},
        )
    ax1.set_xlabel("Label-free property index value (higher = more axioms passed)")
    ax1.set_ylabel("CE-calibrated held-out RPS (all5, lower is better)")
    ax1.set_title("Held-Out Test Judges & Deterministic Controls")
    ax1.set_xlim(-0.05, 1.08)
    ax1.set_ylim(0.122, 0.192)
    ax1.grid(True)
    ax1.legend(loc="upper right", fontsize=7.5, framealpha=0.92)

    order_idx = ["ATI_equal", "ATI_cal_rps", "ATI_min", "C_ms_gold", "1_minus_V_weak", "1_minus_V_rep", "TI_var_trainnorm"]
    oo_map = {r["index"]: r for r in part_b_rows if r["split"] == "lomfo_oof_all"}
    te_map = {r["index"]: r for r in part_b_rows if r["split"] == "test"}
    x = np.arange(len(order_idx))
    w = 0.36
    oo_vals = [oo_map[k]["rps_spearman"] for k in order_idx]
    te_vals = [te_map[k]["rps_spearman"] for k in order_idx]
    ax2.bar(x - w / 2, oo_vals, width=w, color="#2563eb", label="6-fold LOMFO-CV OOF (n=56, zero weight overlap)")
    ax2.bar(x + w / 2, te_vals, width=w, color="#10b981", label="Canonical test split (n=18; 16 unseen-weight)")
    ax2.axhline(0.0, color="black", linewidth=0.9)
    ax2.axhline(0.50, color="#047857", linestyle="--", linewidth=1.0, label="PREREG target (0.50)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([k.replace("_trainnorm", "").replace("ATI_cal_rps", "ATI_cal_rps*") for k in order_idx], rotation=24, ha="right", fontsize=8.5)
    ax2.set_ylabel("Spearman with -CE_fit RPS")
    ax2.set_title("Property-Index Generalization & Control Rejection")
    ax2.set_ylim(-0.45, 1.15)
    ax2.grid(True)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.92)

    fig.suptitle("Experiment 07 (11-26-2026): Idea 3 Property-Index Generalization & Control Rejection", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fig3(path: Path, pc: dict[str, Any]) -> None:
    old_d = pc["datasets"]["old_vb"]
    new_d = pc["datasets"]["new_vb"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2))
    cats = np.arange(6)
    old_h = np.array(old_d["rating_histogram_0_to_5"], dtype=float) / old_d["n_ratings_total"]
    new_h = np.array(new_d["rating_histogram_0_to_5"], dtype=float) / new_d["n_ratings_total"]
    w = 0.36
    ax1.bar(cats - w / 2, old_h, width=w, color="#6366f1", label=f"Old VB (n={old_d['n_ratings_total']} ratings / 75 items)")
    ax1.bar(cats + w / 2, new_h, width=w, color="#ec4899", label=f"New VB (n={new_d['n_ratings_total']} ratings / 100 items)")
    ax1.set_xticks(cats)
    ax1.set_xlabel("Human score (0-5)")
    ax1.set_ylabel("Share of ratings")
    ax1.set_title("Rating Distribution Shift & 5/5 Ceiling Saturation")
    ax1.set_ylim(0.0, 0.85)
    ax1.grid(True)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    labels = ["All items:\nrange >= 1", "All items:\nrange >= 2", "Non-ceiling\n(min<5): range>=1", "Non-ceiling\n(min<5): range>=2", "Ceiling share\nP(r == 5)"]
    old_blks = [old_d["range_ge_1"], old_d["range_ge_2"], old_d["non_ceiling_range_ge_1"], old_d["non_ceiling_range_ge_2"], old_d["rating_weighted_p_eq_5"]]
    new_blks = [new_d["range_ge_1"], new_d["range_ge_2"], new_d["non_ceiling_range_ge_1"], new_d["non_ceiling_range_ge_2"], new_d["rating_weighted_p_eq_5"]]
    x = np.arange(len(labels))
    old_v = [b["value"] for b in old_blks]
    old_err = [[b["value"] - b["ci_lo"] for b in old_blks], [b["ci_hi"] - b["value"] for b in old_blks]]
    new_v = [b["value"] for b in new_blks]
    new_err = [[b["value"] - b["ci_lo"] for b in new_blks], [b["ci_hi"] - b["value"] for b in new_blks]]
    ax2.bar(x - w / 2, old_v, width=w, yerr=old_err, capsize=3, color="#6366f1", label="Old VB (25 tasks)")
    ax2.bar(x + w / 2, new_v, width=w, yerr=new_err, capsize=3, color="#ec4899", label="New VB (25 tasks)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Share [95% task-cluster CI]")
    ax2.set_title("Ceiling Compression vs. Non-Ceiling Human Disagreement")
    ax2.set_ylim(0.0, 1.08)
    ax2.grid(True)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    fig.suptitle("Experiment 07 (11-26-2026): H2 Cross-Regime Human Consistency (Old vs. New VeriBench)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_outputs(summary: dict[str, Any], records: list[dict[str, Any]], part_b_rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)

    # Part A CSV (60 units)
    a_fields = [
        "config", "kind", "harness", "family", "base_model_family", "split", "model_declared", "model_served",
        "rho_heldout_exp3", "eligible", "n_heldout_items", "beats_const_rps", "beats_const_all3", "beats_loro_rps_ce",
        "matched_loro_rps", "matched_loro_ce",
        "raw_all5_rps", "ce_cal_all5_rps", "d_rps_vs_const", "d_rps_ci_lo", "d_rps_ci_hi", "d_rps_p",
        "ce_cal_all5_ce", "d_ce_p", "ce_cal_all5_tvd", "d_tvd_p", "ce_cal_all5_icc_a1", "ce_cal_all5_spearman",
        "V_ms", "V_weak", "V_weak_cal", "V_vac", "V_overc", "V_overc_cal", "V_fmt", "V_fmt_cal", "fmt_cal_tvd",
        "V_bug", "V_rep", "rep_raw_range", "V_rep_heldout", "rep_raw_range_heldout", "rep_cal_ev_range_05", "rep_cal_tvd",
        "ati_equal", "ati_min", "ati_cal_rps_tv", "gate_all_pass", "fails_prereg_h1", "prereg_h1_triggers", "fails_nonbug_h1", "nonbug_triggers",
    ]
    with (out_dir / "part_a_proper_scores_vs_properties.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=a_fields, lineterminator="\n")
        w.writeheader()
        for r in records:
            c5 = r["proper"].get("ce_cal_all5", {})
            ca = r["cal_axioms"]
            rh = r["rep_heldout"]
            w.writerow({
                "config": r["config"], "kind": r["kind"], "harness": r["harness"], "family": r["family"],
                "base_model_family": r["base_model_family"], "split": r["split"],
                "model_declared": r["model_declared"], "model_served": r["model_served"],
                "rho_heldout_exp3": r["rho_heldout_exp3"], "eligible": r["eligible"],
                "n_heldout_items": c5.get("n_heldout_items"),
                "beats_const_rps": r["beats_const_rps"], "beats_const_all3": r["beats_const_all3"],
                "beats_loro_rps_ce": r["beats_loro_rps_ce"],
                "matched_loro_rps": c5.get("matched_loro_rps"), "matched_loro_ce": c5.get("matched_loro_ce"),
                "raw_all5_rps": r["proper"].get("raw_all5_rps"),
                "ce_cal_all5_rps": c5.get("rps", {}).get("value"),
                "d_rps_vs_const": c5.get("d_rps_vs_const", {}).get("value"),
                "d_rps_ci_lo": c5.get("d_rps_vs_const", {}).get("ci_lo"),
                "d_rps_ci_hi": c5.get("d_rps_vs_const", {}).get("ci_hi"),
                "d_rps_p": c5.get("d_rps_vs_const", {}).get("p_val"),
                "ce_cal_all5_ce": c5.get("ce", {}).get("value"),
                "d_ce_p": c5.get("d_ce_vs_const", {}).get("p_val"),
                "ce_cal_all5_tvd": c5.get("tvd", {}).get("value"),
                "d_tvd_p": c5.get("d_tvd_vs_const", {}).get("p_val"),
                "ce_cal_all5_icc_a1": c5.get("icc_a1"),
                "ce_cal_all5_spearman": c5.get("spearman"),
                "V_ms": r["props_raw"]["V_ms"]["value"],
                "V_weak": r["props_raw"]["V_weak"]["value"],
                "V_weak_cal": ca.get("V_weak_cal", {}).get("value"),
                "V_vac": r["props_raw"]["V_vac"]["value"],
                "V_overc": r["props_raw"]["V_overc"]["value"],
                "V_overc_cal": ca.get("V_overc_cal", {}).get("value"),
                "V_fmt": r["props_raw"]["V_fmt"]["value"],
                "V_fmt_cal": ca.get("V_fmt_cal", {}).get("value"),
                "fmt_cal_tvd": ca.get("V_fmt_cal", {}).get("mean_tvd"),
                "V_bug": r["props_raw"]["V_bug"]["value"],
                "V_rep": r["props_raw"]["V_rep"]["value"],
                "rep_raw_range": r["props_raw"]["V_rep"]["raw_range"],
                "V_rep_heldout": rh.get("v_rep"),
                "rep_raw_range_heldout": rh.get("raw_range"),
                "rep_cal_ev_range_05": rh.get("cal_ev_range_05"),
                "rep_cal_tvd": rh.get("cal_tvd"),
                "ati_equal": r["ati_equal"], "ati_min": r["ati_min"],
                "ati_cal_rps_tv": r["ati_cal_rps_tv"], "gate_all_pass": r["gate_all_pass"],
                "fails_prereg_h1": r["fails_prereg_h1"], "prereg_h1_triggers": "|".join(r["prereg_h1_triggers"]),
                "fails_nonbug_h1": r["fails_nonbug_h1"], "nonbug_triggers": "|".join(r["nonbug_triggers"]),
            })

    # Part B CSV
    b_fields = [
        "split", "index", "n", "inference_mode",
        "rps_spearman", "rps_ci_lo", "rps_ci_hi", "rps_p", "rps_within_fam_p",
        "ce_spearman", "ce_ci_lo", "ce_ci_hi", "ce_p", "ce_within_fam_p",
        "tvd_spearman", "tvd_ci_lo", "tvd_ci_hi", "tvd_p", "tvd_within_fam_p",
        "icc_spearman", "icc_ci_lo", "icc_ci_hi", "icc_p", "icc_within_fam_p",
        "exp3_rho_spearman", "exp3_rho_ci_lo", "exp3_rho_ci_hi", "exp3_rho_p", "exp3_rho_within_fam_p",
        "selected_configs", "regret_rps", "regret_ce", "regret_tvd", "q_const_median_index", "ctl_rejected_of_4",
        "q_const_tv_median_index", "ctl_rejected_trainval_q_of_4",
    ]
    with (out_dir / "part_b_idea3_certification.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=b_fields, lineterminator="\n")
        w.writeheader()
        for row in part_b_rows:
            w.writerow({k: row[k] for k in b_fields})

    # Part C CSV
    c_fields = ["dataset", "metric", "value", "ci_lo", "ci_hi", "n", "n_tasks"]
    with (out_dir / "part_c_human_regimes.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=c_fields, lineterminator="\n")
        w.writeheader()
        for ds_name, ds in summary["part_c"]["datasets"].items():
            for mname in (
                "rating_weighted_p_eq_5", "rating_weighted_p_ge_4",
                "item_weighted_p_eq_5", "item_weighted_p_ge_4", "unanimous_range_0", "range_ge_1", "range_ge_2",
                "mean_range", "mean_sd", "pairwise_exact_agree", "irreducible_rps_floor",
                "non_ceiling_range_ge_1", "non_ceiling_range_ge_2", "non_ceiling_mean_range", "non_ceiling_pairwise_exact",
            ):
                blk = ds[mname]
                w.writerow({"dataset": ds_name, "metric": mname, "value": blk["value"], "ci_lo": blk["ci_lo"], "ci_hi": blk["ci_hi"], "n": blk["n"], "n_tasks": blk["n_tasks"]})
            for mname in ("rps", "ce", "tvd"):
                blk = ds["loro_smoothed_" + mname]
                w.writerow({"dataset": ds_name, "metric": "loro_smoothed_" + mname, "value": blk["value"], "ci_lo": blk["ci_lo"], "ci_hi": blk["ci_hi"], "n": blk["n"], "n_tasks": blk["n_tasks"]})

    lines = canonical_lines(summary)
    (out_dir / "canonical_lines.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_fig1(out_dir / "fig1_proper_scores_vs_axiomatic_violations.png", records, summary["part_a"])
    plot_fig2(out_dir / "fig2_idea3_out_of_family_certification.png", records, part_b_rows)
    plot_fig3(out_dir / "fig3_h2_human_ceiling_vs_disagreement.png", summary["part_c"])


def build_summary() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    items = load_items_verified()
    part_a, records, part_b_rows, extra = analyze_part_a_and_b(items)
    part_c = analyze_part_c_human_regimes()
    summary = {
        "date": "2026-11-26",
        "new_model_calls": 0,
        "cost_usd": 0.0,
        "alignment": extra["alignment"],
        "part_a": part_a,
        "part_b": extra["part_b"],
        "part_b_table": part_b_rows,
        "part_c": part_c,
        "records": records,
    }
    return summary, records, part_b_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 07 CPU analysis ($0.00, zero new model calls).")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    summary, records, part_b_rows = build_summary()
    write_outputs(summary, records, part_b_rows, Path(args.out))
    for line in canonical_lines(summary):
        print(line)


if __name__ == "__main__":
    main()
