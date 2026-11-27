#!/usr/bin/env python3
"""Experiment 08 (2026-11-27): task-held-out calibration of the experiment 05 scores.

No model calls. Human ratings are read from an off-repo items.jsonl and are not written back.
Each bootstrap interval uses its own NumPy Generator started at SEED, so one interval does not
depend on whether another interval is computed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

EXP_DIR = Path(__file__).resolve().parents[1]
REPO = EXP_DIR.parents[1]
DEFAULT_SCORES = REPO / "experiments/05_renaming_and_inserted_bug_on_veribench_judge_outputs/results"
DEFAULT_ITEMS = Path("/tmp/offrepo/veribench_e100_items.jsonl")
DEFAULT_OUT = EXP_DIR / "results"

ITEMS_SHA256 = "ad9ed552fc2ffac6328e6010bf6be82215dd74f1eddd791622d433126877a96c"
SEED = 20261127
DRAWS = 2000
EPS = 1e-9
TOL = 0.05
ALPHA = 0.5
CATEGORIES = np.arange(6, dtype=float)
CFGS = ("A", "A2", "B")

# Experiment 05 published point estimates (alignment gate).
PUB_HEAD = {
    "A": {"rps": 0.395, "tvd": 0.859, "ce": 3.53},
    "A2": {"rps": 0.390, "tvd": 0.897, "ce": 3.68},
    "B": {"rps": 0.282, "tvd": 0.863, "ce": 3.55},
    "constant": {"rps": 0.077, "tvd": 0.491, "ce": 1.73},
}
PUB_PROP = {
    "A": {"v_bug": 0.676, "tie": 0.649, "noise": 0.208, "rn_d": 0.083, "id1": 0.917},
    "A2": {"v_bug": 0.757, "tie": 0.649, "noise": 0.042, "rn_d": 0.042, "id1": 0.958},
    "B": {"v_bug": 0.568, "tie": 0.514, "noise": 0.042, "rn_d": 0.375, "id1": 0.996},
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rating_counts(ratings: Sequence[float]) -> np.ndarray:
    c = np.zeros(6, dtype=float)
    if len(ratings) == 0:
        raise ValueError("empty ratings")
    for y in ratings:
        k = int(round(float(y)))
        if k < 0 or k > 5:
            raise ValueError(f"rating out of range: {y}")
        c[k] += 1.0
    return c


def empirical(counts: np.ndarray) -> np.ndarray:
    s = float(counts.sum())
    if s <= 0:
        raise ValueError("empty count vector")
    return counts / s


def smooth_alpha(counts: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    return (counts + alpha) / float(counts.sum() + 6.0 * alpha)


def rps(q: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.cumsum(q)[:5] - np.cumsum(p)[:5]) ** 2))


def tvd(q: np.ndarray, p: np.ndarray) -> float:
    return float(0.5 * np.abs(q - p).sum())


def ce(q: np.ndarray, p: np.ndarray) -> float:
    return float(-(p * np.log(np.clip(q, 1e-12, None))).sum())


def point_mass(score: float) -> np.ndarray:
    p = np.zeros(6, dtype=float)
    r = int(min(5, max(0, np.floor(5.0 * float(score) + 0.5))))
    p[r] = 1.0
    return p


def point_smooth(score: float) -> np.ndarray:
    p = np.full(6, 0.1 / 6.0)
    r = int(min(5, max(0, np.floor(5.0 * float(score) + 0.5))))
    p[r] += 0.9
    return p


def expected_unit(q: np.ndarray) -> float:
    return float(np.dot(CATEGORIES, q) / 5.0)


def load_items(path: Path) -> list[dict[str, Any]]:
    digest = sha256_file(path)
    if digest != ITEMS_SHA256:
        raise SystemExit(f"items.jsonl sha256 mismatch: {digest} != {ITEMS_SHA256}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        ratings = []
        for k in ("p1", "p2", "p3", "p4", "p5"):
            v = it.get(k)
            if v is None:
                continue
            fv = float(v)
            if fv == fv:
                ratings.append(fv)
        it["ratings"] = ratings
        it["counts"] = rating_counts(ratings)
        it["p_emp"] = empirical(it["counts"])
        it["probe_id"] = "CRIT__" + it["item"]
        rows.append(it)
    if len(rows) != 75:
        raise SystemExit(f"expected 75 items, got {len(rows)}")
    return rows


def load_scores(path: Path) -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["probe_id"] in seen:
                raise SystemExit(f"duplicate probe {row['probe_id']} in {path}")
            seen.add(row["probe_id"])
            row["score"] = float(row["score"])
            by[row["kind"]].append(row)
    return by


def _smooth_rows(p: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    p = np.clip(p, eps, None)
    return p / p.sum(axis=1, keepdims=True)


def fit_ce_ordinal(
    train_scores: Sequence[float],
    train_counts: Sequence[np.ndarray],
    l2: float = 1e-3,
) -> tuple[Callable[[Sequence[float]], np.ndarray], float]:
    s = np.asarray(train_scores, dtype=float)
    C = np.asarray(train_counts, dtype=float)
    if C.ndim != 2 or C.shape[1] != 6:
        raise ValueError("train_counts must be (n, 6)")

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
    if not np.all(np.isfinite(opt.x)):
        raise SystemExit("ordinal fit returned a non-finite parameter vector")
    th, b = unpack(opt.x)

    def predict(x: Sequence[float]) -> np.ndarray:
        xa = np.asarray(list(x), dtype=float)
        z = np.clip(th[None, :] - b * xa[:, None], -40.0, 40.0)
        cum = 1.0 / (1.0 + np.exp(-z))
        cum_full = np.concatenate([np.zeros((len(xa), 1)), cum, np.ones((len(xa), 1))], axis=1)
        return _smooth_rows(np.diff(cum_full, axis=1), eps=1e-3)

    return predict, b


def pav_blocks(score_counts: dict[float, np.ndarray]) -> list[dict[str, Any]]:
    groups = [{"scores": [s], "counts": np.asarray(c, dtype=float).copy()} for s, c in sorted(score_counts.items())]
    if not groups:
        raise ValueError("no training scores")

    def mean(c: np.ndarray) -> float:
        return float(np.dot(CATEGORIES, c) / c.sum())

    while True:
        merged = False
        for i in range(len(groups) - 1):
            if mean(groups[i]["counts"]) > mean(groups[i + 1]["counts"]) + 1e-12:
                groups[i]["scores"].extend(groups[i + 1]["scores"])
                groups[i]["counts"] = groups[i]["counts"] + groups[i + 1]["counts"]
                del groups[i + 1]
                merged = True
                break
        if not merged:
            return groups


def iso_lookup(groups: list[dict[str, Any]], score: float) -> np.ndarray:
    for g in groups:
        if any(abs(score - t) <= 1e-12 for t in g["scores"]):
            return smooth_alpha(g["counts"])
    best_g = None
    best_key = None
    for g in groups:
        for t in g["scores"]:
            key = (abs(score - t), t)
            if best_key is None or key < best_key:
                best_key = key
                best_g = g
    return smooth_alpha(best_g["counts"])


def cluster_mean_ci(pairs: Sequence[tuple[str, float]], draws: int = DRAWS, seed: int = SEED) -> dict[str, Any]:
    if not pairs:
        return {"value": None, "ci_lo": None, "ci_hi": None, "n": 0, "n_tasks": 0, "finite_draws": 0}
    groups: dict[str, list[float]] = {}
    for task, val in pairs:
        fv = float(val)
        if not np.isfinite(fv):
            raise ValueError(f"non-finite loss for task {task}")
        groups.setdefault(str(task), []).append(fv)
    keys = sorted(groups)
    arrs = [np.asarray(groups[k], dtype=float) for k in keys]
    point = float(sum(v for _, v in pairs) / len(pairs))
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=float)
    n_keys = len(keys)
    for i in range(draws):
        picks = rng.integers(0, n_keys, size=n_keys)
        boot[i] = float(np.concatenate([arrs[int(k)] for k in picks]).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "value": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": len(pairs),
        "n_tasks": n_keys,
        "finite_draws": draws,
    }


def paired_task_signflip_p(
    task_ids: Sequence[str],
    vals_a: Sequence[float],
    vals_b: Sequence[float],
    draws: int = DRAWS,
    seed: int = SEED,
) -> dict[str, Any]:
    a = np.asarray(vals_a, dtype=float)
    b = np.asarray(vals_b, dtype=float)
    t_arr = np.asarray(task_ids)
    uniq = sorted(set(t_arr.tolist()))
    n_total = float(len(a))
    cluster_sums = np.array([float(np.sum(a[t_arr == t] - b[t_arr == t])) for t in uniq], dtype=float)
    obs = float(np.sum(cluster_sums) / n_total)
    g = len(uniq)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(draws, g))
    perm_means = (signs @ cluster_sums) / n_total
    p_val = float((1 + np.sum(np.abs(perm_means) >= abs(obs) - 1e-12)) / (draws + 1))
    return {"obs_diff": obs, "p_val": p_val, "n_clusters": g, "n_perm": draws}


def excess_ci(
    rn_pairs: Sequence[tuple[str, float]],
    noise_pairs: Sequence[tuple[str, float]],
    draws: int = DRAWS,
    seed: int = SEED,
) -> dict[str, Any]:
    rn: dict[str, list[float]] = defaultdict(list)
    ns: dict[str, list[float]] = defaultdict(list)
    for t, v in rn_pairs:
        rn[str(t)].append(float(v))
    for t, v in noise_pairs:
        ns[str(t)].append(float(v))
    keys = sorted(set(rn) | set(ns))
    point_rn = float(np.mean([v for _, v in rn_pairs]))
    point_ns = float(np.mean([v for _, v in noise_pairs]))
    rng = np.random.default_rng(seed)
    boot = []
    n_keys = len(keys)
    for _ in range(draws):
        picks = rng.integers(0, n_keys, size=n_keys)
        rn_vals = []
        ns_vals = []
        for k in picks:
            task = keys[int(k)]
            rn_vals.extend(rn.get(task, []))
            ns_vals.extend(ns.get(task, []))
        if not rn_vals or not ns_vals:
            continue
        boot.append(float(np.mean(rn_vals) - np.mean(ns_vals)))
    if len(boot) < 1500:
        return {"value": point_rn - point_ns, "ci_lo": None, "ci_hi": None, "finite_draws": len(boot)}
    lo, hi = np.percentile(np.asarray(boot), [2.5, 97.5])
    return {"value": point_rn - point_ns, "ci_lo": float(lo), "ci_hi": float(hi), "finite_draws": len(boot)}


def metric_block(loss_pairs: Sequence[tuple[str, float]]) -> dict[str, Any]:
    return cluster_mean_ci(loss_pairs)


def id1_map(rows: list[dict[str, Any]]) -> dict[str, float]:
    out = {}
    for r in rows:
        if r["task"] in out:
            raise SystemExit(f"two rows for task {r['task']}")
        out[r["task"]] = r["score"]
    return out


def property_block(
    scores: dict[str, list[dict[str, Any]]],
    scalar: Callable[[str, float], float],
) -> dict[str, Any]:
    id1 = id1_map(scores["ID1"])
    id2 = id1_map(scores["ID2"])
    both = sorted(set(id1) & set(id2))
    noise_pairs = []
    for t in both:
        a = scalar(t, id1[t])
        b = scalar(t, id2[t])
        noise_pairs.append((t, 1.0 if abs(a - b) > TOL + 1e-9 else 0.0))
    noise = cluster_mean_ci(noise_pairs)

    def kind_share(kind: str) -> dict[str, Any]:
        rows = [r for r in scores[kind] if r["task"] in id1]
        pairs_05 = []
        pairs_exact = []
        abs_delta = []
        for r in rows:
            base = scalar(r["task"], id1[r["task"]])
            val = scalar(r["task"], r["score"])
            d = abs(val - base)
            pairs_05.append((r["task"], 1.0 if d > TOL + 1e-9 else 0.0))
            pairs_exact.append((r["task"], 1.0 if d > EPS else 0.0))
            abs_delta.append(d)
        share = cluster_mean_ci(pairs_05)
        exact = cluster_mean_ci(pairs_exact)
        return {
            "n": len(rows),
            "share_05": share,
            "share_exact": exact,
            "mean_abs": float(np.mean(abs_delta)) if abs_delta else None,
            "pairs_05": pairs_05,
        }

    rn_d = kind_share("RN_d")
    rn_b = kind_share("RN_b")
    rn_rows = [r for k in ("RN_b", "RN_d") for r in scores[k] if r["task"] in id1]
    rn_pairs = []
    for r in rn_rows:
        base = scalar(r["task"], id1[r["task"]])
        val = scalar(r["task"], r["score"])
        rn_pairs.append((r["task"], 1.0 if abs(val - base) > TOL + 1e-9 else 0.0))
    rn = cluster_mean_ci(rn_pairs)

    bug_ge, bug_gt, bug_tie, bug_mat = [], [], [], []
    for r in scores["BUG"]:
        if r["task"] not in id1:
            continue
        base = scalar(r["task"], id1[r["task"]])
        val = scalar(r["task"], r["score"])
        bug_ge.append((r["task"], 1.0 if val >= base - EPS else 0.0))
        bug_gt.append((r["task"], 1.0 if val > base + EPS else 0.0))
        bug_tie.append((r["task"], 1.0 if abs(val - base) <= EPS else 0.0))
        bug_mat.append((r["task"], 1.0 if val + 1e-9 >= base - TOL else 0.0))
    v_bug = cluster_mean_ci(bug_ge)
    v_strict = cluster_mean_ci(bug_gt)
    v_tie = cluster_mean_ci(bug_tie)
    v_mat = cluster_mean_ci(bug_mat)
    ex = excess_ci(rn_d["pairs_05"], noise_pairs)
    return {
        "ID1_mean": float(np.mean([scalar(t, s) for t, s in id1.items()])),
        "n_id1": len(id1),
        "noise_share_05": noise,
        "RN_d": {k: v for k, v in rn_d.items() if k != "pairs_05"},
        "RN_b": {k: v for k, v in rn_b.items() if k != "pairs_05"},
        "RN": rn,
        "V_bug_exact": v_bug,
        "V_bug_strict": v_strict,
        "tie_exact": v_tie,
        "V_bug_material": v_mat,
        "rn_d_excess": ex,
        "rn_d_pairs_05": rn_d["pairs_05"],
        "bug_pairs_ge": bug_ge,
        "bug_mat_pairs": bug_mat,
    }


def n_flips(a: Sequence[tuple[str, float]], b: Sequence[tuple[str, float]]) -> int:
    if len(a) != len(b):
        raise SystemExit("flip lists differ in length")
    return int(sum(x != y for (_, x), (_, y) in zip(a, b)))


def monotone_order_violations(raw_pairs: Sequence[tuple[float, float]], ev_pairs: Sequence[tuple[float, float]], slope_positive: bool) -> int:
    """Count pairs whose exact order reverses. Same inputs must stay ties."""
    n = 0
    for (ra, rb), (ea, eb) in zip(raw_pairs, ev_pairs):
        if abs(ra - rb) <= EPS:
            if abs(ea - eb) > 1e-8:
                n += 1
        elif slope_positive and (ra - rb) * (ea - eb) < -1e-8:
            n += 1
    return n


def insample_alignment(items_by_id: dict[str, dict[str, Any]], by_cfg: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    crit_ids = [r["probe_id"] for r in by_cfg["A"]["CRIT"]]
    if set(crit_ids) != set(items_by_id):
        raise SystemExit("CRIT probe ids do not match items.jsonl")
    dists = [items_by_id[i]["p_emp"] for i in crit_ids]
    pooled = np.mean(np.stack(dists, axis=0), axis=0)
    report = {}
    ok = True
    for cfg in CFGS:
        sc = {r["probe_id"]: r["score"] for r in by_cfg[cfg]["CRIT"]}
        tasks = [next(r["task"] for r in by_cfg[cfg]["CRIT"] if r["probe_id"] == i) for i in crit_ids]
        raw = [point_mass(sc[i]) for i in crit_ids]
        sm = [point_smooth(sc[i]) for i in crit_ids]
        ps = [items_by_id[i]["p_emp"] for i in crit_ids]
        head = {
            "rps": float(np.mean([rps(q, p) for q, p in zip(raw, ps)])),
            "tvd": float(np.mean([tvd(q, p) for q, p in zip(raw, ps)])),
            "ce": float(np.mean([ce(q, p) for q, p in zip(sm, ps)])),
        }
        const = {
            "rps": float(np.mean([rps(pooled, p) for p in ps])),
            "tvd": float(np.mean([tvd(pooled, p) for p in ps])),
            "ce": float(np.mean([ce(pooled, p) for p in ps])),
        }
        for name, got, pub, tol in (
            ("rps", head["rps"], PUB_HEAD[cfg]["rps"], 0.0005),
            ("tvd", head["tvd"], PUB_HEAD[cfg]["tvd"], 0.0005),
            ("ce", head["ce"], PUB_HEAD[cfg]["ce"], 0.005),
        ):
            if abs(got - pub) > tol:
                ok = False
        for name, got, pub, tol in (
            ("rps", const["rps"], PUB_HEAD["constant"]["rps"], 0.0005),
            ("tvd", const["tvd"], PUB_HEAD["constant"]["tvd"], 0.0005),
            ("ce", const["ce"], PUB_HEAD["constant"]["ce"], 0.005),
        ):
            if abs(got - pub) > tol:
                ok = False
        ident = lambda _t, s: s
        prop = property_block(by_cfg[cfg], ident)
        checks = {
            "v_bug": prop["V_bug_exact"]["value"],
            "tie": prop["tie_exact"]["value"],
            "noise": prop["noise_share_05"]["value"],
            "rn_d": prop["RN_d"]["share_05"]["value"],
            "id1": prop["ID1_mean"],
        }
        for k, pub in PUB_PROP[cfg].items():
            if abs(checks[k] - pub) > 0.001:
                ok = False
        report[cfg] = {"headline": head, "constant": const, "properties": checks, "n_tasks_checked": len(set(tasks))}
    report["pass"] = ok
    return report


def _public_prop(prop: dict[str, Any]) -> dict[str, Any]:
    drop = {"rn_d_pairs_05", "bug_pairs_ge", "bug_mat_pairs"}
    return {k: v for k, v in prop.items() if k not in drop}


def fit_task_maps(items: list[dict[str, Any]], crit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    score_of = {}
    for r in crit_rows:
        score_of[r["probe_id"]] = r["score"]
        by_task[r["task"]].append(r)
    tasks = sorted(by_task)
    if len(tasks) != 25 or any(len(by_task[t]) != 3 for t in tasks):
        raise SystemExit(f"expected 25 tasks of 3 CRIT items, got {[len(by_task[t]) for t in tasks]}")
    item_of = {it["probe_id"]: it for it in items}
    maps = {}
    slopes = {}
    iso_groups = {}
    consts = {}
    for t in tasks:
        train = [r for u in tasks if u != t for r in by_task[u]]
        scores = [score_of[r["probe_id"]] for r in train]
        counts = [item_of[r["probe_id"]]["counts"] for r in train]
        predict, b = fit_ce_ordinal(scores, counts)
        maps[t] = predict
        slopes[t] = b
        pooled: dict[float, np.ndarray] = {}
        total = np.zeros(6, dtype=float)
        for s, c in zip(scores, counts):
            pooled[s] = pooled.get(s, np.zeros(6, dtype=float)) + c
            total = total + c
        iso_groups[t] = pav_blocks(pooled)
        consts[t] = smooth_alpha(total)
    return {"tasks": tasks, "by_task": by_task, "maps": maps, "slopes": slopes, "iso": iso_groups, "const": consts, "item_of": item_of, "score_of": score_of}


def run(items_path: Path, scores_dir: Path, out_dir: Path) -> dict[str, Any]:
    items = load_items(items_path)
    items_by_id = {it["probe_id"]: it for it in items}
    by_cfg = {cfg: load_scores(scores_dir / f"scores_{cfg}.csv") for cfg in CFGS}
    align = insample_alignment(items_by_id, by_cfg)
    if not align["pass"]:
        raise SystemExit("alignment gate failed: " + json.dumps(align))

    summary: dict[str, Any] = {"alignment": align, "configs": {}}
    for cfg in CFGS:
        fitted = fit_task_maps(items, by_cfg[cfg]["CRIT"])
        tasks_l, rps_c, tvd_c, ce_c = [], [], [], []
        rps_raw, tvd_raw, ce_raw = [], [], []
        rps_iso, tvd_iso, ce_iso = [], [], []
        rps_k, tvd_k, ce_k = [], [], []
        held_scores = []
        for t in fitted["tasks"]:
            const_q = fitted["const"][t]
            pred = fitted["maps"][t]
            groups = fitted["iso"][t]
            rows = fitted["by_task"][t]
            ss = [fitted["score_of"][r["probe_id"]] for r in rows]
            cal_q = pred(ss)
            for i, r in enumerate(rows):
                p = fitted["item_of"][r["probe_id"]]["p_emp"]
                s = ss[i]
                raw_q = point_mass(s)
                iso_q = iso_lookup(groups, s)
                tasks_l.append(t)
                held_scores.append(s)
                rps_c.append(rps(cal_q[i], p))
                tvd_c.append(tvd(cal_q[i], p))
                ce_c.append(ce(cal_q[i], p))
                rps_raw.append(rps(raw_q, p))
                tvd_raw.append(tvd(raw_q, p))
                ce_raw.append(ce(point_smooth(s), p))
                rps_iso.append(rps(iso_q, p))
                tvd_iso.append(tvd(iso_q, p))
                ce_iso.append(ce(iso_q, p))
                rps_k.append(rps(const_q, p))
                tvd_k.append(tvd(const_q, p))
                ce_k.append(ce(const_q, p))

        def pack(name: str, losses: list[float], base: list[float] | None) -> dict[str, Any]:
            pairs = list(zip(tasks_l, losses))
            block = cluster_mean_ci(pairs)
            out = {"level": block}
            if base is not None:
                diff = [a - b for a, b in zip(losses, base)]
                out["minus_const"] = cluster_mean_ci(list(zip(tasks_l, diff)))
                out["beats_const"] = bool(out["minus_const"]["ci_hi"] is not None and out["minus_const"]["ci_hi"] < 0)
            return out

        headline = {
            "CAL": {
                "rps": pack("rps", rps_c, rps_k),
                "tvd": pack("tvd", tvd_c, tvd_k),
                "ce": pack("ce", ce_c, ce_k),
            },
            "RAW": {
                "rps": pack("rps", rps_raw, rps_k),
                "tvd": pack("tvd", tvd_raw, tvd_k),
                "ce": pack("ce", ce_raw, ce_k),
            },
            "ISO": {
                "rps": pack("rps", rps_iso, rps_k),
                "tvd": pack("tvd", tvd_iso, tvd_k),
                "ce": pack("ce", ce_iso, ce_k),
            },
            "CONST": {
                "rps": pack("rps", rps_k, None),
                "tvd": pack("tvd", tvd_k, None),
                "ce": pack("ce", ce_k, None),
            },
        }
        headline["CAL"]["rps"]["signflip_p"] = paired_task_signflip_p(tasks_l, rps_c, rps_k)
        slopes = fitted["slopes"]
        min_b = float(min(slopes.values()))
        n_pos = int(sum(b > 0 for b in slopes.values()))

        def scalar_raw(_t: str, s: float) -> float:
            return float(s)

        def scalar_cal(t: str, s: float) -> float:
            return expected_unit(fitted["maps"][t]([s])[0])

        def scalar_iso(t: str, s: float) -> float:
            return expected_unit(iso_lookup(fitted["iso"][t], s))

        raw_prop = property_block(by_cfg[cfg], scalar_raw)
        cal_prop = property_block(by_cfg[cfg], scalar_cal)
        iso_prop = property_block(by_cfg[cfg], scalar_iso)
        valid = raw_prop["ID1_mean"] >= 0.6
        # Order check on bug and RN_d pairs, using the same scalar.
        order_viol = 0
        if n_pos == len(slopes):
            id1 = id1_map(by_cfg[cfg]["ID1"])
            for kind in ("BUG", "RN_d", "RN_b", "ID2"):
                for r in by_cfg[cfg][kind]:
                    if r["task"] not in id1:
                        continue
                    ra, rb = r["score"], id1[r["task"]]
                    ea, eb = scalar_cal(r["task"], ra), scalar_cal(r["task"], rb)
                    order_viol += monotone_order_violations([(ra, rb)], [(ea, eb)], True)

        qual = bool(valid and headline["CAL"]["rps"]["beats_const"])
        summary["configs"][cfg] = {
            "valid": valid,
            "qualifies_rps": qual,
            "qualifies_all3": bool(
                qual and headline["CAL"]["tvd"]["beats_const"] and headline["CAL"]["ce"]["beats_const"]
            ),
            "n_slopes_positive": n_pos,
            "n_tasks": len(slopes),
            "min_slope": min_b,
            "order_violations_when_checked": order_viol,
            "headline": headline,
            "properties": {
                "RAW": _public_prop(raw_prop),
                "CAL": _public_prop(cal_prop),
                "ISO": _public_prop(iso_prop),
            },
            "flips_05": {
                "rn_d_raw_vs_cal": n_flips(raw_prop["rn_d_pairs_05"], cal_prop["rn_d_pairs_05"]),
                "rn_d_raw_vs_iso": n_flips(raw_prop["rn_d_pairs_05"], iso_prop["rn_d_pairs_05"]),
                "bug_material_raw_vs_cal": n_flips(raw_prop["bug_mat_pairs"], cal_prop["bug_mat_pairs"]),
                "bug_material_raw_vs_iso": n_flips(raw_prop["bug_mat_pairs"], iso_prop["bug_mat_pairs"]),
                "bug_exact_raw_vs_cal": n_flips(raw_prop["bug_pairs_ge"], cal_prop["bug_pairs_ge"]),
                "bug_exact_raw_vs_iso": n_flips(raw_prop["bug_pairs_ge"], iso_prop["bug_pairs_ge"]),
            },
        }
    summary["verdict_h1_cal"] = verdict_h1(summary)
    summary["machine_lines"] = machine_lines(summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(out_dir, summary)
    return summary


def _fails_property(block: dict[str, Any], value_key: str) -> bool:
    node = block[value_key]
    return bool(node["value"] is not None and node["value"] >= 0.20 and node["ci_lo"] is not None and node["ci_lo"] > 0.05)


def verdict_h1(summary: dict[str, Any]) -> str:
    qual = [c for c in summary["configs"].values() if c["qualifies_rps"]]
    if not qual:
        return "untestable"

    def supported(c: dict[str, Any]) -> bool:
        cal = c["properties"]["CAL"]
        bug = _fails_property(cal, "V_bug_exact")
        ex = cal["rn_d_excess"]
        rename = bool(ex["value"] is not None and ex["value"] >= 0.20 and ex["ci_lo"] is not None and ex["ci_lo"] > 0.05)
        return bug or rename

    def falsified_one(c: dict[str, Any]) -> bool:
        cal = c["properties"]["CAL"]
        return bool(cal["V_bug_exact"]["value"] < 0.10 and cal["RN_d"]["share_05"]["value"] < 0.10)

    if any(supported(c) for c in qual):
        return "supported"
    if all(falsified_one(c) for c in qual):
        return "falsified"
    return "mixed"


def _fmt(block: dict[str, Any], digits: int = 3) -> str:
    if block["value"] is None or block["ci_lo"] is None:
        return "na"
    return f"{block['value']:.{digits}f} [{block['ci_lo']:.{digits}f}, {block['ci_hi']:.{digits}f}]"


def machine_lines(summary: dict[str, Any]) -> list[str]:
    lines = [f"VERDICT_H1_CAL {summary['verdict_h1_cal']}"]
    for cfg, c in summary["configs"].items():
        h = c["headline"]
        lines.append(
            "CFG {cfg} valid={valid} qualifies_rps={qr} qualifies_all3={qa} "
            "cal_rps={rps} cal_rps_diff={diff} raw_rps_diff={raw} const_rps={const} "
            "cal_tvd_diff={tvd} cal_ce_diff={ce} min_b={b:.4f} n_b_pos={npos}/{nt} "
            "order_viol={ov} Vbug_raw={vbr} Vbug_cal={vbc} RNd_raw={rnr} RNd_cal={rnc} "
            "flips_rn_cal={fr} flips_bugmat_cal={fb}".format(
                cfg=cfg,
                valid=int(c["valid"]),
                qr=int(c["qualifies_rps"]),
                qa=int(c["qualifies_all3"]),
                rps=_fmt(h["CAL"]["rps"]["level"]),
                diff=_fmt(h["CAL"]["rps"]["minus_const"]),
                raw=_fmt(h["RAW"]["rps"]["minus_const"]),
                const=_fmt(h["CONST"]["rps"]["level"]),
                tvd=_fmt(h["CAL"]["tvd"]["minus_const"]),
                ce=_fmt(h["CAL"]["ce"]["minus_const"]),
                b=c["min_slope"],
                npos=c["n_slopes_positive"],
                nt=c["n_tasks"],
                ov=c["order_violations_when_checked"],
                vbr=_fmt(c["properties"]["RAW"]["V_bug_exact"]),
                vbc=_fmt(c["properties"]["CAL"]["V_bug_exact"]),
                rnr=_fmt(c["properties"]["RAW"]["RN_d"]["share_05"]),
                rnc=_fmt(c["properties"]["CAL"]["RN_d"]["share_05"]),
                fr=c["flips_05"]["rn_d_raw_vs_cal"],
                fb=c["flips_05"]["bug_material_raw_vs_cal"],
            )
        )
    lines.append("ALIGN pass={}".format(int(summary["alignment"]["pass"])))
    return lines


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_outputs(out_dir: Path, summary: dict[str, Any]) -> None:
    payload = _jsonable(summary)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out_dir / "canonical_lines.txt").write_text("\n".join(payload["machine_lines"]) + "\n", encoding="utf-8")
    with (out_dir / "headline.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "forecast", "metric", "value", "ci_lo", "ci_hi", "minus_const", "minus_const_lo", "minus_const_hi", "beats_const"])
        for cfg, c in payload["configs"].items():
            for forecast, metrics in c["headline"].items():
                for metric, block in metrics.items():
                    if metric == "signflip_p" or not isinstance(block, dict) or "level" not in block:
                        continue
                    diff = block.get("minus_const") or {}
                    w.writerow([
                        cfg, forecast, metric,
                        block["level"]["value"], block["level"]["ci_lo"], block["level"]["ci_hi"],
                        diff.get("value"), diff.get("ci_lo"), diff.get("ci_hi"),
                        block.get("beats_const"),
                    ])
    with (out_dir / "properties.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "config", "scale", "V_bug_exact", "V_bug_exact_lo", "V_bug_exact_hi",
            "tie_exact", "V_bug_material", "RN_d_share_05", "RN_d_share_05_lo", "RN_d_share_05_hi",
            "noise_share_05", "rn_d_excess", "rn_d_excess_lo", "rn_d_excess_hi", "ID1_mean",
        ])
        for cfg, c in payload["configs"].items():
            for scale, prop in c["properties"].items():
                w.writerow([
                    cfg, scale,
                    prop["V_bug_exact"]["value"], prop["V_bug_exact"]["ci_lo"], prop["V_bug_exact"]["ci_hi"],
                    prop["tie_exact"]["value"], prop["V_bug_material"]["value"],
                    prop["RN_d"]["share_05"]["value"], prop["RN_d"]["share_05"]["ci_lo"], prop["RN_d"]["share_05"]["ci_hi"],
                    prop["noise_share_05"]["value"],
                    prop["rn_d_excess"]["value"], prop["rn_d_excess"]["ci_lo"], prop["rn_d_excess"]["ci_hi"],
                    prop["ID1_mean"],
                ])
    _figures(out_dir, payload)


def _err(ax, x, block, **kw):
    v = block["value"]
    ax.errorbar(x, v, yerr=[[v - block["ci_lo"]], [block["ci_hi"] - v]], fmt="o", capsize=4, **kw)


def _figures(out_dir: Path, summary: dict[str, Any]) -> None:
    cfgs = list(summary["configs"])
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for j, metric in enumerate(("rps", "tvd", "ce")):
        for i, cfg in enumerate(cfgs):
            h = summary["configs"][cfg]["headline"]
            _err(ax[j], i - 0.15, h["RAW"][metric]["level"], color="C0", label="RAW" if i == 0 else None)
            _err(ax[j], i, h["CAL"][metric]["level"], color="C1", label="CAL" if i == 0 else None)
            _err(ax[j], i + 0.15, h["CONST"][metric]["level"], color="0.4", label="CONST train" if i == 0 else None)
        ax[j].set_xticks(range(len(cfgs)))
        ax[j].set_xticklabels(cfgs)
        ax[j].set_title(metric.upper() + " (lower is better)", fontsize=10)
        ax[j].grid(True)
    ax[0].legend(fontsize=8)
    fig.suptitle("Experiment 08 (as of 2026-11-27): task-held-out forecasts, 95% task-cluster intervals", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_calibrated_proper_scores.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, cfg in enumerate(cfgs):
        props = summary["configs"][cfg]["properties"]
        _err(ax[0], i - 0.15, props["RAW"]["V_bug_exact"], color="C0", label="RAW" if i == 0 else None)
        _err(ax[0], i, props["CAL"]["V_bug_exact"], color="C1", label="CAL" if i == 0 else None)
        _err(ax[0], i + 0.15, props["ISO"]["V_bug_exact"], color="C2", label="ISO" if i == 0 else None)
        _err(ax[1], i - 0.15, props["RAW"]["RN_d"]["share_05"], color="C0", label="RAW" if i == 0 else None)
        _err(ax[1], i, props["CAL"]["RN_d"]["share_05"], color="C1", label="CAL" if i == 0 else None)
        _err(ax[1], i + 0.15, props["ISO"]["RN_d"]["share_05"], color="C2", label="ISO" if i == 0 else None)
    for a, title in zip(ax, ("V_bug exact (bug >= original)", "RN_d share with |change| > 0.05")):
        a.set_xticks(range(len(cfgs)))
        a.set_xticklabels(cfgs)
        a.set_ylim(-0.05, 1.05)
        a.set_title(title, fontsize=10)
        a.grid(True)
    ax[0].legend(fontsize=8)
    fig.suptitle("Experiment 08 (as of 2026-11-27): properties before and after calibration", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_properties_before_after.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    summary = run(args.items, args.scores, args.out)
    print("\n".join(summary["machine_lines"]))


if __name__ == "__main__":
    main()
