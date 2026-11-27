#!/usr/bin/env python3
"""Experiment 09 (2026-11-27): selection gap between human correlation and formal properties.

No model calls. Input is the public experiment 03 judge_table.csv only.
Each interval uses its own NumPy Generator started at SEED.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

EXP_DIR = Path(__file__).resolve().parents[1]
REPO = EXP_DIR.parents[1]
DEFAULT_TABLE = REPO / "experiments/03_human_corr_vs_axiomatic_properties_cert_judge/results/judge_table.csv"
DEFAULT_OUT = EXP_DIR / "results"

SEED = 20261127
DRAWS = 2000
PERM = 5000
ATI_PROPS = ("V_ms", "V_weak", "V_vac", "V_overc", "V_fmt", "V_rep")
FAIL_PROPS = ("V_ms", "V_weak", "V_vac", "V_overc", "V_rep")
LAMBDAS = (0.01, 0.1, 1.0, 10.0)
PUB_SPEARMAN = 0.7883251594650563
BASE = {
    "gem31h": "gem31h",
    "qwen36": "qwen",
    "t2_qwen": "qwen",
    "sonnet5m": "sonnet",
    "t2_sonnet5": "sonnet",
    "opus55x": "opus55x",
    "luna6h": "luna",
    "t2_luna": "luna",
    "sol6h": "sol6h",
}


def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    real, controls = [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["kind"] == "control":
                controls.append(row)
                continue
            if row["kind"] != "llm":
                raise SystemExit(f"unexpected kind {row['kind']}")
            if row["rho_heldout"] == "" or float(row["rho_heldout_coverage"]) < 0.90:
                continue
            if row["family"] not in BASE:
                raise SystemExit(f"family {row['family']} has no base-weight block")
            row["rho"] = float(row["rho_heldout"])
            row["base"] = BASE[row["family"]]
            row["pass"] = np.array([1.0 - float(row[p]) for p in ATI_PROPS], dtype=float)
            row["ati"] = float(row["pass"].mean())
            row["ati_bug"] = float((6.0 * row["ati"] + (1.0 - float(row["V_bug"]))) / 7.0)
            row["fail_count"] = int(sum(
                float(row[p]) >= 0.20 and float(row[p + "_ci_lo"]) > 0.05 for p in FAIL_PROPS
            ))
            row["idea1_fail"] = int(
                (float(row["V_weak"]) >= 0.20 and float(row["V_weak_ci_lo"]) > 0.05)
                or (float(row["V_rep"]) >= 0.20 and float(row["V_rep_ci_lo"]) > 0.05)
            )
            real.append(row)
    if len(real) != 56:
        raise SystemExit(f"expected 56 eligible rows, got {len(real)}")
    blocks: dict[str, int] = {}
    for r in real:
        blocks[r["base"]] = blocks.get(r["base"], 0) + 1
    if set(blocks) != set(BASE.values()):
        raise SystemExit(f"base blocks are not a partition: {blocks}")
    return real, controls


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3 or len(set(np.round(xs, 12))) < 2 or len(set(np.round(ys, 12))) < 2:
        return None
    val = float(stats.spearmanr(xs, ys).statistic)
    return val if math.isfinite(val) else None


def pick(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    best = None
    best_key = None
    for r in rows:
        k = float(r[key])
        if best is None or k > best_key + 1e-12 or (abs(k - best_key) <= 1e-12 and r["config"] < best["config"]):
            best = r
            best_key = k
    if best is None:
        raise ValueError("empty family")
    return best


def _solve(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    mu_x = X.mean(axis=0)
    mu_y = float(y.mean())
    A = np.vstack([X - mu_x, math.sqrt(float(lam)) * np.eye(X.shape[1])])
    b = np.concatenate([y - mu_y, np.zeros(X.shape[1])])
    w, _ = optimize.nnls(A, b)
    return w


def fit_weights(train: Sequence[dict[str, Any]]) -> tuple[np.ndarray, float]:
    X = np.stack([r["pass"] for r in train])
    y = np.array([r["rho"] for r in train], dtype=float)
    fams = [r["base"] for r in train]
    uniq = sorted(set(fams))
    best_lam = float(LAMBDAS[0])
    best_cv = -math.inf
    if len(uniq) >= 3:
        fam_arr = np.asarray(fams)
        for lam in LAMBDAS:
            oof = np.empty(len(train), dtype=float)
            ok = True
            for fam in uniq:
                mask = fam_arr == fam
                if int((~mask).sum()) < 3:
                    ok = False
                    break
                w = _solve(X[~mask], y[~mask], lam)
                oof[mask] = X[mask] @ w
            if not ok:
                continue
            sp = spearman(oof, y)
            score = sp if sp is not None else -math.inf
            if score > best_cv + 1e-12:
                best_cv = score
                best_lam = float(lam)
    return _solve(X, y, best_lam), best_lam


def score_rows(rows: Sequence[dict[str, Any]], w: np.ndarray) -> None:
    span = []
    for r in rows:
        r["fitted"] = float(r["pass"] @ w)
        span.append(r["fitted"])
    flat = (max(span) - min(span)) < 1e-8
    for r in rows:
        r["fitted_flat"] = flat


def family_rows(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r["base"], []).append(r)
    return out


def census(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups = family_rows(rows)
    records = []
    for fam in sorted(groups):
        members = groups[fam]
        train = [r for r in rows if r["base"] != fam]
        w, lam = fit_weights(train)
        # Held-out labels must not enter w. Recompute after a label shuffle in tests.
        score_rows(members, w)
        h = pick(members, "rho")
        p = pick(members, "ati")
        flat = bool(members[0]["fitted_flat"])
        f = None if flat else pick(members, "fitted")
        records.append({
            "family": fam,
            "n": len(members),
            "lambda": lam,
            "H": h["config"],
            "P": p["config"],
            "F": None if f is None else f["config"],
            "flat": flat,
            "rho_H": h["rho"],
            "rho_P": p["rho"],
            "rho_F": None if f is None else f["rho"],
            "fail_H": h["fail_count"],
            "fail_P": p["fail_count"],
            "fail_F": None if f is None else f["fail_count"],
            "idea1_H": h["idea1_fail"],
            "idea1_P": p["idea1_fail"],
            "idea1_F": None if f is None else f["idea1_fail"],
            "disagree_HP": int(h["config"] != p["config"]),
            "disagree_HF": None if f is None else int(h["config"] != f["config"]),
            "rho_regret_P": h["rho"] - p["rho"],
            "rho_regret_F": None if f is None else h["rho"] - f["rho"],
            "fail_gap_P": h["fail_count"] - p["fail_count"],
            "fail_gap_F": None if f is None else h["fail_count"] - f["fail_count"],
            "weights": [float(x) for x in w],
        })
    return {"families": records}


def _mean(vals: Sequence[float]) -> float:
    return float(np.mean(vals))


def summarize_census(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    nonflat = [r for r in records if not r["flat"]]
    out = {
        "n_families": len(records),
        "n_flat": int(sum(r["flat"] for r in records)),
        "disagree_HP": int(sum(r["disagree_HP"] for r in records)),
        "disagree_HF": int(sum(r["disagree_HF"] for r in nonflat)),
        "n_fitted": len(nonflat),
        "mean_rho_regret_P": _mean([r["rho_regret_P"] for r in records]),
        "mean_rho_regret_F": _mean([r["rho_regret_F"] for r in nonflat]) if nonflat else None,
        "mean_fail_gap_P": _mean([r["fail_gap_P"] for r in records]),
        "mean_fail_gap_F": _mean([r["fail_gap_F"] for r in nonflat]) if nonflat else None,
        "n_H_idea1_fail": int(sum(r["idea1_H"] for r in records)),
        "n_P_idea1_fail": int(sum(r["idea1_P"] for r in records)),
        "n_F_idea1_fail": int(sum(r["idea1_F"] for r in nonflat)),
    }
    return out


def verdict_gap(disagree: int, fail_gap: float) -> str:
    if disagree >= 3 and fail_gap >= 0.5:
        return "supported"
    if disagree <= 1 or fail_gap <= 0:
        return "falsified"
    return "mixed"


def verdict_pred(point: float | None, lo: float | None, p_val: float | None, n_boot: int, n_perm: int) -> str:
    if point is None or lo is None or p_val is None or n_boot < 1500 or n_perm < 1500:
        return "not_computable"
    if point >= 0.40 and lo > 0 and p_val < 0.05:
        return "supported"
    if point <= 0.15 or (p_val >= 0.05 and point < 0.30):
        return "falsified"
    return "mixed"


def oof_scores(rows: Sequence[dict[str, Any]]) -> tuple[list[float], list[float]]:
    """Refit per family. Returns (oof score, rho) in row order."""
    groups = family_rows(rows)
    score = {r["config"]: None for r in rows}
    rho = {r["config"]: r["rho"] for r in rows}
    for fam, members in groups.items():
        train = [r for r in rows if r["base"] != fam]
        w, _ = fit_weights(train)
        for r in members:
            score[r["config"]] = float(r["pass"] @ w)
    order = [r["config"] for r in rows]
    return [score[c] for c in order], [rho[c] for c in order]


def spearman_ci(xs: Sequence[float], ys: Sequence[float], draws: int = DRAWS, seed: int = SEED) -> dict[str, Any]:
    obs = spearman(xs, ys)
    rng = np.random.default_rng(seed)
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    n = len(x)
    boot = []
    for _ in range(draws):
        idx = rng.integers(0, n, size=n)
        sp = spearman(x[idx], y[idx])
        if sp is not None:
            boot.append(sp)
    lo = hi = None
    if len(boot) >= 1500:
        lo, hi = [float(v) for v in np.percentile(boot, [2.5, 97.5])]
    return {"value": obs, "ci_lo": lo, "ci_hi": hi, "finite_draws": len(boot)}


def spearman_perm(xs: Sequence[float], ys: Sequence[float], draws: int = PERM, seed: int = SEED) -> dict[str, Any]:
    obs = spearman(xs, ys)
    rng = np.random.default_rng(seed)
    y = np.asarray(ys, dtype=float)
    x = np.asarray(xs, dtype=float)
    count = 0
    valid = 0
    for _ in range(draws):
        sp = spearman(x, rng.permutation(y))
        if sp is None:
            continue
        valid += 1
        if abs(sp) >= abs(obs) - 1e-12:
            count += 1
    p_val = None if obs is None or valid < 1500 else (1 + count) / (valid + 1)
    return {"p_val": p_val, "valid": valid, "obs": obs}


def family_cluster_spearman(rows: Sequence[dict[str, Any]], scores: Sequence[float], draws: int = DRAWS, seed: int = SEED) -> dict[str, Any]:
    fams = sorted({r["base"] for r in rows})
    by = {fam: [] for fam in fams}
    for r, s in zip(rows, scores):
        by[r["base"]].append((s, r["rho"]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        picks = rng.integers(0, len(fams), size=len(fams))
        xs, ys = [], []
        for k in picks:
            for s, rho in by[fams[int(k)]]:
                xs.append(s)
                ys.append(rho)
        sp = spearman(xs, ys)
        if sp is not None:
            boot.append(sp)
    lo = hi = None
    if boot:
        lo, hi = [float(v) for v in np.percentile(boot, [2.5, 97.5])]
    return {"ci_lo": lo, "ci_hi": hi, "finite_draws": len(boot), "note": "descriptive; n=6 families; not used in the verdict"}


def run(table: Path, out_dir: Path) -> dict[str, Any]:
    rows, controls = load_rows(table)
    ati = [r["ati"] for r in rows]
    rho = [r["rho"] for r in rows]
    sp = spearman(ati, rho)
    if sp is None or abs(sp - PUB_SPEARMAN) > 1e-9:
        raise SystemExit(f"alignment gate failed: ATI_equal Spearman {sp} != {PUB_SPEARMAN}")
    # Copies so the census fit does not depend on mutation order from a later OOF pass.
    cens = census([dict(r) for r in rows])
    stats_c = summarize_census(cens["families"])
    verdict = verdict_gap(stats_c["disagree_HP"], stats_c["mean_fail_gap_P"])
    if stats_c["n_fitted"] < 4 or stats_c["mean_fail_gap_F"] is None:
        verdict_f = "not_computable"
    else:
        verdict_f = verdict_gap(stats_c["disagree_HF"], stats_c["mean_fail_gap_F"])
    xs, ys = oof_scores(rows)
    pred_ci = spearman_ci(xs, ys)
    pred_p = spearman_perm(xs, ys)
    pred_verdict = verdict_pred(pred_ci["value"], pred_ci["ci_lo"], pred_p["p_val"], pred_ci["finite_draws"], pred_p["valid"])
    fam_ci = family_cluster_spearman(rows, xs)
    med = float(np.median(ati))
    ctrl = []
    for r in controls:
        passes = np.array([1.0 - float(r[p]) for p in ATI_PROPS], dtype=float)
        ctrl.append({"config": r["config"], "ati_equal": float(passes.mean()), "above_real_median": bool(float(passes.mean()) > med)})
    summary = {
        "alignment_spearman": sp,
        "census": stats_c,
        "families": [{k: v for k, v in rec.items() if k != "weights"} | {"weights": rec["weights"]} for rec in cens["families"]],
        "verdict_h_gap": verdict,
        "verdict_h_gap_fitted": verdict_f,
        "h_pred": {"spearman": pred_ci, "permutation": pred_p, "family_cluster": fam_ci, "verdict": pred_verdict},
        "controls_descriptive": ctrl,
        "ati_equal_median_real": med,
    }
    summary["machine_lines"] = machine_lines(summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(out_dir, summary)
    return summary


def machine_lines(summary: dict[str, Any]) -> list[str]:
    c = summary["census"]
    hp = summary["h_pred"]["spearman"]
    pp = summary["h_pred"]["permutation"]
    sp = "na" if hp["value"] is None else f"{hp['value']:.3f} [{hp['ci_lo']:.3f}, {hp['ci_hi']:.3f}]"
    return [
        f"VERDICT_H_GAP {summary['verdict_h_gap']} disagree_HP={c['disagree_HP']}/6 "
        f"fail_gap_P={c['mean_fail_gap_P']:.3f} rho_regret_P={c['mean_rho_regret_P']:.3f} "
        f"idea1_H={c['n_H_idea1_fail']}/6 idea1_P={c['n_P_idea1_fail']}/6",
        f"VERDICT_H_GAP_FITTED {summary['verdict_h_gap_fitted']} disagree_HF={c['disagree_HF']}/{c['n_fitted']} "
        f"n_flat={c['n_flat']} fail_gap_F={c['mean_fail_gap_F']}",
        f"VERDICT_H_PRED {summary['h_pred']['verdict']} spearman={sp} p={pp['p_val']}",
        f"ALIGN spearman={summary['alignment_spearman']:.12f}",
    ]


def write_outputs(out_dir: Path, summary: dict[str, Any]) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "canonical_lines.txt").write_text("\n".join(summary["machine_lines"]) + "\n", encoding="utf-8")
    with (out_dir / "family_picks.csv").open("w", newline="") as f:
        fields = [
            "family", "n", "lambda", "H", "P", "F", "flat", "rho_H", "rho_P", "rho_F",
            "fail_H", "fail_P", "fail_F", "idea1_H", "idea1_P", "idea1_F",
            "disagree_HP", "disagree_HF", "rho_regret_P", "rho_regret_F", "fail_gap_P", "fail_gap_F",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for rec in summary["families"]:
            w.writerow(rec)
    _figure(out_dir, summary)


def _figure(out_dir: Path, summary: dict[str, Any]) -> None:
    fams = [r["family"] for r in summary["families"]]
    x = np.arange(len(fams))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(x - 0.15, [r["rho_H"] for r in summary["families"]], width=0.3, label="selected by human correlation", color="#1d4ed8")
    ax[0].bar(x + 0.15, [r["rho_P"] for r in summary["families"]], width=0.3, label="selected by ATI_equal", color="#c2410c")
    ax[1].bar(x - 0.15, [r["fail_H"] for r in summary["families"]], width=0.3, label="human-correlation pick", color="#1d4ed8")
    ax[1].bar(x + 0.15, [r["fail_P"] for r in summary["families"]], width=0.3, label="ATI_equal pick", color="#c2410c")
    for a, title, ylab in (
        (ax[0], "Held-out human correlation of the selected config", "rho_heldout"),
        (ax[1], "Property gates failed by the selected config", "fail_count (of 5)"),
    ):
        a.set_xticks(x)
        a.set_xticklabels(fams, rotation=20)
        a.set_title(title, fontsize=10)
        a.set_ylabel(ylab)
        a.grid(True)
        a.legend(fontsize=8)
    fig.suptitle("Experiment 09 (as of 2026-11-27): selection gap inside base-weight families", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_selection_gap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    summary = run(args.table, args.out)
    print("\n".join(summary["machine_lines"]))


if __name__ == "__main__":
    main()
