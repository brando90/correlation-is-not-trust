#!/usr/bin/env python3
"""Experiment 05 analysis (dated 2026-11-24). Reads the private probes.jsonl (targets, kinds, tasks) and
scores_<config>.jsonl written by judge_batches.py; writes results/summary.json, results/scores_<config>.csv
(opaque probe id, kind, task, score: no Lean source, no ratings) and figures with grid lines.
Usage: python3 analysis.py --probes probes.jsonl --scores scores_A.jsonl [scores_B.jsonl ...] --out results"""
import argparse, json, collections, csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED, DRAWS, TOL = 20261124, 2000, 0.05

def load_scores(path):
    sc, lanes, models, toks = {}, collections.Counter(), collections.Counter(), collections.Counter()
    for l in open(path):
        r = json.loads(l); sc.update(r["scores"]); lanes[r["lane"]] += 1; models[str(r["model"])] += 1
        for k, v in (r.get("tokens") or {}).items(): toks[k] += v or 0
    return sc, dict(lanes), dict(models), dict(toks)

def cluster_ci(tasks_all, stat, rng):
    """stat(weights per task dict) -> float; percentile bootstrap over tasks."""
    tasks = sorted(set(tasks_all)); idx = {t: i for i, t in enumerate(tasks)}
    vals = []
    for _ in range(DRAWS):
        draw = rng.integers(0, len(tasks), len(tasks))
        w = np.bincount(draw, minlength=len(tasks))
        v = stat({t: w[idx[t]] for t in tasks})
        if v is not None and not np.isnan(v): vals.append(v)
    if not vals: return [None, None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def wmean(vals, tasks, w):
    ws = np.array([w[t] for t in tasks], float)
    return None if ws.sum() == 0 else float((ws * np.array(vals, float)).sum() / ws.sum())

def wspear(x, y, tasks, w):
    xs, ys = [], []
    for a, b, t in zip(x, y, tasks):
        xs += [a] * int(w[t]); ys += [b] * int(w[t])
    if len(set(xs)) < 2 or len(set(ys)) < 2: return None
    return float(spearmanr(xs, ys)[0])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--probes", required=True); ap.add_argument("--scores", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--items", required=True, help="VeriBench items.jsonl with p1..p5 (private, not committed)")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True, parents=True)
    probes = {}
    for l in open(a.probes):
        r = json.loads(l); probes[r["probe_id"]] = r
    hum = {}
    for l in open(a.items):
        it = json.loads(l); cnt = np.zeros(6)
        for k in ("p1", "p2", "p3", "p4", "p5"):
            v = it.get(k)
            try:
                v = float(v)
                if v == v: cnt[int(round(v))] += 1
            except (TypeError, ValueError): pass
        hum["CRIT__" + it["item"]] = cnt / cnt.sum() if cnt.sum() else None
    pooled = np.mean([q for q in hum.values() if q is not None], axis=0); unif = np.ones(6) / 6
    def rps(p, q): return float(np.mean((np.cumsum(p)[:5] - np.cumsum(q)[:5]) ** 2))
    def tvd(p, q): return float(0.5 * np.abs(p - q).sum())
    def ce(p, q): return float(-(q * np.log(np.clip(p, 1e-12, None))).sum())
    def point(s_): 
        p = np.full(6, 0.1 / 6); p[int(min(5, max(0, np.floor(5 * s_ + 0.5))))] += 0.9; return p
    def point_raw(s_):
        p = np.zeros(6); p[int(min(5, max(0, np.floor(5 * s_ + 0.5))))] = 1.0; return p
    summary = {}
    for sp in a.scores:
        cfg = Path(sp).stem.replace("scores_", "")
        sc, lanes, models, toks = load_scores(sp)
        rng = np.random.default_rng(SEED)
        res = dict(lanes=lanes, models=models, tokens=toks)
        kinds = collections.defaultdict(dict)
        for pid, p in probes.items():
            if pid in sc: kinds[p["kind"]][pid] = sc[pid]
        res["n_scored"] = {k: len(v) for k, v in kinds.items()}
        res["n_missing"] = {k: sum(1 for pid, p in probes.items() if p["kind"] == k and pid not in sc) for k in set(p["kind"] for p in probes.values())}
        # CRIT
        crit = [(pid, s) for pid, s in kinds["CRIT"].items()]
        tasks = [probes[pid]["task"] for pid, _ in crit]; x = [s for _, s in crit]
        for tgt in ("target_expert", "target_all5"):
            y = [probes[pid][tgt] for pid, _ in crit]
            pt = wspear(x, y, tasks, {t: 1 for t in tasks})
            ci = cluster_ci(tasks, lambda w, y=y: wspear(x, y, tasks, w), rng)
            res[f"rho_{tgt}"] = dict(value=pt, ci=ci, n=len(crit), distinct_scores=len(set(x)))
        # headline: RPS, TVD, CE vs the human rating distribution
        ids = [pid for pid, _ in crit if hum.get(pid) is not None]
        tk2 = [probes[pid]["task"] for pid in ids]
        M = {}
        for name, fp in (("judge", lambda s_: point_raw(s_)), ("constant", lambda s_: pooled), ("uniform", lambda s_: unif)):
            M[name] = dict(rps=[rps(fp(kinds["CRIT"][pid]), hum[pid]) for pid in ids], tvd=[tvd(fp(kinds["CRIT"][pid]), hum[pid]) for pid in ids])
        M["judge"]["ce"] = [ce(point(kinds["CRIT"][pid]), hum[pid]) for pid in ids]
        M["constant"]["ce"] = [ce(pooled, hum[pid]) for pid in ids]; M["uniform"]["ce"] = [ce(unif, hum[pid]) for pid in ids]
        head = dict(n_items=len(ids))
        for name in M:
            head[name] = {m: dict(value=float(np.mean(v)), ci=cluster_ci(tk2, lambda w, v=v: wmean(v, tk2, w), rng)) for m, v in M[name].items()}
        head["judge_minus_constant"] = {m: dict(value=float(np.mean(np.array(M["judge"][m]) - np.array(M["constant"][m]))),
                                                ci=cluster_ci(tk2, lambda w, m=m: wmean(list(np.array(M["judge"][m]) - np.array(M["constant"][m])), tk2, w), rng)) for m in ("rps", "tvd", "ce")}
        res["headline"] = head
        res["rps_significant"] = bool(head["judge_minus_constant"]["rps"]["ci"][1] is not None and head["judge_minus_constant"]["rps"]["ci"][1] < 0)
        # ID
        id1 = {probes[pid]["task"]: s for pid, s in kinds["ID1"].items()}
        id2 = {probes[pid]["task"]: s for pid, s in kinds["ID2"].items()}
        both = sorted(set(id1) & set(id2))
        noise = [abs(id1[t] - id2[t]) for t in both]; nshare = [float(d > TOL + 1e-9) for d in noise]
        res["ID1_mean"] = float(np.mean(list(id1.values()))) if id1 else None
        res["noise"] = dict(share=float(np.mean(nshare)) if nshare else None, mean_abs=float(np.mean(noise)) if noise else None, n=len(both),
                            share_ci=cluster_ci(both, lambda w: wmean(nshare, both, w), rng) if both else None)
        res["valid_judge"] = bool(id1) and res["ID1_mean"] >= 0.6
        # rename
        for kind in ("RN_b", "RN_d", "RN"):
            pr = [(pid, s) for k in (("RN_b", "RN_d") if kind == "RN" else (kind,)) for pid, s in kinds[k].items() if probes[pid]["task"] in id1]
            if not pr: res[kind] = None; continue
            tk = [probes[pid]["task"] for pid, _ in pr]
            d = [s - id1[probes[pid]["task"]] for pid, s in pr]
            viol = [float(abs(v) > TOL + 1e-9) for v in d]
            res[kind] = dict(n=len(pr), share=float(np.mean(viol)), share_ci=cluster_ci(tk, lambda w: wmean(viol, tk, w), rng),
                             mean_abs=float(np.mean(np.abs(d))), mean_signed=float(np.mean(d)),
                             excess=(float(np.mean(viol)) - res["noise"]["share"]) if res["noise"]["share"] is not None else None)
            if kind == "RN" and res["noise"]["share"] is not None and both:
                nsh = dict(zip(both, nshare))
                def ex(w, viol=viol, tk=tk): 
                    a1 = wmean(viol, tk, w); a2 = wmean([nsh[t] for t in both], both, w); return None if a1 is None or a2 is None else a1 - a2
                res[kind]["excess_ci"] = cluster_ci(sorted(set(tk) | set(both)), lambda w: ex({t: w.get(t, 0) for t in set(tk) | set(both)}), rng)
        # bug
        pr = [(pid, s) for pid, s in kinds["BUG"].items() if probes[pid]["task"] in id1]
        if pr:
            tk = [probes[pid]["task"] for pid, _ in pr]; base = [id1[t] for t in tk]; sv = [s for _, s in pr]
            ge = [float(s >= b - 1e-9) for s, b in zip(sv, base)]; gt = [float(s > b + 1e-9) for s, b in zip(sv, base)]
            res["BUG"] = dict(n=len(pr), V_bug=float(np.mean(ge)), V_bug_ci=cluster_ci(tk, lambda w: wmean(ge, tk, w), rng),
                              V_bug_ID1_at_least_0p5=(lambda m: dict(n=int(sum(m)), V_bug=float(np.mean([g for g, k in zip(ge, m) if k])) if any(m) else None))([b >= 0.5 for b in base]),
                              strict_share=float(np.mean(gt)), tie_share=float(np.mean(ge) - np.mean(gt)),
                              mean_drop=float(np.mean(np.array(base) - np.array(sv))),
                              by_level={str(l): dict(n=sum(1 for pid, _ in pr if probes[pid]["level"] == l),
                                                     V_bug=float(np.mean([g for g, (pid, _) in zip(ge, pr) if probes[pid]["level"] == l])))
                                        for l in sorted(set(probes[pid]["level"] for pid, _ in pr))})
        # verdict pieces
        rho = res["rho_target_expert"]
        res["rho_significant"] = bool(rho["ci"][0] is not None and rho["ci"][0] > 0)  # secondary reading
        summary[cfg] = res
        with open(out / f"scores_{cfg}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["probe_id", "kind", "task", "level", "score"])
            for pid, s in sorted(sc.items()):
                if pid in probes: w.writerow([pid, probes[pid]["kind"], probes[pid]["task"], probes[pid]["level"], s])
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    # figures (grid on)
    cfgs = list(summary)
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    for i, c in enumerate(cfgs):
        s = summary[c]; h = s["headline"]
        for j, (nm, col) in enumerate((("judge", "C0"), ("constant", "gray"), ("uniform", "C3"))):
            v = h[nm]["rps"]; ax[0].errorbar(i + 0.2 * j, v["value"], yerr=[[v["value"] - v["ci"][0]], [v["ci"][1] - v["value"]]], fmt="o", capsize=4, color=col, label=nm if i == 0 else None)
            v = h[nm]["tvd"]; ax[1].errorbar(i + 0.2 * j, v["value"], yerr=[[v["value"] - v["ci"][0]], [v["ci"][1] - v["value"]]], fmt="o", capsize=4, color=col)
        if s.get("RN"): ax[2].errorbar(i, s["RN"]["share"], yerr=[[s["RN"]["share"] - s["RN"]["share_ci"][0]], [s["RN"]["share_ci"][1] - s["RN"]["share"]]], fmt="o", capsize=4, label="rename" if i == 0 else None)
        ax[2].errorbar(i + 0.3, s["noise"]["share"], yerr=[[s["noise"]["share"] - s["noise"]["share_ci"][0]], [s["noise"]["share_ci"][1] - s["noise"]["share"]]], fmt="s", capsize=4, color="gray", label="repeat noise" if i == 0 else None)
        if s.get("BUG"): ax[3].errorbar(i, s["BUG"]["V_bug"], yerr=[[s["BUG"]["V_bug"] - s["BUG"]["V_bug_ci"][0]], [s["BUG"]["V_bug_ci"][1] - s["BUG"]["V_bug"]]], fmt="o", capsize=4)
    for a_, t in zip(ax, ["RPS vs human ratings (lower is better)", "TVD vs human ratings (lower is better)", "Rename: share of |score change| > 0.05", "Bug scored >= original (share)"]):
        a_.set_title(t, fontsize=9); a_.set_xticks(range(len(cfgs))); a_.set_xticklabels(cfgs); a_.grid(True)
    ax[0].legend(fontsize=7); ax[2].legend(fontsize=7)
    fig.suptitle("Experiment 05 (as of 2026-11-24): 95% task-cluster intervals", fontsize=9)
    plt.tight_layout(); plt.savefig(out / "fig_properties.png", dpi=150)
    print(json.dumps(summary, indent=1)[:3000])
main()
