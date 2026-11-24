"""Experiment 02 analysis (PREREG.md). Usage: python analysis.py JUDGE_FREE_EVAL_DIR OUT_DIR
JUDGE_FREE_EVAL_DIR = checkout of brando90/judge-free-eval at d1d53e8c88e32b3695d0c06379c129d65bde63ce"""
import json, sys
import numpy as np
from scipy.stats import spearmanr

root, out = sys.argv[1], sys.argv[2]
base = root + "/experiments/tfa-vs-judge/runs"
rng = np.random.default_rng(20261124)
pool = {}
for l in open(base + "/bon/bon_math__meta-llama__Llama-3.1-8B-Instruct.jsonl"):
    r = json.loads(l); pool[r["problem_id"]] = np.array(r["oracle"], int)

def load(model, cond, rep):
    d = {}
    for l in open(f"{base}/bon_scores_llama/math__Qwen__Qwen2.5-{model}-Instruct__{cond}_{rep}.jsonl"):
        r = json.loads(l); d[r["problem_id"]] = np.array(r["escore"], float)
    return d

def auroc_pairs(s, y):
    """within-problem: share of (correct, wrong) pairs with wrong >= correct, ties 1/2; NaN if one kind missing"""
    c, w = s[y == 1], s[y == 0]
    if len(c) == 0 or len(w) == 0: return np.nan
    diff = w[:, None] - c[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)

def flip(sa, sb, y):
    c, w = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(c) == 0 or len(w) == 0: return np.nan
    da = sa[c][None, :] - sa[w][:, None]; db = sb[c][None, :] - sb[w][:, None]
    ok = (da != 0) & (db != 0)
    if ok.sum() == 0: return np.nan
    return float((np.sign(da[ok]) != np.sign(db[ok])).mean())

def within_spearman(s, y):
    if len(set(y)) < 2 or np.std(s) == 0: return np.nan
    return float(spearmanr(s, y)[0])

def ab_spearman(sa, sb):
    if np.std(sa) == 0 or np.std(sb) == 0: return np.nan
    return float(spearmanr(sa, sb)[0])

def per_problem(A, B, pids):
    M = []
    for p in pids:
        y, a, b = pool[p], A[p], B[p]; m = (a + b) / 2
        M.append([auroc_pairs(m, y), auroc_pairs(a, y), auroc_pairs(b, y), flip(a, b, y), ab_spearman(a, b), within_spearman(m, y)])
    return np.array(M)  # P2_mean, P2_a, P2_b, flip, ab_spearman, within_spearman

def pooled_r(A, B, pids):
    s = np.concatenate([(A[p] + B[p]) / 2 for p in pids]); y = np.concatenate([pool[p] for p in pids])
    return float(np.corrcoef(s, y)[0, 1])

def pooled_auroc(A, B, pids):
    s = np.concatenate([(A[p] + B[p]) / 2 for p in pids]); y = np.concatenate([pool[p] for p in pids])
    from scipy.stats import rankdata
    rk = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

names = ["P2_viol_mean", "P2_viol_a", "P2_viol_b", "P1_wording_flip", "P1_ab_spearman", "C_within_spearman"]
rows = {}
for cond in ("noref", "ref"):
    for model in ("1.5B", "3B", "7B", "14B"):
        A, B = load(model, cond, "a"), load(model, cond, "b")
        pids = sorted(A)
        assert set(pids) == set(B) and all(p in pool for p in pids)
        M = per_problem(A, B, pids)
        est = {"n_problems": len(pids), "C_pooled_pearson": pooled_r(A, B, pids), "C_pooled_auroc": pooled_auroc(A, B, pids)}
        for j, nme in enumerate(names):
            est[nme] = float(np.nanmean(M[:, j]))
        # cluster bootstrap over problems
        boot = {k: [] for k in ["C_pooled_pearson"] + names}
        for _ in range(2000):
            idx = rng.integers(0, len(pids), len(pids)); ps = [pids[i] for i in idx]
            boot["C_pooled_pearson"].append(pooled_r(A, B, ps))
            Mb = M[idx]
            for j, nme in enumerate(names): boot[nme].append(np.nanmean(Mb[:, j]))
        for k, v in boot.items():
            est[k + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        rows[f"{cond}_{model}"] = est
ctrl = rows["ref_14B"]["C_pooled_auroc"]
res = {"alignment_control_ref_14B_pooled_auroc": ctrl, "alignment_control_passed": ctrl >= 0.85, "rows": rows}
keys = list(rows)
C = np.array([rows[k]["C_pooled_pearson"] for k in keys])
for nme in ("P2_viol_mean", "P1_wording_flip"):
    v = np.array([rows[k][nme] for k in keys]); rho = spearmanr(C, v)
    res[f"cross_judge_spearman_C_vs_{nme}"] = {"rho": float(rho[0]), "p": float(rho[1]), "n": len(keys)}
med = float(np.median(C)); res["median_C"] = med
res["rows_with_C_at_or_above_median"] = [k for k in keys if rows[k]["C_pooled_pearson"] >= med]
json.dump(res, open(out + "/results.json", "w"), indent=1)

lines = ["| judge LLM row | C pooled r [95%] | within-problem Spearman | P2 violation (mean score) [95%] | P1 wording flip [95%] | a-b Spearman |", "|---|---|---|---|---|---|"]
f = lambda e, k: f"{e[k]:.3f} [{e[k+'_ci'][0]:.3f}, {e[k+'_ci'][1]:.3f}]"
for k in keys:
    e = rows[k]
    lines.append(f"| Qwen2.5-{k.split('_')[1]}-Instruct, {k.split('_')[0]} | {f(e,'C_pooled_pearson')} | {e['C_within_spearman']:.3f} | {f(e,'P2_viol_mean')} | {f(e,'P1_wording_flip')} | {e['P1_ab_spearman']:.3f} |")
open(out + "/table.md", "w").write("\n".join(lines) + "\n")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for k in keys:
    e = rows[k]; mk = "o" if k.startswith("noref") else "s"
    ax[0].scatter(e["C_pooled_pearson"], e["P2_viol_mean"], marker=mk); ax[0].annotate(k.replace("_", " "), (e["C_pooled_pearson"], e["P2_viol_mean"]), fontsize=7)
    ax[1].scatter(e["C_pooled_pearson"], e["P1_wording_flip"], marker=mk); ax[1].annotate(k.replace("_", " "), (e["C_pooled_pearson"], e["P1_wording_flip"]), fontsize=7)
ax[0].set_xlabel("C: pooled Pearson r with gold correctness"); ax[0].set_ylabel("P2: wrong scored >= correct (within problem)"); ax[0].grid(True)
ax[1].set_xlabel("C: pooled Pearson r with gold correctness"); ax[1].set_ylabel("P1: wording flip rate"); ax[1].grid(True)
fig.suptitle("Qwen2.5 judge LLMs on Llama-3.1-8B MATH solutions (circle noref, square ref)", fontsize=9)
plt.tight_layout(); plt.savefig(out + "/fig_corr_vs_properties.png", dpi=150)
print(open(out + "/table.md").read()); print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
