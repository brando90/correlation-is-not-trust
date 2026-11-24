"""Experiment 01 analysis (PREREG.md). Usage: python analysis.py DICES_CSV OUT_DIR"""
import csv, json, sys, collections
import numpy as np
from scipy.stats import pearsonr as _pr

def pearsonr(x, y):
    # NaN (not an error) when an input is constant; such raters are counted and left out of summaries
    if np.std(x) == 0 or np.std(y) == 0:
        return (np.nan, np.nan)
    return _pr(x, y)

csv_path, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(20261124)
MAP = {"No": 0.0, "Unsure": 0.5, "Yes": 1.0}
rows = list(csv.DictReader(open(csv_path)))
items = sorted({r["item_id"] for r in rows}); raters = sorted({r["rater_id"] for r in rows})
ii = {k: i for i, k in enumerate(items)}; rr = {k: i for i, k in enumerate(raters)}
S = np.full((len(items), len(raters)), np.nan)
for r in rows:
    S[ii[r["item_id"]], rr[r["rater_id"]]] = MAP[r["Q_overall"]]
assert not np.isnan(S).any()
n_items, n_rat = S.shape
LAB = np.rint(S * 2).astype(int)  # 0 No, 1 Unsure, 2 Yes

def pct(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

def modal(L):  # L: items x raters (ints 0..2); returns mode or -1 on tie
    cnt = np.stack([(L == c).sum(1) for c in range(3)], 1)
    m = cnt.argmax(1); top = cnt.max(1)
    tie = (cnt == top[:, None]).sum(1) > 1
    m[tie] = -1
    return m

res = {"n_items": n_items, "n_raters": n_rat, "n_ratings": int(S.size)}
# E1, E2
r_list, agree = [], []
for _ in range(2000):
    p = rng.permutation(n_rat); a, b = p[: n_rat // 2], p[n_rat // 2:]
    r_list.append(pearsonr(S[:, a].mean(1), S[:, b].mean(1))[0])
    ma, mb = modal(LAB[:, a]), modal(LAB[:, b])
    agree.append(np.mean((ma == mb) & (ma >= 0)))
r_list = np.array(r_list)
res["E1_split_half_pearson"] = dict(zip(["mean", "lo", "hi"], pct(r_list)))
sb = 2 * r_list / (1 + r_list)
res["E1_spearman_brown_full_pool"] = dict(zip(["mean", "lo", "hi"], pct(sb)))
res["E2_modal_label_agreement_between_halves"] = dict(zip(["mean", "lo", "hi"], pct(agree)))
# E3
yes = (LAB == 2).mean(1); no = (LAB == 0).mean(1)
minority = np.minimum(yes, no)
res["E3_items_minority_ge_10pct"] = float((minority >= .10).mean())
res["E3_items_minority_ge_25pct"] = float((minority >= .25).mean())
res["E3_items_span_more_than_one_point"] = float(((yes > 0) & (no > 0)).mean())
res["E3_mean_minority_share"] = float(minority.mean())
pair_agree = []
for i in range(n_items):
    c = np.bincount(LAB[i], minlength=3); pair_agree.append(((c * (c - 1)).sum()) / (n_rat * (n_rat - 1)))
res["E3_mean_pairwise_rater_agreement_per_item"] = float(np.mean(pair_agree))
# E4a oracle judge vs single raters
full = S.mean(1); orac_lab = np.rint(full * 2).astype(int)
rp = np.array([pearsonr(full, S[:, j])[0] for j in range(n_rat)])
# leave-one-out oracle (no self-inclusion)
rp_loo = np.array([pearsonr((full * n_rat - S[:, j]) / (n_rat - 1), S[:, j])[0] for j in range(n_rat)])
ag = np.array([(orac_lab == LAB[:, j]).mean() for j in range(n_rat)])
ag_loo = np.array([(np.rint((full * n_rat - S[:, j]) / (n_rat - 1) * 2).astype(int) == LAB[:, j]).mean() for j in range(n_rat)])
res["n_raters_with_constant_labels"] = int((S.std(0) == 0).sum())
def bin_agree(train_mean_yes, train_mean_no, j):
    pred = np.where(train_mean_yes > train_mean_no, 2, 0)
    ok = LAB[:, j] != 1
    return float((pred[ok] == LAB[ok, j]).mean())
yes_all, no_all = (LAB == 2).mean(1), (LAB == 0).mean(1)
ag_bin = np.array([bin_agree(((LAB == 2).sum(1) - (LAB[:, j] == 2)) / (n_rat - 1), ((LAB == 0).sum(1) - (LAB[:, j] == 0)) / (n_rat - 1), j) for j in range(n_rat)])
res["AMENDMENT_E4a_binary_oracle_agreement_leave_self_out_on_yes_no_ratings"] = dict(zip(["mean", "lo", "hi"], pct(ag_bin)))
res["AMENDMENT_share_of_ratings_unsure"] = float((LAB == 1).mean())
res["E4a_oracle_vs_single_rater_pearson"] = dict(zip(["mean", "lo", "hi"], pct(rp)))
res["E4a_oracle_vs_single_rater_pearson_leave_self_out"] = dict(zip(["mean", "lo", "hi"], pct(rp_loo)))
res["E4a_oracle_label_agreement_with_single_rater"] = dict(zip(["mean", "lo", "hi"], pct(ag)))
res["E4a_oracle_label_agreement_leave_self_out"] = dict(zip(["mean", "lo", "hi"], pct(ag_loo)))
# E4b noise ceiling
ceil = {}
for k in [1, 3, 5, 10, 25, 61]:
    v = []
    for _ in range(500):
        p = rng.permutation(n_rat); tr, te = p[:k], p[k:]
        v.append(pearsonr(S[:, tr].mean(1), S[:, te].mean(1))[0] if k > 1 else pearsonr(S[:, tr[0]], S[:, te].mean(1))[0])
    ceil[k] = dict(zip(["mean", "lo", "hi"], pct(v)))
res["E4b_fit_k_raters_corr_with_held_out_mean"] = ceil
# rater-level: each rater vs leave-one-out mean of the others
res["rater_consistency_vs_others_pearson_percentiles"] = {q: float(np.nanpercentile(rp_loo, q)) for q in (0, 5, 25, 50, 75, 95, 100)}
json.dump(res, open(out + "/results.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ks = sorted(ceil)
ax[0].errorbar(ks, [ceil[k]["mean"] for k in ks], yerr=[[ceil[k]["mean"] - ceil[k]["lo"] for k in ks], [ceil[k]["hi"] - ceil[k]["mean"] for k in ks]], marker="o")
ax[0].set_xscale("log"); ax[0].set_xlabel("raters per item the judge is fitted to (k)"); ax[0].set_ylabel("Pearson r with held-out raters' mean"); ax[0].set_title("E4b noise ceiling (DICES-350)"); ax[0].grid(True)
ax[1].hist(minority, bins=20); ax[1].set_xlabel("minority share of raters per item (Yes vs No)"); ax[1].set_ylabel("items"); ax[1].set_title("E3 disagreement"); ax[1].grid(True)
plt.tight_layout(); plt.savefig(out + "/fig_human_inconsistency.png", dpi=150)
print(json.dumps(res, indent=1))
