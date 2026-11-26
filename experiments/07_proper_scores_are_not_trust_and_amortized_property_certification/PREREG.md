# PREREG: 07_proper_scores_are_not_trust_and_amortized_property_certification (date: 2026-11-26)

Committed before any run. Do not change this file after the first run; add a dated "Amendment" section instead.

## Prior-exposure disclosure
Zero-model-call CPU reanalysis (`$0.00`) of existing cached judge scores and anonymized human ratings:
- `/Users/brandomiranda/cert-judge/experiments/39_label_free_proxy_index_train_val_test_over_judge_families/results/` (`panel.json`, `probe_manifest.json`, `units_train_val.json`, `units_test.json`, 60 `harvest/*.json`)
- `/Users/brandomiranda/cert-judge/experiments/42_semantic_weakening_overconstraint_vacuity_probes_on_e39_judges/results/` (`semantic_probe_manifest.json`, `semantic_screen.json`, 60 `harvest/*.json`)
- `/Users/brandomiranda/cert-judge/experiments/68_proper_scoring_expansion_e39_e63_and_vb_ratings_start_24Nov2026/results/study_b_e39_proper_scores.json` (alignment check only)
- `/Users/brandomiranda/veribench/experiments/100_karpathy_recipe_tc_judge_search_human_train_val_test/inputs/items.jsonl` (`75` items across `25` tasks, SHA-256 `ad9ed552fc2ffac6328e6010bf6be82215dd74f1eddd791622d433126877a96c`, `all5` and `expert3` integer ratings `{0..5}`, read-only off-repo; no raw human ratings or Lean code copied to this public repository)
- `/Users/brandomiranda/trustworthy-theorem-judges/experiments/27_build_a_judge_that_beats_the_constant_forecast_on_held_out_tasks_old_new_ratings_start_24Nov2026/data_11-24-2026/{human_ratings_old_11-24-2026.csv,human_ratings_new_11-24-2026.csv}` (`75` old artifacts / `25` tasks and `100` new artifacts / `25` tasks, read-only off-repo; aggregate statistics only, no raw rating rows copied to this public repository)
- `experiments/03_human_corr_vs_axiomatic_properties_cert_judge/results/summary.json` (alignment check only)

Already known from Experiments `03`, `06`, and `cert-judge` Experiment `68` before writing this PREREG:
- In Experiment `03`, `22/22` real Lean 4 judge configurations with `target_expert` `rho_heldout >= 0.70` fail `V_weak`, `V_rep`, or `V_bug` (`22/22` even excluding `V_bug`), and in Experiment `06` the median within-item repeat range across those `22` is `0.097`.
- In `cert-judge` Experiment `68`, `CE_fit` calibration across `E39` `train -> heldout` (`45` items / `15` tasks) improves mean `RPS` from `0.1698` to `0.1491` on `test_llm` (`18` configs), `C_ms_gold` correlates with `-ce_cal_all5_rps` (`Spearman = +0.589`, `p = 0.0163`, `n = 16`) and `-ce_cal_all5_tvd` (`+0.675`, `p = 0.0042`), while `ctl|name_jaccard` attains `C_ms_gold = 1.000` and `ctl|editdist` attains `TI_var_trainnorm = 1.000`.
- In `trustworthy-theorem-judges` Experiment `25`/`27`, `73%` of the `247` New VeriBench ratings are `5`.

Not previously computed and frozen as confirmatory targets here:
1. Which exact real judge configurations strictly beat `CONST_train` (`95%` task-cluster CI of `judge - CONST_train` `< 0`) and match/beat human `LORO` on `CE_fit` `RPS`, `CE`, and `TVD`, whether `100%` of those proper-score-qualified configurations still fail non-bug axiomatic checks (`V_weak`, `V_rep`, `V_overc`, calibrated `V_weak_cal`, calibrated repeat expected-score range `rep_cal_ev_range_05`, and calibrated pairwise repeat `TVD` `rep_cal_tvd`), and exact held-out `crit` 3-repeat raw range reconstructed from `E39` `harvest/*.json`.
2. Out-of-family hold-out generalization (`test` split `n = 18` + strictly unseen-model `n = 16` + 6-fold Leave-One-Model-Family-Out CV grouping `terminus2` judge agents with their base model family across `n = 56`) of multi-property Axiomatic Trust Indices (`ATI_equal`, `ATI_min`, and train-fitted nonnegative Ridge `ATI_cal_rps`), their `RPS`/`CE`/`TVD` selection regret, and rejection of `4/4` deterministic adversarial controls.
3. Cross-regime human consistency and non-ceiling disagreement (`P(range >= 1 | min < 5)`) on New VeriBench (`100` artifacts / `25` tasks) vs. Old VeriBench (`75` artifacts / `25` tasks).

## Hypothesis (one sentence)
**H1-proper (Idea 1 extended from correlation to proper scoring rules):** Real Lean 4 judge LLMs and judge agents whose `CE_fit`-calibrated predictive distributions strictly beat `CONST_train` (and even match or beat the human `LORO` ceiling) on held-out proper scores (`RPS`, `CE`, `TVD`) still violate basic axiomatic properties (`V_weak`, `V_rep`, or `V_overc`, excluding `V_bug`), with stochastic repeat variance persisting as non-trivial `TVD` in the calibrated predictive distributions; **H-Idea3 (amortized calibration, resolving `docs/PROJECT_IDEA.md` Con2):** a multi-property Axiomatic Trust Index (`ATI`) generalizes across unseen model families to predict held-out proper-score alignment (`-RPS`, `-CE`, `-TVD`) and rejects all `4/4` deterministic controls that fool single-property indices (`C_ms_gold`, `TI_var`); **H2-new (human ceiling trap):** on the new 100-artifact VeriBench ratings, `5/5` ceiling saturation compresses proper-score headroom while non-ceiling items still display substantial human disagreement.

## Properties and metrics
All item/task 95% confidence intervals use `2,000`-draw task-cluster percentile bootstrap resampling (`seed = 20261124`, fresh `np.random.default_rng(20261124)` per statistic). Paired contrasts (`judge - CONST_train`) use two-sided task-cluster sign-flip permutation (`2,000` MC draws on `15` held-out tasks, exact `2^G` when `G <= 12`, `seed = 20261124`). Cross-judge meta-associations use `2,000`-draw unit bootstrap 95% CIs and `5,000`-draw two-sided permutation `p-val`s (plus within-family stratified permutation `p-val`s, `seed = 20261124`).

### Part A — Proper Scores (`RPS`, `CE`, `TVD`) vs. Raw & Calibrated Axiomatic Properties (`E39`/`E42`, `60` units)
Proper scoring setup (matching `cert-judge/experiments/68_.../scripts/metrics.py`, `CE_FLOOR = 1e-12`, Laplace `alpha = 0.5`):
- Train split: `30` items (`10` tasks). Held-out evaluation split (`val + test`): `45` items (`15` tasks). Requires `train_cov >= 0.90` and `heldout_cov >= 0.90`.
- Fit `ce_fn_5 = fit_ce_ordinal(tr_s, tr_r_all5)` (and `ce_fn_3` on `expert3`) strictly on train medians `tr_s` (`[0, 1]`), evaluate held-out `RPS`, `CE`, `TVD`, `Brier`, `ICC(A,1)`, `MAE`, `bias_05`, with `Spearman` and `Pearson` as secondary columns.
- **Alignment Gate A1 (`<= 1e-9`):** Recomputed `raw_all5_rps`, `ce_cal_all5_rps`, `ce_cal_all5_ce`, `ce_cal_all5_tvd`, `ce_cal_all5_icc_a1`, `raw_exp3_spearman`, `ce_cal_exp3_rps`, `CONST_train`, and `LORO` baselines must match `study_b_e39_proper_scores.json` within `1e-9` across all `60` units.
- **Alignment Gate A2 (`<= 1e-9`):** Recomputed `V_ms`, `V_weak`, `V_vac`, `V_overc`, `V_fmt`, `V_bug`, `V_rep`, and `rep_mean_range` from raw `E39`/`E42` `harvest/*.json` must match `experiments/03_.../results/summary.json` within `1e-9` across all `60` units.
- Qualification sets (among the `56` real judge configs `kind == "llm"`):
  - `Q_const_rps`: real configs whose paired `ce_cal_all5_rps - const_all5_rps` has upper 95% task-cluster CI `< 0` (`RPS` strictly beats `CONST_train`). Also report `Q_const_all3` (upper 95% CI `< 0` on ALL THREE: `RPS`, `CE`, and `TVD`) and `Q_loro_rps` (subset of `Q_const_rps` with point `ce_cal_all5_rps <= loro_all5_smoothed_rps` AND `ce_cal_all5_ce <= loro_all5_smoothed_ce`). Repeat on `expert3` as sensitivity.
- Calibrated-distribution axiomatic metrics (`ce_fn_5` applied directly to probes and raw repeats):
  - `V_weak_cal`, `V_overc_cal`, `V_vac_cal`: share of valid `E42` probe medians where calibrated expected score `ev(probe) >= ev(ident) - 1e-12`; `V_fmt_cal` and `fmt_cal_tvd` (mean `TVD(q_hat(fmt), q_hat(ident))`).
  - Held-out `E39` `crit` repeat distribution metrics (`45` held-out items with `>= 2` repeats): `V_rep_heldout`, `rep_raw_range_heldout` (mean `max - min` on `[0, 1]`), `rep_cal_ev_range_05` (mean `max(ev_05) - min(ev_05)` across repeats on `[0, 5]`), and `rep_cal_tvd` (mean over held-out items of mean pairwise `TVD(q_hat(r_a), q_hat(r_b))` across distinct repeat pairs).

### Part B — Amortized Property Certification (`ATI`) Out-of-Family Validation (`Idea 3`)
Label-free 6-component axiomatic pass-rate vector on `[0, 1]`:
`f = [1 - V_ms, 1 - V_weak, 1 - V_vac, 1 - V_overc, 1 - V_fmt, 1 - V_rep]`. (`V_bug` excluded by default per Exp 03/06 statement-rubric construct separation; `ATI_plus_bug` reported in sensitivity.)
Indices compared:
1. `ATI_equal`: unweighted mean of `f` (zero fitted parameters).
2. `ATI_min`: `min(f)` (worst-case axiom pass rate).
3. `ATI_cal_rps`: nonnegative Ridge regression (`nnls` augmented with `sqrt(lambda) * I`, `lambda in {0.01, 0.1, 1.0, 10.0}` chosen by inner leave-one-train-family-out CV strictly on the training fold, predicting `-ce_cal_all5_rps`).
4. Single-property comparators: `C_ms_gold`, `1 - V_weak`, `1 - V_rep`, `1 - V_fmt`, `1 - V_bug`, `TI_var_trainnorm`.
5. Hard Axiomatic Gate `Gate_all`: passes iff `V_ms < 0.10`, `V_weak < 0.15`, `V_vac < 0.10`, `V_overc < 0.25`, `V_fmt < 0.10`, and `V_rep < 0.25`.

Evaluations (zero test-family leakage):
- **Canonical `train+val` (`n = 38`) -> `test` (`n = 18`):** Spearman/CI/p-val and within-family `p-val` with `-ce_cal_all5_rps`, `-ce_cal_all5_ce`, `-ce_cal_all5_tvd`, `ce_cal_all5_icc_a1`, and `raw_exp3_spearman`, plus top-1 selection regret (`selected_loss - best_in_pool_loss`). Also report strictly unseen-model test slice (`n = 16`, excluding `t2_sonnet5` which shares `sonnet5m` weights with `train`).
- **6-fold Leave-One-Model-Family-Out CV (`LOMFO-CV`, `n = 56`):** Group `terminus2` judge agents with their base model weights (`sol6h`, `sonnet5m+t2_sonnet5`, `gem31h`, `luna6h+t2_luna`, `qwen36+t2_qwen`, `opus55x`). Report per-fold held-out Spearman, pooled OOF Spearman, and mean/max selection regret across all 6 folds.
- **Deterministic Control Rejection (`n = 4`):** `ctl|count_ratio`, `ctl|name_jaccard`, `ctl|editdist`, `ctl|random`. Count controls rejected continuously (score `< median(Q_const_rps)`) and by `Gate_all` (`0/4` pass).

### Part C — Human Rating Regimes: Old vs. New VeriBench (`H2-new`)
From anonymized multiset CSVs (`human_ratings_old_11-24-2026.csv`, `75` items / `25` tasks; `human_ratings_new_11-24-2026.csv`, `100` items / `25` tasks), with `2,000`-draw task-cluster bootstrap 95% CIs:
- Individual rating share `P(r == 5)` and `P(r >= 4)`.
- Item-level (`>= 2` ratings) unanimous share (`range == 0`), `range >= 1`, `range >= 2`, mean range, mean item SD, and within-item pairwise exact agreement `P(r_a == r_b)` over unordered within-item pairs.
- Non-ceiling conditional disagreement (`min(ratings) < 5`): `P(range >= 1 | min < 5)`, `P(range >= 2 | min < 5)`, mean range, and mean pairwise exact agreement.
- Irreducible `RPS` disagreement floor and `LORO` smoothed/empirical `RPS`, `CE`, `TVD`.

## Data
- Local cached harvest & manifest JSONs (`cert-judge` E39, E42, E68), `items.jsonl` (`veribench` E100), and anonymized `human_ratings_{old,new}_11-24-2026.csv` (`trustworthy-theorem-judges` E27), all read-only off-repo; plus `experiments/03_.../results/summary.json`.
- New model calls: no (`$0.00`).

## Baselines
- `CONST_train` (training marginal forecast on `{0..5}`), `LORO` human ceiling (smoothed & empirical), `4` deterministic non-LLM controls (`ctl|count_ratio`, `ctl|editdist`, `ctl|name_jaccard`, `ctl|random`), and single-property indices (`C_ms_gold`, `TI_var_trainnorm`).
- `pass@k` and `boxed accuracy` do not apply to 0–5 Lean 4 theorem-specification adequacy ratings.

## What would falsify it
- **H1-proper:** Supported if `|Q_const_rps| >= 1` AND at least `1` configuration in `Q_const_rps` violates `V_weak >= 0.20` OR `V_rep >= 0.20` (with task-cluster lower 95% CI `> 0.05`, excluding `V_bug`). Falsified if `|Q_const_rps| >= 1` AND every configuration in `Q_const_rps` satisfies `V_weak < 0.10`, `V_vac < 0.10`, `V_ms < 0.10`, `V_overc < 0.10`, `V_fmt < 0.05`, and `V_rep < 0.10`. Untestable if `|Q_const_rps| == 0`. Mixed otherwise.
- **H-Idea3:** Supported if on held-out `test` real configs (`n = 18`), `ATI_equal` OR `ATI_cal_rps` achieves Spearman `>= 0.50` (`p < 0.05`, two-sided permutation) with `-ce_cal_all5_rps`, `ce_cal_all5_rps` selection regret `<= 0.015`, AND rejects `4/4` deterministic controls (all 4 score below `median(Q_const_rps)` AND `0/4` pass `Gate_all`). Falsified if both `ATI_equal` and `ATI_cal_rps` have test Spearman `< 0.30` or `p >= 0.10` with `-ce_cal_all5_rps`, OR fail to reject `>= 1` control. Mixed otherwise.
- **H2-new:** Supported if on New VeriBench (`100` items / `25` tasks), ceiling share `P(r == 5) >= 0.65` (lower 95% CI `> 0.50`) AND non-ceiling items (`min < 5`) have disagreement share `P(range >= 1 | min < 5) >= 0.60` (lower 95% CI `> 0.40`). Falsified if `P(r == 5) < 0.40` AND unanimous share `>= 0.85`. Mixed otherwise.

## Models and harnesses
- Original harvest harnesses & models (no new calls): `vb103_judge` (single-call judge LLM: `openai/gpt-6-sol-high`, `anthropic/claude-sonnet-5-medium`, `openai/gpt-6-luna-high`, `google/gemini-3.1-pro-preview-high`, `anthropic/claude-opus-5-5-no-fallback`, `alibaba/qwen3.6-27b-thinking`), `terminus2` (judge agent: `openai/gpt-6-luna-high`, `anthropic/claude-sonnet-5-medium`, `alibaba/qwen3.6-27b-thinking`), and `none` (`4` deterministic controls).
- Reanalysis harness: Python 3 (`numpy`, `scipy`, `matplotlib`) in `scripts/analysis.py`.

## Cost estimate
$0.00 (zero new model calls; CPU-only reanalysis).
