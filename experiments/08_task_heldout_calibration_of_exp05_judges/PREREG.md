# PREREG: 08_task_heldout_calibration_of_exp05_judges (date: 2026-11-27)

Commit this file before any run. Do not change it after the first run; add a dated "Amendment" section instead.

Zero new model calls (`$0.00`). CPU-only Python 3 (`numpy`, `scipy`, `matplotlib`). No judge LLM and no judge agent is called. The three judge configurations are the ones already logged in experiment 05.

## Prior-exposure disclosure

Experiment 05 (committed, including Amendments 1 and 2) already reported, on the same 75 `CRIT` files and the same three score files:

- Raw point-mass `RPS` A `0.395`, A2 `0.390`, B `0.282`, against an in-sample constant `0.077` (the unweighted mean of the 75 item rating distributions, fit on the same items). `TVD` A `0.859`, A2 `0.897`, B `0.863`, constant `0.491`. Smoothed `CE` A `3.53`, A2 `3.68`, B `3.55`, constant `1.73`. Every judge was worse than that constant, so Amendment 1's proper-score arm of H1 was untestable.
- Spearman with `target_expert`: A `-0.004`, A2 `-0.062`, B `0.388` with lower bound `0.006`.
- `V_bug` (bug score `>=` gold-as-candidate): A `0.676`, A2 `0.757`, B `0.568`, with exact-tie shares `0.649`, `0.649`, `0.514`. Pooled rename share at `0.05`: A `0.079`, A2 `0.053`, B `0.237`. `RN_d` share: A `0.083`, A2 `0.042`, B `0.375`. Repeat-noise share: A `0.208`, A2 `0.042`, B `0.042`. `ID1` means: A `0.917`, A2 `0.958`, B `0.996`.

Those numbers are an alignment gate, not a result of this experiment. Not previously computed, and the confirmatory targets here: task-held-out ordinal calibration; a constant forecast fit only on the training tasks; `RPS` / `TVD` / `CE` of the calibrated forecast against that held-out constant; and the same bug and declaration-rename rules applied to the calibrated expected score, including which pairs cross the `0.05` band only after calibration.

Experiment 07 calibrated other judges (the experiment 03 panel) on a different item split. It did not calibrate these three experiment 05 score files.

## Hypothesis (one sentence)

H1-cal: after a monotone distributional calibration fit with the evaluated task held out, at least one of the three experiment 05 judge configurations beats the train-task constant forecast on `RPS` and still fails the inserted-bug rule or declaration-rename invariance.

## Properties and metrics

Judge configurations, named as in experiment 05 (harness and model are properties of those logged calls, not of this CPU run):

| Config | Judge LLM | Harness | Rubric |
|---|---|---|---|
| A | GPT-6 Astra (`gpt-6-astra`) | Codex CLI, Vals plan | `P4_CHECKLIST` |
| A2 | GPT-6 Astra (`gpt-6-astra`) | Codex CLI, Vals plan | `P8_STRICT_B` |
| B | Claude Opus 5.5 (`claude-opus-5-5`) | Claude Code CLI, Stanford plan | `P4_CHECKLIST` |

Score files: `experiments/05_renaming_and_inserted_bug_on_veribench_judge_outputs/results/scores_{A,A2,B}.csv` (opaque probe id, kind, task, level, score). Human ratings are read from the off-repo VeriBench experiment 100 `items.jsonl` (sha256 `ad9ed552fc2ffac6328e6010bf6be82215dd74f1eddd791622d433126877a96c`, 75 items, 25 tasks, 3 items per task). The script refuses any other file. That file is not copied into this repository. Outputs contain aggregates only: no per-rater row, no Lean source.

Rating vector of an item: counts of `p1`..`p5` rounded to `{0,1,2,3,4,5}`, missing ratings dropped, not imputed. Probe id `CRIT__` + `item` must match all 75 `CRIT` rows.

### Split

Leave-one-task-out over the 25 `CRIT` tasks. For held-out task `t`, the training items are the 72 `CRIT` items from the other tasks. No rating and no score from task `t` enters the fit for task `t`. The same split is used for the constant.

### Three forecasts

Let `s` be the judge score on `[0, 1]`.

1. `RAW`. Point mass at `r = clip(round(5 s), 0, 5)`. `RPS` and `TVD` use this point mass. `CE` uses the experiment 05 reporting convention `0.9` point mass plus `0.1` uniform, named `ce_smooth`. `RAW` is the uncalibrated reference, not the confirmatory forecast.
2. `CAL` (confirmatory). Cumulative-logit ordinal model, the experiment 07 `fit_ce_ordinal` parameterization: five ordered cutpoints (first cutpoint free, later gaps `exp` of unconstrained parameters, clipped to `[-20, 20]` before the exp), slope `b`, training loss = mean rating-weighted negative log likelihood plus `1e-3 * b^2`, optimized by L-BFGS-B from the marginal-threshold start with `b = 0`. Predicted probabilities are floored at `1e-3` and renormalized per row. Fit on the training items only. If the optimizer does not return a finite parameter vector, the run stops; nothing is imputed.
3. `CONST` (confirmatory baseline). Count-pooled training ratings with additive smoothing `alpha = 0.5`: `q_k = (c_k + 0.5) / (sum_j c_j + 3.0)`. One distribution for every held-out item. It does not depend on `s` and it does not use held-out ratings. The same `q` is used for `RPS`, `TVD`, and `CE`.

Sensitivity forecast `ISO`, not part of the H1-cal qualifier. On the training items, pool rating counts by the exact raw score. Sort the distinct scores. While two adjacent blocks have mean rating decreasing by more than `1e-12`, merge the first such pair (restart after each merge). Pool-adjacent-violators uses rating counts as weights. A query score uses its block if that score was seen (absolute difference `<= 1e-12`); otherwise the nearest training score, ties broken toward the lower score. The block distribution uses the same `alpha = 0.5` smoothing as `CONST`.

### Scores

Against the item's empirical rating distribution `p` (unsmoothed `c / sum c`):

- `RPS = mean_{k=0..4} (F_k - G_k)^2` with `F`, `G` the cumulative sums through category `k`. Lower is better.
- `TVD = 0.5 * sum_k |q_k - p_k|`. Lower is better. `TVD` is a distribution distance, not a strictly proper score. It is reported beside the proper scores.
- `CE = -sum_k p_k log(clip(q_k, 1e-12, None))`, natural log. Lower is better.

Point estimate: unweighted mean over the 75 held-out items (each item is scored once). 95% interval: task-cluster percentile bootstrap, 2,000 draws, seed `20261127`. A draw resamples the 25 tasks with replacement and averages the concatenated item losses, which matches the item mean because every task has 3 `CRIT` items. The paired difference `forecast minus CONST` is bootstrapped the same way on the paired item losses. A forecast beats `CONST` on a metric when that difference's interval lies entirely below 0.

A two-sided task sign-flip `p` of the paired `RPS` difference (2,000 flips, same seed, add-one `(count + 1) / (draws + 1)`) is descriptive. The qualifier is the interval, not the `p`.

### Properties, raw and calibrated

For probes of task `t`, the calibrated scalar is the expected rating of that task's held-out map, divided by 5: `ev(s) = (sum_{k=0..5} k q_k(s)) / 5`, using the same probabilities that enter `RPS` (`CAL` row-smooth, or `ISO` smoothed block). `RAW` properties use the logged score.

Epsilon `1e-9` for exact comparisons. Material band `0.05`, the experiment 05 tolerance.

Per configuration, raw and again for `CAL` and for `ISO`:

- `V_bug_exact`: share of bug probes with `score(bug) >= score(ID1) - 1e-9`. Strict share `>` and exact-tie share (`absolute difference <= 1e-9`) reported beside it. `n` and the task-cluster interval on `V_bug_exact`.
- `V_bug_material`: share with `score(bug) >= score(ID1) - 0.05` (the bug is not materially worse).
- `RN_d` share with absolute change `> 0.05`, and the share with absolute change `> 1e-9`. Same for `RN_b` and for pooled `RN`.
- Repeat noise: share of tasks with `|ID2 - ID1| > 0.05`. Excess = `RN_d` share(`> 0.05`) minus that noise share. The excess interval resamples the union of the `RN_d` tasks and the noise tasks once, and computes both shares on that draw.
- Flip counts: bug pairs and `RN_d` pairs whose `> 0.05` label differs between `RAW` and `CAL`, and between `RAW` and `ISO`.

Each leave-one-task-out fit records slope `b`. Report the number of tasks with `b > 0` and the minimum `b`. A positive slope makes `ev` strictly increasing, so exact ties and exact order are unchanged by `CAL`; the `0.05` labels need not be. That invariance is checked in the output, not assumed by the verdict. `ISO` can merge distinct scores, so it can create ties.

`ID1` mean must be at least `0.6` or the configuration is `NOT_VALID` and out of the verdict. This is the experiment 05 rule.

### Alignment gate (the run stops before a verdict if it fails)

Recompute the experiment 05 in-sample headlines with that experiment's own constant (unweighted mean of the 75 item distributions, fit on all 75 items) and that experiment's point-mass / smoothed-`CE` definitions. Absolute error at most `0.0005` on `RPS` and `TVD`, and at most `0.005` on `CE`, against A `0.395 / 0.859 / 3.53`, A2 `0.390 / 0.897 / 3.68`, B `0.282 / 0.863 / 3.55`, constant `0.077 / 0.491 / 1.73`.

Recompute raw `V_bug_exact`, raw exact-tie share, raw noise share, raw `RN_d` share(`> 0.05`), and `ID1` mean. Absolute error at most `0.001` against the values in the prior-exposure list.

### What would falsify it

A configuration qualifies when it is valid and the `CAL` minus `CONST` `RPS` interval lies entirely below 0. `TVD` and `CE` are reported the same way. `all3` means all three intervals lie entirely below 0. `all3` does not define the qualifier.

- H1-cal supported: at least one qualifying configuration has `V_bug_exact` on `CAL` `>= 0.20` with interval lower bound `> 0.05`, or `RN_d` excess on `CAL` `>= 0.20` with excess interval lower bound `> 0.05`.
- H1-cal falsified: every qualifying configuration has `CAL` `V_bug_exact` `< 0.10` and `CAL` `RN_d` share(`> 0.05`) `< 0.10`.
- H1-cal untestable: no configuration qualifies. The proper-score arm stays untestable after calibration, which is a result.
- Otherwise mixed.

`ISO` does not qualify or disqualify. With three logged configurations this is a pilot about those configurations.

## Data

Public score CSVs above. Off-repo `items.jsonl` above, read-only, sha-pinned, not committed. No new model calls.

## Baselines

`CONST`, fit on the training tasks only. `RAW`, the uncalibrated point mass, as the experiment 05 reference. `ISO`, the monotone pooled-score sensitivity. pass@k and boxed accuracy do not apply to these Lean specification scores.

## Models and harnesses

No model is called in this experiment. The scores being re-scored were produced by the harnesses in the table above.

## Cost estimate

`$0` out of pocket. API-equivalent `$0`. No plan lane is used.
