# PREREG: 03_human_corr_vs_axiomatic_properties_cert_judge (date: 2026-11-24)

Committed before any run. Not changed after the first run; later changes go in a dated "Amendment" section.

## Prior-exposure disclosure
Zero-model-call reanalysis of existing harvested judge scores and human ratings from `brando90/cert-judge` (Experiments 39 and 42, plus `experiments/human_labels/*.tsv` read-only off-repo; no raw human ratings, rater names, Lean source files, or credentials are copied or committed to this public repository). Because these are previously collected datasets, thresholds and hypotheses reflect domain familiarity rather than a fresh prospective sample; fresh zero-exposure testing belongs in a later experiment.

## Hypothesis (one sentence)
H1: Real judge LLMs and judge agents with high rank correlation to human expert ratings (`heldout` Spearman >= 0.70) still violate basic axiomatic properties (bug-insertion monotonicity, missing-specification monotonicity, semantic weakening, vacuity, over-constraint, formatting invariance, or repeat consistency), while H2: on the same Lean 4 theorem-judging domain, human raters themselves show substantial cross-rater disagreement and within-human rubric-shift inconsistency.

## Properties and metrics

### Part A (H1): Human correlation vs. axiomatic property violations across 60 Lean 4 judges/controls
Units (60 total, from `cert-judge` Experiments 39 and 42):
- 50 single-call `vb103_judge` judge LLMs across 6 models (`openai/gpt-6-sol-high`, `anthropic/claude-sonnet-5-medium`, `openai/gpt-6-luna-high`, `google/gemini-3.1-pro-preview-high`, `alibaba/qwen3.6-27b-thinking` × 9 prompts `D0, R4, H0, H1, H2, P1, P4, P8, LEX`; `anthropic/claude-opus-5-5-no-fallback` × 5 prompts `H2, H0, D0, P4, LEX`).
- 6 `terminus2` judge agents across 3 models (`openai/gpt-6-luna-high`, `anthropic/claude-sonnet-5-medium`, `alibaba/qwen3.6-27b-thinking` × 2 prompts `H2, D0`).
- 4 deterministic controls (`harness = none`: `ctl|count_ratio`, `ctl|editdist`, `ctl|name_jaccard`, `ctl|random`).

Human-correlation metric (`rho_human`):
- Primary: `criteria.heldout.spearman` from E39 `units_train_val.json` / `units_test.json` (Spearman rank correlation on 45 held-out VeriBench candidate–gold pairs across 15 `val+test` tasks against the 5-rater VeriBench expert target; 95% task-cluster CI over 10,000 draws; requires `coverage >= 0.90`).
- Secondary: `criteria.all_items.spearman` (75 items / 25 tasks) and `criteria.heldout.pearson` / `criteria.heldout.kendall`.

Axiomatic property violation rates (computed directly from raw `harvest/*.json`, restricted to `valid == True` probes in `probe_manifest.json` / `semantic_probe_manifest.json` and `status in ("measured", "deterministic")`; normalized score scale `[0, 1]`; multi-repeat 3-repeat probes (`E39 ident`, `E39 bm_gold`, `E39 crit`, and all `E42` probes) take median over repeats requiring `>= 2` scored repeats; single-repeat probes (`E39 ms_gold`, `E39 blank`, `E39 alpha`) take the 1 scored repeat; each metric requires `>= 80%` valid-probe coverage else `NOT_COMPUTABLE`):
- `V_bug` (README property c — inserted bug, E39 `bm_gold` levels 1–4 paired with same-task E39 `ident` level 0, 37 valid probes across 11 tasks): share with `score(buggy) >= score(ident)` (also report strict `score(buggy) > score(ident)` and tie `score(buggy) == score(ident)` separately).
- `V_ms` (README property b — missing spec, E39 `ms_gold` levels 1–4 paired with same-task E39 `ident` level 0, 76 valid probes across 24 tasks): share with `score(missing_spec) >= score(ident)` (also report strict `>` and tie `==`).
- `V_weak` (README property b/c — semantic weakening, E42 `weak` levels 1–3 paired with same-task E42 `ident` level 1, 57 valid probes across 19 tasks): share with `score(weakened) >= score(ident)` (= `1 - E42.weakening`).
- `V_vac` (README property b/c — vacuity, E42 `vac_true` + `vac_hyp` paired with E42 `ident` level 1, 38 valid probes across 19 tasks): share with `score(vacuous) >= score(ident)` (= `1 - E42.vacuity`).
- `V_overc` (README property b/c — over-constraint, E42 `overc` levels 1–2 paired with E42 `ident` level 1, 37 valid probes): share with `score(overc) >= score(ident)` (= `1 - E42.overconstraint`).
- `V_fmt` & `fmt_mean_gap` (README property a-family — formatting invariance under comments/layout edits, E42 `fmt_comments` + `fmt_layout` paired with E42 `ident` level 1, 38 valid probes across 19 tasks): `V_fmt` = share with `|score(fmt) - score(ident)| > 1e-12`; `fmt_mean_gap` = mean `|score(fmt) - score(ident)|` (= `E42.formatting`). Note: E39 `alpha` (binder renaming) has only `n = 3` Lean-valid probes and `reorder` has `n = 0`; report `V_alpha` descriptively with `n = 3` caveat and mark `reorder` as unavailable (`n = 0`).
- `V_rep` (README property d — repeat inconsistency, E39 `crit` 75 candidate–gold pairs × 3 repeats across 25 tasks): among items with `>= 2` scored repeats, share with `max(reps) - min(reps) > 1e-12`; also report mean within-item range, within-item population SD, and `ident` repeat inconsistency on the 24 gold tasks.
- `V_blank` (sanity check, E39 `blank` paired with E39 `ident`, 25 valid probes): share with `score(blank) >= score(ident)` and mean `score(blank)`.

Uncertainty and cross-judge checks:
- 95% percentile task-cluster bootstrap intervals (resampling tasks with replacement, 2,000 draws, seed `20261124`) for every per-judge property violation rate.
- Cross-judge Spearman correlation across the 56 real judge configurations (`vb103_judge` + `terminus2`) between `rho_human` (`heldout`) and each property violation rate, with 95% unit bootstrap CI (2,000 draws) and two-sided permutation `p-val` (5,000 draws, seed `20261124`).
- Alignment control check: verify recomputed E42 `1 - V_weak`, `1 - V_vac`, `1 - V_overc`, and `fmt_mean_gap` match `E42/results/semantic_screen.json` within `1e-9` across all 60 units; fail loudly if any mismatch exceeds `1e-9`.

### Part B (H2): Human inconsistency on the same Lean 4 domain
Source: `cert-judge/experiments/human_labels/` (read-only off-repo; scores on 0–5 integer scale across 75 candidate–gold pairs from 25 tasks × 3 solver agents; never mix stacked concatenations `expert_harsh_set_data.tsv` / `lenient_set_data.tsv` with per-rater files):
- Fixed-rubric 5-rater pool (`participant1_data_correct_prompt.tsv` [74/75 scored], `participant2_data_pass_1.tsv` [75/75], `participant3_data_pass_1.tsv` [44/75], `participant4_data_pass_1.tsv` [75/75], `participant5_data_pass_1.tsv` [75/75]; report exact `n` per pair, never impute missing ratings):
  - All 10 unordered rater pairs: Spearman `rho`, exact agreement `|a - b| == 0`, disagreement shares `|a - b| >= 1` and `|a - b| >= 2`, mean absolute difference (MAD), and signed difference.
  - Per-item cross-rater range (`max - min` across all available raters on items with `>= 2` ratings): share with range `>= 1`, `>= 2`, `>= 3`, mean range, and mean item SD.
  - Leave-one-rater-out (LORO): Spearman `rho` and MAE of each rater against the mean of the other available raters on items where at least 2 total raters scored the item.
- Within-human rubric/strictness shifts (same annotator on the 75 pairs / 25 tasks):
  - `old_prompt/participant1_data_pass_1.tsv` vs `old_prompt/participant1_data_pass_2.tsv` (strictness/re-read shift under old prompt), and `participant1_data_correct_prompt.tsv` vs each `old_prompt` pass (rubric-wording shift): Spearman `rho`, exact agreement `|a - b| == 0`, `|a - b| >= 1`, `|a - b| >= 2`, MAD, and signed bias.
- Uncertainty for Part B: 95% percentile task-cluster bootstrap intervals (resampling the 25 `gold_file` tasks with replacement, 2,000 draws, seed `20261124`).

## Data
- `/Users/brandomiranda/cert-judge/experiments/39_label_free_proxy_index_train_val_test_over_judge_families/results/` (`panel.json`, `probe_manifest.json`, `units_train_val.json`, `units_test.json`, 60 `harvest/*.json` files).
- `/Users/brandomiranda/cert-judge/experiments/42_semantic_weakening_overconstraint_vacuity_probes_on_e39_judges/results/` (`semantic_probe_manifest.json`, `semantic_screen.json`, 60 `harvest/*.json` files).
- `/Users/brandomiranda/cert-judge/experiments/human_labels/` (`participant1_data_correct_prompt.tsv`, `participant2_data_pass_1.tsv`, `participant3_data_pass_1.tsv`, `participant4_data_pass_1.tsv`, `participant5_data_pass_1.tsv`, `old_prompt/participant1_data_pass_1.tsv`, `old_prompt/participant1_data_pass_2.tsv`).
- New model calls: no.

## Baselines
- Human 5-rater pool and same-annotator shift baselines (Part B).
- 50 single-call `vb103_judge` judge LLMs, 6 `terminus2` judge agents, and 4 deterministic non-LLM controls (`ctl|count_ratio`, `ctl|editdist`, `ctl|name_jaccard`, `ctl|random`).
- `pass@k` and `boxed accuracy` do not apply to Lean 4 theorem-specification adequacy and are left to coding/math experiments (`02` and planned prospective runs).

## What would falsify it
- **H1**: Supported if at least 1 real judge configuration (`vb103_judge` or `terminus2`) with `heldout` Spearman `>= 0.70` (`coverage >= 0.90`) has `V_bug >= 0.20` OR `V_weak >= 0.20` OR `V_rep >= 0.20` with lower 95% task-cluster CI bound `> 0.05`. Falsified if every real judge with `heldout` Spearman `>= 0.70` satisfies `V_bug < 0.10`, `V_ms < 0.10`, `V_weak < 0.10`, `V_vac < 0.10`, `fmt_mean_gap <= 0.05`, and `V_rep < 0.10`. Mixed otherwise.
- **H2**: Supported if the fixed-rubric 5-rater mean pairwise exact agreement is `< 0.50` OR the share of items with cross-rater range `>= 2` (on 0–5 scale) is `>= 0.50` OR the same-human rubric/strictness shift exact agreement is `< 0.50` (with supporting 95% task-cluster CI). Falsified if fixed-rubric 5-rater mean pairwise exact agreement `>= 0.85` AND share of items with range `>= 2` is `<= 0.10` AND same-human pass/wording exact agreement `>= 0.85`. Mixed otherwise.

## Models and harnesses
- Original harvest harnesses & models (no new calls):
  - `vb103_judge` (single-call judge LLM): `openai/gpt-6-sol-high`, `anthropic/claude-sonnet-5-medium`, `openai/gpt-6-luna-high`, `google/gemini-3.1-pro-preview-high`, `anthropic/claude-opus-5-5-no-fallback`, `alibaba/qwen3.6-27b-thinking`.
  - `terminus2` (judge agent): `openai/gpt-6-luna-high`, `anthropic/claude-sonnet-5-medium`, `alibaba/qwen3.6-27b-thinking`.
  - `none` (deterministic controls): `ctl|count_ratio`, `ctl|editdist`, `ctl|name_jaccard`, `ctl|random`.
- Reanalysis harness: Python 3 (`numpy`, `scipy`, `matplotlib`) in `scripts/analysis.py`.

## Cost estimate
$0.00 (zero new model calls; no tokens spent on any account).

## Amendment (2026-11-24, post-Light QA)
1. **Observed-cluster human bootstrap support:** When resampling task clusters for a human comparison in Part B (`bootstrap_pair_metrics`, `summary_10_pairs`, `loro_summary`), each comparison resamples only its own observed task clusters (`25` tasks for the 5-rater pool and `P1_old_pass1` vs `P1_old_pass2`; `15` for pairs involving `P3`; `24` for `P1_correct` vs `P2/P4/P5`; `21` for exact-key `P1_correct` vs `old_prompt` overlaps), rather than resampling a union of mismatched filename aliases across files.
2. **Scope qualification for `V_bug`:** Because `E39` `bm_gold` mutates executable `def` implementations while `H0`/`H1`/`H2`/`R4`/`D0` score formal theorem-statement coverage (and compilation does not prove semantic spec loss), we additionally report `H1` sensitivity restricted strictly to theorem-statement weakening (`V_weak`) and repeat inconsistency (`V_rep`). Both `22/22` high-correlation judges (`rho_heldout >= 0.70`) still trigger `H1` without `V_bug`.
3. **Seed & dependence notes:** Both bootstrap and cross-configuration permutation routines use seed `20261124`; cross-configuration correlations across the `56` real judge units are conditional/exploratory because configurations share models, prompts, and tasks.
