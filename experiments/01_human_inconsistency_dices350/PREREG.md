# PREREG: 01_human_inconsistency_dices350 (date: 2026-11-24)

Committed before any run. Not changed after the first run; later changes go in a dated "Amendment" section.

## Hypothesis (one sentence)
H2 (cross-rater part): human raters disagree with each other so much, even on one fixed item set, that a judge fitted to the human mean inherits a large noise ceiling, so "fit the humans" is a weak target on its own.

## Properties and metrics
Human property tested: inter-rater consistency (a human label should be reproducible by another human). The within-rater repeat and order properties of the README (d, e) are NOT testable here: DICES has one rating per rater and item and no repeats. They stay open (see README of this folder when written).
Score map: No = 0, Unsure = 0.5, Yes = 1 on the DICES column Q_overall (is the conversation unsafe overall).
- E1 split-half reliability: 2000 random splits of the 123 raters into 61 and 62; per split, Pearson correlation over the 350 items between the two halves' item means; report the mean, the 2.5 and 97.5 percentiles over splits, and the Spearman-Brown corrected value 2r/(1+r) as the reliability of the full pool.
- E2 label agreement: per split, share of items whose modal label (No, Unsure, Yes) is the same in both halves (a tie for the mode counts as a disagreement); same summary.
- E3 minority share: per item, share of raters who pick the less common of Yes and No. Report the share of the 350 items with minority share >= 10% and >= 25%. Also the share of items on which raters span more than one point (at least one Yes and at least one No).
- E4 what a judge fitted to the human mean inherits:
  (a) oracle judge = the full-pool item mean (the best possible fit). Its Pearson correlation with a single rater's 350 labels, and its agreement (mean over raters) of its rounded label with that rater's label; report the mean over the 123 raters and the 2.5 and 97.5 percentiles over raters.
  (b) noise ceiling of fitting k raters per item: for k = 1, 3, 5, 10, 25, 61, 500 draws of k raters; the item mean over those k raters is the "fitted judge"; its Pearson correlation with the mean of the other raters (disjoint). Report the mean and 95% interval over draws.
Seed 20261124. Intervals are over rater sampling, not over items.

## Data
DICES-350 (Google Research, github.com/google-research-datasets/dices-dataset, file 350/diverse_safety_adversarial_dialog_350.csv): 350 conversations, 123 raters, 43,050 ratings, every rater rated every item. Downloaded 2026-11-24. No new model calls.

## Baselines
Human: this experiment. Judge LLM, pass@k, boxed accuracy: not part of this experiment (no model calls); the oracle judge in E4 is the upper bound for any judge LLM fitted to the human mean.

## What would falsify it
H2 (cross-rater part) is falsified if the corrected full-pool reliability in E1 is >= 0.95 AND the mean agreement of the oracle judge with a single rater in E4a is >= 0.95. It is supported if E4a agreement is < 0.80. Between 0.80 and 0.95: reported as mixed, no claim.

## Models and harnesses
None. Python 3 with numpy and scipy, script analysis.py in this folder.

## Cost estimate
$0 (no model calls, no tokens on any account).
