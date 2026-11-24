# RESULTS 01: human inconsistency on DICES-350 (as of 2026-11-24)

Harness: Python 3 + numpy/scipy (analysis.py), no model calls, cost $0. Data: DICES-350, 350 conversations, 123 raters, every rater rated every item, column Q_overall (No/Unsure/Yes), sha256 63cb8620eb04e64957f9e45b1f7cb3d079c4eed8e8b28909f46e0b7038ce4301. Pre-registered in PREREG.md (committed before the run, with a dated post-run amendment, see below).

## Takeaway
- The mean of many humans is stable, a single human is not. Split-half reliability of the item mean is r = 0.952 [0.939, 0.961] (Spearman-Brown full pool 0.975), but one rater correlates only r = 0.49 [0.10, 0.68] with the pool mean, and a judge that matched the pool exactly would agree with one rater's Yes/No label on only 73% of items [36%, 88% across raters].
- Raters disagree on almost every item: the mean minority share (Yes vs No) is 24.8%; 90.6% of items have a minority of at least 10%, 45.4% of at least 25%; mean pairwise rater agreement 0.567.
- Fitting k raters per item caps what a judge can reach on held-out raters: r = 0.69 at k=3, 0.78 at k=5, 0.86 at k=10, 0.93 at k=25.
- Verdict (pre-registered rule, applied to the binary variant per the amendment): H2 cross-rater part SUPPORTED (single-rater agreement 0.73 < 0.80). Not tested: within-rater repeat and order consistency (DICES has no repeats), so the part of H2 about a human contradicting themselves stays open.

## Tables (95% intervals over rater sampling, 2000 splits / 500 draws, seed 20261124)
| Metric | Value [95%] |
|---|---|
| E1 split-half Pearson of item means | 0.952 [0.939, 0.961] |
| E1 Spearman-Brown full pool | 0.975 [0.968, 0.980] |
| E2 modal label agreement between halves | 0.921 [0.883, 0.946] |
| E3 items with minority share >= 10% / >= 25% | 0.906 / 0.454 |
| E3 mean minority share, mean pairwise agreement | 0.248, 0.567 |
| E4a oracle (pool mean) vs one rater, Pearson (leave self out) | 0.479 [0.089, 0.671] (1 of 123 raters constant, left out) |
| E4a oracle vs one rater, label agreement, rounded to No/Unsure/Yes (pre-registered, rounding artifact) | 0.396 [0.155, 0.545] |
| E4a amendment: binary oracle vs one rater, Yes/No ratings only (leave self out) | 0.732 [0.355, 0.876] |
| E4b fit k=1 / 3 / 5 / 10 / 25 / 61 raters, Pearson with held-out mean | 0.49 / 0.69 / 0.78 / 0.86 / 0.93 / 0.95 |

Only 6.3% of ratings are Unsure, so the rounded 3-label agreement mostly measures the rounding. Figure: results/fig_human_inconsistency.png.

## Caveats
- DICES is a safety-rating set (is the conversation unsafe), not code or proof quality; it shows the size of human disagreement on a subjective rating, not on VeriBench. VeriBench human ratings are sealed and off-repo, so the VeriBench version is the next step.
- Raters differ a lot (per-rater Pearson with the others: median 0.51, 5th percentile 0.20, max 0.69): some of the disagreement is a few low-quality or idiosyncratic raters.
- Intervals reflect rater sampling only.

Reproduce: `python analysis.py dices_350.csv results` (about 1 minute).
