# PREREG: 02_judge_properties_qwen_math_wording_and_monotonicity (date: 2026-11-24)

Committed before any run. Not changed after the first run; later changes go in a dated "Amendment" section.

## Hypothesis (one sentence)
H1: a judge LLM can correlate strongly with the gold reference (correct or wrong final answer) and still violate basic properties, here (P1) the same judge with a reworded rubric must give the same ordering of solutions, and (P2) a wrong solution must not score at least as high as a correct one to the same problem.

## Properties and metrics
Judges: Qwen2.5-Instruct 1.5B, 3B, 7B, 14B, each in two conditions: noref (the judge sees no gold answer) and ref (the judge sees the gold reference). 8 judge rows. Each row has two rubric wordings, files `_a` and `_b`; the judge score of a solution is the mean of a and b unless a property says otherwise.
- C (the correlation that people use to trust a judge): pooled Pearson correlation between the judge score and the gold correctness bit over all 176 x 64 solutions (stand-in for "correlation with humans": this data has no human labels, the gold correctness bit is the reference; see caveat). Also the mean within-problem Spearman correlation over problems with both correct and wrong solutions.
- P1 wording stability: (i) mean over problems of the Spearman correlation between the a scores and the b scores of the 64 solutions; (ii) wording flip rate: among within-problem (correct, wrong) pairs with different a-scores and different b-scores, the share where a and b order the pair differently.
- P2 monotonicity violation rate: among within-problem (correct, wrong) pairs, the share where the wrong solution scores >= the correct one (ties count 1/2), computed on the mean score and on a and b separately; average over problems that have both kinds, each problem weighted equally.
All intervals: cluster bootstrap over the problems, 2000 resamples, 95% percentile, seed 20261124.
Cross-judge: Spearman correlation across the 8 rows between C and P2 and between C and P1(ii) (n = 8, descriptive only).

## Data
In brando90/judge-free-eval (commit d1d53e8c88e32b3695d0c06379c129d65bde63ce): experiments/tfa-vs-judge/runs/bon/bon_math__meta-llama__Llama-3.1-8B-Instruct.jsonl (500 MATH-500 problems x 64 samples from Llama-3.1-8B-Instruct, gold correctness bit `oracle`, `level`) and runs/bon_scores_llama/math__Qwen__Qwen2.5-{1.5B,3B,7B,14B}-Instruct__{noref,ref}_{a,b}.jsonl (judge scores `escore`, 64 per problem, on 176 of the 500 problems). No new model calls.
Alignment control: the 14B ref judge must reach pooled AUROC >= 0.85 on the gold bit; if not, the judge files are not aligned to the sample order and the experiment stops with that finding.

## Baselines
Gold-reference correctness and per-problem pass rate are the reference (not rivals). Human labels, boxed-answer accuracy-as-metric and pass@k are the metrics this project asks about; here boxed correctness is the reference bit, so it cannot be a rival in the same table.

## What would falsify it
H1 is supported if at least one judge row with C at or above the median of the 8 rows has a P2 violation rate >= 0.20 or a wording flip rate >= 0.15. H1 is falsified if every row with C at or above the median has P2 < 0.10 and wording flip rate < 0.05. In between: mixed, no claim.

## Models and harnesses
Original scores: the judge-free-eval harness (scripts/bon_judge.py, vLLM) with the Qwen2.5-Instruct judges named above; solver LLM Llama-3.1-8B-Instruct. This experiment only re-analyzes those saved scores (Python 3, numpy, scipy).

## Cost estimate
$0 (no model calls, no tokens on any account).

## Caveat stated before the run
No human labels exist in this data. A positive H1 result here says: a judge with a high correlation to the gold bit still violates a property. The human-correlation version of H1 needs human ratings (sealed, off-repo for the VeriBench 75 outputs) and is the next experiment.
