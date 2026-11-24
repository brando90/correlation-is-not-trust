# RESULTS 05: renaming and inserted-bug properties of VeriBench judges (as of 2026-11-24, partial: configuration B pending)

Estimated cost: about $4 to $8 API-equivalent per configuration (my estimate from the PREREG; measured use per configuration was about 0.86M input tokens, 0.45M of them cached, and about 6k output tokens), $0 out of pocket: all calls ran on Brando's Vals plan (router lane `vals-codex`), no overflow, no extra charge, no agy, no Llama 4. Time: about 12 minutes of calls per configuration.

Harness and model: judge LLM GPT-6 Astra (`gpt-6-astra`) in the Codex CLI on the Vals plan, via Brando's plan router, 6 independent items per call, empty workspace. Configuration A = cert-judge `P4_CHECKLIST` rubric; configuration A2 = `P8_STRICT_B` rubric (added by PREREG Amendment 2 after A finished). Configuration B (Claude Opus 5.5, Claude Code CLI) was planned in the PREREG and is not run yet: the Vals and Stanford Claude lanes are paused until about 4:10 to 4:40 PM PST; this file is updated when it runs.

## Takeaway
- No judge configuration here agrees with the human experts, so the H1 test as pre-registered (a judge that agrees with humans still fails the reference-free properties) cannot be run on these two configurations: pre-registered verdict "no judge to test". Against the human rating distribution of the 75 rated files, both judges are far worse than a constant forecast: RPS 0.395 [0.313, 0.483] (A) and 0.390 [0.304, 0.485] (A2) vs 0.077 [0.055, 0.105] for the constant forecast and 0.088 for uniform; TVD 0.859 [0.789, 0.924] and 0.897 [0.840, 0.949] vs 0.491 [0.446, 0.535] for the constant; CE (smoothed) 3.53 [3.28, 3.77] and 3.68 [3.47, 3.89] vs 1.73 [1.65, 1.82] for the constant and 1.79 for uniform. The constant is the pooled human rating distribution over the same 75 files (one fixed forecast for every file, fit on the same items, so it is a favourable reference); RPS 0.077, TVD 0.491 and CE 1.73 are all for that one constant. Spearman with the expert target is about 0: -0.004 [-0.284, 0.281] (A), -0.062 [-0.369, 0.258] (A2). The judges give 61 of 75 rated files a score of 0.2 or less (A) and 71 of 75 (A2), while the humans' mean rating is 3.0 out of 5 (2.6 for the expert target), so a large part of the gap is level, not only ranking.
- Inserted bug: on gold files with one inserted operator bug, the bugged file scores at least as high as the unmodified gold file in 67.6% [40.5%, 91.4%] of 37 pairs (A) and 75.7% [55.3%, 93.8%] (A2); 64.9% are exact ties (A: 21 at 1.0 and 3 at 0.0; A2: all 24 at 1.0). Mean score drop is only 0.30 (A) and 0.14 (A2). This is a property failure on its own (an inserted bug is not penalized), but it does not test the hypothesis because the judges do not agree with humans.
- Renaming: both judges are invariant to identifier renaming within their own noise. Share of 38 renamed golds whose score moved by more than 0.05: 7.9% [0.0%, 18.0%] (A) and 5.3% [0.0%, 16.7%] (A2), against repeat noise on identical input of 20.8% [4.2%, 37.5%] (A: the same gold file got 1.0 once and 0.0 another time in 5 of 24 cases) and 4.2% [0.0%, 12.5%] (A2). Renaming is not where these judges fail; their repeat noise (A) is.
- Limits: 2 configurations of one model, 24 golds, 6-item batches (not experiment 39's one-item calls), point-mass forecasts with no calibration step for RPS, and the human target is the old expert ratings, not the newer corrected VeriBench-team ratings.

## Headline: agreement with the human rating distribution (75 rated candidate files, 25 tasks)
RPS, TVD and CE of the judge's rating (round(5 x score), point mass) against the empirical distribution of the human raters' 0-5 ratings of each file. CE uses a smoothed forecast (0.9 point mass plus 0.1 uniform), a reporting convention. 95% task-cluster bootstrap, 2,000 draws, seed 20261124.

| judge LLM (harness: Codex CLI, Vals plan) | RPS | TVD | CE (smoothed) | RPS minus constant |
|---|---|---|---|---|
| A: GPT-6 Astra, P4_CHECKLIST | 0.395 [0.313, 0.483] | 0.859 [0.789, 0.924] | 3.53 [3.28, 3.77] | +0.318 [+0.223, +0.407] |
| A2: GPT-6 Astra, P8_STRICT_B | 0.390 [0.304, 0.485] | 0.897 [0.840, 0.949] | 3.68 [3.47, 3.89] | +0.313 [+0.219, +0.404] |
| constant (pooled human distribution, fit on the same items) | 0.077 [0.055, 0.105] | 0.491 [0.446, 0.535] | 1.73 [1.65, 1.82] | 0 |
| uniform | 0.088 [0.065, 0.115] | 0.536 [0.496, 0.580] | 1.79 | n/a |

Secondary (correlation): Spearman with `target_expert`: A -0.004 [-0.284, 0.281], A2 -0.062 [-0.369, 0.258]; with `target_all5`: A 0.028 [-0.257, 0.280], A2 -0.040 [-0.375, 0.295]. n = 75, 6 distinct score values per judge.

## Properties that ignore the gold reference (24 Lean-valid golds; Lean 4 checked every probe)
Probes built: 24 gold-as-candidate (judged twice), 14 binder renames (`RN_b`, cert-judge alpha rename; 10 golds had no safe target), 24 declaration renames (`RN_d`), 37 bug probes (experiment 39 ladder, levels 1 to 4), plus the 75 rated files. No score missing.

| judge | gold scored 1.0 (ID1 mean) | repeat noise share (>0.05) | rename share (>0.05), pooled n=38 | rename excess over noise | bug scored >= gold (V_bug), n=37 | strict > | tie | mean drop |
|---|---|---|---|---|---|---|---|---|
| A | 0.917 | 0.208 [0.042, 0.375] | 0.079 [0.000, 0.180] | -0.129 [-0.261, -0.031] | 0.676 [0.405, 0.914] | 0.027 | 0.649 | 0.297 |
| A2 | 0.958 | 0.042 [0.000, 0.125] | 0.053 [0.000, 0.167] | +0.011 [0.000, 0.034] | 0.757 [0.553, 0.938] | 0.108 | 0.649 | 0.135 |

Rename by kind: `RN_b` 7.1% (A) and 7.1% (A2) of 14; `RN_d` 8.3% (A) and 4.2% (A2) of 24. V_bug by level (A / A2): L1 0.82 / 0.82, L2 0.67 / 0.67, L3 0.56 / 0.78, L4 0.63 / 0.75. Excluding the 4 pairs where the gold itself scored below 0.5: V_bug 0.64 (A) and 0.73 (A2), n = 33.

Pre-registered rule for H1: needs a configuration with significant human agreement. Neither has it under the headline reading (judge RPS minus constant is above 0, not below) nor under the Spearman reading (lower bound below 0). So: no judge to test; the two properties are descriptive for these judges (renaming holds, bug insertion fails).

Figure: results/fig_properties.png (RPS, TVD, rename share vs noise, V_bug; grid lines on).

## Caveats
- The bug probes are single operator swaps inside the gold's implementation; the judge's reference is the gold's last theorem block (experiment 39's protocol), so a judge that compares only theorem statements could miss them. A judge that sees the implementation might penalize them.
- Batches of 6 items per call differ from experiment 39's one-item calls. The original and its copies were never in the same call.
- The judges score the rated candidate files far below the humans' level; a calibration step (as in the cert-judge re-run) was not part of this PREREG, so RPS is for the raw rating. The conclusion "no better than constant" is therefore about raw scores, not about a calibrated judge.
- The human target is the old expert ratings (experiment 100 items); the newer corrected VeriBench-team ratings were not used.
- 3 extra calls were repeated by a driver bug (a log line was read as "no route"), about 51k input tokens; no score from them was used.

## Reproduce
`scripts/build_probes.py` (skampere1, needs the Lean project and cert-judge experiment 39 private probes), `scripts/judge_batches.py --config A --lanes vals-codex` and `--config A2 --rubric P8_STRICT_B`, `scripts/analysis.py`. The probe sources, the ratings and the raw model replies stay off this public repository; results/ holds opaque-id scores and the summary.
