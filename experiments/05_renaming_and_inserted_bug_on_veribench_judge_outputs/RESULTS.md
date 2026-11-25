# RESULTS 05: renaming and inserted-bug properties of VeriBench judges (as of 2026-11-24, three judge configurations: A, A2, B)

Estimated cost: $0 out of pocket (all calls ran on Brando's plans through the plan router; no overflow, no extra charge, no agy, no Llama 4). API-equivalent, my estimate: about $4 for B (Claude Opus 5.5: 0.75M cache-write, 0.44M cache-read, 12k output tokens, at list prices of $5 / $0.20 / $20 per million tokens from third-party pricing pages, not Anthropic's own page) and about $4 to $8 for each of A and A2 (GPT-6 Astra: about 0.86M input tokens each, 0.45M of them cached, about 6k output tokens; the $4 to $8 is the PREREG estimate, GPT-6 Astra list prices not looked up). About 12 minutes of calls per configuration.

Harness and model, one row per configuration, 6 independent items per call, empty workspace, probes and rated files identical in all three:
- A: GPT-6 Astra (`gpt-6-astra`) in the Codex CLI on the Vals plan (router lane `vals-codex`), cert-judge `P4_CHECKLIST` rubric.
- A2: same model, harness and lane, cert-judge `P8_STRICT_B` rubric (added by PREREG Amendment 2 after A finished).
- B: Claude Opus 5.5 (`claude-opus-5-5`) in the Claude Code CLI on the Stanford plan (router lane `stanford-claude`), `P4_CHECKLIST` rubric. Ran 2026-11-24 about 4:20 PM PST, 33 of 33 batches, 6/6 parsed, none missing.

## Takeaway
- Pre-registered verdict (both readings, per Amendment 1). Under the original reading (a judge whose Spearman with the experts has a lower bound above 0) there is one such judge, B (Claude Opus 5.5): Spearman 0.388 [0.006, 0.653], valid (it scores the gold 0.996 against itself), and it still fails the properties: an inserted bug scores at least as high as the unmodified file in 56.8% [40.5%, 76.3%] of 37 pairs (rule threshold 20%), and declaration renames move its score beyond repeat noise (excess 19.5 points [6.6, 34.3]; point estimate just under the 20-point threshold, lower bound above 5). So H1 is supported for this one configuration (pilot, one judge). Under the headline reading (proper scores) no configuration qualifies, so H1 is not testable there: B is also worse than a constant forecast, as in the next bullet.
- Proper scores against the human rating distribution of the 75 rated files, with the Spearman only as the secondary column: RPS A 0.395 [0.313, 0.483], A2 0.390 [0.304, 0.485], B 0.282 [0.227, 0.341], against 0.077 [0.055, 0.105] for the constant forecast (the pooled human distribution, one fixed forecast for every file, fit on the same items, so a favourable reference) and 0.088 for uniform. TVD 0.859 [0.789, 0.924], 0.897 [0.840, 0.949], 0.863 [0.806, 0.916] vs 0.491 [0.446, 0.535] for the constant. CE (smoothed) 3.53 [3.28, 3.77], 3.68 [3.47, 3.89], 3.55 [3.32, 3.75] vs 1.73 [1.65, 1.82] for the constant and 1.79 for uniform. Judge minus constant is above 0 for every judge on all three scores. Spearman with the expert target: A -0.004 [-0.284, 0.281], A2 -0.062 [-0.369, 0.258], B 0.388 [0.006, 0.653]. So the one judge that ranks the files better than chance is still 3.7 times worse than a constant forecast on RPS: correlation does not carry over to calibrated human-distribution agreement. The judges give 61 (A) and 71 (A2) of the 75 files a score of 0.2 or less while the humans' mean rating is 3.0 out of 5 (2.6 for the expert target), so a large part of the gap for GPT-6 Astra is level, not only ranking.
- Inserted bug: on gold files with one inserted operator bug, the bugged file scores at least as high as the unmodified gold in 67.6% [40.5%, 91.4%] of 37 pairs (A), 75.7% [55.3%, 93.8%] (A2) and 56.8% [40.5%, 76.3%] (B). Exact ties: 64.9%, 64.9%, 51.4%. Mean score drop is only 0.30 (A), 0.14 (A2) and 0.10 (B). Every configuration fails this property.
- Renaming: not uniform. Share of 38 renamed golds whose score moved by more than 0.05: A 7.9% [0.0%, 18.0%], A2 5.3% [0.0%, 16.7%], B 23.7% [11.6%, 38.9%], against repeat noise on identical input of 20.8% [4.2%, 37.5%] (A: the same gold got 1.0 once and 0.0 another time in 5 of 24 cases), 4.2% [0.0%, 12.5%] and 4.2% [0.0%, 12.5%]. GPT-6 Astra is invariant within its noise. For B, binder renames (14) never moved the score (0 of 14) but declaration renames to opaque names (24) moved it in 37.5% [20.8%, 58.3%] of cases, always downward (mean signed change -0.048; mean absolute change over all 38 renames only 0.034).
- Limits: three configurations of two models (not a sample of judges), 24 golds, 6-item batches (not experiment 39's one-item calls), point-mass forecasts with no calibration step for RPS, a 0.05 tolerance that B's finer score grid (12 distinct values) trips more easily than GPT-6 Astra's coarse one (my reading), and the human target is the old expert ratings, not the newer corrected VeriBench-team ratings.

## Headline: agreement with the human rating distribution (75 rated candidate files, 25 tasks)
RPS, TVD and CE of the judge's rating (round(5 x score), point mass) against the empirical distribution of the human raters' 0-5 ratings of each file. CE uses a smoothed forecast (0.9 point mass plus 0.1 uniform), a reporting convention. 95% task-cluster bootstrap, 2,000 draws, seed 20261124.

| judge LLM (harness: Codex CLI, Vals plan) | RPS | TVD | CE (smoothed) | RPS minus constant |
|---|---|---|---|---|
| A: GPT-6 Astra, P4_CHECKLIST | 0.395 [0.313, 0.483] | 0.859 [0.789, 0.924] | 3.53 [3.28, 3.77] | +0.318 [+0.223, +0.407] |
| A2: GPT-6 Astra, P8_STRICT_B | 0.390 [0.304, 0.485] | 0.897 [0.840, 0.949] | 3.68 [3.47, 3.89] | +0.313 [+0.219, +0.404] |
| B: Claude Opus 5.5, P4_CHECKLIST (Claude Code CLI, Stanford plan) | 0.282 [0.227, 0.341] | 0.863 [0.806, 0.916] | 3.55 [3.32, 3.75] | +0.205 [+0.140, +0.267] |
| constant (pooled human distribution, fit on the same items) | 0.077 [0.055, 0.105] | 0.491 [0.446, 0.535] | 1.73 [1.65, 1.82] | 0 |
| uniform | 0.088 [0.065, 0.115] | 0.536 [0.496, 0.580] | 1.79 | n/a |

Judge minus constant on TVD: A +0.368 [+0.271, +0.450], A2 +0.406 [+0.328, +0.473], B +0.372 [+0.294, +0.440]; on CE: A +1.81 [+1.46, +2.10], A2 +1.96 [+1.64, +2.22], B +1.82 [+1.49, +2.10]. Positive is worse for all three scores.

Secondary (correlation): Spearman with `target_expert`: A -0.004 [-0.284, 0.281], A2 -0.062 [-0.369, 0.258], B 0.388 [0.006, 0.653]; with `target_all5`: A 0.028 [-0.257, 0.280], A2 -0.040 [-0.375, 0.295], B 0.412 [-0.000, 0.686]. n = 75; 6 distinct score values for A and A2, 12 for B.

## Properties that ignore the gold reference (24 Lean-valid golds; Lean 4 checked every probe)
Probes built: 24 gold-as-candidate (judged twice), 14 binder renames (`RN_b`, cert-judge alpha rename; 10 golds had no safe target), 24 declaration renames (`RN_d`), 37 bug probes (experiment 39 ladder, levels 1 to 4), plus the 75 rated files. No score missing.

| judge | gold scored 1.0 (ID1 mean) | repeat noise share (>0.05) | rename share (>0.05), pooled n=38 | rename excess over noise | bug scored >= gold (V_bug), n=37 | strict > | tie | mean drop |
|---|---|---|---|---|---|---|---|---|
| A | 0.917 | 0.208 [0.042, 0.375] | 0.079 [0.000, 0.180] | -0.129 [-0.261, -0.031] | 0.676 [0.405, 0.914] | 0.027 | 0.649 | 0.297 |
| A2 | 0.958 | 0.042 [0.000, 0.125] | 0.053 [0.000, 0.167] | +0.011 [0.000, 0.034] | 0.757 [0.553, 0.938] | 0.108 | 0.649 | 0.135 |
| B | 0.996 | 0.042 [0.000, 0.125] | 0.237 [0.116, 0.389] | +0.195 [+0.066, +0.343] | 0.568 [0.405, 0.763] | 0.054 | 0.514 | 0.096 |

Rename by kind: `RN_b` 7.1% (A), 7.1% (A2) and 0.0% (B) of 14; `RN_d` 8.3% (A), 4.2% (A2) and 37.5% [20.8%, 58.3%] (B) of 24. V_bug by level (A / A2 / B): L1 0.82 / 0.82 / 0.82, L2 0.67 / 0.67 / 0.33, L3 0.56 / 0.78 / 0.56, L4 0.63 / 0.75 / 0.50. Excluding the pairs where the gold itself scored below 0.5: V_bug 0.64 (A, n = 33) and 0.73 (A2, n = 33); for B all 37 golds qualify, 0.57.

Pre-registered rule for H1 (both readings, Amendment 1). Original reading, Spearman lower bound above 0: only B qualifies (lower bound 0.006, barely above 0; A and A2 do not), and it meets the V_bug arm (0.568, lower bound 0.405, thresholds 0.20 and 0.05) and, on the rename arm, a lower bound of 0.066 above 0.05 with a point estimate of 0.195 just under 0.20: H1 supported for B. Headline reading, judge RPS minus constant entirely below 0: no configuration qualifies (all three are above 0): H1 untestable. The properties are descriptive facts for every configuration: bug insertion is not penalized by any of them; renaming holds for GPT-6 Astra and fails mildly for Claude Opus 5.5 on declaration renames.

Figure: results/fig_properties.png (RPS, TVD, rename share vs noise, V_bug for A, A2 and B; grid lines on).

## Caveats
- The bug probes are single operator swaps inside the gold's implementation; the judge's reference is the gold's last theorem block (experiment 39's protocol), so a judge that compares only theorem statements could miss them. A judge that sees the implementation might penalize them.
- Batches of 6 items per call differ from experiment 39's one-item calls. The original and its copies were never in the same call.
- The judges score the rated candidate files far below the humans' level; a calibration step (as in the cert-judge re-run) was not part of this PREREG, so RPS is for the raw rating. The conclusion "no better than constant" is therefore about raw scores, not about a calibrated judge.
- The human target is the old expert ratings (experiment 100 items); the newer corrected VeriBench-team ratings were not used.
- 3 extra calls were repeated by a driver bug (a log line was read as "no route"), about 51k input tokens; no score from them was used.

## Reproduce
`scripts/build_probes.py` (skampere1, needs the Lean project and cert-judge experiment 39 private probes), `scripts/judge_batches.py --config A --lanes vals-codex` and `--config A2 --rubric P8_STRICT_B`, `--config B --lanes stanford-claude,vals-claude-opus`, `scripts/analysis.py --scores scores_A.jsonl scores_A2.jsonl scores_B.jsonl`. The probe sources, the ratings and the raw model replies stay off this public repository; results/ holds opaque-id scores and the summary.
