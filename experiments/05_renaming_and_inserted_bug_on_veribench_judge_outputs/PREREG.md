# PREREG: 05_renaming_and_inserted_bug_on_veribench_judge_outputs (date: 2026-11-24)

Committed before any judge call and before any probe is built. Not changed after the first run; later changes go in a dated "Amendment" section.

Numbering note: AT-3450 asked for "experiment 04"; experiments 03 (done) and 04 (equivalence, order, gaming; done) were pushed to main by Brando's own agents while this was being planned, so this experiment is 05. Experiment 03's PREREG was already run by them (zero model calls), so it is not run again here.

## Prior-exposure disclosure
Experiment 03 (Brando's agents) already measured an inserted-bug violation rate `V_bug` and a binder-renaming rate `V_alpha` (n = 3, too few) from the cached VeriBench judge outputs of `brando90/cert-judge` experiments 39 and 42. We have seen the summary line of those results, not the numbers behind this experiment's new calls. This experiment makes NEW judge calls so that renaming gets enough probes, and so that the human correlation and both properties are measured on the same judge configuration, in the same run.

## Hypothesis (one sentence)
H1: a judge configuration whose scores correlate significantly with the human expert ratings of VeriBench Lean files still fails properties that ignore the gold reference: its score changes when identifiers are renamed (beyond its own repeat noise), and a gold file with an inserted bug scores as high as the unmodified gold file in at least 20% of pairs.

## Properties and metrics
Task unit: 25 VeriBench tasks (24 with a Lean-valid gold). All 95% intervals: task-cluster percentile bootstrap, 2,000 draws, seed 20261124. Score scale [0, 1] (the judge's own JSON score).

Items per judge configuration (probes are built deterministically from the gold Lean files; a probe enters only if `lake env lean` on skampere1 accepts it; counts are reported):
- `CRIT` (75 items): the 75 VeriBench candidate files vs their gold (experiment 100 items, as in cert-judge experiment 39). Metric `rho_human`: Spearman between the judge score and `target_expert` (the 5-rater VeriBench expert target used by experiments 39 and 03; the same one, so numbers are comparable). Also report Spearman with `target_all5`.
- `ID1`, `ID2` (24 + 24): the gold as its own candidate, judged twice in different batches. `ID1` is the paired baseline. The gap `|ID2 - ID1|` is the judge's repeat noise floor.
- `RN_b`: the gold with one explicit theorem binder renamed to a fresh identifier (cert-judge `alpha_rename_bound_identifier`), Lean-validated. Property (a): the score must not change.
- `RN_d`: the gold with each file-local `def` renamed to an opaque fresh identifier (`ptiFn1`, `ptiFn2`, ...) at every occurrence, Lean-validated; if the full rename does not compile, only the first `def` is renamed; if that fails the probe is dropped. Property (a).
- `BUG` (37 probes in cert-judge experiment 39's validated `bm_gold` ladder, levels 1 to 4; an operator swapped such as `+` to `-`, `<=` to `<`): the buggy gold as candidate. Property (c): must not score as high as `ID1`.
In every call the reference shown to the judge is the gold's last theorem block (cert-judge `legacy.last_theorem_block`) and the candidate is the whole file, as in experiment 39.

Metrics per judge configuration, each with a 95% interval:
- `rho_human` (Spearman, 75 items, task-cluster bootstrap); "significant" means the lower bound > 0.
- `R_rn` = share of valid rename probes (`RN_b` and `RN_d` pooled, also reported separately) with `|score(RN) - score(ID1)| > 0.05`; `R_noise` = share of golds with `|ID2 - ID1| > 0.05`; excess `E_rn = R_rn - R_noise`. Mean `|Δ|` and mean signed Δ also reported.
- `V_bug` = share of bug probes with `score(BUG) >= score(ID1)` of the same task (strict `>` and tie `=` reported separately); mean drop `ID1 - BUG`.
- Alignment control: `mean(ID1)` must be at least 0.6 (a judge that does not credit the gold with itself is not a valid judge for this test); otherwise the configuration is marked `NOT_VALID` and excluded from the H1 verdict.

## Data
- Candidates, golds, human targets: the VeriBench clone on skampere1 (`experiments/100_karpathy_recipe_tc_judge_search_human_train_val_test/inputs/items.jsonl`, 75 items, `target_expert`, `target_all5`; read-only).
- Gold Lean files, validated bug probes (`bm_gold`) and the Lean project: `brando90/cert-judge` experiment 39 (`private/` on skampere1; read-only).
- No raw Lean source, no raw per-rater ratings and no rationales that quote source are committed to this public repository. The repository gets the per-item scores keyed by an opaque probe id, the item-to-target mapping as aggregated numbers, the scripts and the figures.
- New model calls: yes (below).

## Protocol of the judge calls
- Prompt: cert-judge `P4_CHECKLIST` rubric text, unchanged. To make about 200 items per configuration affordable on Brando's plans (each router call costs about 100 s of start-up), each call carries 6 independent items, each with its own reference and candidate, and asks for a JSON list of `{"item": i, "score": x}`; the judge is told to score every item on its own. Items in one call are always from 6 different tasks and from a seeded shuffle (seed 20261124), so an original and its renamed or bugged copy are never in one call. The same batching holds for every item type, so the paired comparisons are fair. This batching differs from experiment 39's one-item calls; it is a stated deviation and a limit of the test.
- Calls run through Brando's plan router on skampere1 (Vals and Stanford lanes first, then personal; no agy, no Llama 4, no key printed; the lane that ran is recorded per call).
- A call whose reply cannot be parsed into 6 scores is retried once; items still without a score are reported as missing, never imputed.

## Baselines
- Human expert target (`target_expert`) as the correlation target; repeat noise (`ID2` vs `ID1`) as the baseline for renaming; `ID1` as the baseline for bugs.
- pass@k and boxed accuracy do not apply to Lean theorem-specification adequacy.

## What would falsify it
- H1 supported: at least one valid judge configuration with significant `rho_human` (lower bound > 0) has `E_rn >= 0.20` (lower 95% bound > 0.05) OR `V_bug >= 0.20` (lower 95% bound > 0.05).
- H1 falsified: every valid configuration with significant `rho_human` has `E_rn < 0.10` AND `V_bug < 0.10`.
- Otherwise mixed. If no configuration has significant `rho_human`, the correlation arm is reported as "no judge to test" and only the two properties are reported per configuration (descriptive).
- With 2 judge configurations (or fewer if a lane is out), this is a pilot: a verdict is about those configurations only.

## Models and harnesses
- Judge LLMs, each in the harness the router gives it (names recorded per call): (A) GPT-6 Astra in the Codex CLI on the Vals plan (`vals-codex`), run now; (B) Claude Opus 5.5 in the Claude Code CLI on the Vals or Stanford plan (`vals-claude-opus`, then `stanford-claude`), run when that lane reopens (Vals Claude is paused until 4:40 PM PST; Stanford Claude until 4:10 PM PST). Judge agents with tools and solver LLMs are not used. The CLIs run in an empty workspace and are told to answer with the JSON only.
- Analysis: Python 3 (numpy, scipy, matplotlib; grid lines on) in `scripts/analysis.py`; probe building and judging scripts in `scripts/` (run on skampere1).

## Cost estimate
About 200 items per configuration in about 35 calls; roughly 0.6M input tokens and 0.1M output tokens per configuration, so about $4 to $8 API-equivalent per configuration and about $8 to $15 for two. Out of pocket: $0 (Brando's Vals plans; no overflow, no extra charge). Time: about 1 hour per configuration.

## Amendment 1 (2026-11-24, 12:25 PST): headline metrics are RPS, TVD and CE, correlation is secondary
Made after Brando's 12:18 instruction that correlation must not be the headline metric. At this time the first judge batch had been launched but no judge score had been read or analysed. Changes (everything else in this PREREG stays):
- Headline human-agreement metrics per judge configuration on the 75 `CRIT` items: `RPS`, `TVD` and `CE` of the judge's rating forecast against the empirical distribution of the human raters' 0-5 ratings of that item (the available ratings of the 5 raters; missing ratings are not imputed). The judge's score `s` becomes a point-mass forecast at the rating level `r = round(5 s)`. `RPS = (1/5) * sum_{k=0..4} (F_k - O_k)^2` over the cumulative distributions of forecast and humans; `TVD = 0.5 * sum_k |p_k - q_k|`; `CE = -sum_k q_k log p_k` with the forecast smoothed to `0.9` point mass plus `0.1` uniform over the 6 levels (a pure point mass has infinite CE; smoothed CE is a reporting convention, not a calibrated forecast). Reference forecasts for context: the constant forecast equal to the pooled human distribution over all 75 items (fit on the same items, so it flatters the constant), and the uniform forecast.
- "Significant human agreement" for the H1 arm now means: `RPS(judge) < RPS(constant)` with the 95% task-cluster interval of the difference (judge minus constant, 2,000 draws, seed 20261124) entirely below 0. Spearman and Pearson with `target_expert` and `target_all5` stay as secondary, and H1 is also reported under the original reading (Spearman lower bound > 0) so both readings appear.
- Figures and RESULTS lead with RPS, TVD, CE; the renaming and bug properties are unchanged.
