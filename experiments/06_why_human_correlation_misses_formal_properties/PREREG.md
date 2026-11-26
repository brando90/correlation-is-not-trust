# PREREG: 06_why_human_correlation_misses_formal_properties (date: 2026-11-26)

Commit this file before any run. Do not change it after the first run; add a dated "Amendment" section instead.

The CPU arm reads only public logs already in this repository. It makes no model calls. The API arm is specified below and is not run under this preregistration; running it spends a plan call and waits for a later decision.

## Prior-exposure disclosure

Experiments 01–05 are already on main, including `docs/MEMO_exp04_05_11-25-2026.md` (commit `6d2e9be`). This design uses those published aggregates as background and does not treat them as new results. In particular, the following were already known while the cuts below were chosen:

- Experiment 05, configurations A, A2, and B: ID1 means about 0.92 to 1.00, bug tie shares about 0.51 to 0.65, mean drops about 0.10 to 0.30, and V_bug about 0.57 to 0.76. Any tie share above one half forces the median of (ID1 − bug) to be 0 when ties have drop 0. That median is a check, not a confirmatory result.
- Experiment 05: A and A2 place 61 and 71 of 75 natural files at 0.2 or below. B is the only configuration whose Spearman with `target_expert` has a lower bound above 0 (0.388 [0.006, 0.653]).
- Experiment 03: marginal Spearman of held-out human correlation with V_weak is −0.747, and with V_bug is −0.164, on 56 judge configurations. All 22 configurations with held-out Spearman at least 0.70 have V_bug strict-greater share 0 (every logged bug violation there is a tie).
- Experiment 04: the two frontier `M1_deployed_tc_v1` rows (`claude-sonnet-4-6`, `gpt-5.4`) were visible in `equiv_invariance_table.csv` while this file was written. On those two rows the point estimates of `harsh_flip_eq10` and `V_equiv` are equal, so their ratio is 1. That ratio is not a confirmatory test. Ratios for the other `M1` rows were not read while writing this file.

Not previously computed, and the confirmatory targets of the CPU arm: the share of experiment 05 bug pairs whose score drop exceeds 0.20; the partial Spearman of human correlation with V_weak given V_bug, and with V_bug given V_weak; the median across the 22 high-correlation configurations of the mean within-item repeat range; the task-level Spearman of natural-file scores with bug drops.

## Hypothesis (one sentence)

Human correlation coexists with formal-property failure because the correlation does not constrain those properties: repeat correlation is a correlation of medians, bug probes are files the correlation never sees, and on the logged bug files the typical failure is not a material score drop, while statement weakening is the component that actually tracks the correlation.

## Properties and metrics

Two different claims are easy to mix up. The lemmas say the statistics are free to diverge. The three confirmatory checks say whether, on these logs, the divergence is a missing material penalty, a large repeat swing, and a split between statement weakening and implementation-bug ties.

### Lemmas (constructed or by file identity; not fit to a judge)

- **Disjoint support.** Let C be the files that enter a Spearman correlation and P the files that enter a property. If C and P are disjoint, replacing the scores on P does not change the Spearman on C. Experiment 05's `CRIT` probe ids and its `BUG`, `RN_b`, `RN_d`, and `ID1` probe ids must be disjoint. The same tasks may appear in both sets; the lemma is about files, not tasks. Report the task overlap so the file split is not mistaken for a task split. Experiment 03's held-out Spearman is on natural candidate files; its V_bug probes are the separate `bm_gold` files. The same lemma applies, and this run checks it on the experiment 05 ids that are actually in the repository.
- **Median slack.** Spearman of the per-item median does not constrain the within-item range. Constructed example, not a logged judge: 45 items, human score and median both equal to `(i+1)/46`, three repeats `{0, median, 1}`. The median Spearman is 1 and the repeat-violation rate at tolerance `1e-12` is 1. Code: `slack_example()` in `scripts/analysis.py`.

### Confirmatory checks (CPU arm)

Scores are on `[0, 1]`. A drop is `ID1 − score(bug)` on the same task. `1e-9` is the same epsilon experiment 05 used. All intervals are 95% percentile bootstrap intervals, 2,000 draws, seed `20261124`. Each interval uses its own NumPy Generator started at that seed, so the draws for one statistic do not depend on whether another statistic is computed. Bug-pair intervals resample tasks and keep every probe of a drawn task (a task with more probes weighs more), matching the probe-level mean used as the point estimate. The partial-correlation interval resamples the 56 judge configurations. The partial-correlation p-value is a two-sided permutation of the correlation values, 5,000 shuffles, same seed on a separate generator, with the experiment 03 add-one form `(count + 1) / (valid + 1)`.

Pair class, first match wins, tolerance 0.05:

1. `inversion` if `score(bug) − ID1 > 0.05`
2. `penalized` if `score(bug) − ID1 < −0.05`
3. `ceiling_tie` if `ID1 ≥ 0.95`
4. `floor_tie` if `ID1 ≤ 0.20` and `score(bug) ≤ 0.20`
5. `mid_tie` otherwise

These five shares are descriptive. They must sum to 1. They are not the confirmatory cut.

**H-material** (configuration B only). B is the only experiment 05 configuration whose Spearman with the experts has a published lower bound above 0. Statistic: share of bug pairs with drop `> 0.20` (more than one point on the 0–5 scale). A and A2 get the same statistic and the same labels, as replication rows, and do not enter the overall verdict.

- Supported: point `≤ 0.15`, interval upper bound `< 0.30`, and all 2,000 draws valid.
- Falsified: point `≥ 0.30` and interval lower bound `> 0.15`.
- Otherwise mixed. If the point or the interval is missing, not computable.

**H-range** (the 22 experiment 03 configurations with held-out Spearman `≥ 0.70` and coverage `≥ 0.90`). Statistic: the 50th percentile, NumPy's linear percentile, of those 22 values of `V_rep.mean_range` (mean over items of max repeat minus min repeat, on `[0, 1]`). This is a census of the logged high-correlation set, not a sample of a wider population, so the verdict uses the percentile itself and not a bootstrap interval. A unit-bootstrap interval may be printed only as a descriptive extra and must be labeled conditional, because the 22 share models, prompts, and tasks.

- Supported (the repeat failures are score swings, not a `1e-12` jitter artifact): median `≥ 0.10`.
- Falsified (jitter artifact): median `≤ 0.02`.
- Otherwise mixed.

**H-split** (the 56 real judge configurations in experiment 03: judge LLMs on `vb103_judge` and judge agents on `terminus2`). Partial Spearman is Pearson correlation of the rank residuals: rank(`rho_heldout`) and rank(`V_weak`) each residualized by ordinary least squares on rank(`V_bug`) and an intercept, and the swapped pair. Ranks use average ranks for ties. If a residual variance is at most `1e-12`, that partial is not computable.

- Supported (statement weakening tracks human correlation and bug ties do not): partial(`rho`, `V_weak` | `V_bug`) has point `≤ −0.40`, interval upper bound `< −0.20`, and permutation `p < 0.05`; and partial(`rho`, `V_bug` | `V_weak`) has absolute point `≤ 0.25` or an interval that covers 0.
- Falsified (bug ties are the channel, statement weakening is not): partial(`rho`, `V_bug` | `V_weak`) has point `≤ −0.40`, interval upper bound `< −0.20`, and `p < 0.05`; and the weakening partial's interval covers 0.
- Otherwise mixed. If either partial has fewer than 1,500 valid bootstrap draws or fewer than 1,500 valid permutations, not computable.

The permutation shuffles these 56 configurations. It is a conditional check on this logged set. The configurations share models, prompts, and tasks. It is not a sample of independent judges.

**Overall**, from H-material on B, H-range, and H-split:

- Supported if at least two are supported and none is falsified.
- Falsified if at least two are falsified and none is supported.
- Mixed otherwise, including any split of supported and falsified.
- Not computable if any of the three is not computable.

### Descriptive quantities (no support cut)

- Class shares, penalized share at the 0.05 tolerance, median drop, and mean drop, for A, A2, and B. The median drop is the corollary check above.
- Task-level Spearman, across tasks that have both a natural-file score and a bug probe, between the task's mean `CRIT` score and its mean bug drop. Report the interval. If fewer than 8 tasks, or the Spearman is undefined, label `low_n` or `undefined`. This does not enter the overall verdict: with about a dozen tasks the interval is expected to be wide.
- Mean `ID1` minus mean `CRIT`, with a task-cluster interval, as the operating-point gap between gold-as-candidate and the files humans rated.
- Rename shares for `RN_b` and `RN_d`: down by more than 0.20, down by more than 0.05, up by more than 0.05, otherwise unchanged.
- Experiment 04, every `M1_deployed_tc_v1` row: point ratio `harsh_flip_eq10 / V_equiv` when `V_equiv > 0`. No new interval, because the pair-level draws are not in the public CSV. The two frontier rows named above are prior exposure. The other rows are descriptive.

### Alignment gates (the run stops if any fails)

- Experiment 05, each of A, A2, B: recomputed V_bug (`score(bug) ≥ ID1 − 1e-9`), tie share, strict share, mean drop, ID1 mean, and kind counts match `results/summary.json` within `1e-9`. Counts of `CRIT` scores `≤ 0.20` are 61 for A and 71 for A2.
- Lanes and model ids in that summary are A and A2: `gpt-6-astra` on `vals-codex`; B: `claude-opus-5-5` on `stanford-claude`.
- Experiment 03: 56 judge configurations with a held-out Spearman; recomputed marginal Spearman of that Spearman with V_weak matches the stored −0.7470214475096882 within `1e-6`; exactly 22 have Spearman `≥ 0.70` and coverage `≥ 0.90`; those 22 are 17 on `vb103_judge` and 5 on `terminus2`; their config ids match `failing_high_corr_llms`; each has V_bug `strict_gt = 0` and `tie_eq` equal to V_bug.
- Class shares sum to 1 within `1e-9`. The slack example returns Spearman 1 and repeat-violation rate 1.

## Data

- `experiments/05_renaming_and_inserted_bug_on_veribench_judge_outputs/results/scores_{A,A2,B}.csv` and `summary.json`. Columns used: `probe_id`, `kind`, `task`, `level`, `score`. No human rating vector is in the public CSV, so this run does not recompute Spearman with the experts.
- `experiments/03_human_corr_vs_axiomatic_properties_cert_judge/results/summary.json` (per-configuration `rho_heldout`, `V_bug`, `V_weak`, `V_rep.mean_range`, harness, model).
- `experiments/04_vb_fsc_equivalence_invariance_and_gaming_violations/results/equiv_invariance_table.csv` for the descriptive ratio only.
- New model calls: no, in the CPU arm.

## Baselines

- Experiment 05 configuration B, judge LLM Claude Opus 5.5 (`claude-opus-5-5`), harness Claude Code CLI, Stanford plan, lane `stanford-claude`, rubric `P4_CHECKLIST`, is the confirmatory human-correlation configuration for H-material.
- A and A2, judge LLM GPT-6 Astra (`gpt-6-astra`), harness Codex CLI, Vals plan, lane `vals-codex`, rubrics `P4_CHECKLIST` and `P8_STRICT_B`, are the replication rows.
- Experiment 03's 56 judge configurations are the set for H-split. The 22 high-correlation configurations are the set for H-range.
- The constant-forecast comparison from experiment 05 is not recomputed. pass@k and boxed accuracy do not apply to these Lean files.

## What would falsify it

The overall rule in "Confirmatory checks" is the falsifier. A mixed overall verdict means the three components do not agree; it is not a license to keep only the component that went the expected way. H-material is falsified when bug pairs on configuration B usually drop by more than 0.20. H-range is falsified when the typical high-correlation repeat range is at most 0.02. H-split is falsified when bug ties track human correlation and statement weakening does not.

## Models and harnesses

CPU arm: no judge LLM and no judge agent is called. The logged names above are the models and harnesses that produced the scores. Analysis harness: Python 3, NumPy, SciPy, matplotlib, in `scripts/analysis.py`. Every figure calls `ax.grid(True)`.

API arm, not run: one new rubric on the same judge LLM and harness as configuration B. See below.

## Cost estimate

CPU arm: $0.00. No tokens, no GPU. There is no GPU training arm in this experiment; nothing here is a fine-tune.

API arm, if it is run later, on Brando's Stanford plan first (`stanford-claude`), then the Vals Claude lane if Stanford is paused (`vals-claude-opus`): about 123 property items in about 21 calls of 6, roughly 60% of an experiment 05 configuration, so about $3 to $5 API-equivalent and $0 out of pocket. No agy lane, no Llama 4, no key in the script or the logs. This preregistration does not by itself authorize launching those calls.

## API arm (specified, not run)

Question the CPU arm cannot close: when configuration B fails to penalize an inserted bug, is that because `P4_CHECKLIST` is applied to the last theorem block as the reference, so an operator change inside a definition is outside what the rubric grades?

Arm S is the logged configuration B scores. Do not call that rubric again.

Arm I calls judge LLM Claude Opus 5.5 (`claude-opus-5-5`) in the Claude Code CLI through the plan router, empty workspace, tools off, same item headers as experiment 05 (`REFERENCE IMPLEMENTATION`, `GENERATED THEOREM`), same batch size 6, same batch seed `20261124`, one probe per task per call. The item pool is only `ID1`, `ID2`, `RN_b`, `RN_d`, and `BUG` (123 items if the experiment 05 probe file is complete). `CRIT` is omitted, so the batches are not the experiment 05 batches; the one-probe-per-task rule still keeps a file and its edit out of the same call. ID1 must be judged under Arm I, not copied from Arm S.

The rubric text is frozen here. `scripts/judge_implementation_rubric.py` must contain this exact string.

<!-- RUBRIC_I_BEGIN -->
```text
Score the candidate Lean file as an implementation of the reference theorem, on a scale from 0.0 to 1.0.

The reference is the theorem statement. The candidate is the whole file, including definitions and proofs.

Give 1.0 only if the definitions denote the function the theorem is about and the theorem is the reference statement. Identifier names may differ when every occurrence is renamed consistently.

If a definition changes an operator, a constant, a comparison, or a bound so that the denoted function can differ on some input, the score must be at most 0.4, even when the theorem statement text matches the reference.

A consistent renaming of binders or of file-local definition names, with the same operators, must receive the same score as the unmodified file.

Ignore comments and whitespace. Do not give credit for a matching theorem statement when the definition implements a different function.
```
<!-- RUBRIC_I_END -->

Arm I verdict, separate from the CPU overall verdict, using experiment 05's V_bug (`score(bug) ≥ ID1 − 1e-9`) and the task-cluster interval (2,000 draws, seed `20261124`):

- Incomplete unless there are 24 ID1 scores and 37 bug scores, the receipt model id contains `claude-opus-5-5`, and the lane is `stanford-claude` or `vals-claude-opus`.
- Not valid if the ID1 mean is below 0.6.
- Supported (the statement rubric was the reason): V_bug point `≤ 0.20`, interval upper bound `< 0.40`, and mean drop `≥ 0.30`.
- Falsified (the judge LLM still fails the property when the rubric asks for the implementation): V_bug point `≥ 0.40` and interval lower bound `> 0.20`.
- Otherwise mixed.

Rename excess on Arm I is descriptive and is not part of this support rule.

The runnable script is `scripts/judge_implementation_rubric.py`. It refuses to call the router without `--confirm-plan-spend`, refuses any lane whose name contains `agy` or `llama`, and refuses to write prompts inside this git repository. Dry-run prints counts only, not candidate files.

```bash
python3 experiments/06_why_human_correlation_misses_formal_properties/scripts/judge_implementation_rubric.py \
  --probes ~/at3450/probes.jsonl \
  --lanes stanford-claude \
  --out ~/at3450/exp06_arm_I \
  --confirm-plan-spend
```

After a completed run, copy back only `scores_I.csv` (probe id, kind, task, level, score) and run:

```bash
python3 experiments/06_why_human_correlation_misses_formal_properties/scripts/analysis.py \
  --arm-i scores_I.csv
```

Do not copy Lean sources, prompts, or raw router receipts into this repository.
