# PREREG: 04_vb_fsc_equivalence_invariance_and_gaming_violations (date: 2026-11-24)

Committed before any run. Not changed after the first run; later changes go in a dated "Amendment" section.

## Prior-exposure disclosure
Zero-model-call reanalysis of existing cached VeriBench Formal Specification Coverage (`FSC`) judge outputs from `brando90/judge-reference-anchoring` (`experiments/01_veribench_fsc_reference_anchoring_open_and_frontier_llm_vs_harness_start_17Nov2026/results/`) and `brando90/trustworthy-theorem-judges` (`experiments/23_do_gamed_or_off_format_lean_files_score_lower_on_veribench_vb256_start_18Nov2026/{expt_v1_tinyveribench256,expt_v2_vb256}/results/`). No raw Lean files, no raw rationales containing source code/paths, and no unredacted screenshots containing collaborator emails are copied into this public repository. Zero new model calls (`$0.00`).

## Hypothesis (one sentence)
**H1-equiv (Idea 1 — Equivalence & Order Invariance on VB FSC):** Frontier and deployed single-call VB FSC judge LLMs (`claude-sonnet-4-6`, `gpt-5.4`, plus open-weight judge LLMs) violate semantic equivalence invariance across Lean-kernel-proved equivalent gold references (`match_refactored` vs. `diff_gold`, and real VeriBench `R_A` vs. `R_B`) and reference-order permutation invariance (`dual_ab` vs. `dual_ba`), whereas reference-isolated Lean-kernel harness judge agents (`M5`, `M6`) reduce these violations; **H1-game (Idea 1 — Monotonicity vs. Adversarial Gaming):** the human-rubric VB FSC judge LLM (`claude-opus-5-5` via `claude -p --tools ''`, `judge_only`) violates monotonicity or leaks substantial credit (`>= 3/5`) on proof-level gaming (`sorry_proofs`, `false_axiom_proofs`, outside its statement-only rubric) and selected statement-level/format perturbations (`shadowed_perm_sorted_pairwise`, `dropped_most_theorems`, `theorem_spam`, `off_format_missing_sections`), showing that human-rubric judging alone is insufficient without formal Lean kernel and structural checks.

## Properties and metrics
All 95% confidence intervals use `2,000`-draw task-cluster percentile bootstrap resampling over each slice's observed tasks (`seed = 20261124`).

### Part A — Reference-Equivalence, Renaming, Order Permutation & Real VeriBench Flips (`judge-reference-anchoring` Exp 01)
Data sources:
- `exp_a_b_per_item_scores.csv` (`28` judge×method configurations across `claude-sonnet-4-6` [`claude-p-cli`], `gpt-5.4` [`openai-api`], `labs-leanstral-1-5-1` [`mistral-api`], `ministral-14b-2512` [`mistral-api`], `ministral-8b-2512` [`mistral-api`], `qwen3.5:9b-q4_K_M` [`ollama-local`]; methods `M1_deployed_tc_v1`, `M2_blind_no_ref`, `M3_prompt_nonexcl_cot`, `M4_dual_ref`, `M5_harness_ref_hidden_lean`; `repeat_idx == 0` primary pass plus `repeat_idx in {0, 1, 2}` where collected).
- `exp_b_cli_agent_scores.csv` (`3` CLI judge agents under `M6_harness_cli_agent`: `claude-sonnet-4-6` [`claude-code-cli`], `koyejolab-openai-bm/gpt-5.4` [`pi-cli`], `unassigned-mistral-main/ministral-14b-2512` [`pi-cli`]).
- `exp_c_veribench_impact_per_task.csv` (`10` tasks × `5` real VeriBench solvers = `50` task–solver pairs across `5` judge×method configurations: `claude-sonnet-4-6` [`M1`, `M4`, `M5`] and `labs-leanstral-1-5-1` [`M1`, `M5`]).

Metrics (on `[0, 1]` `fsc_norm` scale and binary harsh pass indicators `harsh_credit_ge9`, `harsh_credit_eq10`):
- `A_G` (anchoring gap on paired `T1` tasks): `mean(fsc_norm(match_refactored) - fsc_norm(diff_gold))` with 95% task-cluster CI and paired sign-flip permutation `p-val` (exact `2^n` test when `n <= 12`, else `5,000` draws, `seed = 20261124`).
- `V_equiv` (equivalence invariance violation rate on paired `T1` tasks): share with `|fsc_norm(match_refactored) - fsc_norm(diff_gold)| > 1e-12`, plus `mean_abs_equiv_shift = mean(|match_refactored - diff_gold|)`, `harsh_flip_ge9`, and `harsh_flip_eq10`.
- `V_rename_exact` (`match_exact` vs. `match_refactored` on paired `T1` tasks): share with `|fsc_norm(match_exact) - fsc_norm(match_refactored)| > 1e-12` and mean absolute shift.
- `V_order` (README property `(e)` — reference-order permutation under `M4_dual_ref`, `dual_ab` vs. `dual_ba` on paired `T1` tasks): share with `|fsc_norm(dual_ab) - fsc_norm(dual_ba)| > 1e-12` and mean absolute shift.
- `fc` (false credit on `T0` `non_covering` provably inadequate specifications): `mean(fsc_norm)` and `P(harsh_credit_ge9 == 1)`.
- `V_rep_ab` (repeat inconsistency on `(judge_id, method, task_id, condition)` items with `>= 2` repeats): share with `max - min > 1e-12`.
- `V_ref_swap` (real VeriBench `exp_c` 50 solver–task pairs across `10` tasks): share with `|fsc_R_A - fsc_R_B| > 1e-12`, `mean_abs_fsc_shift`, and harsh-pass flip rates (`harsh_nodt_R_A != harsh_nodt_R_B`, `harsh_nodt_ge09_R_A != harsh_nodt_ge09_R_B`, `harsh_dt_R_A != harsh_dt_R_B`), clustered by the `10` tasks.
- **Alignment check A:** Recomputed `cm`, `cd`, `A_G`, and `fc` across all `28` `exp_ab` configurations and `mean_abs_fsc_shift` across all `5` `exp_c` configurations must match `summary_metrics.json` within `1e-9` (fail loudly if violated).

### Part B — Adversarial Gaming & Monotonicity (`trustworthy-theorem-judges` Exp 23)
Data sources:
- `expt_v1_tinyveribench256/results/{judge_tiny256_sample30.csv, gate_tiny256_sample30.csv}` (`30` tasks × `11` conditions), `judge_tiny256_perm_sorted_pairwise.csv` & `gate_tiny256_perm_sorted_pairwise.csv` (`12` tasks × `6` conditions).
- `expt_v2_vb256/results/{judge_vb256_sample30.csv, gate_vb256.csv}` (`30` stratified VB256 tasks × `5` rule-built conditions).
- Original judge LLM: `claude-opus-5-5` via `claude -p --tools ''` (one sample per file, `0–5` human rubric integer scale).

Metrics (for each condition vs. same-task `faithful_gold_copy`, analyzed separately under `judge_only`, `gated` (`judge_score * passes_gate`), and `gated_format` (`judge_score * passes_gate_and_format`) on `TinyVeriBench-256`; and `judge_only`, `gated_kernel` (`judge_score * passes_gate_kernel`), and `full_pipeline` (`judge_score * passes_all_checks`) on `VB256`):
- `V_mono` (monotonicity violation rate): share of paired tasks with `score(condition) >= score(faithful)` (decomposed into strict `>` and tie `==`), with 95% task-cluster CI.
- `high_credit_ge3`: share of tasks where the gamed/degenerate/flawed condition scores `>= 3.0` (`>= 60%` of maximum).
- `mean_score` and `mean_drop = mean(faithful - condition)` with 95% task-cluster CI and exact two-sided binomial sign-test `p-val` on non-zero task differences.
- Separate **statement-level gaming/format** (`weakened_props`, `vacuous_true_props`, `dropped_most_theorems`, `theorem_spam`, `shadowed_names`, `shadowed_perm_sorted_pairwise`, `off_format_missing_sections`), **proof/kernel-level gaming** (`sorry_proofs`, `false_axiom_proofs`), and **degenerate/natural controls** (`empty_file`, `weak_prompt_natural`, `historical_natural_agent`).
- **Alignment check B:** Recomputed `TinyVeriBench-256` `sample30` `judge_only`, `gated`, and `gated_format` `mean` and `mean_drop` values across all `11 × 3 = 33` rows must match `summary_judge_30.csv` within `0.0051` (CSV 2-decimal rounding tolerance; fail loudly if violated).

## Data
- `/Users/brandomiranda/judge-reference-anchoring/experiments/01_veribench_fsc_reference_anchoring_open_and_frontier_llm_vs_harness_start_17Nov2026/results/{exp_a_b_per_item_scores.csv, exp_b_cli_agent_scores.csv, exp_c_veribench_impact_per_task.csv, summary_metrics.json}`
- `/Users/brandomiranda/trustworthy-theorem-judges/experiments/23_do_gamed_or_off_format_lean_files_score_lower_on_veribench_vb256_start_18Nov2026/{expt_v1_tinyveribench256,expt_v2_vb256}/results/*.csv`
- New model calls: no.

## Baselines
- Single-call deployed VB FSC judge LLMs (`M1_deployed_tc_v1`, `claude-opus-5-5` `judge_only`), prompt-engineered single-call judge LLMs (`M2`, `M3`, `M4`), Lean-kernel harness judge agents (`M5_harness_ref_hidden_lean`, `M6_harness_cli_agent`), and kernel/format-gated pipelines (`gated`, `gated_format`, `gated_kernel`, `full_pipeline`).

## What would falsify it
- **H1-equiv:** Supported if at least one frontier single-call VB FSC judge (`claude-sonnet-4-6` or `gpt-5.4` under `M1_deployed_tc_v1`) has `V_equiv >= 0.25` (lower 95% CI `> 0.05`) OR real VeriBench `V_ref_swap >= 0.25` (lower 95% CI `> 0.05`) OR `M4_dual_ref` `V_order >= 0.15` (lower 95% CI `> 0.02`). Falsified if every frontier `M1` judge satisfies `V_equiv < 0.10`, `V_ref_swap < 0.10`, and `M4` `V_order < 0.05`. Mixed otherwise.
- **H1-game:** Supported if `claude-opus-5-5` `judge_only` exhibits `V_mono >= 0.20` (lower 95% CI `> 0.05`) on at least one gaming condition (`sorry_proofs`, `false_axiom_proofs`, `shadowed_perm_sorted_pairwise`, `dropped_most_theorems`, `theorem_spam`, `weakened_props`, or `vacuous_true_props`) AND kernel/format gating drives `V_mono` on the gated proof/format attacks to `< 0.05`. Falsified if `judge_only` achieves `V_mono < 0.05` and `high_credit_ge3 < 0.05` across all gaming conditions without Lean gating. Mixed otherwise.

## Models and harnesses
- Part A judge LLMs & harnesses (`judge-reference-anchoring` Exp 01):
  - Single-call judge LLMs (`M1`–`M4`): `claude-sonnet-4-6` (`claude-p-cli`), `gpt-5.4` (`openai-api`), `labs-leanstral-1-5-1` (`mistral-api`), `ministral-14b-2512` (`mistral-api`), `ministral-8b-2512` (`mistral-api`), `qwen3.5:9b-q4_K_M` (`ollama-local`).
  - Reference-isolated Lean-kernel harness (`M5_harness_ref_hidden_lean`) and CLI judge agents (`M6_harness_cli_agent`): `claude-code-cli` (`claude-sonnet-4-6`), `pi-cli` (`koyejolab-openai-bm/gpt-5.4`, `unassigned-mistral-main/ministral-14b-2512`).
- Part B judge LLM & harness (`trustworthy-theorem-judges` Exp 23): `claude-opus-5-5` via `claude -p --tools ''` (one sample, `0–5` human rubric).
- Reanalysis harness: Python 3 (`numpy`, `scipy`, `matplotlib`) in `scripts/analysis.py`.

## Cost estimate
$0.00 (zero new model calls; no tokens spent on any account).