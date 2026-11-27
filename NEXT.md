# Next (as of 2026-11-27)

## Completed on 2026-11-27

1. **Exp 08 (`experiments/08_task_heldout_calibration_of_exp05_judges/`) — `H1-cal` untestable. `$0.00` CPU.**
   - Leave-one-task-out ordinal calibration of the three experiment 05 score files, with the constant fit only on the training tasks. No configuration beats that constant on `RPS` (A `+0.0007 [-0.0046, +0.0060]`, A2 `+0.0020 [+0.0008, +0.0035]`, B `-0.0066 [-0.0229, +0.0136]`). The raw point-mass gap stays `+0.20` to `+0.31`, so the experiment 05 result was not an artifact of an in-sample constant.
   - Where the slope is positive on all 25 tasks (A and B), exact `V_bug` is unchanged. B's declaration-rename share above `0.05` falls from `0.375 [0.167, 0.542]` to `0.000`, while the exact-change share stays `0.583`. A2 has a negative slope on `5/25` tasks and is reliably worse than the constant on `RPS`, `TVD`, and `CE`.

2. **Exp 09 (`experiments/09_selection_gap_human_correlation_vs_formal_properties/`) — `H-gap` mixed, `H-gap-fitted` falsified, `H-pred` supported. `$0.00` CPU.**
   - On the 56 experiment 03 configurations, grouped so a `terminus2` judge agent stays with its base judge LLM, the highest-`rho_heldout` config and the highest-`ATI_equal` config differ in `3/6` families. Mean Spearman regret `0.158`. Mean extra gate failures `0.167` (the pre-registered support cut was `0.5`). Both picks fail `V_weak` or `V_rep` in `6/6` families.
   - A ridge whose target is `rho_heldout` has failure gap `0` (falsified) and out-of-fold Spearman `0.424 [0.195, 0.607]` (`p = 0.0018`). Experiment 07 already reported the unfitted `ATI_equal` Spearman `0.788` with the same human correlation.

## Completed on 2026-11-26

1. **Exp 06 Arm I — `supported`.** Implementation-equivalence rubric, judge LLM `claude-opus-5-5`, Claude Code CLI, `vals-claude-opus`, `$0.00` extra. `V_bug` `0.108 [0.000, 0.259]`.
2. **Exp 07 — `$0.00` CPU.** `H1-proper` supported (`29/29` `CONST_train`-beating configs still fail `V_weak` or `V_rep`). `H-Idea3` literal supported / leak-free preregistered falsified / `ATI_min` extension supported. `H2-new` supported (New VeriBench rating-weighted `P(r == 5) = 0.729 [0.605, 0.850]`; non-ceiling disagreement `0.886 [0.773, 0.977]`).

## Still blocked (do not fabricate)

These need data or plan lanes that are not on a fresh checkout of this repository. Do not invent scores, and do not send real judge calls through `agy` or Llama 4.

1. **Prospective zero-exposure judge run (the old "Exp 08" plan, now the next numbered folder).** Held-out Lean 4 and/or VeriBench slices covering repeat stability, monotone weakening, gold-as-candidate identity, binder and declaration renaming, order permutation, implementation-bug sensitivity, and a separate length perturbation. Compare proper scores (`RPS`, `CE`) and `TVD` with `ATI_min` / `Gate_all` on judges that were not in the fitting panel. Requires Brando's Vals or Stanford lanes. This VM has no plan router.
2. **Powered rename and order probes for README properties (a) and (e).** A generator is not a powered probe until Lean accepts the renamed and reordered files. The Lean project and the private probe sources are off this checkout (`cert-judge` experiment 39). Experiment 05 already logged `RN_b` / `RN_d` for three configurations. Experiment 03's `reorder` set had `0/75` valid probes.
3. **New VeriBench human annotation.** Experiment 07's New VeriBench file is an interim multiset: `247` ratings, `100` items, `min_n = 2`, `max_n = 4`, no rater ids. Per-rater severity on that wave waits on the rest of the ratings. The raw TSVs stay off this public repository.
4. **Off-repo path for experiment 08's rerun.** The sha-pinned VeriBench experiment 100 `items.jsonl` (`ad9ed552fc2ffac6328e6010bf6be82215dd74f1eddd791622d433126877a96c`) is not in this repository. `scripts/analysis.py` exits if the sha does not match.
