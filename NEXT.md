# Next (as of 2026-11-26)

## Completed on 2026-11-26

1. **Exp 06 Arm I (`experiments/06_why_human_correlation_misses_formal_properties/`) — `supported`**
   - Executed `scripts/judge_implementation_rubric.py` (`RUBRIC_I`, `123` probes in `21` batches) using judge LLM `claude-opus-5-5` via Claude Code CLI (`vals-claude-opus`, `$0.00` extra charges).
   - Switching to implementation equivalence cuts `V_bug` from `0.568 [0.405, 0.765]` (`P4_CHECKLIST`) to `0.108 [0.000, 0.259]`, with `mean_drop = 0.492 [0.347, 0.620]`, `penalized = 32/37` (`0.865 [0.697, 1.000]`), `inversion = 0/37`, and `id1_mean = 0.975 [0.954, 0.994]` (`RN_d` down `> 0.05` drops from `0.375 [0.167, 0.583]` to `0.083 [0.000, 0.208]`).

2. **Exp 07 (`experiments/07_proper_scores_are_not_trust_and_amortized_property_certification/`) — `$0.00` CPU**
   - **Part A (`H1-proper = supported`):** `29/29` Lean 4 judge configs beating `CONST_train` (`0.1739 [0.1424, 0.2085]` `RPS`) on proper score `RPS`, `27/27` beating `CONST_train` across `RPS`, `CE`, and `TVD` distribution distance, and `24/24` matching/beating matched-item human `LORO` (`0.1469 [0.1254, 0.1687]` `RPS` on `45/45`; `0.1525` on `42/45`) still fail `V_weak` or `V_rep` (plus calibrated repeat shifts `rep_cal_tvd`).
   - **Part B (`H-Idea3`: literal `supported` / leak-free PREREG `falsified` / `ATI_min` ext. `supported`):** `ATI` predicts `-CE_fit RPS` on canonical `test` (`n=18`: `ATI_equal` Spearman `0.690 [0.318, 0.881]`, `p=0.0020`, `regret=0.0022`; `ATI_cal_rps` `0.676 [0.247, 0.887]`, `p=0.0022`, `regret=0.0013`) and zero-weight-overlap `test_unseen_weights` (`n=16`: `0.685` / `0.696` / `0.641`) + grouped `6`-fold `LOMFO-CV` (`n=56`: `ATI_equal` `0.721 [0.586, 0.806]`, `ATI_min` `0.750 [0.608, 0.843]`, `ATI_cal_rps` `0.748` descriptive). Under leak-free `train+val` `Q_const_rps`, preregistered `ATI_equal` & `ATI_cal_rps` reject `3/4` controls (`falsified` vs. `4/4` under literal full `Q`), whereas worst-case `ATI_min` (`4/4` leak-free) and hard `Gate_all` (`0/4` pass controls, `0/56` real configs) cleanly reject all `4/4` controls.
   - **Part C (`H2-new = supported`):** New VeriBench (`100` items / `25` tasks) has rating-weighted `P(r == 5) = 0.729 [0.605, 0.850]` (item-weighted `0.717 [0.593, 0.837]`, vs. Old `0.113 [0.028, 0.226]` / `0.147`) while keeping `0.886 [0.773, 0.977]` human disagreement on non-ceiling (`min < 5`, `44/100`) items.

## Next experiment for Grok Bot / Cursor

- **Exp 08 — Prospective zero-exposure multi-property evaluation + order-permutation & verbosity tests on a held-out family**
  - Run prospective unseen judge agents/LLMs (via `vals` / Stanford lanes only; never `agy` or Llama 4) on held-out Lean 4 and/or VeriBench slices covering the established formal trust checks plus new perturbations: (a) repeat stability (`ID2` / `V_rep`), (b) monotone weakening / overconstraint / vacuity (`V_weak`, `V_overc`, `V_vac`), (c) gold-as-candidate identity baseline (`ID1`) + binder/declaration renaming & formatting invariance (`RN_b`, `RN_d`, `V_fmt`), (d) pairwise / multi-reference order permutation invariance, (e) implementation bug sensitivity (`RUBRIC_I` / `V_bug`), and separate controlled length/verbosity perturbations — comparing proper-score calibration (`CE`, `RPS`) and `TVD` distance vs. `ATI_min` / `Gate_all` prospectively with zero exposure during fitting.
  - Read [SYNC.md](SYNC.md) and coordinate via `brando90/agent-sync` before launching any new runs.
