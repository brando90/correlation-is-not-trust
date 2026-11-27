# Next (as of 2026-11-26)

## Completed on 2026-11-26

1. **Exp 06 Arm I (`experiments/06_why_human_correlation_misses_formal_properties/`) — `supported`**
   - Executed `scripts/judge_implementation_rubric.py` (`RUBRIC_I`, `123` probes in `21` batches) using judge LLM `claude-opus-5-5` via Claude Code CLI (`vals-claude-opus`, `$0.00` extra charges).
   - Switching to implementation equivalence cuts `V_bug` from `0.568 [0.405, 0.765]` (`P4_CHECKLIST`) to `0.108 [0.000, 0.259]`, with `mean_drop = 0.492 [0.347, 0.620]`, `penalized = 32/37` (`0.865 [0.697, 1.000]`), `inversion = 0/37`, and `id1_mean = 0.975 [0.954, 0.994]` (`RN_d` down `> 0.05` drops from `0.375 [0.167, 0.583]` to `0.083 [0.000, 0.208]`).

2. **Exp 07 (`experiments/07_proper_scores_are_not_trust_and_amortized_property_certification/`) — all 3 preregistered hypotheses `supported` (`$0.00` CPU)**
   - **Part A (`H1-proper`):** `29/29` Lean 4 judge configs beating `CONST_train` (`0.1739 [0.1424, 0.2085]` `RPS`) on `RPS`, `27/27` beating `CONST_train` across `RPS`, `CE`, and `TVD`, and `24/24` matching/beating human `LORO` (`0.1469 [0.1254, 0.1687]` `RPS`) still fail at least one non-bug property (`V_weak`, `V_rep`, calibrated `V_weak_cal`, calibrated repeat `rep_cal_tvd`, `rep_cal_ev_range_05`).
   - **Part B (`H-Idea3`):** Out-of-family amortized multi-property `ATI` generalizes (`test` `n=18`: `ATI_equal` Spearman `0.690 [0.318, 0.881]`, `p=0.0020`, `regret=0.0022`; `ATI_cal_rps` `0.676 [0.247, 0.887]`, `p=0.0022`, `regret=0.0013`; `6`-fold `LOMFO-CV` `n=56`: `ATI_equal` `0.721 [0.586, 0.806]`, `ATI_min` `0.750 [0.608, 0.843]`, `ATI_cal_rps` `0.748 [0.625, 0.826]`). Leak-free `ATI_min` (`train+val` cutoff) and hard `Gate_all` reject `4/4` deterministic controls (`ATI_equal` rejects `4/4` on full `Q_const_rps` and `3/4` on `train+val`-only `Q_const_rps`; `0/4` controls and `0/56` real configs pass `Gate_all`).
   - **Part C (`H2-new`):** New VeriBench (`100` items / `25` tasks) shows `0.717 [0.593, 0.837]` `5/5` saturation (`0.147 [0.040, 0.280]` on Old VeriBench) while keeping `0.886 [0.773, 0.977]` human disagreement on non-ceiling (`min < 5`, `44/100`) items.

## Next experiment for Grok Bot / Cursor

- **Exp 08 — Prospective zero-exposure multi-property `(a)–(e)` certification + order-permutation test on a held-out family**
  - Run prospective unseen judge agents/LLMs (via `vals` / Stanford lanes only; never `agy` or Llama 4) on held-out Lean 4 and/or VeriBench slices covering all five trust properties `(a)` repeat stability, `(b)` weakening/monotone symmetry, `(c)` order/position permutation invariance (pairwise / multi-reference), `(d)` length/verbosity neutrality (`id1`), and `(e)` bug sensitivity (`RUBRIC_I`), comparing proper-score calibration (`CE`/`RPS`/`TVD`) vs `ATI` gating prospectively with zero exposure during feature/weight fitting.
  - Read [SYNC.md](SYNC.md) and coordinate via `brando90/agent-sync` before launching any new runs.
