# Next (as of 2026-11-26)

## Completed on 2026-11-26

1. **Exp 06 Arm I (`experiments/06_why_human_correlation_misses_formal_properties/`) — `supported`**
   - Executed `scripts/judge_implementation_rubric.py` (`RUBRIC_I`, `123` probes in `21` batches) using judge LLM `claude-opus-5-5` via Claude Code CLI (`vals-claude-opus`, `$0.00` extra charges).
   - Switching to implementation equivalence cuts `V_bug` from `0.568` (`P4_CHECKLIST`) to `0.108` [0.000, 0.259], with `mean_drop = 0.492` [0.347, 0.620], `penalized = 32/37` (`0.865` [0.697, 1.000]), `inversion = 0/37`, and `id1_mean = 0.975` (`RN_d down > 0.05` drops from `0.375` to `0.083` [0.000, 0.208]).

2. **Exp 07 (`experiments/07_proper_scores_are_not_trust_and_amortized_property_certification/`) — all 3 preregistered hypotheses `supported` (`$0.00` CPU)**
   - **Part A (`H1-proper`):** `29/29` Lean 4 judge configs beating `CONST_train` (`0.1739` `RPS`) on `RPS`, `27/27` beating `CONST_train` across all 3 proper scores (`RPS`, `CE`, `TVD`), and `24/24` matching/beating human `LORO` (`0.1469` `RPS`) still fail at least one non-bug property (`V_weak`, `V_rep`, calibrated `V_weak_cal`, calibrated repeat `rep_cal_tvd`, `rep_cal_ev_range_05`).
   - **Part B (`H-Idea3`):** Train-only amortized multi-property `ATI` generalizes out-of-family (`test` `n=18`: `ATI_equal` Spearman `0.690`, `p=0.0020`, `reg=0.0022`; `ATI_cal_rps` `0.676`, `p=0.0022`, `reg=0.0013`; `LOMFO-CV` `n=56`: `ATI_equal` `0.721`, `ATI_cal_rps` `0.748`) and rejects `4/4` deterministic controls (`0/4` pass the multi-property gate).
   - **Part C (`H2-new`):** New VeriBench (`100` items / `25` tasks) shows `71.7%` [59.3%, 83.7%] `5/5` saturation (`14.7%` [4.0%, 28.0%] on Old VeriBench) while keeping `88.6%` [77.3%, 97.7%] human disagreement on non-ceiling (`min < 5`, `44/100`) items.

## Next experiment for Grok Bot / Cursor

- **Exp 08 — Prospective zero-exposure multi-property `(a)–(e)` certification + order-permutation stress test on a held-out family**
  - Run prospective unseen judge agents/LLMs (via `vals` / Stanford lanes only; never `agy` or Llama 4) on held-out Lean 4 and/or VeriBench slices covering all five trust properties `(a)` repeat stability, `(b)` weakening/monotone symmetry, `(c)` order/position permutation invariance (pairwise / multi-reference), `(d)` length/verbosity neutrality (`id1`), and `(e)` bug sensitivity (`RUBRIC_I`), comparing proper-score calibration (`CE`/`RPS`/`TVD`) vs `ATI` gating prospectively with zero exposure during feature/weight fitting.
  - Read [SYNC.md](SYNC.md) and coordinate via `brando90/agent-sync` before launching any new runs.
