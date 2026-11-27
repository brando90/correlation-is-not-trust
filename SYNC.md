# correlation-is-not-trust: shared state (as of 2026-11-27)

## cursor (this run)
- updated: 2026-11-27
- branch: main
- exp08_h1_cal: untestable (task-held-out `CAL` `RPS` minus train-only constant: A `+0.0007 [-0.0046, +0.0060]`, A2 `+0.0020 [+0.0008, +0.0035]`, B `-0.0066 [-0.0229, +0.0136]`; exact `V_bug` unchanged for A and B; B `RN_d` `> 0.05` `0.375` to `0.000`, exact-change share stays `0.583`)
- exp09_h_gap: mixed (`3/6` families disagree, fail gap `0.167`, both picks fail `V_weak` or `V_rep` in `6/6`)
- exp09_h_gap_fitted: falsified (failure gap `0`)
- exp09_h_pred: supported (rho-targeted ridge OOF Spearman `0.424 [0.195, 0.607]`, `p = 0.0018`; unfitted `ATI_equal` remains the experiment 07 value `0.788`)
- doing: results committed on main
- next: prospective zero-exposure judge run on Vals or Stanford lanes; not started here (no plan router on this VM)

## antigravity (unchanged record from 2026-11-26)
- updated: 2026-11-26T16:36-08:00
- exp06_arm_i: supported (`V_bug = 0.108 [0.000, 0.259]`; `claude-opus-5-5`, Claude Code CLI, `vals-claude-opus`)
- exp07_h1_proper: supported (`29/29`, `27/27`, `24/24` fail `V_weak` or `V_rep`)
- exp07_h_idea3: prereg_literal=supported / leak_free_prereg=falsified / leak_free_extension_ati_min=supported
- exp07_h2_new: supported (new VeriBench rating-weighted `P(r == 5) = 0.729 [0.605, 0.850]`)
