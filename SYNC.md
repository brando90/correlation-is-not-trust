# correlation-is-not-trust: shared state (as of 2026-11-26)

## antigravity
- updated: 2026-11-26T16:36-08:00
- branch: main
- exp06_arm_i: supported (`V_bug = 0.108 [0.000, 0.259]`, `mean_drop = 0.492 [0.347, 0.620]`, `id1_mean = 0.975 [0.954, 0.994]`; `claude-opus-5-5`, Claude Code CLI, `vals-claude-opus`, `123/123` probes; Arm S baseline `V_bug = 0.568 [0.405, 0.765]`)
- exp07_h1_proper: supported (`29/29` `Q_const_rps`, `27/27` `Q_const_all3`, `24/24` matched-`LORO` fail `V_weak` or `V_rep`)
- exp07_h_idea3: prereg_literal=supported / leak_free_prereg=falsified (`3/4` controls on `ATI_equal` & `ATI_cal_rps` under `train+val` `Q`) / leak_free_extension_ati_min=supported (`4/4` controls on `ATI_min` & `0/4` passing `Gate_all`; `ATI_equal` test `0.690 [0.318, 0.881]` `p=0.0020`, `LOMFO-CV` `0.721 [0.586, 0.806]` `p=0.0002`; `ATI_cal_rps` test `0.676 [0.247, 0.887]`, `LOMFO-CV` `0.748` descriptive; `0/56` pass `Gate_all`)
- exp07_h2_new: supported (new VeriBench rating-weighted `P(r == 5) = 0.729 [0.605, 0.850]`, item-weighted `0.717 [0.593, 0.837]`, non-ceiling disagreement `0.886 [0.773, 0.977]`)
- doing: 3-stage Mega QA complete (`CRITICAL: 0, MAJOR: 0`, all Stage 1–3 findings fixed & verified); pushing `main` and updating `brando90/agent-sync`
- next: Exp 08 prospective zero-exposure multi-property evaluation (candidate Grok Bot / Cursor lane)
