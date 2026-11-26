# correlation-is-not-trust: shared state (as of 2026-11-26)

## antigravity
- updated: 2026-11-26T15:56-08:00
- commit: f76703c (pre-Exp07 PREREG; Exp 07 results & Mega QA pending commit/push)
- branch: main
- exp06_arm_i: supported (V_bug 0.108 [0.000, 0.259], mean_drop 0.492 [0.347, 0.620], id1_mean 0.975 [0.954, 0.994]; claude-opus-5-5, Claude Code CLI, vals-claude-opus, 123/123 probes)
- exp07_h1_proper: supported (29/29 Q_const_rps, 27/27 Q_const_all3, 24/24 Q_loro fail V_weak or V_rep)
- exp07_h_idea3: supported (ATI_equal test Spearman 0.690 p=0.0020, LOMFO-CV 0.721 p=0.0002; ATI_cal_rps 0.676 / 0.748; 4/4 controls rejected; 0/56 pass Gate_all)
- exp07_h2_new: supported (new VB ceiling P(r==5)=0.717 [0.593, 0.837], non-ceiling disagreement 0.886 [0.773, 0.977])
- doing: finishing Exp 07 docs, running 3-stage Mega QA with agy gemini-4-carbon-high, pushing main, syncing agent-sync
- next: Exp 08 prospective zero-exposure multi-property (a)-(e) eval on fresh code/math/Lean tasks (candidate Grok Bot / Cursor lane)
