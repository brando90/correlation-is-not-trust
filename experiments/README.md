# Experiments index (as of 11-24-2026)

Newest first. Each experiment is experiments/NN_<name>/ with README.md, PREREG.md (committed before any run), scripts, results, RESULTS.md. Start from 00_template/PREREG.md.

| # | Name | Date | Status | Takeaway |
|---|------|------|--------|----------|
| 04 | [vb_fsc_equivalence_invariance_and_gaming_violations](04_vb_fsc_equivalence_invariance_and_gaming_violations/RESULTS.md) | 2026-11-24 | done (H1-equiv & H1-game supported; 3-stage Mega QA fixes applied & verified) | Deployed VB FSC `M1` frontier judge LLMs fail semantic equivalence (`V_equiv` `0.615–0.750`, real `V_ref_swap` `0.486–0.600`); standalone `claude-opus-5-5` `judge_only` confirms expected proof-attack exposure (`V_mono` `0.667–1.000`) & statement-shadowing (`0.333`, closed by format gating; `theorem_spam` stays rubric-bounded at `3.000–3.250`) |
| 03 | [human_corr_vs_axiomatic_properties_cert_judge](03_human_corr_vs_axiomatic_properties_cert_judge/RESULTS.md) | 2026-11-24 | done (H1 & H2 supported; Light QA + 3-stage Mega QA fixes verified) | 22/22 Lean 4 judge configs (17 judge LLMs, 5 judge agents) with `3`-expert `target_expert` `rho_heldout >= 0.70` fail `V_weak`, `V_rep`, or `V_bug` (22/22 even without `V_bug`); 5-rater pool exact agreement `0.294 [0.237, 0.353]` plus within-rater strictness/rubric sensitivity (`0.387`, `0.242`, `0.194`) |
| 02 | judge_properties_qwen_math_wording_and_monotonicity | 2026-11-24 | done (mixed: H1 not supported in this pilot) | Judge LLMs with the highest correlation with gold fail the properties least (Spearman -0.98, n=8); the test shares the gold bit with C |
| 01 | human_inconsistency_dices350 | 2026-11-24 | done (H2 cross-rater part supported) | Pool mean is stable (split-half 0.95) but a perfect fit to it agrees with one rater on only 73% of Yes/No ratings |
