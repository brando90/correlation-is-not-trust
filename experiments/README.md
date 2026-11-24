# Experiments index (as of 2026-11-24)

Newest first. Each experiment is experiments/NN_<name>/ with README.md, PREREG.md (committed before any run), scripts, results, RESULTS.md. Start from 00_template/PREREG.md.

| # | Name | Date | Status | Takeaway |
|---|------|------|--------|----------|
| 03 | [human_corr_vs_axiomatic_properties_cert_judge](03_human_corr_vs_axiomatic_properties_cert_judge/RESULTS.md) | 2026-11-24 | done (H1 & H2 supported; Light QA applied) | 22/22 Lean 4 judges with `rho_heldout >= 0.70` fail `V_weak`, `V_rep`, or `V_bug` (22/22 even without `V_bug`); 5-rater human exact agreement `0.294 [0.237, 0.353]` |
| 02 | judge_properties_qwen_math_wording_and_monotonicity | 2026-11-24 | done (mixed: H1 not supported in this pilot) | Judge LLMs with the highest correlation with gold fail the properties least (Spearman -0.98, n=8); the test shares the gold bit with C |
| 01 | human_inconsistency_dices350 | 2026-11-24 | done (H2 cross-rater part supported) | Pool mean is stable (split-half 0.95) but a perfect fit to it agrees with one rater on only 73% of Yes/No ratings |
