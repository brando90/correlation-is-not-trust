# 09: selection gap between human correlation and formal properties (as of 2026-11-27)

Inside each base-weight family of the 56 experiment 03 judge configurations, does the configuration with the highest held-out human correlation differ from the configuration with the highest formal-property index, and does the correlation pick fail more property gates?

Experiment 07 already measured how well `ATI_equal` tracks `rho_heldout` (Spearman `0.788` on the 56). This experiment does not repeat that claim. The new quantity is the selection disagreement.

- **Preregistration:** [PREREG.md](PREREG.md). Committed before any run.
- **Status:** preregistered, not yet run.
- **Input:** public `experiments/03_human_corr_vs_axiomatic_properties_cert_judge/results/judge_table.csv`.
- **Cost:** `$0`. No judge LLM and no judge agent is called.

## Reproduce

Not runnable until `scripts/analysis.py` is added after this preregistration.
