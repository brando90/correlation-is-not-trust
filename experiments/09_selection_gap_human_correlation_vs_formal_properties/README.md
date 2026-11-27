# 09: selection gap between human correlation and formal properties (as of 2026-11-27)

Inside each base-weight family of the 56 experiment 03 judge configurations, does the configuration with the highest held-out human correlation differ from the configuration with the highest formal-property index, and does the correlation pick fail more property gates?

Experiment 07 already measured how well `ATI_equal` tracks `rho_heldout` (Spearman `0.788` on the 56). This experiment does not repeat that claim. The new quantity is the selection disagreement.

- **Preregistration:** [PREREG.md](PREREG.md), committed at `6e28b12` before the run; Amendment 1 at `1c62824` before the run.
- **Results:** [RESULTS.md](RESULTS.md). **H-gap mixed** (`3/6` families disagree; mean failure gap `0.167`; both picks fail `V_weak` or `V_rep` in `6/6` families). **H-gap-fitted falsified** (failure gap `0`). **H-pred supported** (out-of-fold Spearman `0.424 [0.195, 0.607]`, `p = 0.0018`).
- **Input:** public `experiments/03_human_corr_vs_axiomatic_properties_cert_judge/results/judge_table.csv`.
- **Cost:** `$0`. No judge LLM and no judge agent was called.

## Reproduce

```bash
python3 experiments/09_selection_gap_human_correlation_vs_formal_properties/scripts/analysis.py
python3 -m unittest experiments/09_selection_gap_human_correlation_vs_formal_properties/scripts/test_analysis.py -v
```
