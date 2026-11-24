# Experiment 03: Human correlation vs. axiomatic properties (`cert-judge` E39/E42 + Lean 4 human pool) (date: 2026-11-24)

Zero-model-call reanalysis testing **H1** (whether high human-correlation Lean 4 judges still violate basic axiomatic properties) and **H2** (how inconsistent human raters are on the exact same Lean 4 tasks, both across raters and across passes/rubric wording).

- **Protocol:** [PREREG.md](PREREG.md) (committed at [`b571a7d`](https://github.com/brando90/correlation-is-not-trust/commit/b571a7d7ef8872d3ba1a53ed4d02e5746dbcd934) prior to running)
- **Results:** [RESULTS.md](RESULTS.md)
- **Light QA log:** [QA_LIGHT_2026-11-24.md](QA_LIGHT_2026-11-24.md) (`codex exec`, `gpt-6-astra` / `ultra`, Scope A resolved)
- **Figures:** [results/fig1_human_corr_vs_property_violations.png](results/fig1_human_corr_vs_property_violations.png), [results/fig2_human_inconsistency_lean4.png](results/fig2_human_inconsistency_lean4.png)
- **Machine-readable outputs:** [results/summary.json](results/summary.json), [results/judge_table.csv](results/judge_table.csv), [results/human_consistency.json](results/human_consistency.json)

## Layout
- `PREREG.md`: Frozen hypotheses, metrics, falsification cutoffs, and prior-exposure disclosure.
- `scripts/analysis.py`: Deterministic reanalysis + task-cluster bootstrap (`2,000` draws, seed `20261124`) + `E42` alignment verification (`max_abs_diff <= 1e-9`) + plot generator.
- `scripts/test_analysis.py`: Unit tests on synthetic edge cases and integration/privacy assertions on generated outputs.
- `results/`: Aggregate outputs and plots (no raw human TSVs, rater names, or Lean files stored).

## Reproduce
```bash
python3 experiments/03_human_corr_vs_axiomatic_properties_cert_judge/scripts/analysis.py
python3 -m unittest experiments/03_human_corr_vs_axiomatic_properties_cert_judge/scripts/test_analysis.py -v
```
