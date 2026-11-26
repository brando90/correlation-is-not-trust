# 06: why human correlation misses formal properties (as of 2026-11-26)

Do high human correlations fail formal judge properties because the statistics are free to diverge (different files, medians versus ranges), and do the logged scores show a material bug penalty, a real repeat swing, and a split between statement weakening and bug ties?

- **Protocol:** [PREREG.md](PREREG.md), committed at `f59f959` before the CPU run.
- **Results:** [RESULTS.md](RESULTS.md). Overall CPU verdict mixed; Arm I (`RUBRIC_I`, `claude-opus-5-5`, Claude Code CLI, `vals-claude-opus`, `results/scores_I.csv`) **supported** (`V_bug = 0.108 [0.000, 0.259]`, `mean_drop = 0.492 [0.347, 0.620]`). QA: [QA_MEGA_2026-11-26.md](QA_MEGA_2026-11-26.md).
- **Forward pointer:** [../../NEXT.md](../../NEXT.md).
- **Scripts:** `scripts/analysis.py` (CPU + Arm I aggregation), `scripts/judge_implementation_rubric.py` (Arm I runner, refuses to call the router without `--confirm-plan-spend`).

## Reproduce

```bash
python3 experiments/06_why_human_correlation_misses_formal_properties/scripts/analysis.py \
  --arm-i experiments/06_why_human_correlation_misses_formal_properties/results/scores_I.csv \
  --arm-i-model claude-opus-5-5 \
  --arm-i-lane vals-claude-opus
python3 -m unittest experiments/06_why_human_correlation_misses_formal_properties/scripts/test_analysis.py -v
```

No new model calls in `analysis.py`; Arm I (`scores_I.csv`, 123 probes across 21 batches of 6) ran on Brando's `vals-claude-opus` plan lane (`$0.00` extra charges).
