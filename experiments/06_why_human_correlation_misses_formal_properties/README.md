# 06: why human correlation misses formal properties (as of 2026-11-26)

Do high human correlations fail formal judge properties because the statistics are free to diverge (different files, medians versus ranges), and do the logged scores show a material bug penalty, a real repeat swing, and a split between statement weakening and bug ties?

- **Protocol:** [PREREG.md](PREREG.md), committed before the CPU run.
- **Forward pointer:** [../../NEXT.md](../../NEXT.md).
- **Results:** [RESULTS.md](RESULTS.md) after the CPU run. Until that file exists, the CPU arm has not been run.
- **Scripts:** `scripts/analysis.py` (CPU, public logs only), `scripts/judge_implementation_rubric.py` (API arm, not run; refuses to call the router without `--confirm-plan-spend`).

## Reproduce the CPU arm

```bash
python3 experiments/06_why_human_correlation_misses_formal_properties/scripts/analysis.py
python3 -m unittest experiments/06_why_human_correlation_misses_formal_properties/scripts/test_analysis.py -v
```

No model calls. Cost $0. The API command is in `PREREG.md` and is not part of this reproduce block.
