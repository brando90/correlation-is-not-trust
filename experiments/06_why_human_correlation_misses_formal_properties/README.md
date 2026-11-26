# 06: why human correlation misses formal properties (as of 2026-11-26)

Do high human correlations fail formal judge properties because the statistics are free to diverge (different files, medians versus ranges), and do the logged scores show a material bug penalty, a real repeat swing, and a split between statement weakening and bug ties?

- **Protocol:** [PREREG.md](PREREG.md), committed at `f59f959` before the CPU run.
- **Results:** [RESULTS.md](RESULTS.md). Overall verdict mixed. QA: [QA_MEGA_2026-11-26.md](QA_MEGA_2026-11-26.md).
- **Forward pointer:** [../../NEXT.md](../../NEXT.md).
- **Scripts:** `scripts/analysis.py` (CPU, public logs only; already run), `scripts/judge_implementation_rubric.py` (API arm, not run; refuses to call the router without `--confirm-plan-spend`).

## Reproduce the CPU arm

```bash
python3 experiments/06_why_human_correlation_misses_formal_properties/scripts/analysis.py
python3 -m unittest experiments/06_why_human_correlation_misses_formal_properties/scripts/test_analysis.py -v
```

No model calls. Cost $0. The API command is in `PREREG.md` and is not part of this reproduce block.
