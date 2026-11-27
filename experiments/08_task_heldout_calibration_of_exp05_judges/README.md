# 08: task-held-out calibration of the experiment 05 judges (as of 2026-11-27)

Does a monotone distributional calibration, fit with the evaluated task held out, let any of the three experiment 05 judge configurations beat a constant forecast that was also fit only on the training tasks, and do the inserted-bug failures and declaration-rename failures survive that calibration?

- **Preregistration:** [PREREG.md](PREREG.md), committed at `6e28b12` before the run.
- **Results:** [RESULTS.md](RESULTS.md). **H1-cal untestable.** No configuration's calibrated `RPS` interval lies entirely below the train-only constant. B's point estimate is `-0.0066 [-0.0229, +0.0136]`. Where the slope stays positive, exact `V_bug` is unchanged; B's `0.05` declaration-rename share falls from `0.375` to `0.000` while the exact-change share stays `0.583`.
- **Cost:** `$0`. No judge LLM and no judge agent was called.

## Reproduce

```bash
python3 experiments/08_task_heldout_calibration_of_exp05_judges/scripts/analysis.py --items /path/to/veribench_e100_items.jsonl
python3 -m unittest experiments/08_task_heldout_calibration_of_exp05_judges/scripts/test_analysis.py -v
```

Human ratings stay off this repository (VeriBench experiment 100 `items.jsonl`, sha256 pinned in the preregistration).
