# 08: task-held-out calibration of the experiment 05 judges (as of 2026-11-27)

Does a monotone distributional calibration, fit with the evaluated task held out, let any of the three experiment 05 judge configurations beat a constant forecast that was also fit only on the training tasks, and do the inserted-bug failures and declaration-rename failures survive that calibration?

- **Preregistration:** [PREREG.md](PREREG.md). Committed before any run.
- **Status:** preregistered, not yet run.
- **Inputs:** public `experiments/05_renaming_and_inserted_bug_on_veribench_judge_outputs/results/scores_{A,A2,B}.csv`. Human ratings stay off this repository (VeriBench experiment 100 `items.jsonl`, sha256 pinned in the preregistration).
- **Cost:** `$0`. No judge LLM and no judge agent is called.

## Reproduce

Not runnable until `scripts/analysis.py` is added after this preregistration.
