# 07: proper scores are not trust, amortized certification, and human ceiling regimes (as of 2026-11-26)

Do real Lean 4 judge LLMs and judge agents that strictly beat `CONST_train` and match human `LORO` on proper scoring rules (`RPS`, `CE`, `TVD`) still fail axiomatic trust checks; does a multi-property Axiomatic Trust Index (`ATI`) generalize across unseen model families; and do New VeriBench human ratings hide non-ceiling disagreement behind `5/5` saturation?

- **Protocol:** [PREREG.md](PREREG.md), committed at `f76703c` before any run.
- **Results:** [RESULTS.md](RESULTS.md) (`H1-proper = supported`, `H-Idea3 = supported`, `H2-new = supported`). QA: [QA_MEGA_2026-11-26.md](QA_MEGA_2026-11-26.md).
- **Zero new model calls (`$0.00`):** CPU-only reanalysis (`scripts/analysis.py`) with `<= 1e-9` alignment gates against `cert-judge` Experiment `68` (`436` checks, max diff `3.33e-16`) and Experiment `03` (`480` checks, max diff `0.00e+00`).

## Reproduce

```bash
python3 experiments/07_proper_scores_are_not_trust_and_amortized_property_certification/scripts/analysis.py
python3 -m unittest experiments/07_proper_scores_are_not_trust_and_amortized_property_certification/scripts/test_analysis.py -v
```
