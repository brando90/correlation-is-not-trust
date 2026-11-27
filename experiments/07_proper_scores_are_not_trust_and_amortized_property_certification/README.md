# 07: proper scores are not trust, amortized certification, and human ceiling regimes (as of 2026-11-26)

Do real Lean 4 judge LLMs and judge agents that strictly beat `CONST_train` and match human `LORO` on proper scoring rules (`RPS`, `CE`) and distribution distance (`TVD`) still fail axiomatic trust checks; does a multi-property Axiomatic Trust Index (`ATI`) generalize across unseen model families; and do New VeriBench human ratings hide non-ceiling disagreement behind `5/5` saturation?

- **Preregistration:** [PREREG.md](PREREG.md), committed at [`f76703c`](https://github.com/brando90/correlation-is-not-trust/commit/f76703c) before any run (with dated terminology/scope amendment).
- **Results:** [RESULTS.md](RESULTS.md) (`H1-proper = supported`; `H-Idea3`: `supported` on literal full-`Q_const_rps`, `falsified` at `3/4` controls on leak-free `train+val` cutoff for `ATI_equal`/`ATI_cal_rps`, `supported` on leak-free `ATI_min` + `Gate_all` extension [`ATI_equal` test Spearman `0.690 [0.318, 0.881]`, `ATI_cal_rps` `0.676 [0.247, 0.887]`, `4/4` controls rejected by leak-free `ATI_min` & `Gate_all`]; `H2-new = supported`, New VeriBench rating-weighted `P(r == 5) = 0.729 [0.605, 0.850]`). QA: [QA_MEGA_2026-11-26.md](QA_MEGA_2026-11-26.md).
- **Zero new model calls (`$0.00`):** CPU-only reanalysis (`scripts/analysis.py`) with `<= 1e-9` alignment gates against `cert-judge` Experiment `68` (`436` checks, max diff `3.33e-16`) and Experiment `03` (`480` checks, max diff `0.00e+00`).

## Reproduce

```bash
python3 experiments/07_proper_scores_are_not_trust_and_amortized_property_certification/scripts/analysis.py
python3 -m unittest experiments/07_proper_scores_are_not_trust_and_amortized_property_certification/scripts/test_analysis.py -v
```
