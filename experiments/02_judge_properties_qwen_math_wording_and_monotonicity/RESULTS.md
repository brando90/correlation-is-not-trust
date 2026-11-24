# RESULTS 02: do judge LLMs that correlate with the gold reference fail basic properties? (as of 2026-11-24)

Harness: analysis only (Python 3, numpy, scipy; analysis.py), no model calls, cost $0. Judge LLMs: Qwen2.5-Instruct 1.5B/3B/7B/14B (scores saved earlier by brando90/judge-free-eval scripts/bon_judge.py, vLLM, two rubric wordings a and b, conditions noref and ref). Solver LLM: Llama-3.1-8B-Instruct, 176 MATH-500 problems x 64 solutions. Pre-registered in PREREG.md (commit before the run). Alignment control passed (ref 14B pooled AUROC 0.913 >= 0.85).

## Takeaway
- In this pilot H1 is not supported: the judge LLMs with the highest correlation with the gold reference also fail the properties least. Across the 8 judge rows, pooled correlation C and the wrong-scored-at-least-as-high rate P2 move opposite: Spearman -0.98 (n=8); C and the wording flip rate: -0.93.
- The 4 ref judges (they see the gold answer; C = 0.70 to 0.78) have P2 = 0.06 to 0.09 and wording flip 0.04 to 0.10. The 4 noref judges (C = 0.30 to 0.48) have P2 = 0.23 to 0.32 and flip 0.13 to 0.24.
- Pre-registered verdict: mixed, no claim. The rule "supported" (a row with C at or above the median has P2 >= 0.20 or flip >= 0.15) is not met (best-C rows: P2 <= 0.085, flip <= 0.100); the rule "falsified" needs every such row to have flip < 0.05, but two have 0.100 (3B ref) and 0.050 (1.5B ref).
- Why this test cannot settle H1: P2 uses the same gold bit as C (it is close to 1 minus the within-problem AUROC), so it cannot separate itself from the correlation. Only P1 is independent of the gold bit, and the ref judges are anchored on the gold answer, which makes them stable. A fair H1 test needs properties that do not use the reference and judges not anchored on it: renamed variables, an inserted bug, order, on VeriBench, with human ratings.

## Table (95% cluster bootstrap over 176 problems, seed 20261124)
| judge LLM row | C pooled r | within-problem Spearman | P2 wrong >= correct | P1 wording flip | a-b Spearman |
|---|---|---|---|---|---|
| Qwen2.5-1.5B noref | 0.382 [0.336, 0.425] | 0.344 | 0.287 [0.266, 0.308] | 0.238 [0.225, 0.250] | 0.650 |
| Qwen2.5-3B noref | 0.303 [0.252, 0.349] | 0.291 | 0.318 [0.295, 0.343] | 0.179 [0.168, 0.192] | 0.763 |
| Qwen2.5-7B noref | 0.448 [0.402, 0.489] | 0.409 | 0.246 [0.226, 0.269] | 0.131 [0.121, 0.142] | 0.848 |
| Qwen2.5-14B noref | 0.480 [0.441, 0.517] | 0.429 | 0.233 [0.213, 0.254] | 0.163 [0.150, 0.177] | 0.769 |
| Qwen2.5-1.5B ref | 0.775 [0.746, 0.802] | 0.706 | 0.062 [0.049, 0.076] | 0.050 [0.041, 0.060] | 0.864 |
| Qwen2.5-3B ref | 0.701 [0.665, 0.734] | 0.676 | 0.080 [0.066, 0.096] | 0.100 [0.084, 0.117] | 0.826 |
| Qwen2.5-7B ref | 0.766 [0.732, 0.797] | 0.677 | 0.079 [0.064, 0.095] | 0.043 [0.036, 0.051] | 0.915 |
| Qwen2.5-14B ref | 0.749 [0.717, 0.777] | 0.667 | 0.085 [0.069, 0.103] | 0.060 [0.052, 0.070] | 0.864 |

Figure: results/fig_corr_vs_properties.png. Intervals are cluster bootstrap over problems; the 8 judge rows share the same 176 problems, so they are not independent.

## Caveats
- No human labels exist in this data: C is correlation with gold correctness, not with humans.
- Small judges (1.5B to 14B Qwen) only; frontier judge LLMs and judge agents are untested here.
- The rubric wordings a and b are two wordings, not many; P1 is a lower bound on wording sensitivity.

Reproduce: `python analysis.py <judge-free-eval checkout at d1d53e8c88e32b3695d0c06379c129d65bde63ce> results` (about 3 minutes).
