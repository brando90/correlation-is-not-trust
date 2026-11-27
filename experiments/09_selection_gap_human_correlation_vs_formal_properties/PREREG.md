# PREREG: 09_selection_gap_human_correlation_vs_formal_properties (date: 2026-11-27)

Commit this file before any run. Do not change it after the first run; add a dated "Amendment" section instead.

Zero new model calls (`$0.00`). CPU-only Python 3 (`numpy`, `scipy`, `matplotlib`). The only input is the public experiment 03 table `experiments/03_human_corr_vs_axiomatic_properties_cert_judge/results/judge_table.csv`.

## Prior-exposure disclosure

The association between these property rates and held-out human correlation is already published. It is not a finding of this experiment.

- Experiment 03: `22/22` configurations with `rho_heldout >= 0.70` fail `V_weak`, `V_rep`, or `V_bug`. Marginal Spearman of `rho_heldout` with `V_weak` is `-0.747`, with `V_bug` is `-0.164`, on the 56 real configurations.
- Experiment 06: partial Spearman of `rho_heldout` with `V_weak` given `V_bug` is `-0.766`.
- Experiment 07: `ATI_equal`, the mean of the six pass rates `1 - V` for `V_ms`, `V_weak`, `V_vac`, `V_overc`, `V_fmt`, `V_rep`, has Spearman `0.7883251594650563` with `rho_heldout` on those 56 configurations (the `lomfo_oof_all` / `ATI_equal` / `exp3_rho_spearman` cell). On the canonical test split (`n = 18`) that Spearman is `0.699`. The nonnegative ridge in experiment 07 was fit to proper-score `RPS`, not to `rho_heldout`. Its reported regret is `RPS` regret of one selected configuration, not a per-family comparison of the human-correlation pick against the property pick.

Not previously computed: inside each base-weight family, whether the configuration with the highest `rho_heldout` is the same configuration as the one with the highest property index, and how many more preregistered property gates the correlation pick fails.

## Hypothesis (one sentence)

H-gap: selecting the judge configuration with the highest held-out human correlation disagrees with selecting the configuration with the highest formal-property index, and the correlation-selected configuration fails more property gates.

## Properties and metrics

Eligible rows: `kind == llm`, `rho_heldout` finite, `rho_heldout_coverage >= 0.90`. The run stops if that set is not exactly 56 rows. The four `control` rows are excluded from every selector. They may be described; they do not enter a verdict.

Base-weight families, one held-out block each. A `terminus2` judge agent stays with the judge LLM that uses the same base weights, so a prompt from a family is never in both sides of a fit:

| Family id | Rows |
|---|---|
| `gem31h` | `gem31h` |
| `qwen` | `qwen36`, `t2_qwen` |
| `sonnet` | `sonnet5m`, `t2_sonnet5` |
| `opus55x` | `opus55x` |
| `luna` | `luna6h`, `t2_luna` |
| `sol6h` | `sol6h` |

The run stops if these six blocks are not a partition of the 56 rows.

### Selectors

Inside one family, ties within `1e-12` on the selection key are broken by the lexicographically smallest `config` string. The tie break does not look at property failures or at `rho_heldout`.

- `H`: highest `rho_heldout`. This selector uses the human labels of that family. It is the human-correlation choice.
- `P`: highest `ATI_equal = mean(1 - V)` over `V_ms`, `V_weak`, `V_vac`, `V_overc`, `V_fmt`, `V_rep`. `V_bug` is not in `ATI_equal` (experiment 03's construct boundary on `bm_gold`). `P` does not use `rho_heldout`.
- `F`: highest `w · x` on the six pass rates, where `w` is a nonnegative ridge fit to `rho_heldout` on the other five families only. Parameterization matches experiment 07: center `X` and `y` on the training rows, nonnegative least squares of the centered system stacked with `sqrt(lambda) * I`, `lambda` chosen from `{0.01, 0.1, 1, 10}` by inner leave-one-training-family-out Spearman (higher is better; the first `lambda` wins an exact tie). The intercept does not change the argmax. If `w · x` varies by less than `1e-8` across the held-out family, `F` is `flat` for that family.

`P_bug`, the mean of seven pass rates including `1 - V_bug`, is a descriptive sensitivity. It is not a selector in the verdict.

### Failure gate

A property fails when its published `value >= 0.20` and its published `ci_lo > 0.05` (experiment 03's rule, read from `judge_table.csv`, not recomputed). `fail_count` is the number of failures among the six properties in `ATI_equal`. `idea1_fail` is true when `V_weak` or `V_rep` fails.

### Census (the verdict uses these counts, not a bootstrap)

Six families.

- `disagree_HP`: number of families where `H` and `P` name different configs.
- `disagree_HF`: number of non-flat families where `H` and `F` differ. Families where `F` is flat are counted in `n_flat` and dropped from this fraction's denominator.
- `mean_rho_regret_P`: mean of `rho(H) - rho(P)`. Same for `F` over non-flat families.
- `mean_fail_gap_P`: mean of `fail_count(H) - fail_count(P)`. Same for `F` over non-flat families.
- `n_H_idea1_fail`, `n_P_idea1_fail`: how many of the six `H` picks and six `P` picks have `idea1_fail`.

A family-cluster bootstrap of these means (2,000 draws, seed `20261127`) may be printed only as a descriptive extra. The verdict does not use it. Six families are a census of this logged panel.

### Secondary prediction check

`H-pred` asks whether a ridge whose target is `rho_heldout` (not `RPS`) ranks the held-out families. Out-of-fold score is `w · x` from the other families. Spearman of the 56 out-of-fold scores with `rho_heldout` (SciPy, average ranks for ties). 95% interval: resample the 56 configurations, 2,000 draws, seed `20261127`. Two-sided permutation `p`: shuffle `rho_heldout`, 5,000 shuffles, same seed on a separate generator, add-one `(count + 1) / (valid + 1)` of absolute Spearman. If fewer than 1,500 bootstrap draws or 1,500 permutations are defined, `H-pred` is not computable.

This check is secondary. Experiment 07 already reported Spearman `0.788` for the unfitted index. A large `H-pred` Spearman mostly says the fitted weights recover a similar order.

### Alignment gate (the run stops before a verdict if it fails)

Spearman of `ATI_equal` with `rho_heldout` on the 56 rows equals `0.7883251594650563` within `1e-9`.

### What would falsify it

Primary verdict `H-gap`, selector `P` against selector `H`:

- Supported: `disagree_HP >= 3` and `mean_fail_gap_P >= 0.5`.
- Falsified: `disagree_HP <= 1` or `mean_fail_gap_P <= 0`.
- Otherwise mixed.

Secondary verdict `H-gap-fitted`, selector `F`, same cuts, on the non-flat families only. Not computable if fewer than 4 families are non-flat. It does not override `H-gap`.

Secondary verdict `H-pred`:

- Supported: Spearman point `>= 0.40`, interval lower bound `> 0`, and `p < 0.05`.
- Falsified: point `<= 0.15`, or `p >= 0.05` and point `< 0.30`.
- Otherwise mixed. Not computable when the draw rule above trips.

## Data

`judge_table.csv` only. No human TSV, no Lean source, no new model calls.

## Baselines

Selector `H` is the human-correlation baseline. Selector `P` is the parameter-free property index already defined as `ATI_equal` in experiment 07. Selector `F` is the same functional form as experiment 07's ridge, with the training target changed from `RPS` to `rho_heldout`.

## Models and harnesses

No model is called. The 56 rows are judge LLMs on `vb103_judge` and judge agents on `terminus2`, as labeled in `judge_table.csv`.

## Cost estimate

`$0` out of pocket. API-equivalent `$0`. No plan lane is used.
