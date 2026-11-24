# Experiment 04: `04_vb_fsc_equivalence_invariance_and_gaming_violations` (11-24-2026)

Date: `11-24-2026`

Goal: Test **Idea 1** (*"Human judgements is not sufficient to establish trust on judges – Foundational Stability Properties are needed too"*) on **VeriBench Formal Specification Coverage (`FSC`)** with zero new model calls (`$0.00`).

## Research Provenance

- Motivated by *"Certifying the Formal Judge"*, Slide 2 (Ideas 1–3 on project goal, scientific thesis pitch, and paper pitch; collaborator review discussion recorded on `11-24-2026`; unredacted slide screenshot omitted from this public repo because it displays collaborator email addresses in comment threads).
- Complements [Experiment 03](../03_human_corr_vs_axiomatic_properties_cert_judge/RESULTS.md) (Proof-Adequacy / `E42` dataset) by moving directly to **VeriBench `FSC`** across two cached datasets:
  - **Part A (`judge-reference-anchoring`, Exp 01):** `31` judge LLM / judge agent configurations across `M1_deployed_tc_v1`, `M2_blind_no_ref`, `M3_prompt_nonexcl_cot`, `M4_dual_ref`, `M5_harness_ref_hidden_lean`, and `M6_harness_cli_agent`, plus `5` configurations on real VeriBench solver outputs (`exp_c`).
  - **Part B (`trustworthy-theorem-judges`, Exp 23):** `claude-opus-5-5` via `claude -p --tools ''` (`0–5` human statement-coverage rubric, `judge_only`; `low` reasoning effort on `TinyVeriBench-256` `n = 30` and `perm12` `n = 12` vs. `compile/source-scan` `gated` and `gated_format` gates; `max` effort on `VB256` `n = 30` vs. Lean kernel `collectAxioms` `gated_kernel` and `full_pipeline` gates).

## Files

- [PREREG.md](PREREG.md) — preregistration committed before running analysis, plus dated complete-5-solver and post-Mega QA Stage 3 amendments.
- [RESULTS.md](RESULTS.md) — headline takeaways, full Part A / Part A2 / Part B tables with exact harnesses, models, efforts, and 95% task-cluster bootstrap intervals, plus figures.
- [QA_MEGA_2026-11-24.md](QA_MEGA_2026-11-24.md) — 3-stage sequential Mega QA audit log (planned `Codex -> Claude -> Codex`; executed `Codex -> Antigravity Pro fallback -> Codex` after Claude CLI login expiry).
- [scripts/analysis.py](scripts/analysis.py) — deterministic reanalysis, source-alignment assertions, CSV/JSON summaries, and plotting script (`ax.grid(True)`).
- [scripts/test_analysis.py](scripts/test_analysis.py) — synthetic unit tests + generated-artifact alignment & privacy checks.
- [results/summary.json](results/summary.json), [results/equiv_invariance_table.csv](results/equiv_invariance_table.csv), [results/exp_c_ref_swap_table.csv](results/exp_c_ref_swap_table.csv), [results/gaming_monotonicity_table.csv](results/gaming_monotonicity_table.csv) — aggregate numeric outputs.
- [results/fig1_vb_fsc_equivalence_and_order_invariance.png](results/fig1_vb_fsc_equivalence_and_order_invariance.png), [results/fig2_vb_fsc_gaming_monotonicity.png](results/fig2_vb_fsc_gaming_monotonicity.png) — figures.

## Reproduction

```bash
python3 experiments/04_vb_fsc_equivalence_invariance_and_gaming_violations/scripts/analysis.py
python3 -m unittest experiments/04_vb_fsc_equivalence_invariance_and_gaming_violations/scripts/test_analysis.py -v
```

Requires the local sibling cached datasets referenced in [PREREG.md](PREREG.md) (`$0.00`, zero API calls).
