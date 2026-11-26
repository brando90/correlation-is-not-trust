# Next experiment (as of 2026-11-26)

## Decision

Experiment 06 is the next measurement: why a correlation with human ratings can sit beside a failure of formal judge properties. The cuts are frozen in `experiments/06_why_human_correlation_misses_formal_properties/PREREG.md` (commit `f59f959`, before the CPU run). The CPU results are in that folder's `RESULTS.md`.

A new human-correlation column on the VeriBench formal-specification-coverage runs from experiment 04 would repeat the "whether." Experiments 03 and 05 already show the coexistence. That correlation is the later measurement if a slide needs the link inside the coverage score, and the human ratings are not in this repository.

## CPU findings

Overall verdict: **mixed**. Cost $0. No new model calls.

- The natural-file ids and the property ids in experiment 05 are disjoint, and a constructed 45-item example has median Spearman 1 with a repeat-violation rate of 1. The correlation does not constrain the property.
- **H-material supported** for judge LLM Claude Opus 5.5 (`claude-opus-5-5`), harness Claude Code CLI, Stanford plan, rubric `P4_CHECKLIST`: the share of bug pairs that drop by more than 0.20 is **0.108 [0.000, 0.226]** (37 pairs, 11 tasks). 27 of 37 pairs are ceiling ties.
- **H-range mixed.** Median within-item repeat range on the 22 high-correlation configurations is **0.097** (cut for a clear swing was 0.10; cut for jitter was 0.02). Nine of 22 are at or above 0.10. One is at or below 0.02.
- **H-split mixed** on the 56 judge configurations (judge LLMs on `vb103_judge`, judge agents on `terminus2`). Partial Spearman of held-out correlation with statement weakening, given bug ties, is **−0.766 [−0.847, −0.650]**, p = 0.0002. Partial with bug ties, given weakening, is **+0.302 [0.076, 0.478]**, p = 0.0240. At a fixed weakening rate, higher human correlation goes with more bug ties. The p-values are conditional on this logged set, which shares models, prompts, and tasks.

Gold-as-candidate scores sit about 0.78 above the natural files the correlation uses (B: 0.778 [0.725, 0.826]). B scores 52 of 75 natural files at 0.20 or below while the gold mean is 0.996 [0.990, 1.000].

Every logged `M1_deployed_tc_v1` row in experiment 04 has `harsh_flip_eq10 / V_equiv = 1`, so an equivalence change is also a full-credit bit flip. The two frontier rows were already visible when the preregistration was written. The other four rows were not. Mean absolute shifts still differ (0.16 to 0.64).

## Next command

There is no GPU arm. The unpaid compute is the CPU script already run. The remaining arm is one implementation rubric on the same judge LLM and harness as configuration B. It was not launched. About $3 to $5 API-equivalent, $0 out of pocket, Stanford plan first, then Vals if that lane is paused. No agy, no Llama 4.

```bash
python3 experiments/06_why_human_correlation_misses_formal_properties/scripts/judge_implementation_rubric.py \
  --probes ~/at3450/probes.jsonl \
  --lanes stanford-claude \
  --out ~/at3450/exp06_arm_I \
  --confirm-plan-spend
```

Support on that arm, from the PREREG: V_bug ≤ 0.20 with interval upper bound < 0.40, and mean drop ≥ 0.30, on a complete file of 24 gold scores and 37 bug scores. If the ceiling ties remain, the failure is in the judge LLM after the rubric has asked about the definition. If they move, the failure on the logged `P4_CHECKLIST` scores was the statement target.
