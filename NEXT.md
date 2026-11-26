# Next experiment (as of 2026-11-26)

## Decision

The next experiment is `experiments/06_why_human_correlation_misses_formal_properties/`. It tests why a correlation with human ratings can sit next to a failure of formal judge properties. The cuts are frozen in that folder's `PREREG.md`. The CPU arm had not been run when this file was first committed. Findings from the CPU arm, if they are filled in below, were written after that preregistration commit.

## Why this, and not another correlation table

Experiments 03 and 05 already show the coexistence. On Lean 4, 22 of 22 judge configurations with held-out Spearman at least 0.70 against the 3-expert `target_expert` panel still fail a preregistered property. On the new calls in experiment 05, the one configuration whose Spearman interval lies above 0, judge LLM Claude Opus 5.5 in the Claude Code CLI, still scores an inserted bug at least as high as the gold in 56.8% of pairs. A further human-correlation column on the VeriBench formal-specification-coverage runs from experiment 04 would repeat the "whether." It is the natural later measurement if a slide needs the link inside that coverage score, and it needs human ratings that are not in this repository.

The open scientific question is the mechanism. Three different stories fit the same published tables, and they imply different fixes:

1. The correlation files and the property files are not the same files, so the Spearman cannot constrain the property. Selecting on human correlation never sees the axiom pair.
2. The repeat failures are either large swing around a stable median, or a `1e-12` tolerance counting jitter. Only the first is a property failure worth citing.
3. The bug "failures" are ceiling ties on a statement rubric, or they are real penalties that the tie rate hides. Those two stories disagree about whether the judge LLM was asked the wrong question.

Experiment 02 is the contrast case, not a contradiction: there the property reused the gold bit that the correlation used, so the highest-correlation judge LLMs failed the property least. That is what happens when the property is not free of the correlation. Experiment 06 is the case where the property is free of it.

## What is already specified and not run

There is no GPU arm. Nothing in this experiment is a fine-tune.

The API arm is one implementation rubric on judge LLM Claude Opus 5.5, harness Claude Code CLI, Stanford plan first. The command, the frozen rubric, and the support rule are in the PREREG. It was not launched. Estimated cost if it is launched later: about $3 to $5 API-equivalent, $0 out of pocket, on Brando's Stanford plan, then Vals if that lane is paused. No agy, no Llama 4.

```bash
python3 experiments/06_why_human_correlation_misses_formal_properties/scripts/judge_implementation_rubric.py \
  --probes ~/at3450/probes.jsonl \
  --lanes stanford-claude \
  --out ~/at3450/exp06_arm_I \
  --confirm-plan-spend
```

## Findings (CPU arm)

Not run yet. This section stays empty until `scripts/analysis.py` has been executed, which happens only after the PREREG commit.
