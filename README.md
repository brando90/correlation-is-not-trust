# Correlation is not trust: do AI judges that correlate highly with humans still fail trustworthy judge properties? (as of 2026-11-24)

**What this repo is (and how it differs from its sibling):** this repo asks whether a judge that **correlates highly with human ratings** still **fails basic trustworthy judge properties** (renaming invariance, a worse file must not score higher, an inserted bug must not score as well as the original, consistency across repeats, item order). The target is *bad judge properties despite high correlation*.
The sibling repo [judges-dont-need-psychometrics-or-corr](https://github.com/brando90/judges-dont-need-psychometrics-or-corr) is a different question: **correlation vs cross-entropy (CE)** as the training and selection objective for judges (CE-trained judges, proper scores RPS, CE, TVD; no need for G theory or D theory, really just CE and lots of data, scaling laws as always; working title "No Free Lunch of AI Judges", Brando Nov 24 12:20). That repo is about *which objective to fit*; this repo is about *whether fitting humans is the right target at all*. Read both together.


## The crux (Brando, Nov 24 2026)
Original question title (kept for the record): "is fitting human judgments even a good target for a judge?" Brando asked on Nov 24 12:18 for "trustworthy properties" in the title.
Answer the hypothesis: is fitting human judgments even good? Humans are inconsistent, so do we really want judges (metrics) fitted to them? Axiomatic, fundamental properties of judges and metrics are what seems valuable and interesting to study.

## Hypothesis
H1: A judge or metric can correlate highly with human judgments and still violate basic axiomatic properties, so correlation with humans is not sufficient for trust. H2: humans themselves violate some of these properties (inconsistent across repeats, order, wording), so fitting humans may not even be the right target. Falsifiable: H1 fails if every high-correlation judge passes all the properties; H2 fails if humans pass them all.

## Properties to test (first list, to be frozen in a PREREG before any run)
(a) renaming variables must not change the score; (b) a worse file must not score higher; (c) a file with an inserted bug must not score as well as the original; (d) the same item rated twice gives the same score (consistency); (e) permuting the order of items does not change a score.
Metrics compared: judge LLMs, judge agents, human labels, pass@k, boxed accuracy, gold-reference tests.

## Source
Slides "Certifying the Formal Judge" (https://docs.google.com/presentation/d/1ZYCAQ5gGwmjaipT2nNRTe-mEYxOtgpiLqGogbrgP_Hc/edit?slide=id.g3e67486055a_0_29#slide=id.g3e67486055a_0_29), slide 2. Details: docs/PROJECT_IDEA.md. Deck: [Certifying the Formal Judge](https://docs.google.com/presentation/d/1ZYCAQ5gGwmjaipT2nNRTe-mEYxOtgpiLqGogbrgP_Hc/edit?slide=id.p#slide=id.p), see slide 2.

Slides "Certifying the Formal Judge", slide 2 (Yegor's critique: "strict prerequisite", "spurious proxy"): Idea 1 human judgments are not sufficient to establish trust in judges, foundational stability properties are needed too (the one Brando likes most); Idea 2 properties as a cheaper alternative to humans; Idea 3 calibrate once, deploy forever.

## Layout
experiments/NN_name/ (numbered, dated, PREREG.md before any run), docs/, CLAUDE.md, AGENTS.md.

Next step (as of 2026-11-27): [NEXT.md](NEXT.md). Experiments 08 and 09 are CPU reanalyses: task-held-out calibration of the experiment 05 scores, and the selection gap on the experiment 03 table. The remaining work is a prospective zero-exposure judge run.

## Citation (as of 2026-11-24)
```bibtex
@misc{miranda2026correlationisnottrust,
  author = {Miranda, Brando and Koyejo, Sanmi},
  title  = {Correlation is not trust: do AI judges that correlate highly with humans still fail trustworthy judge properties?},
  year   = {2026},
  note   = {Work in progress, started 2026-11-24},
  url    = {https://github.com/brando90/correlation-is-not-trust}
}
```
Brando Miranda, Sanmi Koyejo. Correlation is not trust: do AI judges that correlate highly with humans still fail trustworthy judge properties? 2026. https://github.com/brando90/correlation-is-not-trust
