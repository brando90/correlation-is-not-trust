# Correlation is not trust: is fitting human judgments even a good target for a judge? (as of 2026-11-24)

## The crux (Brando, Nov 24 2026)
Answer the hypothesis: is fitting human judgments even good? Humans are inconsistent, so do we really want judges (metrics) fitted to them? Axiomatic, fundamental properties of judges and metrics are what seems valuable and interesting to study.

## Hypothesis
H1: A judge or metric can correlate highly with human judgments and still violate basic axiomatic properties, so correlation with humans is not sufficient for trust. H2: humans themselves violate some of these properties (inconsistent across repeats, order, wording), so fitting humans may not even be the right target. Falsifiable: H1 fails if every high-correlation judge passes all the properties; H2 fails if humans pass them all.

## Properties to test (first list, to be frozen in a PREREG before any run)
(a) renaming variables must not change the score; (b) a worse file must not score higher; (c) a file with an inserted bug must not score as well as the original; (d) the same item rated twice gives the same score (consistency); (e) permuting the order of items does not change a score.
Metrics compared: judge LLMs, judge agents, human labels, pass@k, boxed accuracy, gold-reference tests.

## Source
Slides "Certifying the Formal Judge", slide 2 (Yegor's critique: "strict prerequisite", "spurious proxy"): Idea 1 human judgments are not sufficient to establish trust in judges, foundational stability properties are needed too (the one Brando likes most); Idea 2 properties as a cheaper alternative to humans; Idea 3 calibrate once, deploy forever.

## Layout
experiments/NN_name/ (numbered, dated, PREREG.md before any run), docs/, CLAUDE.md, AGENTS.md.
