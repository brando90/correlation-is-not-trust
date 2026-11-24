# Project idea (as of 2026-11-24)

## Copied on 2026-11-24
Source: Google Slides "Certifying the Formal Judge", slide 2, title "Ideas for Project Goal/Thesis/Paper Pitch(es)" (https://docs.google.com/presentation/d/1ZYCAQ5gGwmjaipT2nNRTe-mEYxOtgpiLqGogbrgP_Hc/edit). Body copied word for word (the slide text itself ends mid-sentence at "sub").

Idea1: “Human judgements is not sufficient to establish trust on judges – Foundational Stability Properties are needed too” → most impactful 9.5/10 (paradigm shift)
“Judges use as a metrics/scores for AI Benchmarks can correlate human judgements (& have stability/low variance) yet fail fundamental falsifiable properties  or basic sanity checks (eg equivalence; judge can fail to detect renaming of vars in files vs gold ref, or monotonicity to worse files; fail to detect agent/solver file has bugs vs gold ref)
→ just checking if the humans have these properties
→ I like this idea the most be idk if we have evidence that common judges used in the lit correlate with human judgements &/or are stable yet fail fundamental sanity checks. 
Idea2: “Foundational Stability Properties correlate with Human judgements; so they are a cheaper alternative to human judgments”
Pro: generate gold ref once instead of re asking humans to judge a billion times with a billion agents. The “ablations/bug insertion” can be automated with an LLM + since it’s basic properties it’s easy to trust it will work (plus we’ve checked it work via human judgements)
Con1: for now requires gold ref. Con2: expts have only been applied to benchmarks so idk if it works for scalable oversight in general (like tvd-mi/zach’s) that has a proof of “stability” in general (I think afaik)
Idea3: "Property Certification as Amortized Scalable Oversight: Calibrate Once, Deploy Forever"
Pitch: validating LLM judges against humans is the standard reliability check but doesn't scale — every new judge, prompt, or model needs fresh annotation. Properties (identity, bug-monotonicity, spec-monotonicity, repeat-stability) collectively predict human alignment at Spearman ρ ≈ TODO (n=TODO judges, p=TODO, variance) after a single one-time calibration on gold refs. 
This converts trust validation from O(N labels × M judges) to O(N labels + property scores per new judge for free).
so we validated these properties once against human judgements and some SOTA agent outputs – and claim that you can re-use these properties (PS: we’d probably need to show some generalization on a sub
Pro1: reframes the contribution as economics of oversight, not yet-another-proxy-metric — connects to scalable oversight as a research agenda (Bowman et al. 2022). Pro2: pre-empts Yegor's "strict prerequisite" / "spurious proxy" critique — properties don't replace humans, they extend humans to new judges via calibration ("use humans once, properties forever after"). Pro3: turns the BAD_STRINGY caveat into a feature — calibration is needed precisely because some judges have moderate human-corr (≈0.45) but fail individual properties (12.5% weak monotonicity); calibration identifies which property patterns indicate trustworthy judges. Pro4: methodological contribution (calibration protocol) is stronger than metric contribution → elevates from COLM-tier to NeurIPS-tier.
Con1: still requires gold refs for the one-time calibration step (shared with Idea 2). Con2: generalization claim needs hold-out validation (split 12 judges into 8 calibration plus 4 held-out, show JTI predicts held-out human-corr) — about three to four days of work, not yet run. Con3: doesn't go as far as Idea 1's paradigm-shift; doesn't claim humans are insufficient, only that they're expensive.

## Brando, Nov 24 2026
"the crux is to answer the hypothesis / if fitting human jdugements is even good / humans are inconsistenct etc / do we really want that as our judges? / axiomait cproperties of fudnamentla proeprties of judges / metrics / is what seems valuabel iteresting to study"

## Brando's paste of slide 2, Nov 24 2026 09:46
Verbatim, as he pasted it in Slack (with his formatting marks).

• *Idea1*: “_Human judgements is not sufficient to establish trust on judges – Foundational Stability Properties are needed too_” → most *impactful 9.5/10* (*paradigm shift*)
    ◦ “Judges use as a metrics/scores for AI Benchmarks can correlate human judgements (& have stability/low variance) yet fail fundamental falsifiable properties  or basic sanity checks (eg equivalence; judge can fail to detect renaming of vars in files vs gold ref, or monotonicity to worse files; fail to detect agent/solver file has bugs vs gold ref)
        ▪︎ → just checking if the humans have these properties
    ◦ → I like this idea the most be idk if we have evidence that common judges used in the lit correlate with human judgements &/or are stable yet fail fundamental sanity checks. 
• *Idea2: “_Foundational Stability Properties correlate with Human judgements; so they are a cheaper alternative to human judgments_”*
    ◦ Pro: generate gold ref once instead of re asking humans to judge a billion times with a billion agents. The “ablations/bug insertion” can be automated with an LLM + since it’s basic properties it’s easy to trust it will work (plus we’ve checked it work via human judgements)
    ◦ Con1: for now requires gold ref. Con2: expts have only been applied to benchmarks so idk if it works for scalable oversight in general (like tvd-mi/zach’s) that has a proof of “stability” in general (I think afaik)
• *Idea3*: “_Property Certification as Amortized Scalable Oversight: Calibrate Once, Deploy Forever_”
    ◦ Pitch: *validating LLM judges against humans is the standard reliability check but doesn’t scale* — *every new judge, prompt, or model needs fresh annotation. Properties* (identity, bug-monotonicity, spec-monotonicity, repeat-stability) *collectively predict human alignment at Spearman ρ ≈ TODO (n=TODO judges, p=TODO, variance) after a single one-time calibration on gold refs*. 
        ▪︎ This converts trust validation *from O(N labels × M judges) to O(N labels + property scores per new judge for free)*.
        ▪︎ so we validated these properties once against human judgements and some SOTA agent outputs – and claim that you can re-use these properties (PS: we’d probably need to show some generalization on a sub
    ◦ Pro1: reframes the contribution as economics of oversight, not yet-another-proxy-metric — connects to scalable oversight as a research agenda (Bowman et al. 2022). Pro2: pre-empts Yegor’s “strict prerequisite” / “spurious proxy” critique — properties don’t replace humans, they extend humans to new judges via calibration (“use humans once, properties forever after”). Pro3: turns the BAD_STRINGY caveat into a feature — calibration is needed precisely because some judges have moderate human-corr (≈0.45) but fail individual properties (12.5% weak monotonicity); calibration identifies which property patterns indicate trustworthy judges. Pro4: methodological contribution (calibration protocol) is stronger than metric contribution → elevates from COLM-tier to NeurIPS-tier.
    ◦ Con1: still requires gold refs for the one-time calibration step (shared with Idea 2). Con2: generalization claim needs hold-out validation (split 12 judges into 8 calibration plus 4 held-out, show JTI predicts held-out human-corr) — about three to four days of work, not yet run. Con3: doesn’t go as far as Idea 1's paradigm-shift; doesn’t claim humans are insufficient, only that they’re expensive.
