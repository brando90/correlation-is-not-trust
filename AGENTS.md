# Repo rules (as of 2026-11-24)

Project: correlation-is-not-trust (README title: do AI judges that correlate highly with humans still fail trustworthy judge properties?). Read README.md and docs/PROJECT_IDEA.md first.
Sibling repo, different question: brando90/judges-dont-need-psychometrics-or-corr is correlation vs CE (which objective to train judges on). This repo is bad judge properties despite high correlation. Do not mix their experiments.

- Commit straight to main. No branches, no open PRs, no issues. Keep main current: pull before work, push when done, print the commit link.
- Every experiment is a numbered folder experiments/NN_<name>/ with README.md, PREREG.md (committed before any run), scripts, results, RESULTS.md. Start from experiments/00_template/PREREG.md. Index in experiments/README.md, newest first.
- Date every note and doc (date in the title and at the top; "as of" date on updated parts).
- Reports: takeaway first, then simple tables. Name the harness and the model for every result. Give 95% intervals.
- Terms: say "solver agent", "solver LLM", "judge agent", "judge LLM". Say "scientific thesis" or "written thesis", never "thesis" alone. One precise term per concept.
- Experiments run on Brando's own plans only (Vals and Stanford lanes first, then personal), no extra charges on any account, real experiments never through agy, never Llama 4.
- Never print or commit a key or token.
- matplotlib figures use grid lines (ax.grid(True) / plt.grid(True)).
