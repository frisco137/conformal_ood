# Tier 3 — Robustness — NOT RUN

This directory is **empty because nothing in it has ever been run.** It is kept,
rather than deleted, so that the gap is visible in the tree rather than only in
a document.

`FINAL_NUMBERS.md` §8 ("Explicitly not measured") is the authoritative list.
`experiments.md` defines each experiment: what to run, the pass/fail threshold
fixed in advance, the falsifier, and the reviewer objection it closes.

Do not rename or renumber the experiments — the IDs are referenced across
`experiments.md`, `theory_and_claims.md` and `FINAL_NUMBERS.md`.

| ID | what it would establish | why it matters |
|---|---|---|
| E3.1 | context distribution: GP-drawn vs SCM-generated vs real OpenML | `experiments.md` calls this **the largest standing risk** and says the objection "you audited the model off-distribution" is **fatal if unaddressed**. Phase 2 §9 found TabICL's `tr J` reaching 100 on real data against `‖J−I‖_F/‖J‖_F = 0.85` on the audit's synthetic contexts — a context-family effect of exactly the kind E3.1 exists to find. That is a reason to run it, not a result about it. |
| E3.2 | in-sample vs query regime | the Jacobian asks the model to predict *at* its own context locations |
| E3.3 | context-design sweep over `d`, noise, duplicate count, `n` | one recipe throughout: `n=100, d=5, sigma=1.0`, 10 duplicated rows |
| E3.4 | query row vs context block consistency | curvature is measured on one, symmetry on the other |
| E3.5 | seed and checkpoint variation | "one checkpoint" |

Everything reported to date is a single context family.
