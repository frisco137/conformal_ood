# Tier 4 — The value channel — NOT RUN

This directory is **empty because nothing in it has ever been run.** It is kept,
rather than deleted, so that the gap is visible in the tree rather than only in
a document.

`FINAL_NUMBERS.md` §8 ("Explicitly not measured") is the authoritative list.
`experiments.md` defines each experiment: what to run, the pass/fail threshold
fixed in advance, the falsifier, and the reviewer objection it closes.

Do not rename or renumber the experiments — the IDs are referenced across
`experiments.md`, `theory_and_claims.md` and `FINAL_NUMBERS.md`.

| ID | what it would establish |
|---|---|
| E4.1 | test MSE, NLL, calibration, coverage, martingale/sequential-consistency, prequential log loss — on the **same contexts** as the audit |
| E4.2 | the same battery pushed through the wrapped imitator |

**No MSE, NLL, calibration, coverage, martingale or prequential number exists
for any model on the audit's contexts.** `FINAL_NUMBERS.md` §8 states this.

`experiments.md` calls E4.1 the headline and says its figure goes **before** the
audit results in the paper: *a model that passes every published test and fails
the parameter-free structural conditions, on the same data.* Without it, T1 is a
proof about an adversarial construction rather than an empirical claim.

**Partially anticipated elsewhere, but not a substitute.** `phase_2` measures
coverage, interval score, NLL and a martingale test — but on **12 OpenML
datasets**, not on the audit's synthetic contexts, so it does not close E4.1's
"same model, same contexts" requirement. See `phase_2/PHASE2.md` Parts A and C.
