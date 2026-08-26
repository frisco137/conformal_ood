# Tier 5 — Positive control — NOT RUN

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
| E5.1 | exact GP and hierarchical GP on **real** tabular data, through the identical pipeline including the projection, passing A1/A2/A3 |
| E5.2 | TabPFN v2 against the Biloš prediction, pre-registered |
| E5.3 | adaptive-bandwidth Nadaraya–Watson — the probing rebuttal |
| E5.4 | McCarter duplication-anomaly reproduction |

**E5.1 is blocking.** `experiments.md`: "Without this the paper does not go out",
and "Nothing is reported before a real Bayesian model passes on real data." Its
falsifier is that if the controls fail on real data, the pipeline rather than the
models is producing the violations, and nothing else in the paper survives.

Every control measured to date is on synthetic GP contexts
(`FINAL_NUMBERS.md` §2). `phase_2` runs an exact-GP and hierarchical-GP control
on real OpenML data, but as an **oracle baseline for the uncertainty and
detection experiments**, not as an A1/A2/A3 audit through the projection. It
does not close E5.1.

**E5.2 note.** It was pre-registered as "TabPFN v2 materially closer to passing
than TabICL v2". The measured values run the other way — TabPFN is worse on both
A1 (`1.312` vs `0.580`) and A2 (`0.404` vs `0.177`), which is E5.2's stated
falsifier. But those numbers were **not produced by a registered E5.2 run**, and
`FINAL_NUMBERS.md` §8 lists "E5.2 as a registered test" as not run. The
distinction is what makes it a test rather than a fishing expedition, and it
should be stated carefully in the paper.
