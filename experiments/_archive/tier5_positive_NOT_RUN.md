# Tier 5 — E5.1 RUN, E5.2–E5.4 outstanding

> **E5.1 was run on 2026-08-26.** `e5_1_positive_control.py` → `e5_1_results.json`,
> reported in `FINAL_NUMBERS.md` **Section 11**. A1 and A2 pass **120/120** across
> both wrapped controls on 60 real OpenML contexts through the identical
> pipeline — `negeig` exactly zero throughout. A3 passes 60/60 for the exact GP
> and 60/60 for the hierarchical GP unwrapped, but the *wrapped* hierarchical GP
> fails 3/60, which establishes an artifact floor for A3 (see the note added to
> `FINAL_NUMBERS.md` Section 5). Tangential curvature separates hierarchical from
> exact by 8.1e6x, as predicted.
>
> **The pipeline does not manufacture A1 or A2 violations.** The blocking gate is
> cleared. E5.2–E5.4 below are still outstanding.


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

**E5.1 was blocking and is now cleared** — see the banner above and
`FINAL_NUMBERS.md` Section 11.

**E5.2 note.** It was pre-registered as "TabPFN v2 materially closer to passing
than TabICL v2". The measured values run the other way — TabPFN is worse on both
A1 (`1.312` vs `0.580`) and A2 (`0.404` vs `0.177`), which is E5.2's stated
falsifier. But those numbers were **not produced by a registered E5.2 run**, and
`FINAL_NUMBERS.md` §8 lists "E5.2 as a registered test" as not run. The
distinction is what makes it a test rather than a fishing expedition, and it
should be stated carefully in the paper.
