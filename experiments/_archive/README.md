# `_archive/` — history, retained and out of the navigation

**Nothing here is cited by the paper.** Nothing here is deleted either, and two items are retained
for specific reasons that matter.

Live documents are in [`../phase_1/`](../phase_1/) and [`../phase_2/`](../phase_2/).

| file | what it is | why it is kept |
|---|---|---|
| `experiments.md` | **the dated pre-registration.** States every threshold and falsifier *before* the runs. Frozen, never edited. | **The E5.2 claim depends on the Biloš prediction being demonstrably dated.** Without this file the paper can only say the audit's numbers are inconsistent with a mechanism story; with it, the prediction is on record in advance. |
| `DEVIATIONS.md` | one row per experiment: run as specified, run with a stated deviation, not run, or falsifier fired | **The only record that E0.6's halt condition was breached knowingly** (`r₁ ≳ 1e-1` was pre-registered as "halt"; TabPFN measured `0.51`, TabSwift `0.91`, and the audit continued). Also the only place the E5.2 distinction is written out in full. |
| `phase_2_RESULTS.md` · `phase_2_LOCALISATION.md` · `phase_2_SEQUENTIAL.md` · `phase_2_UNCERTAINTY_EXPERIMENT.md` | the four Phase 2 source documents | Consolidated **unchanged** into `../phase_2/phase_2_record.md`. Kept so the trail from part document → consolidated record is checkable. |
| `phase_2_README_old.md` | Phase 2's old navigation file | Superseded by `../phase_2/experiments.md`. Carries a correction banner about the imitator fix. |
| `README_old_navigation.md` | the old `experiments/` navigation file | Describes the pre-2026-08-26 tree. |
| `tier2_audit_signpost.md` | signpost left when `tier2_audit/` was quarantined | Points at `quarantine/tier2_audit/`. |
| `tier3_robustness_NOT_RUN.md` | what Tier 3 would have established | Folded into `../phase_1/phase_1_results.md` §9.1. |
| `quarantine/` | **known-wrong code and outputs.** `__init__.py` raises on import. | Several retraction claims in `FINAL_NUMBERS.md` §1.3 are claims *about* this code; deleting it would make them uncheckable. Manifest in `quarantine/QUARANTINE.md`. |

## Why the pre-registration is not the live document

`experiments.md` here is a **pre-registration**: its value comes entirely from being dated and from
stating thresholds before the runs. Amending it destroys that value, so it is frozen — including its
inline **Status** lines, which were stale the day they were written.

The live descriptions of what was actually run are `../phase_1/experiments.md` and
`../phase_2/experiments.md`. Experiments that were planned and never run appear in neither; they are
in each phase's results document, under what is not established.
