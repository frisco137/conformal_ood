# Quarantine

Code and outputs that are known wrong or known superseded. **Nothing here may
produce a number for the paper.** It is retained because several retraction
claims in `FINAL_NUMBERS.md` §1.3 are claims *about* this code, and deleting it
would make those claims uncheckable.

`__init__.py` raises on import, so no live module can reach any of this through
the package path.

Quarantined 2026-08-26. Nothing was deleted; every item is recoverable from git
history and from its path below.

---

## 1. `tier2_audit/` — the entire directory

**What is wrong.** Every `asym` under this directory was computed as
`‖(J − Jᵀ)/2‖_F / ‖J‖_F`, **exactly half** the canonical definition fixed in
`theory_and_claims.md`. Compounding that:

| defect | where | effect |
|---|---|---|
| halved `asym` | `batch1_tabicl.py`, `batch2_tabpfn_5seed.py`, `batch3_tabswift.py`, `batch4_lastrun.py`, `audit_phase1_metrics.py`, `instrument_c_d.py`, `rebuttal_5_standardize.py` | every A1 number is 2× too small |
| `negfrac` returned under the name `negeig` | `instrument_c_d.py:89` | consumed by eight scripts; propagated to `audit_e_results.json`, `instrument_c_d_results.json` |
| unseeded `get_Q` | 13 separate local copies | no measurement in here is reproducible |
| profile residual always `0.00e+00` | `item3.py:95` | rank-deficient design; numpy returns an empty residuals array. True residual on the same fit is `4.786` |
| `sum(neg eigs)/sum(eigs)` reported as `negfrac` | `batch4_lastrun.py:69` | a third, undocumented convention |
| cannot execute at all | `batch4_lastrun.py:16` | imports `experiments.core.synthetic`, which does not exist |

**What supersedes it.** `FINAL_NUMBERS.md` in full. §1.2 states plainly that no
number in that document is taken from these files, and every model figure was
re-measured through `core/metrics.py`.

**Two exceptions you must know about — this directory is still cited.**
`FINAL_NUMBERS.md` deliberately carries two *historical* values from here,
labelled as historical at each appearance and shown only beside a value measured
in the current run (§9.2):

| FINAL_NUMBERS § | value | old path | new path |
|---|---|---|---|
| 3.2, 4.2, 9.2 | TabPFN `t=1e-2` `asym 1.3309 ± 0.0443`, `negeig 0.4003` | `tier2_audit/results_tabpfn_5seed.json` | `_quarantine/tier2_audit/results_tabpfn_5seed.json` |
| 3.1 | TabICL reproduction check `asym 0.5802 ± 0.0784`, `negeig 0.1769 ± 0.0464` | `tier2_audit/results_batch1.json` | `_quarantine/tier2_audit/results_batch1.json` |

Both are **data files, not code**, both are quoted with the doubling stated, and
the TabICL pair was independently re-measured this run and reproduces to four
significant figures. They are safe to keep citing. A signpost is left at the old
location (`experiments/tier2_audit/README.md`) so that following
`FINAL_NUMBERS.md` §9.1 lands on a pointer rather than a missing directory.

**`FINAL_NUMBERS.md` was not edited for this move.** See "Open items" below.

### 1a. `tier2_audit/e2_6_mechanism.py` — called out separately

Inside the directory above, so already quarantined, but it fails for its own
reason. Its verdict is

```python
if min(err_jac_gp, err_jac_ridge, err_jac_nw, err_jac_nn) > 0.5:
    report += "...This confirms that TabICL v2 is not implementing any of these..."
```

a **hardcoded 50% threshold**, which is precisely the invented tolerance E2.6
exists to avoid. `experiments.md` E2.6 and `theory_and_claims.md` N5 both say the
surrogate's own value-fit must supply the tolerance and that nothing may be
invented. It is also single-seed, single-model, and compares **ambient**
Jacobians with no `Q` projection, so the N1(i) normaliser confound is left in.

**Superseded by:** nothing. E2.6 has never been run correctly. `e2_6_report.md`
travelled with it and must not be cited.

## 2. `e1_4_positional.py` — was `tier1_confounds/`

**What is wrong.** Standardises the target before probing
(`y_std`, line 48) and builds `Q` with a local **unseeded** `get_Q`, so the run
is not reproducible. Reports two marginal means where E1.4 requires a **paired**
per-context statistic. It also writes `e1_4_report.md` with a hardcoded
`abs(mean_raw - mean_zero) < 0.05` verdict.

**Superseded by:** `FINAL_NUMBERS.md` §3.6 and
`tier0_instrument/e14_paired.json`, which use raw `y`, `get_Q(y, seed=0)`, and
report the paired difference with a CI: `mean −0.002233`, `95% CI
[−0.028050, +0.023585]`. The CI contains zero — RoPE is **not** the source of
TabICL's asymmetry. `e1_4_report.md` was left in `tier1_confounds/` and is
marked superseded there; the two runs agree to three decimals on the marginal
means, so the old report is misleading rather than wrong.

## 3. `root_scratch/` — files that were loose in the repository root

| file | what it is |
|---|---|
| `results.json` | `tier2_audit/agg.py` output, written to the repo root by a relative path. Halved `asym` (TabICL `A1 = 0.3028` is half the canonical `0.5803`) and retracted `negfrac = 0.0918` on every row. |
| `results_final_fixed.json` | `tier2_audit/batch4_lastrun.py:175` output, same conventions. The "fixed" in the name refers to an unrelated fix and is not a claim of canonical convention. |
| `scratch.py`, `scratch2.py` | ad-hoc debug scripts that `from experiments.tier2_audit.instrument_c_d import get_Q, compute_reduced_jacobian, get_metrics` — i.e. they import the module whose `get_metrics` returned `negfrac` under the name `negeig`. |
| `debug_shape.py` | was `scratch/debug_shape.py`. |

**Superseded by:** `FINAL_NUMBERS.md` for the two JSONs; nothing needs the
scratch scripts.

---

## `negfrac` — retracted, but deliberately still computable

`negfrac` is **not** moved here, and that is intentional.

**What is wrong with it.** It counts the rank deficiency of the *context*, not a
property of the model. `core/context.py::generate_audit_context` duplicates the
first 10 rows of `X` exactly, so `K` is rank-deficient and
`W = K(K+σ²I)⁻¹` inherits a null space. The exact GP's reduced spectrum has nine
eigenvalues at machine zero and then a hard gap to `2.2e-2`; any noise tips them
negative. Control and TabICL v2 both read `9/98 = 0.0918367`, on every seed, at
every amplitude, in every dither configuration.

**Why it stays.** `FINAL_NUMBERS.md` §7.4 uses `negfrac` as *evidence for its own
retraction*: it verifies N3 by showing `negfrac` is pinned at `0.0918367` while
`negeig` falls by a factor of 106 over the same matrices. Removing the function
would make §7.4 unreproducible.

**What was done instead.** `core/metrics.py::negfrac` now carries a RETRACTED
banner and emits a `RetractedMetricWarning` on every call unless called with
`acknowledge_retracted=True`. The return value is unchanged, bit for bit, so no
recorded number moves.

**Where retracted `negfrac` values sit on disk.** `FINAL_NUMBERS.md` §1.3 lists
them: `results_batch1.json`, `results_tabpfn_5seed.json`, `results_tabswift.json`,
`results_phase1.json` (all now under `_quarantine/tier2_audit/`), plus
`report.md`. Live files that still *contain* a `negfrac` field —
`tier0_instrument/chunk1_results.json`, `chunk3_results.json`,
`exp3_reprobe.json`, `exp4_results.json` — are fine to keep, because
`FINAL_NUMBERS.md` §8 lists `negfrac, all models` under "explicitly not
measured" and never reports one as a model result.

---

## Suspect but deliberately retained — your call, not mine

These were **not** quarantined, because doing so would destroy the only record of
a result or break a citation. Flagging rather than deciding.

### `tier1_confounds/e1_preprocessing_confounds.py` and `tier1_report.md`

Uses its own **unseeded** local `get_Q` (line 12), the same reproducibility
defect that got `e1_4_positional.py` quarantined. Its E1.1 and E1.2 numbers are
superseded — E1.1 by `FINAL_NUMBERS.md` §2.7 (which measures all three models
plus two controls at five shift/scale amplitudes, against this script's single
pair), E1.2 by §2.2 and §2.9.

But its **E1.3 (ensemble-of-one, `n_estimators=1`) and E1.5 (Euler defect,
`e_ratio = 4.1287e-04`) are the only record of those two items anywhere in the
tree**, and neither appears in `FINAL_NUMBERS.md`. Quarantining would lose them.
`tier1_report.md` now carries a header marking which of its five lines are
superseded and which are load-bearing.

**Recommendation:** re-run E1.3 and E1.5 through the seeded instrument before the
paper cites either, then quarantine.

### `tier0_instrument/exp1_a3_models.py`

`FINAL_NUMBERS.md` §9.3 states it is "superseded by `exp12_…`" and "not a source
of any number here" — so by output it belongs in quarantine. But §5's
specification of the A3 regression quotes this file **as source code**
(`exp1_a3_models.py:52-58`, called at `:137`), so moving it breaks a live
citation in a document I was asked not to edit. Retained and marked in place.

---

## Open items for you

1. **`FINAL_NUMBERS.md` cites quarantined files, but almost entirely by bare
   filename, not by path.** Checked line by line:

   | § | citation | form | still resolves? |
   |---|---|---|---|
   | 1.3 | `results_batch1.json`, `results_tabpfn_5seed.json`, `results_tabswift.json`, `results_phase1.json` | bare filename | yes — filenames unchanged |
   | 3.1, 3.4, 4.1 | `results_batch1.json` | bare filename | yes |
   | 3.2, 9.2 | `results_tabpfn_5seed.json` | bare filename | yes |
   | 1.3, 9.3 | `instrument_c_d.py:89`, `item3.py:95` | bare filename | yes |
   | 9.3 | `` `tier2_audit/instrument_c_d.py` `` | **full path** | **stale** |

   So exactly **one** row in the whole document carries a stale path, and it is
   in §9.3's "scripts written or modified in this run" table — a descriptive
   row, not a number's provenance. **§9.1, the provenance table that every
   reported number traces through, cites nothing under `tier2_audit/`.** I did
   not edit the document. The signpost at `experiments/tier2_audit/README.md`
   covers the one stale path; say the word if you want that row updated.

2. **`e2_6_report.md`** is quarantined with its script and is cited nowhere in
   `FINAL_NUMBERS.md`, but it *is* the only E2.6 output that exists. If any draft
   text leans on "TabICL's Jacobian differs by over 50% from all fitted
   surrogates", that sentence has no valid support.

3. **Nine documents cited across `FINAL_NUMBERS.md` and the script docstrings do
   not exist anywhere** — not in the working tree, not in `intermediate/`, and
   not in any commit in git history. `experiments/` was untracked until
   2026-08-26, so nothing captured them.

   | missing document | cited by |
   |---|---|
   | `final_audit_report.md` | FINAL_NUMBERS §3.3 — "`0.702` appears only in `final_audit_report.md`" |
   | `RESULTS_REPORT.md` | FINAL_NUMBERS §1.3 (rev.1 §4.2), §5.3 note; `exp3_reprobe.py:14`; `exp12_ambient_jacobians.py` |
   | `CONTROL_VALIDATION.md` | `chunk1_control_validation.py:5` — "see CONTROL_VALIDATION.md, 0.4"; FINAL_NUMBERS §6.1 |
   | `CLARIFICATIONS.md` | FINAL_NUMBERS front matter — "everything raised in CLARIFICATIONS.md ... is folded in here" |
   | `E11_WRAPPER_FORM.md` | FINAL_NUMBERS front matter — same sentence |
   | `e2_5_report.md` | FINAL_NUMBERS §1.3 retraction row |
   | `tier2_report.md` | FINAL_NUMBERS §1.3 retraction row |
   | `e1_4_report.md` | FINAL_NUMBERS §3.6 — "`e1_4_report.md` reports `0.5801 → 0.5781`" |
   | `models/ARCHITECTURE_REPORT.md` | `models/REMEDIATION_REPORT.md:1` — "Companion to ..." |

   **No number is lost.** `FINAL_NUMBERS.md` is self-contained by construction
   and §9.1 traces every reported value to a JSON or `.npz` that exists and is
   now tracked. What is lost is the *supersession narrative*: the documents that
   explain what was retracted and why are gone, and only FINAL_NUMBERS' summary
   of them survives. Several retraction claims in §1.3 are now unverifiable
   against their sources.

   Nothing to do about it beyond knowing it. Flagged because the paper's
   methods section will want to say "X superseded Y", and Y is not there.
