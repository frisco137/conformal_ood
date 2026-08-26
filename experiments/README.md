# `experiments/` — navigation

Start here.

This directory holds **two separate research programmes**. They share an
instrument and nothing else. Keep them separate when reading and when writing.

| | **The audit** | **Phase 2** |
|---|---|---|
| asks | do these models' label-Jacobians satisfy the conditions every posterior mean satisfies? | does the violation *cost* anything downstream? |
| data | synthetic, one family: `generate_audit_context(n=100, d=5, sigma=1.0)`, 10 duplicated `X` rows | 12 real OpenML regression datasets, 5 splits each |
| lives in | `core/`, `tier0_instrument/` … `tier5_positive/` | `phase_2/` |
| authoritative record | **`FINAL_NUMBERS.md`** | **`phase_2/PHASE2.md`** |

Models under audit throughout: TabPFN v2, TabICL v2, TabSwift — frozen
regressors, loaded via `models.load()`. `models/registry.py` is the single source
of truth for architecture and capabilities.

---

## 1. Where the authoritative numbers live, and what supersedes what

```
                      theory_and_claims.md   ← claims (C-T*, C-N*, C-I*, C-E*, C-X*)
                      experiments.md         ← battery (E0.1 … E5.4), thresholds, falsifiers
                             │  both are PLANNING documents, dated August 2026,
                             │  and are NOT a record of what was measured
                             ▼
   THE AUDIT      ►   FINAL_NUMBERS.md       ← AUTHORITATIVE. Revision 2.
                             ▲                 Supersedes CLARIFICATIONS.md and
                             │                 E11_WRAPPER_FORM.md (both now missing, §5)
                             │
                       §9.1 traces every number to a .json / .npz on disk

   PHASE 2        ►   phase_2/PHASE2.md      ← AUTHORITATIVE. All 7 experiments,
                             │                 Parts A–D, content unchanged from
                             │                 the four source documents
                             ├─ RESULTS.md               Part A  (exp 2.1)
                             ├─ LOCALISATION.md          Part B  (exp 2.2, 2.3)
                             ├─ SEQUENTIAL.md            Part C  (exp 2.4–2.7)
                             └─ UNCERTAINTY_EXPERIMENT.md Part D (exp 2.1 full record)
```

**Rules that matter:**

- **`FINAL_NUMBERS.md` wins over every other audit document.** Its §1.2 states
  that no number in it is taken from the quarantined `tier2_audit/`, and its
  §9.2 lists the only two values carried from an earlier run.
- **`experiments.md` and `theory_and_claims.md` are plans, not results.** Their
  inline "Status" lines are stale. `theory_and_claims.md` §0 does fix the
  canonical `asym` and `negeig` definitions, and those are current.
- **Canonical metric definitions live in `core/metrics.py`** and nowhere else.
  `asym(J) = ‖J−Jᵀ‖_F/‖J‖_F`. Any value that is half this came from
  `_quarantine/tier2_audit/` and must be doubled or discarded.
- **`phase_2` changes nothing in the audit.** It re-derives no audit number.

---

## 2. Which script produces which result

### The instrument — `core/`

| file | what it is |
|---|---|
| `metrics.py` | `asym`, `negeig`, `central_jacobian`, `get_Q(y, seed)`, `reduced_jacobian`, `profile_jacobian`, `extract_variance`. **`negfrac` is retracted** and warns on call. |
| `surrogates.py` | `ExactGP`, `Ridge`, `NadarayaWatson`, `OneNN`, `HierarchicalGP`, `Imitator` |
| `controls.py` | `wrap` (the N1 normaliser), `quantize`, `ImitatorWrapped`, `TargetedImitator`. **Single definition** — these used to live in `tier0_instrument/chunk1_control_validation.py`. |
| `context.py` | `generate_audit_context` and two older context generators |
| `test_controls.py` | regression tests: every control's `inner_jacobian` must be the derivative of its `inner`. `pytest experiments/core/test_controls.py` |

### The audit — `tier0_instrument/`

Despite the name, this directory holds the Tier 0 **and** the current Tier 2
measurements; the Tier 2 numbers in `FINAL_NUMBERS.md` §3–§6 were all re-measured
here after `tier2_audit/` was found to be using a halved convention.

| script | writes | feeds FINAL_NUMBERS § |
|---|---|---|
| `chunk1_control_validation.py` | `chunk1_results.json` | 2.1–2.4 controls, curvature calibration |
| `chunk2_a3_controls.py` | `chunk2_results.json` | 5.5 reference heads, 7.1 T2, 7.2 T4 |
| `chunk3_dither_controls.py` | `chunk3_results.json` | 2.6 quantisation artifact table |
| `e11_wrapper_form.py` | `e11_wrapper_form.json` | 2.7 shift/scale identities (E1.1) |
| `exp12_ambient_jacobians.py` | `exp12_ambient_jacobians.npz`, `exp12_ambient_meta.json` | the raw ambient Jacobians everything else reads |
| `exp1_extractor_validation.py` | `exp1_extractor_validation.json` | 5.1 variance-extractor validation |
| `exp2_ambient_profiling.py` | `exp2_ambient_profiling.json` | 6.2–6.4 row/column mechanism profiling |
| `exp3_reprobe.py` | `exp3_reprobe.json`, `exp3_reduced_jacobians.npz` | 3.2, 3.3, 4.1 headline A1/A2 |
| `exp4_quanta_and_tabicl.py` | `exp4_results.json`, `exp4_tabicl_reduced.npz` | 3.1, 3.4, 4.2 TabICL + artifact floors |

**Never run, no saved output:** `e0_2_analytic_control.py` (E0.2, the N5 analytic
battery — covers 6 of 9 rows), `e0_model_instrument.py` (E0.1 determinism, E0.3,
E0.6), `d_diagnostic.py`.

**Superseded, retained for its source-code citation:** `exp1_a3_models.py` — see
its banner.

### The audit — `tier1_confounds/`

| file | status |
|---|---|
| `e1_preprocessing_confounds.py` → `tier1_report.md` | E1.1/E1.2 superseded by FINAL_NUMBERS §2.7/§2.2/§2.9. **E1.3 and E1.5 are the only record of those items in the tree**, and were computed with an unseeded `Q`. See the header on `tier1_report.md`. |
| `e1_4_positional.py` | **quarantined.** Superseded by FINAL_NUMBERS §3.6 / `tier0_instrument/e14_paired.json`. |

### The audit — `tier2_audit/`, `tier3_robustness/`, `tier4_value/`, `tier5_positive/`

`tier2_audit/` is a **signpost only** — the code is quarantined. The other three
are **empty because nothing in them has ever been run**; each carries a
`NOT_RUN.md` saying what it would establish and why it matters. See §4 below.

### Phase 2 — `phase_2/`

`phase_2/README.md` has the full layout, the reproduce order with runtimes, and
the two decisions that changed mid-run. In brief:

| script | writes | experiment |
|---|---|---|
| `chunk1_validate.py` | `chunk1_results.json` | estimator validation on known-answer maps |
| `chunk2_data.py` | `chunk2_datasets.json`, `arrays/chunk2_data.npz` | OpenML selection + splits |
| `chunk2_stepsize.py` | `chunk2_stepsize_*.json` | probe step on the frozen models |
| `chunk2_estimators.py` | `chunk2_estimator_<model>.json`, `arrays/chunk2_arrays_<model>.npz` | the four uncertainty estimators (GPU, ~2.5 h) |
| `chunk3_measure.py` | `chunk3_results.json` | coverage, interval score, NLL, rank corr → **exp 2.1** |
| `chunk4_aux.py` | `chunk4_results.json` | SURE, directional monotonicity |
| `exp22_loo.py` + `exp22_analyse.py` | `exp22_*.json` | **exp 2.2** — does the A2 violation localise? |
| `exp23_corrupt.py` / `_controls.py` / `_loo.py` / `_analyse.py` | `exp23_*.json` | **exp 2.3** — corrupted-label detection |
| `exp24_acquisition.py` | `exp24_results.json` | **exp 2.4** — acquisition-argmax sensitivity |
| `exp25_martingale.py` / `_controls.py` / `_analyse.py` | `exp25_*.json` | **exp 2.5** — martingale self-consistency |
| `exp27_bo.py` + `exp27_analyse.py` | `exp27_*.json` | **exp 2.6 + 2.7** — the BO loop |

`phase_2/arrays/*.npz` hold the full 100×100 context Jacobians and per-point
predictions. **Any re-analysis that is a function of `J` or the stored
predictions needs no GPU.**

---

## 3. What is quarantined, and why

`_quarantine/` — full manifest with per-item reasons in
**`_quarantine/QUARANTINE.md`**. `_quarantine/__init__.py` raises on import, so
no live module can reach it through the package path.

| item | defect | superseded by |
|---|---|---|
| `tier2_audit/` (all) | every `asym` is **half** canonical; `negfrac` returned as `negeig`; 13 unseeded `get_Q`; profile residual always `0.00e+00` | `FINAL_NUMBERS.md` in full |
| `tier2_audit/e2_6_mechanism.py` | verdict is a **hardcoded 50% threshold** — the invented tolerance E2.6 exists to avoid; single-seed, single-model, ambient (no `Q`) | **nothing. E2.6 has never been run correctly.** |
| `e1_4_positional.py` | standardised targets, unseeded `Q`, marginal instead of paired | `FINAL_NUMBERS.md` §3.6 |
| `root_scratch/` | tier2 outputs written to the repo root; scratch scripts importing quarantined modules | `FINAL_NUMBERS.md` |
| `negfrac` **(not moved)** | reads the context's rank deficiency, not the model — pinned at `9/98` on control and model alike | `negeig`. Kept computable because FINAL_NUMBERS §7.4 uses it as evidence for its own retraction; now warns on call. |

---

## 4. What has genuinely never been run

`FINAL_NUMBERS.md` §8 is authoritative. Consolidated, with the blocking items
first:

**Blocking, per `experiments.md`'s own sequencing:**

- **E5.1 — positive control on real data.** *"Without this the paper does not go
  out."* Every control to date is on synthetic GP contexts.
- **E4.1 / E4.2 — the value-level battery.** No MSE, NLL, calibration, coverage,
  martingale or prequential number exists for any model **on the audit's
  contexts**. `experiments.md` calls this the headline figure.

**Tier 0 instrument gaps:**

- E0.1 determinism on the models — needs a GPU refit-twice pass on the audit path
- E0.2 N5 analytic battery — script exists, no saved output, covers 6 of 9 rows
- E0.4 autograd cross-check — no code

**Tier 3 robustness, E3.1–E3.5 — none of it.** One context family throughout.
E3.1 is the objection `experiments.md` calls fatal if unaddressed.

**Tier 5:** E5.2 as a *registered* test, E5.3, E5.4.

**Per-model gaps:** E1.4 for TabPFN v2 (needs source modification of the
attention path) · A3 for TabSwift (`Linear(384,1)` point head, no predictive
distribution — report as unevaluable, never as a failure) · N1(ii) exactness for
TabSwift (satisfies neither the shift nor the scale identity).

**Measured but not reportable:** curvature on models (single amplitude only; the
artifact scales as `t⁻²` while genuine curvature is flat in `t`, so one amplitude
cannot separate them, and the metric is not the M0 tangential great-circle
curvature with geodesic correction) · A4 decay law (out of scope wherever A1 or
A2 fail) · `negfrac` on all models · `cv_s ≤ cv_J` as a pass (permutation-
invariant) · reduced-basis profiling.

---

## 5. Two provenance hazards to know about before writing

**(a) Ten result files have no producing script in the tree.** They were written
in-session and the code was never saved. `FINAL_NUMBERS.md` §9.3 acknowledges
this in its last row. Affected:

```
exp_stress_tests.json      exp_negeigvec.json      exp_n1ii.json
exp1_a3_final.json         exp1_tabicl_a3.json     e14_paired.json
exp4_floor_brackets.json   clarif_item4_6.json     clarif_item5.json
chunk1_profile_control.json  chunk3_control_matrix.json
```

These carry FINAL_NUMBERS §2.5 (seven stress tests), §2.8 (`r₁`), §2.9 (ambient
floor), §3.4/§4.2 (floor brackets), §3.6 (E1.4 paired), §4.3 (negative
eigenvector, `#(J_ii<0)`), §5.2/§5.3 (A3 regressions), §5.8 (no-intercept fits),
§7.3 (N1(ii)).

**They are recomputable without a GPU** — the raw Jacobians they were derived
from are in `exp12_ambient_jacobians.npz` and `exp3_reduced_jacobians.npz`, both
now tracked. Re-deriving them into scripts is the cheapest reproducibility win
available and needs no new measurement.

**(b) Nine cited documents no longer exist** — not in the working tree, not in
`intermediate/`, and not in any git commit (`experiments/` was untracked until
2026-08-26):

`final_audit_report.md` · `RESULTS_REPORT.md` · `CONTROL_VALIDATION.md` ·
`CLARIFICATIONS.md` · `E11_WRAPPER_FORM.md` · `e2_5_report.md` ·
`tier2_report.md` · `e1_4_report.md` · `models/ARCHITECTURE_REPORT.md`

No number is lost — `FINAL_NUMBERS.md` is self-contained and §9.1 traces every
value to a file that exists. What is lost is the supersession narrative: several
§1.3 retraction claims can no longer be checked against their sources. Details in
`_quarantine/QUARANTINE.md`, "Open items".

---

## 6. Conventions that bite

- **Probe amplitude is per-model and not interchangeable.** TabICL `t=1e-3` (no
  dither), TabPFN `t=1e-1` (dither halfwidth `δ=6.87e-4`, `N=10`), TabSwift
  `t=1e-1` (no dither, **float16** output). `FINAL_NUMBERS.md` §3.5 gives the
  reason for each.
- **Phase 2 uses a *relative* step**, `h = h_frac · std(y_ctx)`, because all
  three models renormalise the target internally. An absolute step on a dataset
  with `std(y) ≈ 1083` lands inside the quantisation floor and returns
  `tr J = 243` against `n = 100`.
- **`get_Q` must be seeded.** `get_Q(y, seed=0)` throughout. `asym`/`negeig` are
  basis-invariant; `profile_jacobian` is not, which is why it matters.
- **Ambient and reduced `asym` are different objects.** N1(i) puts an asymmetric
  rank-one term in the ambient Jacobian of any normaliser-wrapped map. Never
  compare an ambient number to a reduced one.
- **The tier directory names do not match the tier of the work inside them.**
  Current Tier 2 results live in `tier0_instrument/`. Renaming would break
  `FINAL_NUMBERS.md` §9.1's paths, so the names stay.

---

## 7. Running things

```bash
.venv/bin/python -m pytest experiments/core/test_controls.py -q   # instrument regression tests
.venv/bin/python experiments/tier0_instrument/chunk1_control_validation.py   # CPU, ~2 min
```

GPU scripts load models through `models.load(model_id, task="regression",
device="cuda")` and refit on every perturbed `y`, so a 100×100 ambient Jacobian
is 200 fits. Two RTX 6000 Ada are available; long runs were launched with
`setsid` and capped at 16 BLAS threads.
