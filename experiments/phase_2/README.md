# phase_2 — downstream consequences of the Jacobian audit

Self-contained. Nothing outside this directory is modified, and no number in `FINAL_NUMBERS.md` or
the audit's reports is changed or re-derived here.

## Read this first

| file | what it is |
|---|---|
| **`PHASE2.md`** | **all seven experiments in one file** — Parts A–D, the four documents below consolidated with their content unchanged. Start here. |
| `RESULTS.md` | Experiment 2.1, Jacobian-derived predictive variance. Measurement in Sections 0–8, interpretation confined to Section 9. |
| `LOCALISATION.md` | Experiments 2.2 and 2.3 — does the A2 violation localise, and does the Jacobian find corrupted labels. Measurement in Sections 0–3, interpretation confined to Section 4. |
| `SEQUENTIAL.md` | Experiments 2.4–2.7 — acquisition sensitivity, martingale self-consistency, the BO loop, and the sign of the update at acquired points. Measurement in Sections 0–5, interpretation confined to Section 6. |
| `UNCERTAINTY_EXPERIMENT.md` | the full record: chunk by chunk, with the gate arithmetic, every checklist item, the provenance table, and the log of rejected settings. |
| `README.md` | this file — layout and how to reproduce. |

## Layout

```
phase_2/
  PHASE2.md                   all seven experiments, consolidated (Parts A-D)
  RESULTS.md                  exp 2.1 results, then interpretation
  LOCALISATION.md             exp 2.2 + 2.3 results, then interpretation
  SEQUENTIAL.md               exp 2.4-2.7 results, then interpretation
  UNCERTAINTY_EXPERIMENT.md   exp 2.1 full chunk-by-chunk record + provenance
  src/                        all scripts
    paths.py                  every path this experiment reads or writes
    estimators.py             the shared library
    chunk1_validate.py        chunk 1 -- estimator validation on known-answer maps
    chunk2_data.py            chunk 2 -- OpenML selection and splits
    chunk2_stepsize.py        chunk 2 -- probe-step sweep on the frozen models
    chunk2_estimators.py      chunk 2 -- the four estimators, the long GPU run
    chunk3_measure.py         chunk 3 -- coverage, interval score, sharpness, NLL, rank corr
    chunk4_aux.py             chunk 4 -- SURE and directional monotonicity
    exp22_loo.py              exp 2.2 -- leave-one-out influence and label perturbation
    exp22_analyse.py          exp 2.2 -- loading, baselines, targets, controls
    exp23_corrupt.py          exp 2.3 -- corrupted-context Jacobians
    exp23_controls.py         exp 2.3 -- exact GP and hierarchical GP controls
    exp23_loo.py              exp 2.3 -- actual leave-one-out on corrupted contexts
    exp23_analyse.py          exp 2.3 -- detection AUC, precision@k, baselines
    exp24_acquisition.py      exp 2.4 -- acquisition argmax sensitivity (no GPU)
    exp25_martingale.py       exp 2.5 -- drift and variance shrinkage under self-sampling
    exp25_controls.py         exp 2.5 -- analytic GP gate, T1 imitator, quadratic drift control
    exp25_analyse.py          exp 2.5 -- drift aggregation and the J_** stratification
    exp27_bo.py               exp 2.7 -- the BO loop, with 2.6 riding on it
    exp27_analyse.py          exp 2.6 + 2.7 -- regret, sign violations, audit watch
  results/                    JSON measurement records
  arrays/                     .npz stores: data splits, Jacobians, per-point predictions
  logs/                       run logs, kept
```

`src/paths.py` is the single source of truth for locations; no script builds a path of its own.

## Reproduce

Run from the repository root, in this order. Chunks 1 and 3 need no GPU.

```
python experiments/phase_2/src/chunk1_validate.py              # ~7 min, CPU
python experiments/phase_2/src/chunk2_data.py                  # OpenML download + splits
python experiments/phase_2/src/chunk2_stepsize.py <models...>  # GPU; fixes the probe step
python experiments/phase_2/src/chunk2_estimators.py <models...># GPU; ~2.5 h for all three
python experiments/phase_2/src/chunk3_measure.py               # instant, CPU
python experiments/phase_2/src/chunk4_aux.py <models...>       # GPU for 4.2 only, ~25 min

python experiments/phase_2/src/exp22_loo.py <models...>        # GPU, ~2.4 h all three
python experiments/phase_2/src/exp22_analyse.py                # CPU
python experiments/phase_2/src/exp23_corrupt.py <models...>    # GPU, ~4.7 h all three
python experiments/phase_2/src/exp23_controls.py               # CPU
python experiments/phase_2/src/exp23_loo.py <models...>        # GPU, ~1.5 h all three
python experiments/phase_2/src/exp23_analyse.py                # CPU

python experiments/phase_2/src/exp24_acquisition.py            # CPU, instant
python experiments/phase_2/src/exp25_martingale.py <models...> # GPU, ~2 h all three
python experiments/phase_2/src/exp25_controls.py               # CPU, ~20 min
python experiments/phase_2/src/exp25_analyse.py                # CPU
python experiments/phase_2/src/exp27_bo.py <objectives...>     # GPU, ~3.5 h all five
python experiments/phase_2/src/exp27_analyse.py                # CPU
```

`exp27_bo.py` takes objective names and writes files keyed by them, so it can be split across GPUs;
it was run as `branin2 hartmann3 hartmann6` on one and `ackley5 rosenbrock5` on the other.

Long GPU runs were launched with `setsid` so they survive a session ending, and capped at 16 BLAS
threads per process — two uncapped runs drove a 192-core machine to load average 209.

`chunk2_stepsize.py` and `chunk2_estimators.py` take model ids as arguments and were run
concurrently on two GPUs via `CUDA_VISIBLE_DEVICES`; each writes files keyed by model, so concurrent
invocations do not collide. `chunk2_estimators.py` also accepts `--limit=N` to run the first `N`
datasets only.

Chunks 1, 3 and 4 were re-run after this directory was reorganised and reproduce their pre-move
outputs (Section "Reproduction check" below).

## What is reused, not reimplemented

From `experiments/core/`: `asym`, `negeig`, `get_Q` (seeded), `extract_variance`,
`central_jacobian`, `ExactGP`, `HierarchicalGP`, `NadarayaWatson`, `Imitator`,
`generate_audit_context`. From `experiments/tier0_instrument/`: the `wrap` normaliser and the
`TargetedImitator` control.

`src/estimators.py` adds only what the audit's objects lack: out-of-sample prediction and closed-form
query predictive variance for the two GPs (`GPTrainTest`, `HierGPTrainTest`, checked against the
audit's in-sample surrogates by `check_matches_audit_surrogate`, run as item 1.0 — worst deviation
`9.659e-15`), the appended-query Jacobian, the noise-scale estimator, split conformal, and the
interval metrics.

## Two decisions changed mid-run, both recorded

1. **Estimator form.** The brief's `sigma2_hat * (1 + J_**)` is measured and stored everywhere as
   `s2_jac_specified`, but is not the primary: it carries 42–52% median relative error on the exact
   GP and is bounded above by `2 sigma^2`. The primary is `sigma2_hat / (1 - J_**)`, exact to
   `8e-13` on the same test. Derivation and measurement: `RESULTS.md` §1.

2. **Probe step.** A first pass used the audit's amplitudes as absolute numbers and was discarded
   before any estimator number was read off it. The models renormalise the target internally, so the
   output quantum in original target units scales with `std(y)`, and the datasets span target SDs
   from 0.75 to 1083. The step is `h = h_frac * std(y_ctx)`. Evidence: `RESULTS.md` §2.

A third correction is local to `src/chunk4_aux.py`: `TargetedImitator.inner_jacobian` in
`tier0_instrument/` returns `-c*vvᵀ` where the map applies `-(c/2)*vvᵀ`, the same doubling
`FINAL_NUMBERS.md` §2.3 records for `audit_phase2_imitator.py`. It is corrected here rather than by
editing the tier-0 control, which other results depend on.

> **CORRECTION, 2026-08-26.** The sentence that stood here — "The tier-0 file still carries the bug"
> — **is no longer true.** The bug was fixed at source in `experiments/core/controls.py`, now the
> single definition of `TargetedImitator`; this script's inline workaround and the two others
> (`exp22_analyse.py`, `tier0_instrument/chunk3_dither_controls.py`) call the corrected method.
> Verified **bit-identical** to the inline form — max abs difference exactly `0.0` on all five seeds
> — so **no number in this directory changed.** Regression test at
> `experiments/core/test_controls.py`. The record documents (`RESULTS.md`, `LOCALISATION.md`,
> `SEQUENTIAL.md`, `UNCERTAINTY_EXPERIMENT.md`, `PHASE2.md`) still describe the pre-fix state and
> are left as history.

## Reproduction check

After the move, with the pre-move outputs kept for comparison:

| script | outcome |
|---|---|
| `chunk3_measure.py` | byte-identical JSON |
| `chunk1_validate.py` | identical on every numeric field; only `seconds` timing fields differ |
| `chunk4_aux.py` | identical on every numeric field; only `seconds` timing fields differ |

`chunk2_data.py` and `chunk2_estimators.py` were not re-run (OpenML re-download and ~2.5 h of GPU
time); their path resolution was checked by import and by the fact that chunks 3 and 4 read their
outputs from the new locations and reproduce.

## Note on the arrays

`arrays/chunk2_arrays_<model>.npz` holds the full `100x100` context Jacobian for every dataset and
split, plus per-point predictions, variances and calibration residuals. Any re-analysis that is a
function of `J` or of the stored predictions needs no GPU — `chunk3_measure.py` and item 4.1 of
`chunk4_aux.py` are both pure functions of these files.
