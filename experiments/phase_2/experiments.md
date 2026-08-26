# Phase 2 — The experiments

**What was run, why it exists, what it was measured against, and how it was judged.**

This is a live document describing the experiments that were actually run and are going in the paper.
Experiments that were planned and never run are **not** here — they are in
[`phase_2_results.md`](phase_2_results.md) §5, under what is not established.

Results: [`phase_2_results.md`](phase_2_results.md). Detailed per-split record:
[`phase_2_record.md`](phase_2_record.md), which wins any dispute with the results document.

---

## What Phase 2 is for

[Phase 1](../phase_1/experiments.md) established that all three models violate A1 and A2 by factors of
`57×` to `22 711×` over the instrument's artifact floor. That is a structural statement about the
label-Jacobian. It is not, by itself, a statement that anything goes wrong.

Phase 2 asks the follow-up a reviewer will ask immediately: **does it cost anything?**

Seven experiments, all on **real** data, in three groups:

| group | question |
|---|---|
| **2.1** | Can the Jacobian be *used* — does a variance built from it beat the model's own head? |
| **2.2, 2.3** | Does the violation *localise* — can it point at the context points responsible? |
| **2.4–2.7** | Does the violation reach a *decision* — does it change what a sequential loop does? |

Two of the three answer negatively, and that is reported as the result rather than buried. The
honest summary is that **one** downstream quantity carried signal (2.3, from A1) and **three
independent attempts to convert A2 into a per-point diagnostic all returned null.**

**Phase 2 establishes no Phase 1 number and changes none.** Where it appears to speak to Phase 1 —
finding TabICL near `J = I` on real data — that is flagged as *a reason to run E3.1*, not a result
about it.

---

## The standing setup

**Data.** 12 OpenML regression datasets × 5 splits, `n_ctx = 100`, `n_cal ≤ 100`, `n_test ≤ 50`, seeds
`[42, 100, 200, 300, 400]`. Selection rule fixed in advance: `200 ≤ n ≤ 2000`, `2 ≤ d ≤ 50`, at least
20 unique target values, OpenML id ascending, first 12 accepted. Features standardised on **context
rows only**; targets left in original units, which span sd `0.75` to `1083`.
*Script:* [`src/chunk2_data.py`](src/chunk2_data.py) → `results/chunk2_datasets.json`, `arrays/chunk2_data.npz`.

**The probe step is relative:** `h = h_frac · std(y_ctx)`. A first pass used Phase 1's amplitudes as
absolute numbers and was **discarded before any estimator number was read off it**. All three models
renormalise the target internally, so the output quantum in original units scales with `std(y)`; on
`places` (`std(y) = 1083`) an absolute step landed deep inside the quantisation floor and returned
`tr J = 243` against `n = 100`. `h_frac` is Phase 1's own amplitude divided by its mean context label
sd of `1.457782`. *Script:* [`src/chunk2_stepsize.py`](src/chunk2_stepsize.py).

**The instrument is Phase 1's, reused unchanged**, imported as `experiments.phase_1.core`: `asym`,
`negeig`, `get_Q` (seeded at 0), `extract_variance`, `ExactGP`, `HierarchicalGP`, `TargetedImitator`.
Nothing is reimplemented. What [`src/estimators.py`](src/estimators.py) adds is only what the audit's
objects lack — out-of-sample prediction, closed-form query predictive variance, the appended-query
Jacobian, split conformal, and the interval metrics.

---

## 2.1 — Is a Jacobian-derived variance any good?

**Measures.** Four uncertainty estimators on the identical mean channel: the model's **native** head,
a **Jacobian**-derived variance, **split conformal**, and an **oracle GP**. Scored by coverage,
Winkler interval score, sharpness at matched coverage, Gaussian NLL, and rank correlation between
each variance and the realised squared residual.

**Exists because** Phase 1's A3 result says the reported variance does not satisfy `s² = σ²(1 + Jᵢᵢ)`.
The obvious implication — that a variance built from `J` instead would be *better* — is testable, and
if true would be the practical payoff of the whole audit. It is also the stated precondition for the
BO study: if this route does not improve calibration, there is no reason to try it as an acquisition
signal.

**Measured against** the oracle GP (marginal-likelihood fit, per split) as a reference predictor, and
against split conformal as the cheap distribution-free incumbent. Paired per dataset with bootstrap
CIs and Wilcoxon tests, so a win is a win across datasets and not an artifact of one.

**Judged by** win/loss over the 12 datasets on interval score at 90%, with the CI required to exclude
zero.

**The estimator form was corrected before use.** The specified `σ̂²(1 + J_**)` does not estimate the
held-out predictive variance — appending `(x_*, y_*)` conditions on a label that was *invented*, so
T2 returns the post-observation variance, and the form is **bounded above by `2σ²`** while its target
is unbounded. The inversion `σ²/(1 − J_**)` is exact on an exact GP (`~1e-13` against `~0.45` for the
specified form). Both are computed and stored; the inverted one is primary.

**TabSwift has no native head** — `Linear(384,1)`. No variance was synthesised for it; the row is
reported empty.

*Scripts:* [`src/chunk2_estimators.py`](src/chunk2_estimators.py) → `results/chunk2_estimator_<model>.json`,
`arrays/chunk2_arrays_<model>.npz` (GPU, ~2.5 h); [`src/chunk3_measure.py`](src/chunk3_measure.py) → `results/chunk3_results.json`.
→ [Results §1](phase_2_results.md)

### Auxiliary — SURE, and directional monotonicity
**Measures.** Whether SURE's in-sample risk estimate tracks held-out risk; and, directly, whether
`uᵀJu < 0` along the leading negative eigendirection — *raising the context labels along a direction
and watching the prediction move the opposite way.*
**Exists because** it converts A2 from a matrix property into an observable behaviour, at five probe
amplitudes so the reader can see it is a derivative statement and not an artifact of one step size.
**Measured against** an exact GP (`uᵀJu ≥ 0` necessarily) and the targeted imitator (constructed
non-PSD; must go negative).
*Script:* [`src/chunk4_aux.py`](src/chunk4_aux.py) → `results/chunk4_results.json`.
→ [Results §1.5](phase_2_results.md)

---

## 2.2 — Does the A2 violation localise?

**Measures.** The per-context-point loading `wᵢ = uᵢ²` of the leading negative eigendirection
`u = Qv`, and whether it predicts anything a practitioner would act on: the model's own residual,
local held-out error, and **actual leave-one-out influence** — 100 refits per split, the expensive
gold standard for "how much does this point matter".

**Exists because** if the A2 violation concentrated on identifiable points, the audit would yield a
*diagnostic tool* rather than a description. That would be the most useful thing the project could
produce.

**Measured against** four baselines computed on the same splits — `|Jᵢᵢ|`, the column norm, plain
residual magnitude, and feature leverage — plus an oracle GP control, plus a **power check**: a
scaled imitator whose constructed negative direction is made to dominate, to establish that the score
*can* recover a direction when one is there.

**Judged by** Spearman against each target, and by whether the high-loading set is geometrically
stable across splits of the same dataset relative to random subsets of the same size.

**The power check is what makes the null interpretable.** The shipped imitator recovers only to
`|cos| = 0.63` — but its competition ratio is `0.406`, so it never presented a dominant direction to
find. Scaled until it does, recovery is `0.998`. **The null is about the models, not the score.**

*Scripts:* [`src/exp22_loo.py`](src/exp22_loo.py) → `results/exp22_loo_<model>.json`,
`arrays/exp22_loo_<model>.npz` (GPU, ~2.4 h); [`src/exp22_analyse.py`](src/exp22_analyse.py) → `results/exp22_results.json`.
→ [Results §2](phase_2_results.md)

---

## 2.3 — Does the A1 violation find corrupted labels?

**Measures.** Whether a score built from the **row-minus-column asymmetry** of `J` detects
deliberately corrupted context labels, across three corruption schemes and three rates.

**Exists because** it is the sharpest available test of whether A1 has *practical content*. For any
symmetric `J` the row norm and the column norm are the same number; **the gap between them is the A1
violation itself**, used as a detector. A score that is exactly zero for every Bayes-consistent map
by construction is a test of the condition, not a generic outlier statistic.

**Measured against** three things, and the design needed all three:
- **residual magnitude**, the incumbent a practitioner would actually use;
- **actual leave-one-out**, the expensive gold standard;
- **two Bayesian controls.** The exact GP's Jacobian is *label-blind* — `K(K+σ²I)⁻¹` contains no `y` —
  so corrupting labels leaves it bit-identical and every Jacobian score is at chance *by
  construction*. That makes it a control against spurious signal but **not** the control the design
  needs. The **hierarchical GP** is: Bayes-consistent *and* with a label-dependent Jacobian that
  genuinely responds to corruption, so its asymmetry score sitting at chance is informative.

**Judged by** detection AUC and precision@k against the chance rate, paired over all 180
combinations against the residual baseline.

**Scope, fixed in advance:** run on 4 of the 12 datasets. The full grid was 108 540 model calls per
model — about 10.6 h for TabICL alone. All schemes and rates kept; the 8 dropped datasets are named
in the output.

**One caveat the design surfaced:** the A2 loading `w` is **not** a pure function of `J`, because
`Q = get_Q(y)` is built from the labels. The label-blind exact GP still returns `w` AUC `0.5625`.
That is `w`'s floor in this experiment, and the models do not clear it.

*Scripts:* [`src/exp23_corrupt.py`](src/exp23_corrupt.py) (GPU, ~4.7 h),
[`src/exp23_controls.py`](src/exp23_controls.py), [`src/exp23_loo.py`](src/exp23_loo.py),
[`src/exp23_analyse.py`](src/exp23_analyse.py) → `results/exp23_results.json`.
→ [Results §3](phase_2_results.md)

---

## 2.4 — Does the variance defect reach a decision?

**Measures.** Whether the native and Jacobian variance channels pick the **same candidate** under
standard acquisition functions, the regret gap when they differ, and the rank correlation between the
two acquisition surfaces.

**Exists because** calibration metrics aggregate over a dataset; an acquisition function takes an
`argmax`. A defect can be invisible in the former and decisive in the latter, and that difference is
what "decision-relevant" means.

**Measured against** pure exploitation (greedy on the mean) as the floor, and against the oracle GP.

**Two corrections made before running.** Naive EI underflows to exactly `0.0` for `z < −39` — on 3 of
60 TabPFN splits *every* pool point returned `EI = 0` and `argmax` picked index 0; replaced by log-EI.
And the frontier gap must be measured in EI space, not log-EI space, since `log EI` carries `−z²/2`
and so grows as `z²` by construction.

*Script:* [`src/exp24_acquisition.py`](src/exp24_acquisition.py) → `results/exp24_results.json` (CPU).
→ [Results §4.1](phase_2_results.md)

---

## 2.5 — Are the models sequentially self-consistent?

**Measures.** Martingale self-consistency: sample a hypothetical next observation from the model's
own predictive, append it, re-predict, and test whether the mean is unchanged in expectation —
`E[μ_{n+1}] = μ_n`. Reported as `|drift| / SE` at the probe location and at other locations, plus the
total-variance identity.

**Exists because** it is the strongest *value-channel* coherence test available, and because the
project's own theory (T1) says value functionals cannot separate the classes with a uniform margin —
so what this experiment finds, and what it cannot claim, are both informative.

**Measured against** four controls, each added for a reason the run itself exposed:
- **exact GP, analytic** — a martingale to `1.07e-11`, the numerical gate;
- **exact GP, sampled** — the Monte Carlo floor;
- **hierarchical GP, sampled from its own mixture** — the control that matters, Bayesian *with* a
  label-dependent Jacobian. Sampling it from the *wrong* law produced `|drift|/SE = 23.065` for a
  Bayesian map, which is how the sampling bug was caught;
- **the T1 imitator** — ε-close in values, so its drift is bounded by `2ε` regardless of how
  non-Bayesian it is. It does not drift, which demonstrates T1 rather than failing the test. A
  **quadratic map** was added as the control that genuinely does drift.

**Judged by** `|drift|/SE` against the controls, with `K` sensitivity reported.

**Also tests the A2-localised prediction:** drift should be larger and signed wrong where `J_** < 0`.
**TabSwift's entire 2.5 result rests on a Gaussian surrogate** for a predictive distribution it does
not have, and is not a statement about its own sampling law.

*Scripts:* [`src/exp25_martingale.py`](src/exp25_martingale.py) (GPU, ~2 h),
[`src/exp25_controls.py`](src/exp25_controls.py), [`src/exp25_analyse.py`](src/exp25_analyse.py) → `results/exp25_results.json`.
→ [Results §4.2, §4.3](phase_2_results.md)

---

## 2.6 — Does the model un-learn from its own acquisitions?

**Measures.** At each BO acquisition, whether the model finds a point **better** than it predicted and
becomes **less** optimistic about it — and the shrinkage factor
`(μ_{n+1} − μ_n)/(y_t − μ_n)`, which a posterior mean keeps in `[0, 1]`.

**Exists because** it is the cleanest *behavioural* statement the audit can make: not a matrix
property but a thing the model does, in a loop, that no Bayesian can do.

**Measured against** a refitted GP on the identical protocol — same pool, same seeds, same
acquisitions.

**Rides on the 2.7 loop** at no extra model cost.

*Scripts:* [`src/exp27_bo.py`](src/exp27_bo.py), [`src/exp27_analyse.py`](src/exp27_analyse.py) → `results/exp27_results.json`.
→ [Results §4.5](phase_2_results.md)

---

## 2.7 — The BO loop

**Measures.** Simple regret over a 20-iteration Bayesian optimisation loop on five synthetic
objectives (`branin2`, `hartmann3`, `hartmann6`, `ackley5`, `rosenbrock5`), 5 seeds, 128-point pool,
with four arms: native variance, Jacobian variance, conformal, and a fitted GP.

**Exists because** it is the end-to-end version of 2.1 and 2.4 — the place where a calibration defect
either costs something measurable or does not.

**Measured against** the fitted GP as the reference method, paired per objective with bootstrap CIs
and Wilcoxon tests.

**Also carries an audit watch:** `tr J / n` and `n − tr J` at every iteration, because 2.1 found
TabICL nearly interpolating at `n = 100` and BO contexts start small and grow.

**Scope, fixed in advance:** TabPFN v2 only. The Jacobian arm costs `2 × n_pool` calls per iteration —
`12.4 h` for TabICL and `3.9 h` for TabSwift, not run and stated. Knowledge gradient was not run.

*Scripts:* [`src/exp27_bo.py`](src/exp27_bo.py) (GPU, ~3.5 h), [`src/exp27_analyse.py`](src/exp27_analyse.py).
→ [Results §4.4, §4.6](phase_2_results.md)

---

## Reproducing

Run from the repository root, in this order. Chunks 1 and 3 need no GPU.

```bash
.venv/bin/python experiments/phase_2/src/chunk1_validate.py              # ~7 min, CPU
.venv/bin/python experiments/phase_2/src/chunk2_data.py                  # OpenML download + splits
.venv/bin/python experiments/phase_2/src/chunk2_stepsize.py <models...>  # GPU; fixes the probe step
.venv/bin/python experiments/phase_2/src/chunk2_estimators.py <models...># GPU; ~2.5 h for all three
.venv/bin/python experiments/phase_2/src/chunk3_measure.py               # instant, CPU
.venv/bin/python experiments/phase_2/src/chunk4_aux.py <models...>       # GPU for 4.2 only, ~25 min

.venv/bin/python experiments/phase_2/src/exp22_loo.py <models...>        # GPU, ~2.4 h all three
.venv/bin/python experiments/phase_2/src/exp22_analyse.py                # CPU
.venv/bin/python experiments/phase_2/src/exp23_corrupt.py <models...>    # GPU, ~4.7 h all three
.venv/bin/python experiments/phase_2/src/exp23_controls.py               # CPU
.venv/bin/python experiments/phase_2/src/exp23_loo.py <models...>        # GPU, ~1.5 h all three
.venv/bin/python experiments/phase_2/src/exp23_analyse.py                # CPU

.venv/bin/python experiments/phase_2/src/exp24_acquisition.py            # CPU, instant
.venv/bin/python experiments/phase_2/src/exp25_martingale.py <models...> # GPU, ~2 h all three
.venv/bin/python experiments/phase_2/src/exp25_controls.py               # CPU, ~20 min
.venv/bin/python experiments/phase_2/src/exp25_analyse.py                # CPU
.venv/bin/python experiments/phase_2/src/exp27_bo.py <objectives...>     # GPU, ~3.5 h all five
.venv/bin/python experiments/phase_2/src/exp27_analyse.py                # CPU
```

[`src/paths.py`](src/paths.py) is the single source of truth for locations; no script builds a path of
its own. Scripts taking model ids or objective names write files keyed by them, so concurrent
invocations on different GPUs do not collide. Long runs were launched with `setsid` and **capped at 16
BLAS threads** — two uncapped runs drove a 192-core machine to load average 209.

**`arrays/chunk2_arrays_<model>.npz` holds the full 100×100 context Jacobian for every dataset and
split**, plus per-point predictions, variances and calibration residuals. Any re-analysis that is a
function of `J` or of the stored predictions **needs no GPU** — `chunk3_measure.py` and item 4.1 of
`chunk4_aux.py` are both pure functions of these files.

### Dependency on Phase 1

Phase 2 imports `experiments.phase_1.core` for the instrument, and reads two of Phase 1's stored
arrays — `tier0_instrument/exp3_reduced_jacobians.npz` and `exp_negeigvec.json` — to reproduce the
audit's localisation in 2.2. The dependency is one-directional. Phase 1 touches Phase 2 in exactly one
place: `e5_1_positive_control.py` reads `phase_2/arrays/chunk2_data.npz` rather than re-downloading
the OpenML splits.
