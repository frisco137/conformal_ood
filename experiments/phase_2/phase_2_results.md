# Phase 2 — Results

**Source of truth for the paper's downstream half.** Self-contained: every number carries the script
that produced it and the file it lives in, inline.

*Compiled 2026-08-26 from `phase_2_record.md` (Parts A–D) and the JSON records under
`results/`.*

> **Read [`experiments.md`](experiments.md) first** — it describes each experiment, why it exists,
> what it was measured against and how it was judged. This document carries the numbers.
>
> **Naming, resolved 2026-08-26.** The four-part consolidated measurement record this document is
> compiled from was called `PHASE2.md` and is now [`phase_2_record.md`](phase_2_record.md), which
> removes the collision with this file. Where they differ, **the record wins.** Where either
> disagrees with [`../phase_1/FINAL_NUMBERS.md`](../phase_1/FINAL_NUMBERS.md) on a Phase 1 number,
> **the ledger wins** (see §7).
>
> **Filename note.** Instructions to this project have twice referred to a `PHASE2_RESULTS.md`.
> **No file of that name has ever existed.** Phase 2's documents are `RESULTS.md` (Part A),
> `LOCALISATION.md` (Part B), `SEQUENTIAL.md` (Part C) and `UNCERTAINTY_EXPERIMENT.md` (Part D).
> `LOCALISATION.md` §0 records the same confusion.

---

## 0. What Phase 2 asks

Phase 1 established that all three models violate A1 and A2 far above the instrument floor. Phase 2
asks the obvious follow-up: **does any of it cost anything?** Seven experiments, all on **real** data.

| exp | question | answer |
|---|---|---|
| 2.1 | Is a Jacobian-derived predictive variance better than the model's own head? | **No**, on any model, on any dataset |
| 2.2 | Does the A2 violation localise onto identifiable context points? | **No** — null |
| 2.3 | Does the A1 violation find corrupted labels? | **Yes**, control-validated — but residual is still better |
| 2.4 | Does the variance defect change an acquisition decision? | **Yes** |
| 2.5 | Are the models martingale-self-consistent? | **No**, with controls clean |
| 2.6 | Does the model un-learn from its own acquisitions? | **Yes**, `6–8%` of the time |
| 2.7 | Does BO on the native head match a fitted GP? | **No**; conformal-wrapped does |

**Standing configuration.** 12 OpenML regression datasets × 5 splits, `n_ctx = 100`, `n_cal ≤ 100`,
`n_test ≤ 50`, seeds `[42, 100, 200, 300, 400]`. Features standardised on **context rows only**;
targets left in original units (sd `0.75` to `1083`).
*Script:* `src/chunk2_data.py` → `results/chunk2_datasets.json`,
`arrays/chunk2_data.npz`.

Datasets: `stock`(223), `pwLinear`(229), `places`(509), `pm10`(522), `mu284`(540), `no2`(547),
`strikes`(549), `bodyfat`(560), `fri_c0_250_5`(579), `fri_c3_500_25`(581), `fri_c1_500_25`(582),
`fri_c1_1000_50`(583).

### 0.1 Two decisions changed mid-run, both recorded

**1. The estimator form.** The brief specified `s²(x_*) = σ̂²(1 + J_**)`. That form does **not**
estimate the held-out predictive variance: appending `(x_*, y_*)` makes the query a context point, so
T2 gives `σ²J_** = Var(f_*|y, y_*)` — conditioned on a label that was *invented*. The target is
`Var(f_*|y) + σ²`. Writing `v = Var(f_*|y)`, the Gaussian update gives `J_** = v/(v+σ²)`, inverting to
**`s² = σ²/(1 − J_**)`**. The specified form is **bounded above by `2σ²`** however uncertain the query
is; the target is unbounded.

Measured on the exact GP against its closed form, using the true `σ²`, 5 seeds
(*script:* `chunk1_validate.py` → `results/chunk1_results.json`):

| form | median rel. error | p90 | max |
|---|---|---|---|
| `σ²(1 + J_**)` as specified | `0.417 – 0.520` | `0.596 – 0.628` | `0.640` |
| **`σ²/(1 − J_**)` as used** | **`4.7e-14 – 3.3e-13`** | `1.3e-13 – 8.3e-13` | `1.4e-12` |

`σ̂²(1 + J_**)` is still computed and stored at every query as `s2_jac_specified`. Cost of the
Gaussian assumption in the inversion: on the hierarchical GP (a mixture, hence non-Gaussian) the p90
relative error is `[2.56e-2, 7.30e-2, 1.65e-1, 4.05e-2, 1.37e-2]`, max `1.135`. **That is the ceiling
on this route even when everything else goes right.**

**2. The probe step is relative:** `h = h_frac · std(y_ctx)`. A first pass used the audit's amplitudes
as absolute numbers and was **discarded before any estimator number was read off it**. All three
models renormalise the target internally, so the output quantum in original units scales with
`std(y)`. On `places` (`std(y) = 1083`) an absolute `h = 1e-1` is `9e-5` of a label SD — deep inside
the quantisation floor — and returned `tr J = 110–243` against `n = 100`, i.e. `n − tr J < 0` and
`σ̂²` undefined, with 31–37 negative diagonal entries. The same split at `h = 0.0686·std(y)` returns
`tr J = 16.5` and 2 negative entries, stable to within 5% from `0.01·std` to `0.5·std`.
*Script:* `chunk2_stepsize.py` → `results/chunk2_stepsize_*.json`.

`h_frac` = the audit's own amplitude divided by its mean context label sd `1.457782`:
`1e-3/1.457782` for TabICL, `1e-1/1.457782` for TabPFN and TabSwift.

---

## 1. Experiment 2.1 — Jacobian-derived predictive variance

*Scripts:* `chunk2_estimators.py` → `results/chunk2_estimator_<model>.json`,
`arrays/chunk2_arrays_<model>.npz` (GPU, ~2.5 h); `chunk3_measure.py` → `results/chunk3_results.json`.

Four estimators. The mean channel is **identical** for 1–3; only the uncertainty differs.
1 **native** (the model's own head) · 2 **Jacobian** (`σ̂²/(1−J_**)`) · 3 **conformal** (split
conformal on absolute residuals) · 4 **oracle GP** (sklearn, marginal-likelihood fit — the control).

### 1.1 Headline, aggregated over 12 datasets × 5 splits

*Key:* `chunk3_results.json → aggregate.<model>.<estimator>`.

| model | estimator | cov@90 | IS@90 | width@90 | NLL | ρ(var, sq.resid) |
|---|---|---|---|---|---|---|
| **TabICL v2** | native | `0.9047` | **`515.8`** | `378.5` | **`2.238`** | **`0.300`** |
| | Jacobian | `0.7755` | `772.0` | `582.1` | `871.7` | `0.054` |
| | conformal | `0.8830` | `523.6` | **`352.0`** | `10.68` | n/a |
| | **oracle GP** | `0.9073` | `545.1` | `386.1` | `2.708` | `0.182` |
| **TabPFN v2** | native | `0.9057` | **`515.9`** | `375.5` | **`2.119`** | **`0.283`** |
| | Jacobian | `0.9030` | `547.4` | `416.1` | `40.95` | `0.124` |
| | conformal | `0.8957` | `519.9` | `356.6` | `14.08` | n/a |
| | **oracle GP** | `0.9073` | `545.1` | `386.1` | `2.708` | `0.182` |
| **TabSwift** | native | — | — | — | — | — |
| | Jacobian | `0.9337` | `1818.0` | `1714.0` | `18.09` | `0.042` |
| | conformal | `0.9153` | **`1553.0`** | **`1370.0`** | **`3.261`** | n/a |
| | **oracle GP** | `0.9073` | `545.1` | `386.1` | `2.708` | `0.182` |

TabSwift has **no native head** — `Linear(384,1)` point output,
`models/registry.py has_predictive_distribution=False`. **No variance was synthesised for it.**
The oracle GP row is identical across blocks by construction; it does not depend on the model.

### 1.2 Paired comparisons, interval score at 90%

Win/loss over the 12 datasets, with bootstrap CI and Wilcoxon on the paired per-dataset difference.
*Key:* `chunk3_results.json → comparisons.<model>.<a>_vs_<b>.per_level."0.90"`.

| comparison | TabICL v2 | TabPFN v2 | TabSwift |
|---|---|---|---|
| Jacobian vs native | `1–11`, CI `[+3.9,+682.1]`, `p=9.8e-4` | `0–12`, CI `[+0.3,+85.6]`, `p=4.9e-4` | n/a |
| Jacobian vs conformal | `0–12`, CI `[+2.7,+671.3]`, `p=4.9e-4` | `4–8`, `p=0.52` | `1–11`, CI `[+3.0,+764.9]`, `p=2.4e-3` |
| Jacobian vs oracle GP | `2–10`, CI `[+1.2,+615.5]`, `p=0.027` | `8–4`, `p=0.18` | `5–7`, `p=0.092` |
| native vs conformal | `9–3`, CI `[−28.9,+5.8]`, `p=0.27` | `8–4`, CI `[−26.2,+16.0]`, `p=0.13` | n/a |
| native vs oracle GP | `10–2`, CI `[−67.8,−1.2]`, `p=4.9e-3` | `9–3`, CI `[−67.8,−1.4]`, `p=9.3e-3` | n/a |

**Positive `mean_diff` = the first arm is worse** (interval score is negatively oriented).

### 1.3 What follows

**The Jacobian-derived variance does not beat the native head, on any model that has one, on any
dataset** — `0–12` and `1–11`, CIs excluding zero, `p ≤ 0.001`, same direction on NLL. Phase 1's A3
finding is that the reported variance does not satisfy `s² = σ²(1+J_ii)`. This says that building a
variance out of `J` instead makes calibration **worse**. Those are consistent — A3 is a structural
identity, not a claim about predictive quality — but **the downstream cost this experiment was
designed to find is not present in this form**, and the A3 result should be stated as a structural
violation without an implied practical penalty that has now been looked for and not found.

**Split conformal is the practical recommendation**: lowest IS on TabICL, within noise of native on
TabPFN (`519.9` vs `515.9`, CI `[−26.2,+16.0]`, `p=0.13`), and `11–1` over the Jacobian route on
TabSwift. Cheap, distribution-free, and indifferent to every structural violation Phase 1 measured.

**The native heads carry real information about where the model errs** — ρ with squared residual of
`0.300` and `0.283` against `0.054`/`0.124` for the Jacobian route and `0.182` for a fitted GP,
reaching `0.68–0.72` on `mu284` and `strikes`. **A variance channel that fails a structural identity
is not thereby uninformative**, and the paper must not imply that it is.

### 1.4 TabICL nearly interpolates on real data — a Phase 1 finding, discovered here

`tr J` reaches `99.98–100.03` against `n = 100`, flat across four decades of step size, so `n − tr J`
goes negative and **`σ̂²` is undefined on 10 of 60 splits**, with `J_** ≥ 1` on up to `73%` of queries.
**This is the `J = I` row of the N5 catalogue** — the row that passes A1/A2 vacuously and the reason
the non-degeneracy check exists. It is dataset-dependent: `tr J` is `45.7` on `pwLinear` against
`87.3` on average.

**Flagged for Phase 1:** the audit's synthetic contexts put TabICL *far* from `J = I`
(`‖J−I‖_F/‖J‖_F = 0.85`), while several real datasets put it close. **That is a context-family effect
of exactly the kind E3.1 exists to find. It is a reason to run E3.1, not a result about it.**

### 1.5 Auxiliary — directional monotonicity makes A2 behavioural

*Script:* `chunk4_aux.py` → `results/chunk4_results.json`.
On **58 of 60 splits for TabPFN v2** and **49 of 60 for TabSwift**, raising the context labels along
one direction moves the prediction the **opposite** way along that same direction — `uᵀJu < 0` — with
both controls (exact GP, targeted imitator) behaving as they must. The effect is amplitude-dependent
and vanishes by `t = 0.3‖y − ȳ‖` for TabPFN and TabICL, which is what a derivative statement should
do and is why five amplitudes are reported rather than one.

---

## 2. Experiment 2.2 — Does the A2 violation localise? *(null)*

*Scripts:* `exp22_loo.py` → `results/exp22_loo_<model>.json`, `arrays/exp22_loo_<model>.npz` (GPU,
~2.4 h); `exp22_analyse.py` → `results/exp22_results.json`.

**The score.** `v` = unit eigenvector of `λ_min(sym(QᵀJQ))`, `u = Qv`, per-point loading
`w_i = u_i²` normalised to sum to 1, `Q = get_Q(y, seed=0)`.

### 2.1 The audit reproduces exactly

All ten rows of Phase 1 §4.2 recovered from `tier0_instrument/exp3_reduced_jacobians.npz` to five
decimal places with **identical top-5 sets**. *Key:* `exp22_results.json → 2.2.1_reproduction.<tag>[]`.
No disagreement, no configuration difference to name.

### 2.2 On real data it does not point at anything actionable

Mean over 60 splits per model. *Key:* `exp22_results.json → per_split[]`.

| model | mass top5 | participation | `λ_min` mean | `λ_min > 0` |
|---|---|---|---|---|
| TabICL v2 | `0.4918` | `17.69` | `−0.1195` | **`27/60`** |
| TabPFN v2 | `0.3929` | `22.97` | `−1.6583` | `0/60` |
| TabSwift | `0.4707` | `17.46` | `−0.4746` | `0/60` |
| **gp_oracle (control)** | `0.8004` | `5.41` | `+0.2452` | `60/60` |

**Spearman against four targets**, mean over 60 splits — the loading `w` against the baselines:

| target | `w` | `\|J_ii\|` | col norm | residual | leverage |
|---|---|---|---|---|---|
| own residual (2.2.4) | `0.249 / 0.173 / 0.093` | `−0.14/−0.02/−0.34` | `−0.16/0.00/−0.36` | `1.000` | `0.07/−0.02/0.10` |
| local held-out error (2.2.6) | `0.032 / 0.020 / −0.068` | — | — | `0.123/0.092/0.343` | `0.074/0.081/0.111` |
| **actual LOO influence (2.2.5)** | **`0.089 / 0.048 / 0.143`** | — | — | **`0.425/0.178/0.151`** | `0.202/0.122/0.117` |

*(order: TabICL / TabPFN / TabSwift; `gp_oracle` control at `−0.003 / −0.053 / —` respectively.)*

**On two of three models the loading is the worst of the five scores at predicting LOO influence** —
worse than plain feature leverage. The high-loading set is also **not a stable region**: across splits
of the same dataset its points are no closer together than random subsets of the same size
(ratio `0.955 / 0.957 / 0.976`) while the GP control's are (`0.756`).

Matched-pair perturbation is null for TabICL (`p = 0.76`) and TabPFN (`p = 0.16`), positive only for
TabSwift (`p = 0.019`).

### 2.3 The null is about the models, not the score

Two facts establish that. **First, the score carries its own information**: it anti-correlates with
`|J_ii|` at `−0.28/−0.15/−0.13` while `|J_ii|` and the column norm are near-duplicates of each other
at `+0.83/+0.59/+0.72`. **Second, the power check settles it** — once the constructed direction
actually dominates, recovery is `|cos| = 0.998` and Spearman `0.990`. The shipped `TargetedImitator`
recovers only to `0.63` because its competition ratio `(c/2)/λ_max(QᵀWQ)` is `0.406`: **it never
presented a dominant direction to find.** *Key:* `exp22_results.json → 2.2.11b_strong_imitator.mult_<m>[]`.

**A per-context-point diagnostic does not fall out of A2.**

---

## 3. Experiment 2.3 — Does the A1 violation find corrupted labels? *(yes, narrowly)*

*Scripts:* `exp23_corrupt.py` (GPU, ~4.7 h), `exp23_controls.py`, `exp23_loo.py`,
`exp23_analyse.py` → `results/exp23_results.json`.

4 datasets × 5 splits × 3 schemes × 3 rates = **180 combinations per model**. Chance rate `0.117`.
*(Reduced from the full 12-dataset grid, which was 108 540 model calls per model — about 10.6 h for
TabICL alone. The 8 dropped datasets are named in `exp23_corrupt_<model>.json → _config.dids_dropped`.)*

### 3.1 Detection AUC

*Key:* `exp23_results.json → aggregate.<model>.overall.<score>.{auc,prec}`.

| score | TabICL v2 | TabPFN v2 | TabSwift |
|---|---|---|---|
| **row-minus-column asymmetry** | **`0.690`** | `0.627` | `0.525` |
| row norm | `0.642` | `0.630` | `0.492` |
| column norm | `0.437` | `0.483` | `0.474` |
| `w` (the A2 loading) | `0.571` | `0.602` | `0.572` |
| **residual magnitude (baseline)** | **`0.685`** | **`0.740`** | **`0.644`** |
| **actual LOO (gold standard)** | **`0.746`** | **`0.744`** | **`0.667`** |

Precision@k for the asymmetry score: `0.378 / 0.311 / 0.163` against chance `0.117`.

**The mechanism is visible in the pair of norms it is built from**: the row norm detects, the column
norm does not. **For any symmetric `J` those two are the same number**, and on the hierarchical GP
they are — `0.5303` both. **The gap between them *is* the A1 violation, used as a detector.**

### 3.2 Controls make the signal credible

*Key:* `exp23_results.json`, `exp23_controls.json`.

- **Exact GP**: its Jacobian is label-blind — `J = K(K+σ²I)⁻¹` contains no `y`. Measured:
  `‖J(y_corrupt) − J(y_clean)‖_F/‖J_clean‖_F` has **mean `0.000000e+00`, max `0.000000e+00`** over all
  180 combinations, and its four Jacobian scores return values identical to four decimal places across
  all three corruption schemes. **No Jacobian score carries spurious corruption signal.**
- **Hierarchical GP** — the control the design actually needs, Bayes-consistent **and** with a
  label-dependent Jacobian. `J` changes by relative Frobenius **mean `0.2659`, max `2.3522`** under
  corruption while staying symmetric at `asym(J) = 2.938e-14`. **Its asymmetry score still sits at
  `0.515`** — chance.
- **Uncorrupted baseline**: no score concentrates on an arbitrary subset — `[0.486, 0.518]` across all
  ten scores and three models.

**One caveat on `w`.** It is not a pure function of `J`, because `Q = get_Q(y)` is built from the
labels. The exact GP — whose Jacobian provably does not move — still returns `w` AUC `0.5625` overall
and `0.6516` under the noise scheme. **That is the floor**, and against it the models' `0.571/0.602/0.572`
add nothing.

### 3.3 But residual is still the better tool, and that must be said plainly

Paired over all 180 combinations, the asymmetry score **loses decisively** on TabPFN (`−0.113`,
`p = 3.1e-18`) and TabSwift (`−0.119`, `p = 2.9e-17`), and **ties** on TabICL (`+0.005`, 96 of 180,
`p = 0.55`). The aggregate `0.6901` vs `0.6851` on TabICL **is not a win and reading it as one would
be wrong.** Actual LOO beats the asymmetry score on all three (`p` between `3.6e-10` and `2.9e-5`).

Ordering that survives: **100 refits per split > one residual vector > any Jacobian score**, with the
asymmetry score reaching the residual only on TabICL.

**Still, this is the most substantive positive result downstream of the audit**: the first time a
structural violation was converted into a working diagnostic rather than a description. The score is
**exactly zero for every Bayes-consistent map by construction**, which is what makes it a test of A1's
practical content rather than a generic outlier statistic.

### 3.4 A1 and A2 are not interchangeable diagnostics

The model where the asymmetry score works best is **TabICL, which has the weakest A2 violation** of
the three (`λ_min > 0` on 27 of 60 real splits — no negative eigenvalue at all). The model with the
most negative diagonals, **TabSwift, is where the asymmetry score fails** (`0.525`). Whatever the
row-column gap tracks, **it is not the object the negative eigendirection tracks**, and the habit of
reporting A1 and A2 side by side as two readings of one violation is **not supported** by their
downstream behaviour.

---

## 4. Experiments 2.4–2.7 — Sequential coherence and BO

*Scripts:* `exp24_acquisition.py` → `results/exp24_results.json` (CPU);
`exp25_martingale.py`/`exp25_controls.py`/`exp25_analyse.py` → `results/exp25_*.json` (GPU, ~2 h);
`exp27_bo.py`/`exp27_analyse.py` → `results/exp27_*.json` (GPU, ~3.5 h).

### 4.1 The variance defect reaches the decision (2.4)

Native and Jacobian channels pick the **same** candidate on `0.160` of TabICL splits and about half
of TabPFN's, and the Jacobian channel's regret is worse than pure exploitation everywhere. The
acquisition surfaces remain strongly rank-correlated (`+0.53` to `+0.89`) — **the channels agree
about the shape of the landscape and disagree about the argmax**, which is what a decision-relevant
defect looks like. *Key:* `exp24_results.json → aggregate.<model>.<acq>`.

### 4.2 The martingale is violated (2.5)

Self-consistency drift, `|drift|/SE`, at targets other than the probe:

| | `\|drift\|/SE` | probes beyond 2 SE |
|---|---|---|
| **TabICL v2** | **`7.4`** | `81%` |
| **TabPFN v2** | **`14.2`** | `90%` |
| exact GP, analytic (control) | `1.07e-11` | — |
| exact GP, sampled (control) | `0.694` | — |
| **hierarchical GP, sampled (control)** | **`0.792`** | — |

The hierarchical GP is the control that matters — Bayesian **with** a label-dependent Jacobian — and
it sits at `0.792` once sampled from its own mixture rather than from the wrong law.

**Two unpredicted features.** First, the violation is much clearer at `x_* ≠ x'` than at `x_* = x'`:
TabICL's drift at the probe itself is inside Monte Carlo error (`0.808`). **The incoherence is in how
an observation propagates to *other* locations, not in how the model absorbs it locally.** Second,
**the variance side nearly closes** — the total-variance identity has relative gaps of `0.1%` to
`1.6%` — while the mean side does not. A3 concerns the coupling of those two channels, and here the
coupling identity holds while the mean identity fails.

> **The martingale test is a value functional, and T1 says value functionals cannot separate the
> classes with any margin.** That it detects something anyway is not a contradiction — T1 is about
> worst-case separation with a uniform margin, not about whether a particular non-Bayes map happens
> to drift. But it does mean **this drift is evidence about *these checkpoints*, not a general
> instrument.** The T1 imitator control makes that concrete: it is ε-close in values and its drift is
> bounded by `2ε` regardless of how non-Bayesian it is.

### 4.3 The A2-localised prediction fails a third time (2.5.4, 2.6.3)

The novel claim was that drift should be larger and signed wrong where `J_** < 0`. Result: **one
significant signed effect, on one model, in one target configuration** — TabPFN at the probe itself,
`p = 0.020` — with no magnitude effect (`p = 0.083`), an **opposite-signed** effect on TabSwift, and
**no effect at all on TabICL because its negative stratum is empty: 0 of 3000 held-out probes**.
Spearman between drift and `J_**` across all probes is `+0.02` to `+0.11`.

**2.6.3 could not be tested at all**: `J_**` at the acquired point was **non-negative on all 500
acquisitions**. Acquisition functions select points the model is optimistic and uncertain about, and
those are not where the appended-query diagonal goes negative. **The mechanism by which a negative
diagonal was supposed to compound in a loop has no opportunity to act, at least under EI.**

Taken with 2.2 and 2.3, **this is the third independent attempt to convert A2 into a per-point
diagnostic and the third null. The A1-derived asymmetry score of 2.3 remains the only downstream
quantity that has carried signal.**

### 4.4 The BO loop reproduces 2.1's ordering (2.7)

TabPFN v2 only, 5 synthetic objectives (`branin2`, `hartmann3`, `hartmann6`, `ackley5`,
`rosenbrock5`), `T = 20`, `S = 5`, 128-point pool.
*Key:* `exp27_results.json → comparisons.<a>_vs_<b>`.

- **Conformal-wrapped TabPFN beats the native head on all five objectives** — `0–5`, mean normalised
  difference `+0.132`, CI `[+0.039, +0.266]`, `p = 0.0064` — and is **statistically indistinguishable
  from a properly fitted GP** (`p = 0.45`, CI straddling zero).
- The **native head is significantly worse than the GP** (`p = 0.042`); the Jacobian arm worse still
  (`p = 0.0056`).

So the brief's question — does native-variance BO match or beat the GP — **answers no**, and 2.1's
recommendation carries over unchanged: **wrap the model, do not trust its head.**

*Note on 2.4 vs 2.7:* conformal was a **degenerate** channel in 2.4, reducing exactly to
greedy-on-the-mean, because its width is constant across candidates at one step. In the loop the
calibration set is refitted each iteration and the width tracks the growing context. The two results
are consistent: constant width is useless for choosing *among* candidates, and a well-scaled width is
what a sequential loop needs.

### 4.5 What TabPFN does in a BO loop that a Bayesian would not (2.6)

The cleanest behavioural statement in Phase 2. On **`6.0%` to `7.8%` of acquisitions** the model finds
a point **better** than it predicted and becomes **less optimistic** about it — against **`1.0%`** for
a refitted GP on the identical protocol. The shrinkage factor `(μ_{n+1} − μ_n)/(y_t − μ_n)` lies
outside `[0,1]` on **`19%` to `25%`** of acquisitions against the GP's `5%`, and its **median is
`0.55` against the GP's `0.99`** — the model absorbs about half of each surprise where a posterior
mean absorbs nearly all of it.

**Whether that costs anything is not established here**: the per-trajectory correlation between
violation count and final regret changes sign across objectives for every arm, and with 5 trajectories
per cell those correlations carry little weight.

### 4.6 One thing that did not go wrong

2.1 found TabICL nearly interpolating at `n = 100`, leaving `σ̂²` undefined on 10 of 60 splits. The
`n − tr J` watch was added because BO contexts start small and grow. On TabPFN across 500 iterations
at `n = 10` to `29`, `tr J / n` rises from `0.474` to `0.614` and **never exceeds `0.941`**, minimum
`n − tr J` is `+1.35`, and **`σ̂²` is defined at every single iteration.**

---

## 5. What is NOT established

### 5.1 Scope limits, per experiment

**2.1** — All of it at `n_ctx = 100`, 12 datasets, `d ≤ 50`, numeric features only. **The append
perturbation was never measured on the frozen models**, where `n` changes from 100 to 101 and
behaviour is `n`-dependent in ways a GP's is not. The `σ̂²` route inherits a **~18% single-split
sampling spread** at this `n`. A variant taking `σ²` from the calibration split instead of from
`tr J` would remove that spread at the cost of no longer being derivable from the model alone; **it
was not run.**

**2.2** — The null is well-powered (see §2.3) but is a null about *these three models on these 12
datasets*.

**2.3** — Ran on **4 of the 12 datasets**, so dataset-level generalisation rests on **4 points per
model**, and per-dataset spread is wide (the asymmetry score ranges `0.601` to `0.794` on TabICL
alone). Item 2.3.7's actual-LOO baseline ran **at the 10% rate only**; 5% and 20% are NOT MEASURED.
**The corruption schemes are synthetic and known** — nothing here speaks to naturally occurring label
noise, whose structure is not a flip, a shuffle, or additive Gaussian noise.

**2.4–2.7** — The BO study is **one model, five synthetic objectives**, `T = 20`, `S = 5`, 128-point
pool: 25 paired trajectories per comparison, so objective-level win counts rest on **five points
each**. **TabICL v2 and TabSwift were never run in the loop** (12.4 h and 3.9 h respectively for the
Jacobian arm). **Knowledge gradient was not run at all** — the acquisition most exposed to what 2.5
measures, because it is literally an expectation over the model's predictive future. **No real
objective was included.** **TabSwift's entire 2.5 result rests on a Gaussian surrogate for a
predictive distribution it does not have**, and is not a statement about its own sampling law. The
drift in 2.5 is at `n = 100`; the loop operates at `n = 10` to `29`, and **nothing here connects the
two.**

### 5.2 Cross-cutting

- **Phase 2 establishes no Phase 1 number.** It re-derives none and changes none. Where it appears to
  speak to Phase 1 — §1.4's `tr J → 100` finding — it is flagged as **a reason to run E3.1, not a
  result about it**.
- **The one genuine Phase 1 → Phase 2 dependency is `TargetedImitator`**, used as a control in 2.2 and
  in chunk 4. See §7.
- **2.6's mechanism was untestable, not tested and passed.** `J_** ≥ 0` on all 500 acquisitions.

---

## 6. Discrepancies found and how they were resolved

| # | discrepancy | resolution |
|---|---|---|
| 1 | `phase_2/README.md:123` (now `../_archive/phase_2_README_old.md`) states **"The tier-0 file still carries the bug"** about `TargetedImitator.inner_jacobian`; `UNCERTAINTY_EXPERIMENT.md:908` and `phase_2_record.md:2944` say "the tier-0 file is unchanged". | **STALE as of 2026-08-26.** The bug was fixed at source in `../phase_1/core/controls.py`; the three inline workarounds now call the corrected method. **Verified bit-identical** (max abs diff exactly `0.0` on all five seeds), so **no Phase 2 number changes.** A correction banner was added to that README before it was archived; the record documents are left as history. |
| 2 | Instructions have twice cited a **`PHASE2_RESULTS.md`**. | **No such file has ever existed.** Mapping used: `phase2.md` → `UNCERTAINTY_EXPERIMENT.md`, `PHASE2_RESULTS.md` → `RESULTS.md`. `LOCALISATION.md` §0 records the same confusion arising once before. |
| 3 | A brief attributed **participation ratios of 3–33/98** to `PHASE2_RESULTS.md` §8.2. | `RESULTS.md` §8.2 is directional monotonicity and reports **no** participation ratio; **no Phase 2 document reports one**. The figures are `FINAL_NUMBERS.md` §4.3, measured on the audit's **synthetic** contexts. Phase 2 §2.1 reproduces them exactly and then measures the same quantity on real data, where it differs. |
| 4 | The brief for 2.3.9 required the exact GP to show "chance-level AUC on asymmetry and **above-chance on the column norm**". | **Impossible by construction** and corrected before running. The exact GP's `J` contains no `y`, so corrupting labels leaves it **bit-identical** and *every* Jacobian score is exactly at chance. Kept as a no-spurious-signal control; the **hierarchical GP** added as the control the brief actually needed. |
| 5 | The imitator control's partial recovery (`\|cos\| = 0.63`) looked like a score failure. | It is **a property of the construction, not of the score** — competition ratio `0.406`. Quantified in §2.3 with a power check reaching `0.998`. |

**Unresolved: none.** Every disagreement found had a determinable answer.

---

## 7. What this supersedes, and what happens to the old documents

**This document and [`../phase_1/phase_1_results.md`](../phase_1/phase_1_results.md) are the source of
truth for the paper.** Everything else is either the layer beneath them or history, and **nothing was
deleted**:

| document | where it is now | status |
|---|---|---|
| `../phase_1/FINAL_NUMBERS.md` | `phase_1/` | **the Phase 1 ledger.** Per-seed values, the retraction table, the full provenance index. **Wins any dispute with `phase_1_results.md`.** |
| `phase_2_record.md` | `phase_2/` | **the Phase 2 record**, Parts A–D, per-split detail. Was `PHASE2.md`. **Wins any dispute with this file.** |
| `../phase_1/theory_and_claims.md` | `phase_1/` | **live**, revision 2, amendment log at its §5. |
| `../phase_1/experiments.md`, `experiments.md` | each phase folder | **live**, describing what was run. |
| `phase_2_RESULTS.md`, `phase_2_LOCALISATION.md`, `phase_2_SEQUENTIAL.md`, `phase_2_UNCERTAINTY_EXPERIMENT.md` | `../_archive/` | **history.** The four source documents consolidated unchanged into `phase_2_record.md`. Not cited by the paper. |
| `experiments.md` (the dated pre-registration), `DEVIATIONS.md` | `../_archive/` | **history, deliberately retained.** The E5.2 claim depends on the prediction being demonstrably dated, and the deviations log is the only record that E0.6's halt condition was breached knowingly. **Archived, not cited, out of the navigation.** |
| `quarantine/` | `../_archive/` | known-wrong code, import barrier intact. |

**Why everything is retained.** The audit trail from raw JSON → part document → consolidated record →
this file is complete and checkable at every hop, and that is worth more than the tidiness of deleting
the intermediate layers — particularly given that **nine other cited documents in this project have
already been lost entirely** (`../phase_1/phase_1_results.md` §9.2).
