# Phase 1 — Results

**Source of truth for the paper's audit half.** Self-contained: every number below carries the
script that produced it and the file it lives in, inline. Nothing here requires opening
`FINAL_NUMBERS.md`, though that remains the underlying ledger and wins any dispute.

*Compiled 2026-08-26 from `FINAL_NUMBERS.md` revision 2 plus Sections 10 and 11.*

> **Read [`experiments.md`](experiments.md) first** — it describes each experiment, why it exists,
> what it was measured against and how it was judged. This document carries the numbers.
>
> **What this supersedes.** This document replaces `FINAL_NUMBERS.md` as the reading entry point for
> Phase 1. [`FINAL_NUMBERS.md`](FINAL_NUMBERS.md) stays as the ledger — it holds per-seed values, the retraction table,
> and the full provenance index — and **where the two disagree, the ledger wins.** Nine older
> documents (`final_audit_report.md`, `RESULTS_REPORT.md`, `CONTROL_VALIDATION.md`,
> `CLARIFICATIONS.md`, `E11_WRAPPER_FORM.md`, `e2_5_report.md`, `tier2_report.md`, `e1_4_report.md`,
> `ARCHITECTURE_REPORT.md`) are cited across the tree but **do not exist anywhere** — not in the
> working tree, not in `intermediate/`, not in any git commit. No number is lost; the supersession
> narrative is. See §9.2.
>
> **Companion:** [`../phase_2/phase_2_results.md`](../phase_2/phase_2_results.md).

---

## 0. What was measured, and the standing configuration

Three frozen tabular regressors — **TabPFN v2, TabICL v2, TabSwift** — audited against three
conditions that every posterior mean satisfies for every prior, under a Gaussian channel.

| condition | statement | parameter-free? |
|---|---|---|
| **A1** symmetry | `asym(QᵀJQ) = ‖J−Jᵀ‖_F/‖J‖_F ≈ 0` | yes |
| **A2** positivity | `negeig(QᵀJQ) = −min(0, λ_min(sym J))/‖J‖₂ ≈ 0` | yes |
| **A3** cross-channel | `s²ᵢ = σ²(1 + Jᵢᵢ)` | one profiled scalar |

**Context, everywhere in Phase 1 except §7 (E5.1):**
`generate_audit_context(n=100, d=5, sigma=1.0, seed=s)`, seeds `[42, 100, 200, 300, 400]`.
90 unique `X ~ N(0,I₅)` rows plus **10 exact duplicates**, deliberately, to force smoothing.
Reduced basis `get_Q(y, seed=0)`, shape `(100, 98)`.
*Script:* `core/context.py`, `core/metrics.py`.

**Canonical metric definitions**, and the only implementations:
`core/metrics.py:3` (`asym`), `:13` (`negeig`), `:135` (`get_Q`, seeded).

> **Convention warning.** Every `asym` computed under the old `tier2_audit/` directory was
> `‖(J−Jᵀ)/2‖_F/‖J‖_F`, **exactly half** the canonical value. That directory is quarantined at
> `../_archive/quarantine/tier2_audit/`. No number in this document comes from it. Two historical
> values are quoted beside current ones and are doubled where they appear (§3.1, §3.2).

---

## 1. The instrument is valid

Nothing downstream is interpretable until these pass.

### 1.1 The controls read zero

Exact GP through the identical pipeline, reduced basis, `t=1e-3`, no dither, 5 seeds.
*Script:* `tier0_instrument/chunk1_control_validation.py` → `chunk1_results.json`,
key `1A.<name>.measured`.

| control | `asym` | `negeig` |
|---|---|---|
| ExactGP, unwrapped | `2.877e-12` (spread `2.65–3.37e-12`) | `4.31e-16` |
| ExactGP, **wrapped** in the N1 normaliser | `3.141e-12` (spread `2.71–3.76e-12`) | `5.00e-16` |

**`3.141e-12` is the reduced-basis floor** and is the denominator for every ratio in §3 and §4.

### 1.2 The projection transmits violations rather than laundering them

The load-bearing control (E1.2(b), C-I3). `TargetedImitator` is Bayes-impossible by construction,
wrapped in the same normaliser, and must **survive** the projection.
*Script:* `core/controls.py` (the map), `chunk1_control_validation.py` → `chunk1_results.json`.

| seed | `‖J_meas − QᵀGQ‖_F / ‖·‖_F` | `negeig` measured | `negeig` analytic |
|---|---|---|---|
| 42 | `2.131e-12` | `0.088194` | `0.088194` |
| 100 | `2.356e-12` | `0.059000` | `0.059000` |
| 200 | `2.470e-12` | `0.064744` | `0.064744` |
| 300 | `2.082e-12` | `0.089827` | `0.089827` |
| 400 | `2.808e-12` | `0.120723` | `0.120723` |

`asym` measured `0.6005769 … 0.6004114` against analytic identical to 7 significant figures.

> **Correction of record, 2026-08-26.** `TargetedImitator.inner_jacobian` returned `−c·vvᵀ` for a map
> applying `−(c/2)⟨v,u⟩v` — **twice** the rank-one term. Against finite differences the old closed
> form was off by `5.6%`; corrected it agrees to `6.6e-10`. **The map was always correct; only the
> analytic reference was wrong**, so no measured value moved. The bug was independently rediscovered
> and worked around inline in three scripts and left unfixed in the one that wrote to disk, which is
> why `chunk1_results.json`'s stored `analytic` block read `negeig` mean `0.318842` where the truth
> is `0.084498`. Fixed at source in `core/controls.py`; regression test at
> `core/test_controls.py` (22 tests). The table above is the corrected version and is
> what `FINAL_NUMBERS.md` §2.3 always reported.

### 1.3 The instrument's own error floor

**Row-sum residual `r₁ = ‖J·1 − 1‖₂/‖1‖₂`**, exactly zero for any shift-equivariant map, so any
deviation is pure instrument error.
*Script:* `tier0_instrument/recompute_orphans.py` → `clarif_item4_6.json`, key `<tag>_r1`.
Computed from `exp12_ambient_jacobians.npz`.

| | `r₁` mean | reading |
|---|---|---|
| ExactGP wrapped (control) | `3.70e-12` | the floor |
| ExactGP unwrapped (control) | `8.18e-02` | not shift-equivariant, as expected |
| **TabICL v2** | `0.0099` | ~1% instrument error at `h=1e-3` |
| **TabPFN v2** | `0.5114` | ~51% instrument error at `h=1e-1` with `N=10` dither |
| **TabSwift** | `0.9071` | **not an error bound** — see §1.5 |

### 1.4 The quantisation artifact, and why the probe amplitude differs per model

A quantised wrapped exact GP has true `asym`, `negeig` and curvature **all exactly zero**, so every
nonzero value it returns is manufactured. *Script:* `chunk3_dither_controls.py` → `chunk3_results.json`;
`exp3_reprobe.py` → `exp3_reprobe.json`, key `artifact_t1e-01`.

| `t` | dither | `asym` | `negeig` |
|---|---|---|---|
| `1e-2` | off | `0.285744` | `0.020461` |
| `1e-2` | `±δ`, N=10 | `0.096875` | `0.001893` |
| **`1e-1`** | **`±δ`, N=10** | **`0.009744`** | **`0.0000178`** |
| — | unquantised | `3.174e-13` | `5.563e-16` |

The artifact falls as `1/t`. Dither count is the wrong knob: `asym` decays as `N^-0.30`, so removing
it that way needs `N ~ 2e4`.

**Amplitude per model**, and the reason (`FINAL_NUMBERS.md` §3.5):

| model | `t` | dither | why |
|---|---|---|---|
| TabICL v2 | `1e-3` | none | float32 output, median nonzero jump `9.83e-7`; measured sweep flat to `0.21%` across `t ∈ {1e-3,1e-2,1e-1}` |
| TabPFN v2 | `1e-1` | `±δ=6.87e-4`, N=10 | bar-distribution quantisation; artifact `0.0969 → 0.0097` from `1e-2` |
| TabSwift | `1e-1` | none | **float16** output; `1e-3` is inside the derivative-collapse regime |

### 1.5 Preprocessing, E1.1 — and the one model that fails it

Forward passes only, no derivatives, no dither. Worst value over 5 seeds.
*Script:* `tier0_instrument/e11_wrapper_form.py` → `e11_wrapper_form.json`.
Threshold: `< 1e-3` relative.

| | shift `c=1.0` | scale `c=2.0` | verdict |
|---|---|---|---|
| ExactGP wrapped (control) | `1.854e-16` | `3.124e-16` | pass |
| **TabPFN v2** | `7.270e-07` | `0.000e+00` | **pass** |
| **TabICL v2** | `2.171e-06` | `0.000e+00` | **pass** |
| ExactGP unwrapped (control) | `8.122e-02` | `0.000e+00` | fails shift, as a linear map must |
| **TabSwift** | `3.379e+00` | `9.165e-01` | **FAILS BOTH** |

**Target-preprocessing path, read from source:** TabPFN v2 `tabpfn/regressor.py:877-880`;
TabICL v2 `tabicl/_sklearn/regressor.py:412-413`; **TabSwift `models/vendor/tabswift/regressor.py:319-324`
and `:526` — the target `StandardScaler` is commented out in both directions**, `self.scaler_ = None`
at `:165`. No target transform is applied.

**Consequence, and it is load-bearing for TabSwift.** N1(ii)'s identity `QᵀJQ = QᵀGQ` is derived only
for `m(y) = ȳ·1 + s_y·g(u)`. It is licensed for TabPFN v2 and TabICL v2 and **is not established for
TabSwift**. TabSwift's A1 number in §3 is therefore the asymmetry of the *compressed Jacobian*, not
of a recovered inner map. **A2 is unaffected for all three**: a compression `QᵀJQ` of a PSD matrix is
PSD for any orthonormal `Q`, so a positivity violation on a subspace is a violation of the full
condition regardless.

### 1.6 N1(ii) verified numerically

`QᵀJ_wrappedQ` vs `QᵀJ_unwrappedQ` on the exact GP, relative Frobenius, 5 seeds:
`[2.680, 2.744, 2.981, 2.606, 3.305] × 10⁻¹²`, mean `2.863e-12`.
*Script:* `recompute_orphans.py` → `exp_n1ii.json`.

### 1.7 Seven stress tests

*Script:* `recompute_orphans.py` → `exp_stress_tests.json`. All at machine precision:
estimator recovers a known asymmetric `J` to `6.15e-13`; transpose/orientation `8.88e-16`; basis
invariance across `Q` seeds `{0,1,2}` — `asym` spread `0.0`, `negeig` spread `1.53e-16`; null
perturbation `0.0`; determinism of the analytic pipeline `0.0`; reduced-probe vs
ambient-then-project `2.05e-12`; permutation consistency `0.0`.

### 1.8 Theory checks

*Script:* `chunk2_a3_controls.py` → `chunk2_results.json`.
**T2** (`σ²J = Cov(f|y)`) on the exact GP: relative Frobenius `[6.38, 6.31, 6.18, 5.64, 8.35] × 10⁻¹³`.
**T4** (hedging decomposition) on the hierarchical GP: `[1.21e-9, 1.47e-9, 4.17e-8, 3.10e-9, 7.54e-9]`.

---

## 2. Non-degeneracy — checked before any verdict

`J ≈ I` is N5's 1-NN row and passes A1/A2 **vacuously**; `J ≈ 0` is instrument noise. Both are
reportable findings, neither is a pass.
*Script:* `exp4_quanta_and_tabicl.py` → `exp4_results.json`; `exp3_reprobe.py` → `exp3_reprobe.json`.

| model | `‖J−I‖_F/‖J‖_F` | `‖J‖_F` vs `√98 = 9.899` | verdict |
|---|---|---|---|
| TabICL v2 | mean `0.8513`, per-seed `[0.714, 0.649, 1.376, 0.585, 0.933]`, **min `0.585`** | `6.990` (ratio `0.706`) | non-degenerate |
| TabPFN v2 | — (see note) | — | non-degenerate at `t=1e-1` |
| TabSwift | — | `4.0598` at `t=1e-1` | non-degenerate at `t=1e-1` only |

**TabSwift caveat.** `‖J‖_F` collapsed `392.8 → 6.09` across the `h` sweep; `t=1e-3` is inside the
derivative-collapse regime and any number measured there is void. `t=1e-1` is the only valid
amplitude for this model.

---

## 3. A1 — Symmetry

**Threshold** (`experiments.md` E2.3): pass `≤ max(3F, 0.05)`; fail `> max(10F, 0.15)`; between is
inconclusive. `F` = the artifact floor beside each model.

### 3.1 Measured, reduced basis, 5 seeds

*Script:* `exp3_reprobe.py` → `exp3_reprobe.json` (TabPFN, TabSwift);
`exp4_quanta_and_tabicl.py` → `exp4_results.json` (TabICL).

| model | `asym` mean ± sd | artifact floor | **ratio** | verdict |
|---|---|---|---|---|
| **TabPFN v2** `t=1e-1`, dither | **`1.312414 ± 0.091648`** | `0.009744` | **`134.7×`** | **FAIL** |
| **TabSwift** `t=1e-1` | **`1.020249 ± 0.182063`** | `0.003857` – `0.017996` | **`264.5×` – `56.7×`** | **FAIL** |
| **TabICL v2** `t=1e-3` | **`0.580331 ± 0.078422`** | `0.001018` – `0.004258` | **`569.8×` – `136.3×`** | **FAIL** |

Per-seed: TabPFN `[1.382, 1.349, 1.280, 1.402, 1.149]`; TabSwift `[1.207, 1.270, 0.903, 0.905, 0.816]`;
TabICL `[0.607, 0.533, 0.705, 0.470, 0.587]`.

TabICL and TabSwift get a **bracket** rather than a point floor because neither output is binned the
way TabPFN's is: the floor is computed at the `p10` and the median of the measured nonzero output
jumps. Measured quanta, 501-point scan, seed 42 (`exp4_results.json`, keys `quantum_tabicl`,
`quantum_tabswift`): TabICL median `9.8348e-07`, p10 `2.3842e-07`, dtype float32;
TabSwift median `4.2725e-04`, p10 `9.1553e-05`, dtype **float16**.

### 3.2 Step-size stability (C-I5)

| model | sweep | drift |
|---|---|---|
| TabICL v2 | `t = 1e-3 / 1e-2 / 1e-1` → `0.580331 / 0.580057 / 0.579137` | `0.21%` over two decades |
| TabPFN v2 | `t = 1e-2 / 3e-2 / 1e-1 / 3e-1` → `1.373 / 1.333 / 1.312 / 1.275` | `6.6%` over 1.5 decades |
| TabSwift | `t = 1e-2 / 1e-1 / 1e0` → `1.340 / 1.020 / 0.951` | `24%` from `1e-2` to `1e-1` |

*Historical cross-check:* `results_tabpfn_5seed.json` (now quarantined) at `t=1e-2` gives
`0.6655 ± 0.0221` halved-convention = **`1.3309 ± 0.0443` canonical**, against this run's
`1.373350 ± 0.023903`. TabICL's historical `0.5802 ± 0.0784` reproduces the current
`0.580331 ± 0.078422` to four significant figures despite an unseeded `Q` in the original.

### 3.3 Positional encoding is not the cause — E1.4, TabICL v2, paired

RoPE zeroed at `model_.row_interactor.tf_row.rope = None`, same contexts, `h=1e-3`, canonical `asym`.
*File:* `e14_paired.json`. **No producing script exists** — see §9.1.

| seed | raw | zeroed | paired diff |
|---|---|---|---|
| 42 | `0.606412` | `0.622545` | `+0.016132` |
| 100 | `0.532348` | `0.524374` | `−0.007974` |
| 200 | `0.705529` | `0.707124` | `+0.001595` |
| 300 | `0.470630` | `0.435577` | `−0.035052` |
| 400 | `0.587082` | `0.601217` | `+0.014135` |

**Paired mean `−0.002233`, sd `0.020796`, se `0.009300`, 95% CI `[−0.028050, +0.023585]`.**
The CI contains zero. RMSE change `−0.005144 ± 0.032665`. **RoPE is not the source of the asymmetry.**
TabSwift has no positional encodings. **E1.4 for TabPFN v2 is NOT MEASURED** — it needs source
modification of the attention path.

---

## 4. A2 — Positivity *(leads, per N3)*

**Threshold** (`experiments.md` E2.2): pass `≤ max(3F, 0.02)`.
**Why this leads:** N3 proves an affine normaliser *cannot* manufacture a negative eigenvalue in the
projected block, though it demonstrably can manufacture raw asymmetry. A2 is strictly more robust to
preprocessing than A1.

### 4.1 Measured, reduced basis, 5 seeds

*Script:* `exp3_reprobe.py` → `exp3_reprobe.json`; `exp4_quanta_and_tabicl.py` → `exp4_results.json`.

| model | `negeig` mean ± sd | artifact floor | **ratio** | verdict |
|---|---|---|---|---|
| **TabPFN v2** `t=1e-1` | **`0.404109 ± 0.091393`** | `0.0000178` | **`22 711×`** | **FAIL** |
| **TabSwift** `t=1e-1` | **`0.264443 ± 0.101490`** | `2.894e-06` – `6.969e-05` | **`91 381×` – `3 795×`** | **FAIL** |
| **TabICL v2** `t=1e-3` | **`0.176660 ± 0.046088`** | `1.990e-07` – `4.076e-06` | **`887 607×` – `43 339×`** | **FAIL** |

Per-seed: TabPFN `[0.462, 0.408, 0.396, 0.513, 0.241]`; TabSwift `[0.271, 0.356, 0.169, 0.132, 0.394]`;
TabICL `[0.242, 0.159, 0.220, 0.129, 0.133]`.

Amplitude sweeps: TabPFN `0.439/0.417/0.402/0.361` at `t=1e-2/3e-2/1e-1/3e-1`;
TabSwift `0.510/0.264/0.243` at `1e-2/1e-1/1e0`; TabICL `0.17666/0.176656/0.175941`.

**Against the imitator control:** the targeted imitator, built to violate A2 deliberately, reads
`negeig` mean `0.0845` (§1.2). **All three models violate A2 more strongly than the control built to
violate it.**

### 4.2 Where the violation sits

Eigenvector of `λ_min(sym QᵀJQ)` mapped back through `Q`; participation ratio
`(Σw²)²/Σw⁴` on squared loadings, out of 98. *Script:* `recompute_orphans.py` → `exp_negeigvec.json`.

| model, seed | `λ_min` | top-5 mass | participation |
|---|---|---|---|
| TabPFN 42 | `−3.30369` | `55.2%` | `12.3/98` |
| TabPFN 100 | `−0.65786` | `35.3%` | `23.9/98` |
| TabPFN 200 | `−2.76867` | `43.2%` | `17.8/98` |
| TabPFN 300 | `−9.34969` | `28.9%` | `32.8/98` |
| TabPFN 400 | `−0.84400` | `49.1%` | `14.8/98` |
| TabSwift 42–400 | `−0.437 … −1.437` | `32.2–74.6%` | `3.0 – 29.0/98` |

**Ambient negative diagonals**, `#(J_ii < 0)` out of 100 (`exp1_a3_final.json`, `clarif_item4_6.json`):
TabPFN `[4, 4, 2, 14, 0]`; TabSwift `[7, 15, 3, 1, 3]`; **TabICL `0` on every seed**.

In interpretable form: *raising the context labels along this direction lowered the model's estimate
along the same direction.* No posterior mean can do that.

### 4.3 `negfrac` is retracted and is not reported

It counts the **context's** rank deficiency, not the model's. The audit context duplicates 10 rows of
`X` exactly, so the exact GP's reduced spectrum has nine machine-zero eigenvalues then a hard gap to
`2.2e-2`; any noise tips them negative. The quantised control and TabICL v2 **both** read
`9/98 = 0.0918367`, on every seed, at every amplitude, in every dither configuration — it does not
respond to the probe at all, while `negeig` falls by `106×` between `t=1e-2` and `t=1e-1` over the
same matrices. `core/metrics.py::negfrac` now warns on call.

---

## 5. A3 — Cross-channel law

`s²ᵢ = a + b(1 + Jᵢᵢ)`, ambient diagonal, OLS over the 100 context points. The law requires `a = 0`
and `b = σ²`. **Threshold** (`experiments.md` E2.5): `R² ≥ 0.90`, intercept within 10% of `σ̂²`.

> ### ⚠ 5.0 A3 has an artifact floor, and it is not small
>
> **Established 2026-08-26 by E5.1 (§7).** A hierarchical GP — a map that satisfies A3 *exactly*, by
> construction — was pushed through this exact procedure on 60 real contexts. Unwrapped it returns
> `R² = 1.000000` on all 60. **Wrapped in the same normaliser the audited models wear, its `R²` falls
> as low as `0.675` and it fails the `R² ≥ 0.90` gate on 3 of 60.** The exact GP does not show this,
> because its inner map is linear; the effect needs a **nonlinear** inner map, which all three
> audited models are.
>
> **Therefore an `R²` below ~`0.68` cannot be attributed to the model alone.** The measured values
> are an order of magnitude below that floor, so the verdict stands — but the floor must be quoted
> beside them, and the bare claim "the models fail A3" is not licensed without it. The derivation
> explaining this is **N7, pending** — see §9.1.

### 5.1 The variance channel, per model

- **TabICL v2** — `output_type="quantiles"` on 9999 uniform levels in `[1e-4, 1-1e-4]`, integrated by
  `core/metrics.py::extract_variance`. **Not** `output_type="variance"`, which is
  `raw_quantiles.var(dim=-1)` (`tabicl/_model/tabicl.py:582`), a spread over the quantile grid: on
  seed 42 that reads `0.2507` where the integrated second moment reads `0.5282`, a factor of `2.107`.
- **TabPFN v2** — `FullSupportBarDistribution.variance(logits)`
  (`tabpfn/architectures/base/bar_distribution.py:606`), the integrated second moment with
  half-normal tails on the outer bins, 5000 bins.
- **TabSwift** — **NOT COMPUTABLE.** `Linear(384,1)` point head,
  `models/registry.py:134 has_predictive_distribution=False`. **A3 is unevaluable for TabSwift and
  must be reported as such, never as a failure.**

**Extractor validated** against the exact GP's closed-form predictive variance: worst-case relative
error `0.0031` on the grid used, gate `< 5%`.
*Script:* `exp1_extractor_validation.py` → `exp1_extractor_validation.json`.

### 5.2 Measured

*File:* `exp1_a3_final.json`. **No producing script existed**; recomputed 2026-08-26 by
`recompute_orphans.py` from `exp12_ambient_jacobians.npz` — **all 105 values bit-identical**.

| model | slope mean ± sd | `R²` mean ± sd | intercept range | verdict |
|---|---|---|---|---|
| **TabICL v2** `h=1e-3` | `−0.0994 ± 0.1561` | **`0.0199 ± 0.0146`** | `[0.596, 1.735]` | **FAIL** |
| **TabPFN v2** `h=1e-1` | `+0.2684 ± 0.2547` | **`0.1065 ± 0.1310`** | `[1.050, 1.808]` | **FAIL** |
| exact GP (control) | `0.250000000` | **`1.000000000000`** | `−2e-14 … +3e-14` | pass |
| hierarchical GP (control) | `0.250000001` | **`1.000000000000`** | ~0 | pass |
| wrapped exact GP (control) | — | `0.99999115` | — | pass |
| **wrapped hierarchical GP (control, §7)** | — | **min `0.675`** | — | **the floor** |

TabICL's slope is **negative on three of five seeds**, which the law forbids outright — `b = σ² > 0`.
Fitted intercepts sit near each seed's `mean(s²)`, the signature of a regressor carrying no
information about the response.

**Leverage was adequate**, so this is not a design failure: `cv_J` is `0.225–0.330` (TabICL) and
`0.465–1.291` (TabPFN) against a gate of `> 0.05`; `range(J_ii)` is `[0.0025, 1.118]` and
`[−0.638, 0.796]`.

### 5.3 No-intercept fit, the form the law actually specifies

*File:* `clarif_item5.json`, recomputed bit-identical by `recompute_orphans.py`.

| | `b` | `R²` uncentred | `R²` centred |
|---|---|---|---|
| exact GP | `0.250000000` | `1.000000000` | `+1.000000` |
| hierarchical GP | `0.250000000` | `1.000000000` | `+1.000000` |
| TabICL v2 | `0.723787` | `0.94371` | **`−0.418772`** |
| TabPFN v2 | `1.528936` | `0.98969` | **`−4.955435`** |

The uncentred `R²` of `0.94`/`0.99` is **not** a fit-quality statement — `s²` is strictly positive and
far from zero, so any `b` of roughly the right scale explains most of `Σy²`. The centred figure is
the informative one and is negative for both.

### 5.4 `cv_s ≤ cv_J` is reported as a secondary diagnostic only, never as a pass

It **holds** for both models — TabICL `0.197 ≤ 0.289`, TabPFN `0.051 ≤ 0.800`. It is nonetheless
**permutation-invariant**: permuting `s²` across context points leaves `cv_s` bit-identical
(`0.0894008` in both rows) while `R²` falls from `1.000000000000` to `0.010468752`.
*Script:* `chunk2_a3_controls.py` → `chunk2_results.json`.
**N4 is proven and true; it has no instrument power against the failure that actually occurred.**

---

## 6. Mechanism — which row of the N5 catalogue, if any

Ambient coordinates. Reduced-basis profiling was abandoned: Nadaraya–Watson is row-scaled symmetric
by construction and profiles to `9.503e-16` ambient, but in the reduced basis its asymmetry *rises*
from `0.249240` to `0.259940`. *Script:* `exp2_ambient_profiling.py` → `exp2_ambient_profiling.json`.

**Controls, and the gates they had to clear:**

| control | requirement | measured |
|---|---|---|
| Nadaraya–Watson, row-profiled | `< 1e-12` | **`9.503e-16`** (residual `1.549e-15`) |
| wrapped ExactGP inflation, profiled/raw | `< 2` | **`0.9784`** |
| targeted imitator (must survive) | not of diagonal-scaling form | `raw 0.597 → col 0.594 / row 0.594`, residual `0.956` |

**Models, 5 seeds:**

| model | raw `asym` | **column** residual | **row** residual | col `asym` | row `asym` |
|---|---|---|---|---|---|
| TabICL v2 | `0.7727 ± 0.127` | `0.5699 ± 0.042` | identical | `0.5002` | `0.5978` |
| TabPFN v2 | `1.2643 ± 0.106` | `0.6161 ± 0.139` | identical | `1.1399` | `1.2237` |
| TabSwift | `1.2457 ± 0.117` | `0.4761 ± 0.056` | identical | `0.9109` | `1.1160` |

**Neither system fits.** Residuals of `0.476–0.616` against `1.5e-15` on the NW positive control.
Row and column residuals are identical by construction (`b_row = −b_col`), so only post-profile
`asym` separates them; column scaling gives the lower value for all three.

**Recovered profiles span implausible ranges**: `87×`, `68×` and `169×` for the three models against
`2.4×` and `1.7×` for the two controls.

**What this excludes, and what it does not.** The column system is the form
`J = Cov(f|y)Σ⁻¹` takes under heteroscedastic noise (N6); the row system is the form `J = D⁻¹K` takes
for a kernel smoother or attention-weighted label vote (N5). Neither admits a positive solution with
small residual. **This is an exclusion of the two *diagonal-scaling* hypotheses at the measured
residual, not a calibrated exclusion of the published mechanisms** — that would require E2.6, which
**has never been run correctly**. See §9.1.

---

## 7. E5.1 — The positive control on real data *(the blocking gate)*

`experiments.md`: *"Nothing is reported before a real Bayesian model passes on real data."*
*Script:* `tier5_positive/e5_1_positive_control.py` → `e5_1_results.json`. CPU, 55 s.

**Setup.** Exact GP and hierarchical GP, wrapped in the same normaliser, through the **identical**
pipeline — same `reduced_jacobian`, same `get_Q(y, seed=0)`, same `asym`/`negeig`, same A3 OLS.
Data: the Phase 2 OpenML selection from the tracked `phase_2/arrays/chunk2_data.npz`, **12 datasets ×
5 splits = 60 contexts per control**, `n_ctx=100`, features standardised on context rows.

**Probe step: relative**, `h = 6.860e-4 · std(y_ctx)`. These analytic controls have no output
quantisation so an absolute step would not have broken them; relative is used because the claim is
"identical pipeline" and an easier probe would void the comparison. A four-decade sweep moves `asym`
from `1.15e-9` to `1.18e-13` while `negeig` stays exactly `0` and A3 `R²` stays `0.999982`.

**Lengthscale: the median heuristic**, to avoid a *vacuous* pass — a fixed lengthscale `1.0` kernel on
standardised features at `d=25` or `50` is near-diagonal and lands on N5's `J=I` row. Swept `×0.25–×4`;
nothing moves.

### 7.1 Result

| | A1 `asym` max | A2 `negeig` max | non-degenerate | A1 | A2 | A3 |
|---|---|---|---|---|---|---|
| exact GP, wrapped | `1.702e-11` | **`0.000e+00`** | `60/60` | **`60/60`** | **`60/60`** | **`60/60`** |
| hierarchical GP, wrapped | `2.643e-08` | **`0.000e+00`** | `53/60` | **`60/60`** | **`60/60`** | `57/60` |
| hierarchical GP, **unwrapped** | `2.642e-08` | `0.000e+00` | `53/60` | `60/60` | `60/60` | **`60/60`** |

Gates: A1 `≤ 0.05`, A2 `≤ 0.02`, A3 `R² ≥ 0.90` and slope within `±25%`, non-degeneracy
`‖J−I‖_F/‖J‖_F > 0.3`.

**A1 and A2 pass 120/120 across both wrapped controls.** `negeig` is **exactly zero** on all 120, not
merely small. Exact-GP A3 `R²` min `0.999974`, mean `0.999990`, slope within `7.8e-3` of `σ²s_y²`.

**Tangential curvature separates the two controls by `8.1e6`×** — hierarchical `3.122e-3` against
exact `3.848e-10`, the T3/T4 prediction. *Caveat:* this is the second difference along `Q`'s columns,
which preserves `ȳ` exactly and `‖y−ȳ‖` to second order. It is **not** the full M0 great-circle
construction with the geodesic correction.

### 7.2 The one failure is the normaliser, not the pipeline

| condition | A1 | A2 | A3 | A3 `R²` min |
|---|---|---|---|---|
| wrapped, shipped lengthscale ladder | `60/60` | `60/60` | `57/60` | `0.675` |
| wrapped, ladder rescaled to the data | `60/60` | `60/60` | `55/60` | `0.843` |
| **unwrapped** | `60/60` | `60/60` | **`60/60`** | **`1.000000`** |

Rescaling the lengthscale does **not** fix it; removing the wrapper fixes it **completely**.

### 7.3 Verdict

**The pipeline does not manufacture A1 or A2 violations on real data.** Two real Bayesian predictors,
120 real contexts, identical pipeline, `negeig` exactly zero throughout and `asym` at `1e-11`–`1e-8`
against models measuring `0.58`–`1.31`. **The blocking gate is cleared.**
**A3 is a weaker instrument than previously implied** — §5.0 carries the floor.

---

## 8. E4.1 — The value channel *(is the audit confounded?)*

The objection: if the models simply perform badly on the audit contexts, the violations are entangled
with plain out-of-distribution failure.
*Script:* `tier4_value/e4_1_value_battery.py` → `e4_1_results.json`. 3 models × 5 seeds on
the **exact** audit contexts, plus five reference predictors. Held-out queries drawn from the GP
conditional `f_*|f` at 200 fresh `X_* ~ N(0,I₅)` per seed, so `(X, y, f)` stay bit-identical.

> **Read this before judging any MSE.** Unit signal variance and `sigma = 1.0` means SNR is **1 by
> design**, and at `d=5` with `lengthscale=1.0` the kernel is nearly diagonal — median off-diagonal
> `K_ij` is `0.0158` (seed 42), only `17–20%` of pairs above `0.1`. **The Bayes-optimal oracle itself
> reaches only `R² = 0.1425` held out.** Raw MSE is uninterpretable, so everything is reported as
> **efficiency**: `(MSE_constant − MSE_model)/(MSE_constant − MSE_oracle)`, so `0` = predicting the
> context mean and `1` = the Bayes-optimal oracle.

| | `MSE(m(y), f)` in-sample | /oracle | **eff_ctx** | eff_test | NLL | cov@90 |
|---|---|---|---|---|---|---|
| oracle GP (true model, `σ=1.0`) | `0.3288` | `1.00` | `1.000` | `1.000` | `1.6764` | `0.894` |
| hierarchical GP | `0.3514` | `1.07` | `0.968` | `0.805` | `1.6963` | `0.906` |
| audit control GP (`σ=0.5`) | `0.4293` | `1.32` | `0.858` | `0.741` | `1.9203` | **`0.722`** |
| **TabICL v2** | `0.4936` | `1.51` | **`0.768`** | `0.431` | `1.7554` | `0.896` |
| **TabSwift** | `0.5876` | `1.82` | **`0.636`** | `0.367` | N/A | N/A |
| **TabPFN v2** | `0.6703` | `2.06` | **`0.519`** | `0.145` | `1.7584` | `0.886` |
| constant `mean(y)` | `1.0387` | `3.37` | `0.000` | `0.000` | `1.7666` | `0.914` |
| 1-NN | `0.9567` | `2.99` | `0.116` | `−4.314` | N/A | N/A |

Held-out `R²`: oracle `0.1425`, TabICL `0.0402`, TabSwift `0.0486`, **TabPFN `−0.0042`**, constant `−0.0019`.
TabSwift's NLL and coverage are **N/A, not zero and not synthesised**.

**Verdict.** All three denoise substantially and beat every trivial baseline, so **the audit is not
confounded with out-of-distribution failure**. But plainly: **they are `1.51×`, `1.82×` and `2.06×`
the oracle's in-sample error**, and TabPFN's held-out `R²` is at the constant baseline.

**The calibration result is the one that matters for the paper.** Coverage at 90% is `0.896`
(TabICL) and `0.886` (TabPFN) against the oracle's `0.894`; NLL is `1.755`/`1.758` against `1.676`.
The misspecified `σ=0.5` control GP is the **worst-calibrated map in the table** at `0.722`, which
confirms the coverage statistic has power here. So on these contexts the two models with a predictive
distribution are **well calibrated and near-oracle in NLL while failing A1 and A2 by 57×–22 711× over
the artifact floor**. That is **C-X2 instantiated empirically** rather than by construction.

---

## 9. What is NOT established

Everything above is measurement. This section is the boundary.

### 9.1 Gaps in the audit itself

| gap | status |
|---|---|
| **N7 — the A3 wrapper floor derivation** | **PENDING.** §5.0 establishes the effect *empirically* (a map satisfying A3 exactly reads `R²` as low as `0.675` through the wrapper) but there is no derivation explaining it. Until N7 exists, §5's exclusion rests on an empirical floor, not a theorem. |
| **E2.6 — calibrated mechanism exclusion** | **NEVER RUN CORRECTLY.** The only artefact, `e2_6_mechanism.py`, is quarantined: its verdict is a **hardcoded 50% threshold** — precisely the invented tolerance E2.6 exists to avoid — and it is single-seed, single-model, and compares **ambient** Jacobians with no `Q` projection. §6 excludes the two diagonal-scaling hypotheses; it does **not** deliver the calibrated exclusion of published mechanisms that N5(3) requires. Any draft sentence resting on "TabICL's Jacobian differs by over 50% from all fitted surrogates" has **no valid support.** |
| **One context family, throughout** | Every Phase 1 number except §7 is `n=100, d=5, sigma=1.0`, GP-drawn, 10 duplicated `X` rows. **Tier 3 (E3.1–E3.5) has never been run.** `experiments.md` calls the off-distribution objection **fatal if unaddressed**. Phase 2 §9 found TabICL's `tr J` reaching `100` on real data against `‖J−I‖_F/‖J‖_F = 0.85` on the audit's contexts — a context-family effect of exactly the kind E3.1 exists to find. |
| **Curvature and M0 on models** | **NEVER MEASURED.** Measured at a single amplitude only, where the artifact scales as `t⁻²` (ExactGP floor `3.98e-10 → 3.97e-16` across `t = 1e-3…1`) while genuine curvature is flat in `t` (hierarchical GP `2.868e-03 → 2.764e-03`), so one amplitude cannot separate them. Separately the quantity computed is **not** the M0 tangential great-circle curvature with the geodesic correction. |
| **A4 / the decay law** | Out of scope wherever A1 or A2 fail. A decay exponent on a map excluded from the hierarchical-posterior class has no interpretation attached. Stated, not measured. |
| **E1.4 for TabPFN v2** | NOT MEASURED. Needs source modification of the attention path. |
| **`e14_paired.json` is not recomputable** | §3.3's paired RoPE analysis is the **one** orphaned result that cannot be regenerated from the tracked arrays — it needs fresh forward passes through a **modified** TabICL, and no stored Jacobian can stand in. ~2000 GPU fits, roughly an hour. All other orphaned results were recovered: `recompute_orphans.py`, **292 of 293 checks reproduce, none differs** (262 bit-identical, 25 to `≤1e-12`, 5 pairs both at machine precision). |
| **E0.1 determinism on models** | NOT MEASURED. Needs a GPU refit-twice pass on the audit path. |
| **E0.2 N5 analytic battery** | Script exists (`e0_2_analytic_control.py`), no saved output, covers 6 of 9 rows. |
| **E0.4 autograd cross-check** | No code. |
| **E4.2 — imitator on the value battery** | Not run. E4.1 is done (§8); E4.2 is not. |
| **E5.2** | **Not run as a registered test.** See §9.2 — this is the one most easily overclaimed. |
| **E5.3** — adaptive-bandwidth Nadaraya–Watson | Not run. A *fixed*-bandwidth NW is used as the profiling positive control (§6), but the adaptive-bandwidth rebuttal — bandwidth linearly decodable and causally patchable, yet averaging over nothing, so it passes A2, fails A1 in the row-scaled way, and has zero curvature — was never constructed. It is the direct answer to "probing found the kernel, so the model is Bayesian". |
| **E5.4** — McCarter reproduction | Not run. No reproduction of the duplication anomaly on the current checkpoints, and no test of whether the leading negative eigenvector aligns with the direction it identifies. Cheap, and would convert a blog-post curiosity into a prior sighting of the violation. |
| **E3.2–E3.5** | Not run. In-sample vs query regime; context-design sweep over `d`, noise, duplicate count and `n`; query-row vs context-block consistency; and **checkpoint variation** — five *seeds* were run everywhere but only one checkpoint per model, so the "one checkpoint" objection is not closed. |

### 9.2 Scope limits that must appear in the paper

**TabSwift fails shift and scale equivariance (§1.5), and this propagates.** Its target scaler is
commented out in source. N1(ii)'s exactness guarantee for `QᵀJQ` is **not established** for it, so its
A1 value is the asymmetry of the compressed Jacobian, not of a recovered inner map. **A2 is
unaffected** — a compression of a PSD matrix is PSD. **A3 is unevaluable**, not failed. Any sentence
treating TabSwift's three verdicts as equivalent to the other two models' is wrong.

**E5.2 was not run as a registered test.** The prediction was stated in advance and the falsifier
fired as stated — TabPFN v2 is **worse** than TabICL v2 on both A1 (`1.312` vs `0.580`) and A2
(`0.404` vs `0.177`), where the pre-registration predicted it would be *materially closer to
passing*. But those numbers came from the general audit, not from a registered E5.2 run, and
`FINAL_NUMBERS.md` §8 lists "E5.2 as a registered test" as not run. **The honest claim is that the
audit's numbers are inconsistent with the Biloš readout prediction, not that a pre-registered test
falsified it.** The distinction is what separates a test from a fishing expedition.

**Nine cited documents do not exist** — `final_audit_report.md`, `RESULTS_REPORT.md`,
`CONTROL_VALIDATION.md`, `CLARIFICATIONS.md`, `E11_WRAPPER_FORM.md`, `e2_5_report.md`,
`tier2_report.md`, `e1_4_report.md`, `models/ARCHITECTURE_REPORT.md`. Not in the tree, not in
`intermediate/`, not in any commit. No number is lost — `FINAL_NUMBERS.md` §9.1 traces every value to
a file that exists and is tracked. What is lost is the supersession narrative: several §1.3 retraction
claims can no longer be checked against their sources.

**Two other things it would be wrong to claim:**
- **The `1` and `u` directions are permanently unidentifiable** (N2). `M` has rank `n−2`, so `G`
  restricted to `span{1,u}` is not recoverable from `J` by any measurement. This is an identifiability
  limit, not a shortcoming of the projection.
- **`negfrac`, on any model** (§4.3), and **`cv_s ≤ cv_J` as a pass** (§5.4).

### 9.3 The three gaps that most constrain what the paper may claim

Ranked, because they are not equivalent:

1. **E2.6 — the calibrated mechanism exclusion.** Never run correctly. The pre-registration calls its
   scatter of value agreement against derivative agreement **"the headline figure"**. It does not
   exist. §6 excludes two *diagonal-scaling* hypotheses; it does not exclude the field's published
   mechanisms, which requires fitting each as a surrogate to the model's own outputs so that the
   surrogate's own value-fit supplies the tolerance.
2. **E3.1 — the context-distribution objection.** Never run. The pre-registration calls it **fatal if
   unaddressed**. Every Phase 1 number except §7 is one context family, and Phase 2 found TabICL's
   `tr J → 100` on real data against `‖J−I‖_F/‖J‖_F = 0.85` here — a context-family effect of exactly
   the kind E3.1 exists to find.
3. **E1.5 / M0 — the radial vs tangential gate.** Never evaluated. Its pre-registered rule was that
   *if curvature is mostly radial, the hedging interpretation dies and the finding becomes "apparent
   hedging is preprocessing."* Only an Euler-defect ratio exists, from a script with an unseeded `Q`.
   The great-circle construction with the geodesic correction was never built.

### 9.4 What is quarantined

`../_archive/quarantine/`, with `__init__.py` raising on import. Manifest at
`../_archive/quarantine/QUARANTINE.md`.

| item | defect |
|---|---|
| `tier2_audit/` entire | every `asym` **half** canonical; `negfrac` returned under the name `negeig`; 13 unseeded `get_Q`; profile residual always `0.00e+00` |
| `e2_6_mechanism.py` | hardcoded 50% verdict threshold (see §9.1) |
| `e1_4_positional.py` | standardised targets, unseeded `Q`, marginal instead of paired; superseded by §3.3 |
| `root_scratch/` | tier2 outputs written to the repo root, carrying halved `asym` and retracted `negfrac` |
