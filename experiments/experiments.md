# Experiment Battery

**PFN Interpretability Project · August 2026**
Companion to `01_theory_and_claims.md`. Claim IDs (C-T*, C-I*, C-E*, C-X*) refer to that document.

Every entry has four fields. The third is the one that is usually skipped and is the reason this
document exists.

1. **What to run.**
2. **Pass / fail thresholds**, fixed before the run.
3. **Falsifier** — what result would kill the claim.
4. **Objection closed** — which reviewer question this answers. *An experiment that closes no
   objection and supports no claim does not belong in the battery.*

**Models under audit:** TabPFN v2, TabICL v2, TabSwift — frozen regressors.
**Standing rules:** every model number reported beside its control number; mean ± spread over ≥5
contexts, never a bare number; anything landing in an inconclusive band is reported as inconclusive
with no threshold adjusted; all numbers logged to the results ledger with model, checkpoint, config,
n, h, dtype, context family, seed, commit.

---

## Tier 0 — Instrument validity

*Nothing downstream is interpretable until these pass. Run in order; halt on failure.*

### E0.1 Determinism
**Run.** Identical context twice through each model; confirm bit-identical output. Identify and pin
every nondeterminism source (ensembling over configurations, decoder sampling, dropout,
nondeterministic kernels).
**Pass.** Max abs difference `< 1e-9`.
**Falsifier.** Persistent jitter above `1e-6` — finite differencing at small `h` differences numbers
agreeing to many decimals, and jitter destroys the measurement.
**Closes.** "Your Jacobian is noise."

### E0.2 Analytic control battery — the N5 catalogue
**Run.** Implement each row of the N5 table behind the model interface and push it through the
**identical** finite-difference pipeline: exact GP, ridge, k-step GD (`X p(XᵀX) Xᵀ`), Nadaraya–Watson
with fixed and with data-adaptive bandwidth, 1-NN, hierarchical GP, imitator. Compare measured `J`
against the closed form entry-wise.
**Pass.** Relative error vs closed form `< 1e-6` (float64). Verdicts match the N5 table: GD/ridge/GP
symmetric PSD zero-curvature; NW asymmetric but with positive real spectrum and recoverable *row*
scaling; hierarchical GP nonzero curvature with slope in `[−1.3, −0.7]`; imitator flagged non-PSD.
**Falsifier.** Any row whose measured verdict contradicts its closed form → the pipeline is wrong,
or the algebra in N5 is wrong. Either way, halt.
**Closes.** "How do we know your instrument reports the right answer on maps whose answer is known?"
**Note.** This doubles as numerical verification of N5. It is the single highest-value Tier 0 item.

### E0.3 Step-size plateau
**Run.** `h ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}` on exact GP and on each frozen model. Report dtype.
**Pass.** `asym` and `negeig` vary `< 20%` across ≥3 consecutive `h`. Take the spread as
`A_floor_model`.
**Falsifier.** Systematic `h`-dependence (e.g. growth as `h → 0`) → the violation is discretisation
or decoder noise, and A1/A2 must be re-derived at the stable `h`.
**Closes.** "Your violation is a finite-difference artifact." *(C-I5)*

### E0.4 Autograd cross-check
**Run.** If the frozen forward pass supports JVPs, compute one Jacobian column by autograd and
compare against the finite-difference column.
**Pass.** Relative agreement `< 1e-3`.
**Falsifier.** Disagreement → finite differencing is not measuring the derivative.
**Closes.** "Finite differences ≠ derivative." *(C-I1)*

### E0.5 Smoothness scan in the measured region
**Run.** Fine grid scan of `m(y + t v)` in `t` across the perturbation range; look for kinks from the
bar-distribution binning, whose borders move affinely with context target statistics.
**Pass.** No kink within the measurement region, or kinks demonstrably smoothed by ensembling.
**Falsifier.** Kinks inside the region → restrict the region or exclude, explicitly.
**Closes.** "The bar-distribution head is piecewise-smooth; your derivatives straddle a bin border."

### E0.6 Row-sum residual as instrument-error bound
**Run.** `r₁ = ‖J·1 − 1‖/‖1‖` per context. Parameter-free; must hold exactly for any
shift-equivariant map, so any deviation is pure instrument error on the real model.
**Report.** `r₁` mean ± spread, and the ratio of measured violations to `r₁`.
**Falsifier.** `r₁ ≳ 1e-1` → the Jacobian is corrupted; halt.
**Closes.** "What is your actual noise floor on a float32 transformer?" — replaces the fallback
`0.05` constant with a measured bound. *(C-I4)*

---

## Tier 1 — Confound control

### E1.1 Preprocessing identification
**Run.** Read each model's target-preprocessing path. Report exactly what it does. Independently
confirm empirically: shift equivariance `m(y + c1) = m(y) + c1` and scale homogeneity
`m(cy) = c·m(y)`.
**Pass.** Both hold to `< 1e-3` relative → affine wrapper confirmed, projection justified.
**Falsifier.** Rank- or quantile-based target transform → **halt on A1 for that model**; no linear
projection removes an order-statistic transform, and A3 becomes the load-bearing instrument.
**Closes.** "You assumed the wrapper is affine." *(C-N1)*

### E1.2 Projection validation, both directions
**Run.** (a) Wrap the exact GP in the same normaliser; confirm raw asymmetry is O(1) and
`asym(J_red)` collapses. (b) Wrap the imitator; confirm `negeig(J_red)` survives.
**Pass.** (a) `asym(J_red) < 0.02`. (b) `negeig(J_red) ≥ 10× A_floor`.
**Falsifier.** (a) fails → projection wrong. (b) fails → **the projection launders violations**, and
no statement about any model is licensed.
**Closes.** "Does your projection remove a confound, or remove the signal?" *(C-I2, C-I3)*
**Note.** (b) is load-bearing. Without it a clean projected result means nothing.

### E1.3 Ensemble-of-one
**Run.** Locate the ensemble setting per model; report the default `K`. Recompute at `K = 1`, where
one normaliser exists and the projection is exact with no residual assumption. Compare to default-`K`.
**Pass.** Agreement within context spread → ensemble irrelevant. Divergence → `K=1` numbers are
authoritative, and the divergence localises per-member normalisation (itself reportable).
**Falsifier.** Cannot be set to 1 → carry projection validity as a stated assumption, explicitly.
**Closes.** "Per-member normalisation makes your projection leak." *(C-N1(iii))*

### E1.4 Positional-encoding zeroing
**Run.** Biloš et al. identify localised positional parameters that break permutation invariance and
can be zeroed at inference with no accuracy loss. Positional leakage is an independent generator of
asymmetry — processing row `i` differently from row `j` by position is effectively assigning them
different precisions. Run the audit with these zeroed. **Use paired per-context differences**, not
marginal spreads.
**Pass.** Report the paired difference with its CI, and confirm accuracy is unchanged.
**Falsifier.** Asymmetry drops to threshold → the violation was positional leakage.
**Closes.** "Your asymmetry is positional encoding, not the readout."
**Status.** Done for TabICL (0.5218 → 0.4359 unpaired; **paired analysis outstanding**). TabSwift has
no positional encodings. **Outstanding for TabPFN v2.**

### E1.5 M0 radial/tangential gate
**Run.** Euler defect `e(y) = yᵀ∇m_*(y) − m_*(y)`, reported as `|e|/|m_*|`. Then great-circle
tangential curvature in the centred subspace with the geodesic correction `−∇m·γ''` subtracted.
Validate the construction on the hierarchical GP, which has no normaliser.
**Pass.** Correction validated to `< 10%` against uncorrected directional curvature on the
hierarchical GP.
**Falsifier.** Curvature is mostly radial → pre-registered: the hedging interpretation dies and the
finding becomes "apparent hedging is preprocessing."
**Closes.** "Your curvature is z-normalisation." *(C-E5)*

---

## Tier 2 — The audit, per model

Run for **TabPFN v2, TabICL v2, TabSwift**. Contexts designed to force smoothing: substantial label
noise, ~10 near-duplicate `x` pairs with conflicting labels.

### E2.1 Non-degeneracy (runs before any verdict)
**Run.** `‖J − I‖_F/‖J‖_F`, `‖J‖_F` against expected `√n`, effective rank, `max|J_ij|`.
**Pass.** `‖J − I‖_F/‖J‖_F > 0.3` **and** `‖J‖_F` of order `√n`.
**Falsifier.** Either fails → the audit has no purchase on these contexts. `J ≈ I` is the 1-NN row of
N5 and passes A1/A2 vacuously; `J ≈ 0` is instrument noise. Both are reportable findings, neither is
a pass.
**Closes.** "Your pass is vacuous" / "your fail is noise." *(C-I6)*
**Status.** TabSwift outstanding and likely decisive — `‖J‖_F` collapsed 392.8 → 6.09 across the `h`
sweep, and 6.09 against `√n ≈ 10` is signal, not near-zero. The earlier `asym = 1.99` and Euler
defect 4.888 were measured in the noise regime and are void.

### E2.2 A2 — Positivity *(leads, per N3)*
**Run.** `negeig(Qᵀ J Q)`, full spectrum of `sym(QᵀJQ)`, sign of every diagonal entry.
**Pass.** `negeig ≤ max(3F, 0.02)` where `F = max(A_floor_norm, A_floor_model)`.
**Falsifier.** `negeig ≤ F` → no positivity violation; A2 passes.
**Closes.** "A normaliser could produce this" — it provably cannot. *(C-E3, C-N3)*
**Also report.** The context points on which the leading negative eigenvector concentrates, and the
violation in interpretable form: *raising labels along this direction lowered the model's estimate
along the same direction.* Compare against the imitator's `negeig` explicitly.

### E2.3 A1 — Symmetry
**Run.** `asym(Qᵀ J Q)`, with raw `asym(J)` reported alongside so the confound size is visible.
**Pass.** `asym ≤ max(3F, 0.05)`. **Fail.** `> max(10F, 0.15)`. Between → inconclusive.
**Falsifier.** Passing after E1.2 validated → the mean map is symmetry-consistent.
**Closes.** *(C-E2)*

### E2.4 Mechanism profiling — heteroscedastic vs smoother *(if A1 fails)*
**Run.** Solve both systems in log-space by least squares and report residuals:
- **Column scaling** (heteroscedastic Bayes): `Jᵢⱼσⱼ² = Jⱼᵢσᵢ²`
- **Row scaling** (Nadaraya–Watson / attention vote): `dᵢJᵢⱼ = dⱼJⱼᵢ`
**Pass.** Either admits a positive solution with residual below the A1 threshold → that mechanism is
compatible; report the recovered profile.
**Falsifier.** Neither → both classes excluded, which is the stronger result.
**Closes.** "It's heteroscedastic, so your symmetry condition is the wrong one" and "it's just a
kernel smoother." *(C-N6, C-E6)*
**Caveat to state.** In reduced coordinates `Q` mixes context points, so a recovered diagonal is a
heuristic indication rather than a fitted noise model. If promising, re-derive the condition through
the normaliser — a theory task.

### E2.5 A3 — Cross-channel law
**Run.** (a) Parameter-free: `cv_J = std(Jᵢᵢ)/|mean Jᵢᵢ|` and `cv_s = std(s²)/mean(s²)`; check
`cv_s ≤ cv_J`. (b) Regression of `s²ᵢ` on `(1 + Jᵢᵢ)`: slope, intercept, R², and `σ̂²` consistency
across ≥5 contexts and ≥2 data scales.
**Pass.** (a) `cv_s ≤ cv_J` within error. (b) `R² ≥ 0.90`, intercept within 10% of `σ̂²`, `σ̂²`
consistent to ±25%.
**Falsifier.** (a) holding → no parameter-free violation, and the regression result must be
re-examined for leverage rather than reported as failure.
**Closes.** "Your regression had no leverage" — (a) needs none. *(C-E4, C-N4)*
**Required fixes.** Variance must be the integrated second moment over the **full** predictive
distribution, not the 0.005/0.995 quantile spread — the most tail-sensitive estimator available,
applied where the bar distribution's outer bins are coarsest. Validate against the exact GP's
closed-form predictive variance, not a Gaussian mock. TabSwift has a `Linear(384,1)` head and no
predictive distribution: **A3 is unevaluable and must be reported as such**, not as a failure.
**Also report.** Two independent corroborations if the law fails: Nagler & Rügamer state PFNs mix
epistemic and aleatoric uncertainty without decomposition; Johnson et al. find TabPFN's RRMSE highly
variable at small `n` while intervals stay calibrated. And the project's own earlier finding that the
mean and variance channels occupy **separate causal subspaces** — a structural prediction that
nothing couples the two channels.

### E2.6 Mechanism surrogate fitting — the calibrated exclusion
**Why this exists.** The N5 closed forms are derivations; they are true whether or not any model
implements the mechanism, and E0.2 already validates the pipeline against them. Neither of those
licenses the inference "the measured Jacobian ≠ `D⁻¹K`, therefore the smoother hypothesis is
excluded." **Nobody claims these models are *exactly* a smoother or *exactly* a solver.** Miftachov's
surrogate matches on 55 datasets; Biloš describes an attention-weighted vote at ~L9. These are
approximation claims, and "approximately" is doing all the work. Excluding them requires a tolerance,
and inventing one would be indefensible.

**Run.** For each frozen model and each candidate mechanism in N5, fit the mechanism as a **surrogate
to the model's own outputs** on a given context — bandwidth for Nadaraya–Watson, `λ` for ridge, step
count and `η` for the GD family, kernel hyperparameters for GP, prototype construction for the
distance readout. Then report two numbers per surrogate:

    value agreement:      ‖m_model − m_surrogate‖ / ‖m_model‖
    derivative agreement: ‖J_model − J_surrogate‖_F / ‖J_model‖_F

**The surrogate's own value-fit sets the tolerance.** No threshold is invented: the claim becomes
"this surrogate reproduces the model's predictions to X% and its Jacobian to Y%", and the exclusion
is quantitative.

**Pass (mechanism compatible).** Derivative agreement of the same order as value agreement.
**Fail (mechanism excluded).** Derivative agreement an order of magnitude worse than value agreement,
consistently across contexts and models.
**Falsifier.** Some surrogate matches in both channels → that mechanism *is* the model's computation
to within its own fit quality, and the paper's verdict becomes mechanism-identification rather than
exclusion. A better outcome, and one the design must be able to return.

**Closes.** "You excluded mechanisms nobody claimed held exactly." *(C-N5, C-E6)*

**Why this is the strongest form of the thesis.** Every one of these surrogates was validated in the
value channel and published on that basis. Showing they agree there and diverge in the derivative
channel is value-blindness demonstrated on the field's actual best explanations rather than on an
adversarial sine construction. **The headline figure is this scatter: value agreement on one axis,
derivative agreement on the other, with the published mechanisms in the bottom-right corner.**

**Scope caveats to state in the paper.**
- Camp A studies constructions and trained toys, not frozen tabular models. Von Oswald never claims
  TabICL performs gradient descent. Excluding the linear-solver family is a **bridging** claim — the
  family that dominates ICL theory has a derivative signature, and the production models do not carry
  it — not a refutation of those authors. Only accounts making direct mechanistic claims about these
  checkpoints are targets.
- Biloš's vote and prototype readouts are described for **classification**; we audit regressors. The
  label-vote form `Σᵢ wᵢ(x,xᵢ)yᵢ` transfers naturally to continuous targets; the prototype-distance
  readout arguably does not. Transfer to the regression head is a stated assumption, and if TabPFN v2
  fails identically to TabICL v2, "the readout distinction does not transfer to regression" is a live
  alternative to "the mechanism story is wrong."

---

## Tier 3 — Robustness

*These are what survive review. Each closes a specific "but you only tested…" objection.*

### E3.1 Context distribution — the largest standing risk
**Run.** Three context families: **GP-drawn** (current), **SCM-generated** (matching TabPFN's actual
prior family), **real tabular** (OpenML). Full audit on each.
**Pass.** Verdict stable across all three.
**Falsifier.** Violation present off-distribution and absent in-distribution → a different and
arguably better paper: Bayes-consistent where trained, breaking down outside.
**Closes.** "You audited the model off-distribution; of course it fails." **Unaddressed, this
objection is fatal.** *(C-E7)*

### E3.2 In-sample regime
**Run.** The Jacobian asks the model to predict *at* its own context locations — the right object
(Brown's identity concerns `E[f_i | y]`) but a regime the model was not trained for. Compare `m` and
`Jᵢᵢ` at `xᵢ` against a query at `xᵢ + δ` for small `δ`.
**Pass.** Agreement as `δ → 0`.
**Falsifier.** Discontinuity → the in-sample and query regimes differ and the object measured is not
the object the model computes at test time.
**Closes.** "You measured a regime the model never sees."

### E3.3 Context-design sweep
**Run.** Vary `d`, noise level, duplicate-pair count, label distribution, context size `n`, one at a
time from the base recipe.
**Pass.** Violation stable, or its dependence characterised.
**Falsifier.** Violation appears only at one setting → it is a property of the context recipe.
**Closes.** "Your result is an artifact of one synthetic setup." *(C-E7)*

### E3.4 Query row vs context block consistency
**Run.** Curvature is measured on the query row, symmetry on the context block. Confirm they describe
the same map: compare `∂m_*/∂y` against the corresponding extension of the context-block Jacobian.
**Pass.** Consistent.
**Falsifier.** Inconsistent → the two measurements are not about the same object.
**Closes.** "Your two measurements are of different things."

### E3.5 Seed and checkpoint variation
**Run.** Where multiple checkpoints or seeds exist, repeat the headline measurement.
**Closes.** "One checkpoint."

---

## Tier 4 — The value channel *(Pareek's marginal note; currently the largest gap)*

### E4.1 Value-level battery on the same model and the same contexts
**Run.** On the identical contexts used for the audit: test MSE, NLL, calibration curves, coverage,
sequential-consistency / martingale statistics, and prequential log loss against a Bayesian
reference.
**Pass.** The model performs at or near its published level on all of them.
**Falsifier.** The model performs badly on value tests too → the contexts are adversarial and the
audit result is confounded with plain out-of-distribution failure. **This would be a serious
problem, which is precisely why it must be run.**
**Closes.** Everything. *(C-E1, C-X2)*
**Why this is the headline.** T1 is currently a proof about an adversarial construction. This makes
it empirical: *a model that passes every published test and fails the parameter-free structural
conditions, on the same data.* This figure goes **before** the audit results in the paper, not after.

### E4.2 Imitator on the value battery
**Run.** Push the wrapped imitator through the same battery.
**Pass.** It passes the value tests to `O(ε)` while failing the audit.
**Closes.** Instantiates T1 end-to-end with a known-impossible map. *(C-X2)*

---

## Tier 5 — Positive control *(protect at all costs)*

### E5.1 A predictor that passes, on real data
**Run.** Exact GP and hierarchical GP on real tabular data (OpenML), through the **identical**
pipeline including projection. Must pass A1, A2, A3; hierarchical GP additionally shows nonzero
tangential curvature.
**Pass.** All conditions pass, with the same thresholds applied to the frozen models.
**Falsifier.** Fails on real data → the pipeline, not the models, is producing the violations. Nothing
else in the paper survives.
**Closes.** "Everything fails, so your test is too strict." **Without this the paper does not go
out.** *(C-X1)*

### E5.2 TabPFN v2 against the Biloš prediction — pre-register before running
**Prediction, stated in advance.** Biloš et al. find TabPFNv2 and Mitra form an attention-weighted
label vote at ~L9, which is the `D⁻¹K` row of N5: row-scaled symmetric, so A1 fails in raw form but
E2.4's **row** system should recover a clean profile, and A2 should pass. TabICLv2 instead uses a
nearest-class-prototype distance readout at L11, quadratic in the labels, with no reason to be
symmetric or PSD.
**Pass.** TabPFN v2 materially closer to passing than TabICL v2, with row-scaling recovered.
**Falsifier.** TabPFN v2 fails identically to TabICL v2 → the readout distinction does not govern
derivative structure, and the mechanism story in N5 needs revision.
**Closes.** "You have one failing model and no discrimination." Either outcome is a result; the
pre-registration is what makes it a test rather than a fishing expedition. **Highest-value single
experiment on the board.** *(C-E6)*

### E5.3 Adaptive-bandwidth Nadaraya–Watson — the probing rebuttal
**Run.** A smoother whose bandwidth is data-adaptive: bandwidth linearly decodable and causally
patchable, yet averaging over nothing.
**Pass.** Passes A2, fails A1 in the row-scaled way, zero curvature.
**Closes.** "Probing found the kernel, so the model is Bayesian" — presence ≠ integration, now
demonstrated empirically rather than argued. Directly relevant to reconciling the project's own
spectral paper.

### E5.4 McCarter reproduction
**Run.** Reproduce the reported duplication anomaly — duplicating samples of one class shifting the
boundary the wrong way — on the current checkpoints, and check whether the leading negative
eigenvector of `QᵀJQ` aligns with the direction it identifies.
**Pass.** Alignment.
**Closes.** Converts a blog-post curiosity into a prior sighting of the violation. *(C-X4)*
**Note.** Cheap, high narrative value: "this has been observed before and dismissed as an artifact;
our framework identifies it as a violation of a condition every posterior mean satisfies."

---

## Sequencing

**Blocking, in order:** Tier 0 → E1.1–E1.3 → E5.1. Nothing is reported before a real Bayesian model
passes on real data.

**Then, in parallel:** E5.2 (TabPFN v2, pre-registered) · Tier 2 for all three models · E4.1.

**Then:** Tier 3 robustness · E2.4 mechanism profiling · **E2.6 surrogate fitting** · E5.3, E5.4.

**Outstanding from prior runs:** E1.4 paired analysis for TabICL · E2.1 for TabSwift · all numbers
recomputed under the canonical `asym` definition · E2.5 variance estimator fixed.

**Out of scope, stated not measured:** A4 / the decay law wherever A1 or A2 fail — the hedging
reading of curvature presupposes the map is a hierarchical posterior mean, which the audit excludes,
so a decay exponent would be a number with no interpretation attached.