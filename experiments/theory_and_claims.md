# Theory and Claims

**PFN Interpretability Project · working document · revision 2 · 2026-08-26**

> **This document is LIVE and is amended.** Unlike `experiments.md`, which is a frozen
> pre-registration, this one is updated as results land — but **as recorded amendments, never as
> silent edits**. Section 5 is the amendment log; every change since revision 1 is listed there with
> its cause. Revision 1 was August 2026.

This document fixes canonical numbering, states every result the paper will rely on, and lists every
claim the paper will assert. The two existing notes (the dense theorem–proof note and the study
guide) number results differently; **this numbering supersedes both.** Nothing here is an
experiment; the companion document `02_experiment_battery.md` attaches experiments to claims.

---

## 0. Setup and standing assumptions

Context inputs `X = (x₁, …, xₙ)` are held fixed throughout; `y ∈ Rⁿ` is the context label vector;
`x_*` is a query. A frozen model supplies a mean map `m(y) ∈ Rⁿ` (its predictions *at* the context
locations; query predictions extend the map by extra rows) and a predictive variance `s²(y)`. All
derivatives are with respect to `y`.

**Assumption A (Gaussian channel).** `y = f + ε`, `f ~ P` for an arbitrary prior `P` on `Rⁿ`,
`ε ~ N(0, σ²I)` independent of `f`, `σ² > 0`.

**Assumption B (Integrability).** `E_P‖f‖ < ∞`; where higher derivatives are taken the corresponding
posterior moments exist and dominated differentiation under the integral is valid. Automatic for
sub-Gaussian priors, hence for every Gaussian-mixture and hierarchical-GP prior.

Under A–B the marginal `p(y) = ∫ φ_σ(y − f) dP(f)` is strictly positive and C^∞, and the exact
posterior mean is `μ*(y) = E[f | y]`.

**Notation fixed once.** `J(y) = ∂m/∂y`. Asymmetry is reported as

    asym(J) = ‖J − Jᵀ‖_F / ‖J‖_F

and positivity violation as

    negeig(J) = −min(0, λ_min(sym J)) / ‖J‖₂,    sym J = (J + Jᵀ)/2.

**These are the definitions. Every number in the paper is computed through them.** Three different
denominators have appeared in working notes; all prior numbers are to be recomputed. The choice of
`‖J‖_F` over `‖sym J‖_F` in the denominator is deliberate: the latter diverges when the symmetric
part vanishes, which is precisely the regime where a near-zero Jacobian produces an uninterpretable
ratio.

---

## 1. Inherited results

These are established in the existing notes. They need proof-checking, not rediscovery.

### T1 — Value-blindness

For any prior's posterior mean `μ*` and any `ε > 0` there exists `g_ε ∈ C^∞` with
`‖g_ε − μ*‖_∞ ≤ ε` that is not Bayes-realisable, whose Jacobian antisymmetry (for `n ≥ 2`) and
whose PSD failure (for `n ≥ 1`) are bounded below **uniformly in ε**. Consequently no value
functional — any statistic depending on the map only through its values, continuously in sup-norm —
nor any finite family nor any continuous aggregation of such, can separate Bayes-realisable maps
from their complement with any margin.

Test risk, NLL with a fixed variance head, calibration and coverage statistics,
martingale/sequential-consistency statistics, prequential log loss and cumulative predictive KL, and
probe accuracies computed from predictions are all value functionals.

*Construction:* `g_ε(y) = μ*(y) + ε sin(⟨a,y⟩/ε) b`.

### T2 — Master identity (Brown / Tweedie)

Under A–B, for every prior,

    σ² J_{μ*}(y) = Cov(f | y)

Hence `J_{μ*}` is **symmetric** and **positive semi-definite** everywhere, and its diagonal satisfies
the **cross-channel law**

    s²(xᵢ) = σ²(1 + [J_{μ*}]ᵢᵢ)

for the model's own predictive variance at a fresh observation at `xᵢ`.

*Slicker form, worth carrying:* Tweedie gives `μ*(y) = y + σ²∇log p(y)`, so `μ*(y) − y` is a
**gradient field** and its Jacobian is a Hessian. Symmetry is the statement that the map is
conservative — derived from the single scalar potential `log p`, the log marginal likelihood. A map
not obtained by integrating against a prior has no such potential.

### T3 — Curvature is the third posterior cumulant

    ∂²μ*ᵢ / ∂y_j ∂y_k = σ⁻⁴ κ₃(fᵢ, f_j, f_k | y)

and inductively the k-th derivative tensor is the (k+1)-th posterior cumulant over `σ^{2k}`.
**Corollary.** A Gaussian posterior has zero third cumulant, so any fixed Gaussian hypothesis gives
an affine map; nonzero label-curvature certifies a non-Gaussian posterior. Under a hierarchical GP,
mixing over hyperparameters is the only source of non-Gaussianity, so curvature directly measures
residual hypothesis mixing.

### T4 — Hedging decomposition

For a hierarchical prior `θ ~ π`, `f | θ ~ N(0, K_θ)`, with `μ_θ(y) = K_θ(K_θ + σ²I)⁻¹y`:

    σ² J_{μ*}(y) = E_{θ|y}[C_θ]  +  Cov_{θ|y}(μ_θ(y))
                   └ fixed-hypothesis ┘   └ hedging term H(y) ⪰ 0 ┘

(a) The map is affine exactly when the hedging term vanishes (point-mass hypothesis posterior).
(b) Excess predictive variance over the best committed model equals `diag H`.
Curvature and excess variance are two read-outs of one operator.

### T5 — BvM decay

Under hierarchical Bernstein–von Mises regularity and local Lipschitzness of `θ ↦ μ_θ`,
`H_**(y) = O_P(1/n)`.

### T6 — Converse (characterisation)

A smooth `m` is some prior's posterior mean **iff** (A) `(m(y) − y)/σ²` is a gradient field
(equivalently `J` symmetric everywhere), and (B) `e^φ` is, up to a positive constant, the Gaussian
mollification of a positive measure. The audit is therefore not merely necessary but essentially
characterising; the representing measure is the recovered prior.

---

## 2. New results

These are the paper's theoretical contribution. **All six need to be written in theorem–proof form
and independently checked before any prose is written around them.**

### N1 — Normaliser lemma

Let a predictor apply affine target normalisation by context statistics:

    m(y) = ȳ·1 + s_y · g(u),    u = (y − ȳ·1)/s_y,    ȳ = mean(y),  s_y = std(y)

Then with `G = ∂g/∂u`,

    J = (1/n)·1 1ᵀ  +  (1/n)· g uᵀ  +  G M,    M = I − (1/n)1 1ᵀ − (1/n) u uᵀ

**(i) The confound is O(1), not O(1/n).** `‖(1/n) g uᵀ‖_F = (1/n)‖g‖‖u‖ ≈ (1/n)·√n·√n = O(1)`; the
prefactor is cancelled by two vectors of norm √n. The term is rank-one and asymmetric unless
`g(u) ∝ u`, and `GM` is asymmetric whenever `G` and `M` fail to commute, which is generic.
**Therefore an exact Bayesian composed with a context-statistic normaliser has an O(1) asymmetric
Jacobian, and raw symmetry tests on any such model are uninformative.**

**(ii) The projection is exact.** Let `Q ∈ R^{n×(n−2)}` be an orthonormal basis of `span{1, u}^⊥`.
Then `Qᵀ1 = 0`, `uᵀQ = 0`, `MQ = Q`, and

    Qᵀ J Q = Qᵀ G Q

exactly. Test symmetry and positivity on the reduced `(n−2)×(n−2)` matrix, not on `PJP` in ambient
coordinates — the latter's two zero eigenvalues make the PSD test trivially pass.

**(iii) Ensembling.** A single *outer* normaliser around an ensemble is harmless: `(1/K)Σ g_k` is
itself one inner map. The failure mode is *per-member* normalisation with differing statistics, in
which case each member contributes a rank-one term along its own `span{1, u_k}` and the projection
leaks. Verified absent when the ensemble is set to one member.

### N2 — Identifiability of the normaliser directions

`M` has rank `n−2`, so `G` restricted to `span{1, u}` is **not recoverable from `J` by any
measurement.** This is a genuine identifiability limit, not a shortcoming of the projection.

Moreover, the two identities determined by the wrapper,

    J·1 = 1        J·u = g = (m(y) − ȳ·1)/s_y

are together exactly equivalent to **shift-equivariance plus degree-1 homogeneity** (since
`Jy = ȳ·J1 + s_y·Ju = m(y)`). Both properties survive averaging over an ensemble, so these identities
have **no power** as a test of per-member normalisation — they test the wrapper form only. Their
utility is different: `J·1 = 1` is parameter-free and must hold exactly, so `‖J·1 − 1‖/‖1‖` is a
free measurement of instrument error on the real model.

**Consequence for scope.** The two untestable directions are where scale and shift inference live.
Theory (Section 3, radial/tangential) independently argues that hypothesis averaging must appear
*tangentially*. The unidentifiable sector is therefore the one where signal was not expected — a
point to state rather than bury.

### N3 — Asymmetric robustness of A1 and A2

Since `QᵀJQ = QᵀGQ` and `QᵀGQ` is PSD whenever `G` is, **an affine normaliser cannot manufacture a
negative eigenvalue in the projected block**, though it demonstrably can manufacture raw asymmetry.
The two conditions are not interchangeable diagnostics: **A2 is strictly more robust to
preprocessing and should lead.**

### N4 — Parameter-free cross-channel inequality

Under the cross-channel law, `s²ᵢ = σ²(1 + Jᵢᵢ)`, so `mean(s²) = σ²(1 + mean Jᵢᵢ)` and
`std(s²) = σ² std(Jᵢᵢ)`. Hence

    cv_s = std(Jᵢᵢ)/(1 + mean Jᵢᵢ)  ≤  std(Jᵢᵢ)/mean(Jᵢᵢ) = cv_J

whenever `mean(Jᵢᵢ) > 0` (guaranteed under A2). **No fitted σ², no regression, no design leverage
required.** The reported variance cannot vary more than the model's own label sensitivity permits.

### N5 — Jacobian catalogue for published candidate mechanisms

Every mechanistic account of these models in the literature has a closed-form label-Jacobian. This
turns the audit from a Bayes/not-Bayes binary into a **discriminating instrument over the field's own
explanations.**

| Mechanism (source) | `J` at context points | A1 | A2 | Curvature |
|---|---|---|---|---|
| One GD step (von Oswald, W₀=0) | `(η/N)·XXᵀ` (Gram) | ✓ | ✓ | 0 |
| k GD steps / preconditioned / Newton | `X p(XᵀX) Xᵀ` | ✓ | ✓ if `p ≥ 0` on spec | 0 |
| Ridge, exact GP, kernel ridge | `K(K + λI)⁻¹` | ✓ | ✓ | 0 |
| Richardson + Jacobi (Kang et al., core) | `X p(XᵀX) Xᵀ` form | ✓ | ✓ | 0 (head only) |
| Nadaraya–Watson / attention vote (Craig & Tibshirani; Miftachov; Biloš TabPFNv2, Mitra) | `D⁻¹K`, `D = diag(K1)` | ✗ **row-scaled** | ✓ (similar to `D^{-1/2}KD^{-1/2}`) | 0 |
| 1-NN at context points (McCarter) | `I` | ✓ trivially | ✓ trivially | 0 |
| Prototype-distance readout (Biloš TabICLv2) | labels enter via group means; readout quadratic in them | no reason | no reason | ≠ 0 |
| Hierarchical Bayes | `Cov(f\|y)/σ²` | ✓ | ✓ | ≠ 0, `O(1/n)` |
| Imitator (T1) | `J_{μ*} + cos(·) baᵀ` | ✗ | ✗ | — |

**Two consequences worth stating in the paper.** First, the entire Camp A linear-solver family
satisfies A1 and A2 *by construction* — so a model failing them is not implementing any of those
algorithms. Second, 1-NN at context points gives `J = I`, which passes both conditions vacuously;
this is why a non-degeneracy check (`‖J − I‖_F/‖J‖_F` bounded away from 0, and `‖J‖_F` of order √n)
is mandatory before any pass is reported.

**N5 does three distinct jobs, with different evidential status. Keep them separate.**

1. **Analytic.** The closed forms are derivations about the mechanisms, true whether or not any model
   implements them. No experiment required. This is what makes the audit a *discriminating*
   instrument rather than a Bayes/not-Bayes binary.
2. **Instrument validation.** Pushing each closed form through the finite-difference pipeline (E0.2)
   validates the pipeline against known answers and simultaneously verifies the algebra above.
3. **Mechanism exclusion on frozen models.** This does **not** follow from a mismatch between the
   measured Jacobian and a closed form. Nobody claims these models are *exactly* a smoother or
   *exactly* a solver — the published accounts are approximation claims, so exclusion needs a
   calibrated tolerance. It is obtained by fitting each mechanism as a surrogate to the model's own
   outputs and reporting value agreement and derivative agreement on the same context (E2.6); the
   surrogate's own value-fit supplies the tolerance, and nothing is invented.

**Scope caveats attached to the catalogue rows.**
- Camp A studies constructions and trained toys, not frozen tabular models. Excluding the
  linear-solver family is a **bridging** claim — the family dominating ICL theory has a derivative
  signature these models do not carry — not a refutation of authors who never claimed otherwise.
- The vote and prototype readouts are described for **classification**; the audit is on regressors.
  The vote form transfers naturally to continuous targets; the prototype-distance readout arguably
  does not. Transfer to the regression head is a stated assumption, not a fact.

### N6 — Heteroscedastic generalisation, and row-vs-column scaling

Under `ε ~ N(0, Σ)`, `Σ = diag(σᵢ²)`, differentiating gives

    J = Cov(f | y) Σ⁻¹      ⟹  **JΣ symmetric** (not J)

i.e. a **column** rescaling of a symmetric matrix: `Jᵢⱼσⱼ² = Jⱼᵢσᵢ²`. Contrast Nadaraya–Watson,
`J = D⁻¹K`, a **row** rescaling: `dᵢJᵢⱼ = dⱼJⱼᵢ`. These are different linear systems in the unknown
diagonal, both overdetermined, both solvable by least squares in log-space.

**Therefore the measured Jacobian discriminates "heteroscedastic Bayesian" from "kernel smoother",**
and Gate 2d stops being a caveat and becomes a mechanism test. If neither system admits a positive
solution with small residual, both classes are excluded.

---

## 3. The audit

Conditions, in the order they are run.

**M0 — Radial/tangential gate (runs first in any campaign).**
Decompose curvature by direction. Radial: along `y ↦ cy`; measured in closed form by the Euler
defect `e(y) = yᵀ∇m_*(y) − m_*(y)`, exactly zero for any degree-1 homogeneous map. Tangential: along
the sphere `‖y‖ = const`, holding `ȳ` and `s_y` exactly fixed via a great circle in the centred
subspace. Only tangential curvature can be hypothesis averaging.
*Pre-registered rule: if curvature is mostly radial, the hedging interpretation dies and the finding
becomes "apparent hedging is preprocessing."*
*Note the geodesic correction:* `d²/dt²[m(γ(t))] = ∇²m[γ',γ'] + ∇m·γ''` with `γ'' = −(γ(t) − c)`;
the second term must be subtracted or second derivatives are contaminated at leading order.

**A1 — Symmetry** (parameter-free). `Qᵀ J Q` symmetric within instrument error.
**A2 — Positivity** (parameter-free). `Qᵀ J Q ⪰ 0`. *Leads, per N3.*
**A3 — Cross-channel law** (one profiled scalar, plus the parameter-free N4 inequality).
**A4 — Decay** (a rate, given a scope). *Out of scope for v1 where A1/A2 fail — see below.*

Each failure of A1–A3 excludes the entire Bayesian class; A4 excludes only the concentrating
hierarchical subclass.

---

## 4. Claim inventory

Every sentence the paper will assert. Experiments attach to these in `02_experiment_battery.md`.

### Theory claims (T-claims)

- **C-T1.** Value functionals cannot certify Bayes-realisability. *(T1, proved)*
- **C-T2.** Every exact posterior mean under a Gaussian channel has symmetric PSD label-Jacobian
  with `s² = σ²(1 + Jᵢᵢ)`, for every prior. *(T2, proved)*
- **C-T3.** Label-curvature equals the third posterior cumulant; nonzero curvature certifies a
  non-Gaussian posterior. *(T3, proved)*
- **C-T4.** Curvature and excess predictive variance are two read-outs of one hedging operator.
  *(T4, proved)*
- **C-T5.** The conditions are essentially characterising, not merely necessary. *(T6, proved)*
- **C-N1.** Context-statistic normalisation contributes an O(1) asymmetric rank-one term; an exact
  Bayesian wearing a normaliser fails raw A1. *(N1, new)*
- **C-N2.** The projection recovers the inner map's Jacobian exactly, and the two removed directions
  are permanently unidentifiable and carry no model information. *(N1–N2, new)*
- **C-N3.** A normaliser cannot manufacture negative eigenvalues in the projected block; A2 is
  strictly more robust than A1. *(N3, new)*
- **C-N4.** `cv_s ≤ cv_J` is a parameter-free consequence of the cross-channel law. *(N4, new)*
- **C-N5.** Every published candidate mechanism has a closed-form Jacobian; the linear-solver family
  satisfies A1/A2 by construction; smoothers are row-scaled symmetric. *(N5, new)*
- **C-N6.** Under heteroscedastic noise the symmetric object is `JΣ`; row-vs-column scaling
  discriminates smoother from heteroscedastic Bayesian. *(N6, new)*

### Instrument claims (I-claims)

- **C-I1.** The finite-difference pipeline reproduces known Jacobians to a stated precision.
- **C-I2.** The projection removes the confound on a normaliser-wrapped exact Bayesian.
- **C-I3.** The projection preserves the violation on a normaliser-wrapped Bayes-impossible map.
- **C-I4.** The measured violations exceed instrument error by a stated factor.
- **C-I5.** The measured violations are step-size invariant.
- **C-I6.** The Jacobian is non-degenerate — not `≈ I`, not `≈ 0`.

### Empirical claims (E-claims), per model

For each of **TabPFN v2, TabICL v2, TabSwift**:

- **C-E1.** The model passes the value-level battery at or near published performance.
- **C-E2.** Verdict on A1, with heteroscedastic and smoother profiling if raw A1 fails.
- **C-E3.** Verdict on A2, reported against the imitator control.
- **C-E4.** Verdict on A3, including the parameter-free `cv` inequality.
- **C-E5.** Verdict on M0 (radial vs tangential).
- **C-E6.** Which row of the N5 catalogue, if any, is compatible with the measured Jacobian.
- **C-E7.** Robustness: the verdict is stable across step size, ensemble configuration, positional
  encoding zeroing, context distribution, dimension, noise level, and context size.

### Cross-cutting claims

- **C-X1.** At least one predictor passes the full audit through the identical pipeline, on real
  tabular data. *(Without this, "everything fails" reads as "the test is too strict.")*
- **C-X2.** The value-level battery and the derivative channel disagree on the same model, same
  contexts — T1 instantiated empirically.
- **C-X3.** The derivative channel appears once in the PFN literature (Nagler, as a magnitude bound
  used to prove variance decay) and its structural content has never been examined; across 31
  surveyed papers, none states, derives, or measures the label-dependence of the predictive map.
- **C-X4.** McCarter's reported anomaly — duplicating samples of one class shifts the boundary the
  wrong way — is a discrete shadow of a negative Jacobian diagonal, observed and dismissed as an
  artifact because no framework said it was Bayes-impossible.

### Explicitly not claimed

- Anything about A4 / the decay law where A1 or A2 fail. The hedging reading of curvature
  presupposes the map is a hierarchical posterior mean; a decay exponent measured on a map excluded
  from that class has no interpretation attached. **Stated in scope, not measured.**
- Anything about the `1` and `u` directions.
- Non-diagonal noise structures.
- Classification.
- Any claim that these models are poor predictors.

---

## 5. Amendment log

Every change since revision 1, with its cause. Nothing above was edited silently; each amended item
carries a pointer to its entry here.

### Revision 2 — 2026-08-26

#### A1. N4 — proven, and without instrument power against the failure that occurred

**N4 stands as proved.** `cv_s = std(J_ii)/(1 + mean J_ii) ≤ std(J_ii)/mean(J_ii) = cv_J` whenever
`mean(J_ii) > 0`, with no fitted `σ²`, no regression and no design leverage. Nothing about the
derivation changes.

**What is added is a limitation on its use as an instrument.** `cv_s ≤ cv_J` **holds** for both
models with a variance channel — TabICL `0.197 ≤ 0.289`, TabPFN `0.051 ≤ 0.800` — so it returns **no
violation** on maps that fail A3 badly by regression (`R² = 0.0199` and `0.1065`). The reason is that
**it is permutation-invariant**: permuting `s²` across context points leaves `cv_s` bit-identical
(`0.0894008` in both rows of the control table) while `R²` falls from `1.000000000000` to
`0.010468752`.

**Both facts stand.** N4 is a true and parameter-free *necessary* consequence of the cross-channel
law. It is **not sufficient**, and it has **no power against the specific failure these models
exhibit**, which is a *reordering* of the variance channel relative to the sensitivity channel rather
than an inflation of its spread. C-N4 is amended below accordingly.
*Evidence:* `chunk2_a3_controls.py` → `chunk2_results.json`; `PHASE1.md` §5.4.

#### A2. N5 — the exclusion inference is softened to what was actually measured

**Amended.** N5(3) previously licensed reading a mismatch between the measured Jacobian and a closed
form as *mechanism exclusion*, subject to a calibrated tolerance from E2.6. **E2.6 was never run
correctly**, so that tolerance does not exist. The N5 table's closed forms are unaffected — they are
derivations and remain true.

**What §7's measurement actually supports**, and the strongest form the claim may take:

> Fitting the two **diagonal-scaling** hypotheses to the measured Jacobian by least squares in log
> space, **neither admits a positive solution with small residual.** The column system
> (`J_ij σ_j² = J_ji σ_i²`, the heteroscedastic-Bayes form of N6) and the row system
> (`d_i J_ij = d_j J_ji`, the Nadaraya–Watson / attention-vote form of N5) return residuals of
> `0.476`–`0.616` across the three models, against **`1.5e-15` on a Nadaraya–Watson positive control
> pushed through the identical procedure**. The recovered profiles span factors of `87×`, `68×` and
> `169×` against `2.4×` and `1.7×` on the two controls.

That is an exclusion of **two diagonal-scaling hypotheses at the measured residual, calibrated by the
NW control**. It is **not** the calibrated exclusion of the field's published mechanisms that N5(3)
describes, because that requires fitting each mechanism as a **surrogate to the model's own outputs**
and reporting value agreement beside derivative agreement — E2.6, which does not exist.

**Consequence for the paper.** The scatter of value agreement against derivative agreement, which
`experiments.md` calls "the headline figure", **cannot be drawn**. C-N5 and C-E6 are amended below.
*Evidence:* `exp2_ambient_profiling.py` → `exp2_ambient_profiling.json`; `PHASE1.md` §6, §9.1.

#### A3. N7 — the cross-channel law through the affine wrapper *(NEW — DERIVATION PENDING)*

> **PLACEHOLDER. The derivation has not been supplied and must not be invented.**
> This entry records the *empirical* fact that N7 is required to explain, and reserves its number.

**The fact requiring a derivation.** A hierarchical GP satisfies A3 **exactly** by construction: T2
gives `σ²J = Cov(f|y)` for every prior, so `s²_i = σ²(1 + J_ii)` holds pointwise. Measured
**unwrapped** through the audit's own A3 procedure on 60 real contexts, it returns `R² = 1.000000` on
all 60. Measured **wrapped** in the affine context-statistic normaliser of N1 — the same wrapper the
audited models wear — its `R²` falls as low as **`0.675`** and it fails the `R² ≥ 0.90` gate on
**3 of 60**.

**What is already known about the effect:**
- It is **not** the lengthscale ladder: rescaling the ladder to the data leaves it (`R²` min `0.843`,
  A3 `55/60`).
- It is **not** real data or the pipeline: A1 and A2 pass `60/60` in every condition, and `negeig` is
  **exactly zero** throughout.
- It **requires a nonlinear inner map.** The exact GP, whose inner map is linear, shows `R²` min
  `0.999974` wrapped. All three audited models are nonlinear.

**What N7 must supply.** A statement of what `s²_i = σ²(1 + J_ii)` becomes under
`m(y) = ȳ·1 + s_y·g(u)` — i.e. how the ambient diagonal `[J]_ii` of the wrapped map relates to the
inner `G_ii`, given `J = (1/n)11ᵀ + (1/n)g uᵀ + GM` with `M = I − (1/n)11ᵀ − (1/n)uuᵀ`. The diagonal
picks up `1/n + (1/n)g_i u_i` plus the row-sum corrections inside `(GM)_ii`, none of which vanish for
nonlinear `g`.

**Until N7 exists,** `PHASE1.md` §5.0 carries the floor as an **empirical** bound and the A3 verdict
rests on the measured values being an order of magnitude below it — not on a theorem.
*Evidence:* `tier5_positive/e5_1_positive_control.py` → `e5_1_results.json`; `PHASE1.md` §5.0, §7.2.

#### A4. C-X2 — promoted from claim to result

**Amended.** C-X2 previously read as a claim awaiting evidence: *the value-level battery and the
derivative channel disagree on the same model, same contexts.* **E4.1 instantiates it empirically.**

On the **exact** audit contexts, the two models with a predictive distribution are **well calibrated
and near-oracle in NLL** — coverage at 90% of `0.896` (TabICL) and `0.886` (TabPFN) against the
Bayes-optimal oracle's `0.894`, NLL `1.755`/`1.758` against `1.676` — **while failing A1 and A2 by
`57×`–`22 711×` over the artifact floor.** The statistic has power: the deliberately misspecified
`σ = 0.5` control GP is the **worst-calibrated map in the table** at `0.722`.

**T1 is no longer instantiated only by an adversarial construction.** It is instantiated by three
production checkpoints on the data the audit was run on.

**Scope, which must travel with the claim.** One context family, deliberately hard. E4.1 does **not**
establish that the models perform at their *published* level, because published benchmarks are real
tabular data with exploitable structure rather than a near-white `d=5` GP at SNR 1. It establishes the
narrower thing: on the contexts where the violations were measured, the models work, are calibrated,
and beat every trivial baseline. E3.1 remains the experiment that settles the distributional question.
*Evidence:* `tier4_value/e4_1_value_battery.py` → `e4_1_results.json`; `PHASE1.md` §8.

---

### Amended claim statements

Replacing the corresponding entries in Section 4. The originals are left in place above; these are
the versions the paper uses.

- **C-N4 (amended).** `cv_s ≤ cv_J` is a parameter-free *necessary* consequence of the cross-channel
  law. It is **not sufficient** and is **permutation-invariant**, so it has no power against a
  reordering of the variance channel — which is the failure these models exhibit. It holds for both
  models and is reported as a secondary diagnostic, never as a pass. *(A1)*
- **C-N5 (amended).** Every published candidate mechanism has a closed-form Jacobian; the
  linear-solver family satisfies A1/A2 by construction; smoothers are row-scaled symmetric. **The
  measured Jacobians admit neither a row nor a column diagonal rescaling**, at residuals of
  `0.476`–`0.616` against `1.5e-15` on a Nadaraya–Watson positive control. **This excludes the two
  diagonal-scaling classes at the measured residual. It is not a calibrated exclusion of the
  published mechanisms, which requires E2.6 and was not run.** *(A2)*
- **C-E6 (amended).** Which row of the N5 catalogue is compatible with the measured Jacobian is
  **answered only negatively and only for the diagonal-scaling rows.** No surrogate fit exists.
- **C-N7 (new, pending).** The cross-channel law is not invariant under the affine context-statistic
  normaliser for a nonlinear inner map. Empirically a map satisfying A3 exactly reads `R²` as low as
  `0.675` through the wrapper. **Derivation pending.** *(A3)*
- **C-X2 (amended — now a result).** The value-level battery and the derivative channel disagree on
  the same models on the same contexts: near-oracle calibration and NLL alongside A1/A2 violations of
  `57×`–`22 711×` over the artifact floor. **Measured, not claimed.** *(A4)*
