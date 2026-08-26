# Theory and Claims

**PFN Interpretability Project · working document · August 2026**

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