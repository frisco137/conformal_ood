# T3.1 — Unknown noise scale `(f, σ²)`

**Blocking for every A1 number the paper quotes.** Derivation, then numerical verification.

Verification: [`src/t3_1_noise_scale.py`](src/t3_1_noise_scale.py) → `results/t3_1_noise_scale.json`.
Label: **derivation + numerics.** The proofs below are complete; the numerics confirm them at `n = 60`
and are reported beside each step.

---

## 1. Why the audited class is the wrong one

Assumption 1 fixes `σ²` **known**, and Brown's identity then gives
`σ²J_{μ*}(y) = Cov(f | y)` — symmetric, PSD, diagonal pinned to the predictive variance. Every A1
and A2 threshold in Phases 1–2 is calibrated against that.

**The prior these models were trained on does not fix `σ²`.** A reviewer will therefore invoke the
class of posterior means under a **joint** prior on `(f, σ²)`. That class is strictly larger, and
**Brown's identity does not apply to it directly** — for a `σ²`-mixing prior, `E[f | y]` is not of the
form `y + σ²∇log p(y)`, and there is no reason for its Jacobian to be symmetric.

That is not a hypothetical. Measured below: a `σ²`-mixing **exact posterior mean** has ambient
asymmetry `0.2830` on the audit contexts. **The honest class-level A1 floor is `O(0.1)`, not
`10⁻³`.**

## 2. Setup

`y | f, σ² ~ N(f, σ²I)`, joint prior `P(f, σ²)`, marginal
`p(y) = ∬ φ_{σ²}(y − f) dP(f, σ²)`, strictly positive and `C^∞` under Assumption B. Define the
**precision-weighted** quantities

```
h(y) = E[σ⁻² f | y]        w(y) = E[σ⁻² | y]        g(y) = E[σ⁻²(y − f) | y] = w(y)·y − h(y)
```

## 3. The three identities

### (1) Generalised Tweedie — `∇log p(y) = −g(y)`

Differentiating `p` under the integral,

```
∇_y p(y) = ∬ (−(y − f)/σ²) φ_{σ²}(y − f) dP  =  −p(y)·E[σ⁻²(y − f) | y]
```

so `∇log p(y) = ∇p/p = −g(y) = h(y) − w(y)·y`. **Verified: max relative error `7.8e-10`.**

Note this reduces to ordinary Tweedie when `σ²` is degenerate at `σ₀²`: `w ≡ σ₀⁻²` and
`h = σ₀⁻²E[f|y]`, giving `E[f|y] = y + σ₀²∇log p(y)`.

### (2) The symmetric PSD core — `Cov(σ⁻²(y−f) | y) = w(y)I + ∇²log p(y)`

Write `N_i(y) = ∬ σ⁻²(y_i − f_i) φ dP`, so `g = N/p`. Then

```
∂N_i/∂y_j = δ_ij ∬ σ⁻² φ dP  −  ∬ σ⁻⁴(y_i − f_i)(y_j − f_j) φ dP
```

and by the quotient rule, using `∇log p = −g` from (1),

```
∂g_i/∂y_j = δ_ij w(y) − E[σ⁻⁴(y−f)(y−f)ᵀ | y]_ij + g_i g_j
          = w(y)δ_ij − Cov(σ⁻²(y−f) | y)_ij
```

Since `g = −∇log p`, the left side is `−∇²log p`, giving the identity. The right side is a
**covariance matrix**, hence symmetric and PSD, for every joint prior.

**Verified:** identity to `3.8e-6` relative (limited by the finite-difference Hessian, not the
algebra); the measured covariance is symmetric to `3.1e-15` and has minimum eigenvalue `−1.2e-15`,
i.e. **PSD to machine precision.**

### (3) The contamination is rank one along `y` — `J_h = Cov(σ⁻²(y−f)|y) + y ∇w(y)ᵀ`

From (1), `h(y) = w(y)·y + ∇log p(y)`. Differentiating,

```
J_h = y ∇w(y)ᵀ + w(y)I + ∇²log p(y)  =  Cov(σ⁻²(y−f) | y)  +  y ∇w(y)ᵀ
```

by (2). **So `J_h` is a symmetric PSD matrix plus a rank-one term whose range is spanned by `y`.**

**Verified:** `J_h − Cov` has `σ₂/σ₁ = 3.5e-10` — rank one to machine precision — and its top left
singular vector aligns with `y/‖y‖` at `|cos| = 1.000000` on every context.

## 4. Why the projection and the loop are both blind to it

This is the consequence that matters, and it is exact.

**The projection.** `Q` is an orthonormal basis of `span{1, u}^⊥` with `u = (y − ȳ1)/s_y`. Since
`y = ȳ1 + s_y·u`, we have **`span{1, u} = span{1, y}`**, hence `Qᵀy = 0` and

```
Qᵀ J_h Q  =  Qᵀ Cov(σ⁻²(y−f)|y) Q  +  (Qᵀy)(∇wᵀQ)  =  Qᵀ Cov Q
```

**The contamination is annihilated exactly**, and what remains is a compression of a symmetric PSD
matrix — so A2 in particular is untouched by noise-scale mixing.

**The loop.** For planes with `a, b ⊥ span{1, y}`, the rank-one term contributes

```
aᵀ(y∇wᵀ − ∇w yᵀ)b = (aᵀy)(∇wᵀb) − (aᵀ∇w)(yᵀb) = 0
```

since `aᵀy = yᵀb = 0`. **The circulation instrument is exactly blind to it too** — which is why
`circulation.centred_basis` takes an `exclude` argument and why the P1 reconciliation draws planes
orthogonal to both `1` and the centred `y`.

> **This is a second, independent reason the projection and the loop are the right instruments.**
> The first was N1: they remove the normaliser confound. The second is T3.1: they remove the
> noise-scale confound. Neither was designed for the other.

## 5. The consequence that must be stated honestly

**`μ*(y) = E[f | y]` under a `σ²`-mixing prior is a genuine Bayesian object, and its ambient
Jacobian is asymmetric.** `MEASURED`, exact `σ²`-mixing posterior over a 9-point log-uniform grid on
`[0.25, 4.0]`, `n = 60`, 5 audit contexts:

| quantity | value |
|---|---|
| **ambient `asym(J_{μ*})`** | **mean `0.2830`**, range `[0.2418, 0.3380]` |
| mean `\|aᵀ(J−Jᵀ)b\|`, planes ⊥ `span{1,y}` | `3.2e-03` – `4.6e-03` |
| mean `\|aᵀ(J−Jᵀ)b\|`, planes containing `y` | `6.7e-02` – `9.6e-02` |
| **suppression** | **mean `22.2×`**, range `[14.8×, 28.0×]` |

*Deviation from the plan, reported not tuned:* the plan states `0.287` ambient and `~30×`
suppression. Ambient reproduces almost exactly (`0.2830` vs `0.287`); **suppression comes in lower,
`22.2×` against `~30×`**. The direction and order of magnitude hold; the specific factor does not.

### What this does to the reported floors

Phase 1 quotes TabICL's A1 as `0.580` against a **quantisation-artifact** floor of `0.001–0.004`, a
ratio of `136×–570×`. That floor answers "what does the instrument manufacture on a map with no
asymmetry". **It does not answer "what does a Bayesian in the class the reviewer means produce".**

Against the class-level floor measured here, the same `0.580` is a **~2× excess in ambient terms**,
not `570×`. The projected comparison is the one that survives — the projection annihilates this
contamination exactly (§4) — but **the ambient ratios in Phase 1 are not comparisons against the
right null and must not be quoted as if they were.**

**The reported A1 floors are wrong by roughly ten orders of magnitude at the class level.** This is
what P6 predicts and what the Tier 1 noise-hyperprior GP control measures directly. Every symmetry
ratio in the paper is to be re-quoted against that control once it is run.

**A2 is not affected.** §4 shows the projected block is a compression of a symmetric PSD matrix
regardless of noise-scale mixing, so TabICL's `negeig = 0.177` stands as measured. That asymmetry
between the two conditions — A1 needs re-floor, A2 does not — is exactly N3's claim that A2 is the
more robust condition, now extended from preprocessing to noise-scale mixing.

## 6. What is still open

- The derivation assumes differentiation under the integral is valid, which Assumption B gives for
  sub-Gaussian priors. Stated, not re-proved here.
- `∇²log p` is verified by finite differences, so V2's `3.8e-6` is a **numerical** bound, not an
  algebraic one. The algebra in §3(2) is exact.
- The class-level floor measured here uses one prior family (GP mean, log-uniform `σ²` grid). The
  Tier 1 noise-hyperprior GP control (P5/P6) is the registered measurement and covers all six rungs.
