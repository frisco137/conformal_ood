# Phase 1 — The experiments

**What was run, why it exists, what it was measured against, and how it was judged.**

This is a live document describing the experiments that were actually run and are going in the paper.
Experiments that were planned and never run are **not** here — they are in
[`phase_1_results.md`](phase_1_results.md) §9, under what is not established.

Results: [`phase_1_results.md`](phase_1_results.md). Ledger: [`FINAL_NUMBERS.md`](FINAL_NUMBERS.md),
which wins any dispute. Theory: [`theory_and_claims.md`](theory_and_claims.md), revision 2.

---

## What Phase 1 is for

Three frozen tabular foundation models — **TabPFN v2, TabICL v2, TabSwift** — are widely described as
doing approximate Bayesian inference in context. That description makes a testable commitment.

If a model's prediction `m(y)` at the context locations is the posterior mean `E[f | y]` under *any*
prior, with Gaussian observation noise, then Brown's identity forces
**`σ²·∂m/∂y = Cov(f | y)`**. A covariance matrix is symmetric and positive semi-definite, and its
diagonal is pinned to the predictive variance. So three things must hold, for every prior, with no
free parameters:

| | condition | what it means |
|---|---|---|
| **A1** | `J` is symmetric | how much point *i*'s label moves prediction *j* must equal the reverse |
| **A2** | `J` is PSD | raising labels along any direction cannot lower the prediction along it |
| **A3** | `s²ᵢ = σ²(1 + Jᵢᵢ)` | the variance channel and the sensitivity channel are one object |

Phase 1 measures `J = ∂m/∂y` by finite differences and checks all three.

**Why this is worth doing.** T1 (`theory_and_claims.md`) proves that no *value* functional — test
error, NLL, calibration, coverage, martingale statistics, probe accuracy — can separate
Bayes-realisable maps from their complement with any margin. Everything the field currently measures
is a value functional. The derivative channel is not, and across 31 surveyed papers nobody has
looked at it.

**The whole argument therefore rests on the instrument.** If the measurement pipeline manufactures
asymmetry, there is no result. Tier 0 and Tier 1 exist entirely to close that off, and Tier 5 exists
to prove a real Bayesian passes the same pipeline. They are not preliminaries; they are the load
that the conclusion sits on.

---

## The standing setup

**Contexts.** `generate_audit_context(n=100, d=5, sigma=1.0, seed=s)`, seeds
`[42, 100, 200, 300, 400]`. 90 unique `X ~ N(0, I₅)` rows plus **10 exact duplicates**, and
label noise at `sigma = 1.0`. The duplicates and the noise are deliberate: they force the model into
a regime where it must smooth, which is where a Bayesian's structure is most constrained.
*Script:* [`core/context.py`](core/context.py).

**The projection.** All three models normalise the target by context statistics before predicting.
N1 proves this contributes an **O(1) asymmetric rank-one term** to the ambient Jacobian — so a
*perfect* Bayesian wearing a normaliser fails a raw symmetry test, and raw tests are uninformative.
N1(ii) proves the fix is exact: with `Q` an orthonormal basis of `span{1, u}^⊥`,
**`QᵀJQ = QᵀGQ`** recovers the inner map's Jacobian exactly. Every A1 and A2 number is measured on
that reduced block. *Script:* [`core/metrics.py`](core/metrics.py) `get_Q(y, seed=0)`, `reduced_jacobian`.

**Metric definitions**, fixed once and implemented once:
`asym(J) = ‖J − Jᵀ‖_F / ‖J‖_F` and `negeig(J) = −min(0, λ_min(sym J)) / ‖J‖₂`.

---

## Tier 0 — Is the instrument sound?

### E0.3 · Step-size plateau
**Measures.** Whether the measured violation is a real derivative or a finite-difference artifact.
**Exists because** a violation that grows as `h → 0` is discretisation noise, not structure.
**Measured against** a sweep of probe amplitudes per model, each chosen from that model's own measured
output quantum. **Judged by** drift across the sweep.
*Script:* [`tier0_instrument/exp3_reprobe.py`](tier0_instrument/exp3_reprobe.py),
[`exp4_quanta_and_tabicl.py`](tier0_instrument/exp4_quanta_and_tabicl.py) → `exp3_reprobe.json`, `exp4_results.json`.
→ [Results §3.2](phase_1_results.md)

### E0.5 · Output-quantum measurement
**Measures.** The discretisation step of each model's output, by a 501-point fine scan of `m(y + tq₁)`.
**Exists because** the artifact floor cannot be computed without it, and the three models differ by
three orders of magnitude — TabICL is float32, **TabSwift is float16**.
**Judged by** the jump distribution: a quantised output concentrates jumps at multiples of a step, a
continuous one spreads them around the local slope.
*Script:* [`tier0_instrument/exp4_quanta_and_tabicl.py`](tier0_instrument/exp4_quanta_and_tabicl.py) → `exp4_results.json`.
→ [Results §3.1](phase_1_results.md)

### E0.6 · Row-sum residual `r₁`
**Measures.** `‖J·1 − 1‖₂/‖1‖₂`. **Exists because** it is parameter-free and must hold *exactly* for
any shift-equivariant map, so any deviation is pure instrument error — a free noise-floor measurement
on the real model, replacing a guessed constant.
**Measured against** wrapped and unwrapped exact GPs, which separate instrument error from genuine
absence of the property. **Judged by** magnitude, read together with E1.1.
*Script:* [`tier0_instrument/recompute_orphans.py`](tier0_instrument/recompute_orphans.py) → `clarif_item4_6.json`.
→ [Results §1.3](phase_1_results.md)

### Instrument stress tests
**Measures.** Seven properties the estimator must have: it recovers a known asymmetric `J`; it has the
right orientation (not transposed); `asym`/`negeig` are invariant to the choice of `Q` within the
subspace; a constant map gives `J = 0`; the analytic pipeline is deterministic; probing along `Q`
agrees with probing ambient and then projecting; and permuting the context permutes the Jacobian.
**Exists because** each is a way the pipeline could be silently wrong.
**Judged by** machine precision.
*Script:* [`tier0_instrument/recompute_orphans.py`](tier0_instrument/recompute_orphans.py) → `exp_stress_tests.json`.
→ [Results §1.7](phase_1_results.md)

### Control validation and the artifact floor
**Measures.** What the pipeline reports on maps whose true answer is known to be **exactly zero** — an
exact GP (`asym = 0`, `negeig = 0` by construction) and a *quantised* exact GP, whose every nonzero
reading is manufactured by output discretisation.
**Exists because** every model number in this phase is reported as a ratio to one of these floors. A
violation is only a violation relative to what the instrument produces on a map that has none.
**Judged by** how far the floor sits below the model reading.
*Scripts:* [`tier0_instrument/chunk1_control_validation.py`](tier0_instrument/chunk1_control_validation.py) → `chunk1_results.json`;
[`chunk3_dither_controls.py`](tier0_instrument/chunk3_dither_controls.py) → `chunk3_results.json`;
[`recompute_orphans.py`](tier0_instrument/recompute_orphans.py) → `exp4_floor_brackets.json`.
→ [Results §1.1, §1.4](phase_1_results.md)

### Theory verification — T2, T4, N1(ii)
**Measures.** That the identities the audit rests on actually hold numerically: `σ²J = Cov(f|y)` on an
exact GP (T2), the hedging decomposition on a hierarchical GP (T4), and that the projection recovers
the inner Jacobian exactly (N1(ii)).
**Exists because** the conditions are derived from these identities; if the algebra were wrong the
audit would be measuring nothing.
**Judged by** relative Frobenius error against the closed form.
*Scripts:* [`tier0_instrument/chunk2_a3_controls.py`](tier0_instrument/chunk2_a3_controls.py) → `chunk2_results.json`;
[`recompute_orphans.py`](tier0_instrument/recompute_orphans.py) → `exp_n1ii.json`.
→ [Results §1.6, §1.8](phase_1_results.md)

---

## Tier 1 — Is the violation a preprocessing confound?

### E1.1 · Preprocessing identification
**Measures.** Whether each model's target transform is genuinely **affine**, by testing shift
equivariance `m(y + c1) = m(y) + c1` and scale homogeneity `m(cy) = c·m(y)` directly, plus reading
the preprocessing path out of each library's source.
**Exists because** the projection is only licensed for an affine wrapper. A rank- or quantile-based
transform would make N1(ii) inapplicable and no linear projection could remove it.
**Measured against** wrapped and unwrapped exact GPs — the unwrapped one satisfies scale but not
shift, showing the two identities are independent and the test has power.
**Judged by** `< 1e-3` relative.
**Outcome that matters:** TabPFN and TabICL pass; **TabSwift fails both**, and its scaler is
commented out in source. This is why TabSwift's A1 is reported as the asymmetry of the compressed
Jacobian rather than of a recovered inner map.
*Script:* [`tier0_instrument/e11_wrapper_form.py`](tier0_instrument/e11_wrapper_form.py) → `e11_wrapper_form.json`.
→ [Results §1.5](phase_1_results.md)

### E1.2 · Projection validation, both directions
**Measures.** (a) that wrapping an exact GP produces O(1) raw asymmetry which the projection
collapses; (b) that wrapping a **Bayes-impossible** map produces a violation which the projection
**preserves**.
**Exists because** (b) is load-bearing. Without it, a clean projected result could mean the
projection removed the signal rather than the confound, and no statement about any model would be
licensed.
**Measured against** `TargetedImitator` — an exact GP plus an antisymmetric term and a negative
rank-one term, sized in advance, with a closed-form Jacobian to check against.
**Judged by** `asym(J_red) < 0.02` for (a) and `negeig ≥ 10 × A_floor` for (b).
*Scripts:* the map is [`core/controls.py`](core/controls.py);
[`tier0_instrument/chunk1_control_validation.py`](tier0_instrument/chunk1_control_validation.py) → `chunk1_results.json`.
Regression test: [`core/test_controls.py`](core/test_controls.py).
→ [Results §1.2](phase_1_results.md)

### E1.4 · Positional-encoding zeroing (TabICL v2)
**Measures.** Whether TabICL's asymmetry survives when RoPE is zeroed at inference.
**Exists because** positional leakage is an independent generator of asymmetry — processing row *i*
differently from row *j* by position is effectively assigning them different precisions, which would
produce the signature without any Bayesian failure.
**Measured against** itself, **paired per context**, not as two marginal means.
**Judged by** whether the 95% CI on the paired difference contains zero. It does.
*File:* `tier0_instrument/e14_paired.json`. **This is the one result in Phase 1 with no producing
script** — see [Results §9.1](phase_1_results.md).
→ [Results §3.3](phase_1_results.md)

---

## Tier 2 — The audit

### E2.1 · Non-degeneracy
**Measures.** `‖J − I‖_F/‖J‖_F`, `‖J‖_F` against `√n`, and the effective rank.
**Exists because** a pass can be vacuous. `J ≈ I` is the 1-NN row of the N5 catalogue and satisfies
A1 and A2 trivially; `J ≈ 0` is instrument noise. **This check runs before any verdict is reported**,
and it is what identified `t = 1e-3` as void for TabSwift, whose `‖J‖_F` collapses from `392.8` to
`6.09` across the amplitude sweep.
**Judged by** `‖J − I‖_F/‖J‖_F > 0.3` and `‖J‖_F` of order `√n`.
→ [Results §2](phase_1_results.md)

### E2.2 · A2, positivity — *leads*
**Measures.** `negeig(QᵀJQ)`, the full spectrum, and where the leading negative eigenvector
concentrates on context points.
**Exists because** N3 proves an affine normaliser **cannot** manufacture a negative eigenvalue in the
projected block, though it demonstrably can manufacture raw asymmetry. **A2 is strictly more robust
to preprocessing than A1 and therefore leads.**
**Measured against** the quantised-exact-GP artifact floor at each model's own probe configuration,
and against the imitator, which was built to violate A2 deliberately.
**Judged by** `negeig ≤ max(3F, 0.02)`.
*Scripts:* [`tier0_instrument/exp3_reprobe.py`](tier0_instrument/exp3_reprobe.py),
[`exp4_quanta_and_tabicl.py`](tier0_instrument/exp4_quanta_and_tabicl.py),
[`recompute_orphans.py`](tier0_instrument/recompute_orphans.py) → `exp_negeigvec.json`.
→ [Results §4](phase_1_results.md)

### E2.3 · A1, symmetry
**Measures.** `asym(QᵀJQ)`, with raw ambient `asym` reported alongside so the size of the N1 confound
is visible.
**Measured against** the same artifact floors. **Judged by** `≤ max(3F, 0.05)` to pass,
`> max(10F, 0.15)` to fail, inconclusive between.
*Same scripts as E2.2.*
→ [Results §3](phase_1_results.md)

### E2.4 · Mechanism profiling — heteroscedastic vs smoother
**Measures.** Whether the measured Jacobian is a **diagonal rescaling** of a symmetric matrix, solved
by least squares in log space, in two forms: **column** scaling `Jᵢⱼσⱼ² = Jⱼᵢσᵢ²` (what
`J = Cov(f|y)Σ⁻¹` looks like under heteroscedastic noise, N6) and **row** scaling `dᵢJᵢⱼ = dⱼJⱼᵢ`
(what `J = D⁻¹K` looks like for a kernel smoother or attention-weighted label vote, N5).
**Exists because** it converts two standing objections — "it's heteroscedastic, so your symmetry
condition is the wrong one" and "it's just a kernel smoother" — into a mechanism test with an answer.
**Measured against** a Nadaraya–Watson positive control, which is row-scaled symmetric *by
construction* and must profile to zero; and a wrapped exact GP, which must not be inflated.
**Judged by** whether either system admits a positive solution with residual below the A1 threshold.
Neither does.
**Run in ambient coordinates**, because reduced-basis profiling was shown to have no power — the NW
control's asymmetry *rises* under projection.
*Script:* [`tier0_instrument/exp2_ambient_profiling.py`](tier0_instrument/exp2_ambient_profiling.py) → `exp2_ambient_profiling.json`.
→ [Results §6](phase_1_results.md)

### E2.5 · A3, cross-channel law
**Measures.** Regression of the model's own predictive variance `s²ᵢ` on `(1 + Jᵢᵢ)`, per context,
both with and without an intercept — the law requires intercept zero and slope `σ²`.
**Exists because** it is the only condition coupling the two output channels, and the project's own
earlier finding was that mean and variance occupy separate causal subspaces, which predicts exactly
this failure.
**Measured against** an exact GP and a hierarchical GP (both `R² = 1.000000000000`), a permuted
variance head, a constant head, and a scaled head — the permuted control is what shows the
parameter-free `cv_s ≤ cv_J` inequality has no power, since permuting leaves `cv_s` bit-identical
while `R²` collapses.
**Judged by** `R² ≥ 0.90` and intercept within 10% of `σ̂²`.
**The variance must be the integrated second moment over the full predictive distribution** — not a
quantile spread, which is the most tail-sensitive estimator available applied where the bins are
coarsest. The extractor is validated against the exact GP's closed form.
**TabSwift has no predictive distribution** (`Linear(384,1)` point head), so A3 is **unevaluable**
for it and is reported as such, never as a failure.
*Scripts:* [`tier0_instrument/exp12_ambient_jacobians.py`](tier0_instrument/exp12_ambient_jacobians.py) → `exp12_ambient_jacobians.npz`;
[`exp1_extractor_validation.py`](tier0_instrument/exp1_extractor_validation.py);
[`chunk2_a3_controls.py`](tier0_instrument/chunk2_a3_controls.py);
[`recompute_orphans.py`](tier0_instrument/recompute_orphans.py) → `exp1_a3_final.json`, `clarif_item5.json`.
→ [Results §5](phase_1_results.md)

---

## Tier 4 — Is the audit confounded with plain failure?

### E4.1 · Value-level battery, same models, same contexts
**Measures.** Conventional predictive performance on the **exact** audit contexts: MSE against the
latent `f` at the context points (the object the Jacobian differentiates), held-out MSE at 200 fresh
queries, Gaussian NLL, coverage and interval score at four nominal levels, and rank correlation
between each variance channel and realised squared error.
**Exists because** it closes the objection that would otherwise sink the whole result: *if the models
simply perform badly on these adversarial contexts, the structural violations are entangled with
out-of-distribution failure.* It is also what turns T1 from a proof about a construction into an
empirical claim.
**Measured against** five references on the identical contexts — the **Bayes-optimal oracle GP** (true
kernel, true `σ`), a hierarchical GP, the misspecified `σ = 0.5` audit control, a constant predictor,
and 1-NN.
**Judged by** *efficiency*, `(MSE_constant − MSE_model)/(MSE_constant − MSE_oracle)`, because raw MSE
is uninterpretable here: the contexts are near-information-free by design and the oracle itself
reaches only `R² = 0.1425`.
**Held-out queries are drawn from the GP conditional `f_*|f`**, so the audit contexts are left
bit-identical to what every other script sees.
*Script:* [`tier4_value/e4_1_value_battery.py`](tier4_value/e4_1_value_battery.py) → `e4_1_results.json`.
→ [Results §8](phase_1_results.md)

---

## Tier 5 — Does a real Bayesian pass?

### E5.1 · Positive control on real data — *the blocking gate*
**Measures.** An exact GP and a hierarchical GP, wrapped in the same normaliser the models wear,
pushed through the **identical** pipeline — same probe, same `Q`, same metrics, same A3 regression —
on 60 real OpenML contexts.
**Exists because** without it, "all three models fail" reads as "the test is too strict". Its
falsifier is the one that ends the project: if a real Bayesian fails here, the **pipeline** rather
than the models is producing the violations and nothing downstream survives.
**Measured against** the same thresholds applied to the frozen models, plus a non-degeneracy check on
every split — a GP is Bayes-realisable for any hyperparameters, so A1/A2 cannot be tuned into
passing, but they *can* pass **vacuously** if the kernel is near-diagonal. The lengthscale is the
median heuristic to prevent that, and both the lengthscale and the step size are swept.
**Judged by** A1 `≤ 0.05`, A2 `≤ 0.02`, A3 `R² ≥ 0.90` and slope within `±25%`, non-degeneracy
`‖J−I‖_F/‖J‖_F > 0.3`.
**The probe step is relative**, `h = h_frac · std(y_ctx)`. These analytic controls have no output
quantisation so an absolute step would not have broken them — relative is used because the claim is
"identical pipeline", and handing the control an easier probe than the models got would void the
comparison.
**What it also produced:** the wrapped hierarchical GP fails A3 on 3/60 while passing 60/60
unwrapped, which **located an artifact floor in A3 itself** that was not previously known. That floor
now qualifies every A3 number in this phase.
*Script:* [`tier5_positive/e5_1_positive_control.py`](tier5_positive/e5_1_positive_control.py) → `e5_1_results.json`.
→ [Results §7, and the floor at §5.0](phase_1_results.md)

---

## Reproducing

```bash
# CPU only, no model inference — controls, theory checks, and the recomputed orphans
.venv/bin/python -m pytest experiments/phase_1/core/test_controls.py -q      # 22 tests
.venv/bin/python experiments/phase_1/tier0_instrument/chunk1_control_validation.py
.venv/bin/python experiments/phase_1/tier0_instrument/chunk2_a3_controls.py
.venv/bin/python experiments/phase_1/tier0_instrument/chunk3_dither_controls.py
.venv/bin/python experiments/phase_1/tier0_instrument/recompute_orphans.py   # 293 checks vs the record
.venv/bin/python experiments/phase_1/tier5_positive/e5_1_positive_control.py # ~55 s

# GPU — model inference. Each refits on every perturbed y, so a 100x100 ambient
# Jacobian is 200 fits.
.venv/bin/python experiments/phase_1/tier0_instrument/exp12_ambient_jacobians.py
.venv/bin/python experiments/phase_1/tier0_instrument/exp3_reprobe.py
.venv/bin/python experiments/phase_1/tier0_instrument/exp4_quanta_and_tabicl.py
.venv/bin/python experiments/phase_1/tier0_instrument/exp2_ambient_profiling.py
.venv/bin/python experiments/phase_1/tier0_instrument/e11_wrapper_form.py
.venv/bin/python experiments/phase_1/tier4_value/e4_1_value_battery.py       # ~1 min
```

`recompute_orphans.py` and `e5_1_positive_control.py` are pure functions of arrays already on disk
and of analytic NumPy, so they need no GPU. Everything else loads models through
`models.load(model_id, task="regression", device="cuda")`.

**Phase 2 imports this phase's instrument** — `experiments.phase_1.core` — and reads two of
`tier0_instrument/`'s stored arrays to reproduce the audit's localisation. That dependency is
one-directional: nothing in Phase 1 imports Phase 2, except `e5_1_positive_control.py`, which reads
Phase 2's OpenML splits from `../phase_2/arrays/chunk2_data.npz` rather than re-downloading them.
