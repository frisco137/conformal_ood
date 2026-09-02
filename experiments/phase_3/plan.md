# PHASE 3 — In-family audit of TabICL, and a downstream result with a manipulated cause

**Audience:** the repo agent, working autonomously.
**Scope decision:** Phase 3 is **TabICL only**. TabPFN v2 and TabSwift appear only where a
shared-context number falls out for free. Do not re-audit them.
**Mode:** run autonomously. Do not stop for approval. Halt only on the conditions in §7.
**Live deliverable:** `experiments/phase_3/phase_3_results.md`, written *as you go* (§6).

---

## 0. Why TabICL, and what Phase 3 is for

Phase 1–2 established a parameter-free structural test (Brown/Tweedie: `σ²J = Cov(f|y)`, so the
label-Jacobian of any posterior mean is symmetric PSD with its diagonal pinned to the predictive
variance), an instrument validated against analytic controls, and failures on three frozen
regressors — measured on a single GP-drawn context family, with a mixed downstream picture.

Two objections remain unanswered and they are the two that decide whether this is a paper:

1. **Off-family.** Every audit number sits on GP-drawn contexts at `n=100, d=5, σ=1, ℓ=1`.
   The models were trained on SCM-style priors. A reviewer says: the models approximate the PPD
   *on their prior's support*; approximation error is unbounded off-support; your `O(1)` violations
   are off-support approximation error, not a structural claim.
2. **So what.** Phase 2's downstream section measures in seven places and manipulates nothing.

Phase 3 answers both with **one design**: make in-family-ness the independent variable, and
make the violation something we can dial and repair.

TabICL is the right and only choice. Its prior sampler is the one available, and it is the
*mildest* violator (asym 0.580, negeig 0.177, `λmin > 0` on 27/60 real splits), so auditing it
in-family gives the best-behaved model home advantage. Do **not** claim "in-family for all
three" anywhere. The claim is: one within-prior audit of TabICL, plus shared-context comparison.

---

## 1. Standing rules

- **Flexibility is granted on method, not on scope.** If a better estimator, control or
  parameterisation presents itself, take it and log the decision in
  `phase_3_results.md` under `Decisions`. Do not add new research questions.
- **No tangents.** If something interesting but off-plan appears, write it in
  `Open threads` in the results file and move on.
- **Budget.** Target ≤ 40 GPU-hours and ≤ 1 week wall clock for T0–T2. If a task threatens
  that, cut *power* (fewer trajectories/objectives), never cut a *registered comparison*.
- **Every number is traceable.** No value enters the results file that is not also in a saved
  output artifact. Quote the producing script and output path inline beside the number.
- **A result that looks too good is a bug hypothesis first.** Log the falsification you ran.
- **Pin the model.** Record TabICL package version, checkpoint filename and hash, and the exact
  regression-head configuration, in `phase_3_results.md §0` before any measurement.
  Phase 1–2 called it "TabICL v2"; if that name is not what the release calls it, fix the name
  everywhere and note the correction.

---

## 2. Tier 0 — gates. Run these first, concurrently.

### T0.1 Prior and checkpoint provenance  *(blocking for Tier 1; half a day)*

Answer, from source and release notes, not from memory:

- Does the released TabICL prior sampler emit **continuous** targets? (TabICL shipped as a
  classifier; our audited object reads 9999 quantile levels, so a regression variant is in play.)
- Was the audited regression checkpoint trained on **that** prior, or an extended/different one?
- Which knobs are exposed: DAG structure, mechanism family, feature distributions, `n`, `d`,
  categorical fraction, noise mechanism and magnitude?

**Deliverable:** `experiments/phase_3/provenance_prior.md` — what is exposed, what is not, with
file paths and line references into the installed package.

**Branches:**
- **PASS** (continuous targets, prior matches checkpoint, noise mechanism controllable) → Tier 1 as written.
- **PARTIAL** (prior available but not provably the training prior) → run Tier 1 anyway, and
  rename the rung everywhere from *in-family* to **prior-family**. Every claim gets
  "contexts drawn from the released TabICL prior, which we could not verify is the exact
  pre-training prior for this checkpoint". Do not silently keep the stronger word.
- **FAIL** (no usable continuous prior) → skip to T1.7 (nano-PFN) as the primary in-family arm
  and demote the ladder to rungs C–F.

### T0.2 Determinism  *(blocking for everything; ~1 GPU hour)*

Closes Phase 2 gap E0.1. Run the audit path twice on identical inputs, TabICL only.
Report max relative difference in `m(y)`, in `Q^T J Q`, and in `asym`/`negeig`.

**Acceptance:** relative difference in `asym` and `negeig` ≤ 1% of the reported values.
**Fail →** halt condition H1 (§7).

### T0.3 The circulation instrument  *(blocking for Tier 1 breadth; ~2 days incl. controls)*

Symmetry has an integral form that needs no derivatives. On a loop of constant `ȳ` and constant
`‖y − ȳ1‖` — a circle in the centred sphere spanned by orthonormal `a, b ⊥ 1` —

```
y(t) = ȳ1 + r(cos t · a + sin t · b),      t ∈ [0, 2π)
I    = ∮ m(y) · dy                          (forward passes only)
I / (π r²) = aᵗ (J − Jᵗ) b                  exactly, for linear m
```
and in general, by Stokes, `I/(πr²)` is the area-average of the antisymmetric 2-form over the
disc. Four properties make this the right instrument for Phase 3:

- **The normaliser confound vanishes identically.** `m(y) = ȳ1 + s_y g(u)` is constant in `ȳ, s_y`
  along such a loop, so Theorem 12's `O(1)` rank-one term contributes nothing. No projection needed.
- **The unidentifiable sector is automatically excluded.** `span{1, u} = span{1, y}` is exactly
  the set of directions the loop cannot move along, so Theorem 13's limit becomes a design feature.
- **Quantisation-immune.** Output quantum enters additively and averages as `1/√N` instead of
  being amplified by `1/t`. This is what retires the dither, the per-model probe amplitude, and
  TabPFN's `r₁ = 0.51` bind.
- **Frobenius recovery.** Over random orthonormal pairs in the `m = n−1` dimensional centred
  subspace, `E[(aᵗ(J−Jᵗ)b)²] = 2‖J−Jᵗ‖_F² / (m(m−1))`, so ~40 planes recover the Frobenius
  asymmetry.

Implement it, then **reconcile against the stored Phase 1–2 Jacobians on the five existing audit
contexts.** This is the whole point of T0.3: two independent estimators of the same quantity,
which is also the independent cross-check Phase 2's E0.4 asked for without needing autograd hooks.

**Registered predictions (derived analytically, then verified numerically by me at n=60; treat as
targets, and report deviations rather than tuning toward them):**

| ID | Prediction | Falsifier |
|---|---|---|
| P1 | Loop-recovered `‖J−Jᵗ‖_F` on TabICL's audit contexts agrees with the finite-difference value within 20% | disagreement > 2×; then one estimator is wrong and finding out which becomes the task |
| P2 | Exact GP **behind the normaliser**, no projection: `\|I\|/(πr²) ≤ 10⁻⁸` | any larger; the wrapper-invariance argument is wrong |
| P3 | Loop estimate changes ≤ 5% as an artificial output quantum sweeps `10⁻⁵ → 10⁻²`, where central differences at `t=10⁻³` collapse to exactly 0 by `δ=10⁻³` | loop degrades comparably to FD |
| P4 | Frobenius recovery from 40 random planes within 20% of the true centred `‖J−Jᵗ\|_F` on analytic maps | worse than 50% |

**Keep the Jacobian pipeline.** Positivity has no integral form and A2 is the strongest condition
(Theorem 15, and see P6 below). Circulation replaces the *symmetry* measurement; `negeig` still
comes from `Q^T J Q`.

### T0.4 Repair the literature artifact  *(half a day, CPU, unblocks §2 of the paper)*

`literature_extraction_all.md` is two concatenated camps: camp `a` (line 1) whose manifest claims
12 files and lists 10 while 20 `##` sections follow, and camp `c` (line 1616) with no manifest and
no date. **Camp `b` is absent entirely.** Also: ~20 of 33 entries have `[NOT FOUND]` venue/ID, and
`TabSwift` and `TabICL v2` appear nowhere in it.

Fix the camp structure and manifests, locate camp `b`, and add one column per paper:
**which channel supplied the evidence** — value / internal-representation / derivative /
discrete-perturbation. That column *is* the C-X3 survey artifact the paper needs, and it is what
licenses the sentence "none states, derives, or measures the label-dependence of the predictive map".

Before writing that sentence, read von Oswald et al. (ICML 2023) directly: their extraction records
"comparison of the partial derivatives of the transformer predictions against the analytical
sensitivities of gradient descent steps". Establish whether that is w.r.t. inputs/parameters
(expected) or labels (would weaken our gap claim), and record the answer.

---

## 3. Tier 1 — the in-family diagnostic

### T1.1 Context generator

Sample DAG, mechanisms, feature distributions, `d` and `n` from the released TabICL prior;
**force the terminal noise to additive `N(0, σ²)` with `σ²` recorded per context.**

This is the key design move and it needs no change to the theory: Theorem 5 only requires that `P`
be *some* prior on `ℝⁿ`, and the pushforward of the SCM prior onto the sampled locations is such a
prior. So contexts can be in-family in every respect the training distribution controls, while
still satisfying Assumption 1.

It also forces us to confront the noise-scale issue rather than inherit it — see T3.1.

### T1.2 The ladder — 60 contexts per rung, matching the existing 12×5 convention

| Rung | Contexts | Assumption 1 | Instruments |
|---|---|---|---|
| A | TabICL prior, Gaussian terminal noise forced | holds | A1 (circulation + FD), A2, A3, value channel |
| B | TabICL prior, **native** noise mechanism | violated | circulation + value channel only; A2/A3 reported as uninterpretable, not as failures |
| C | Prior with mechanisms perturbed (deeper DAG, OOD feature distributions) | holds | full |
| D | GP-drawn `n=100, d=5, σ=1, ℓ=1` — **the existing Phase 1–2 family** | holds | full (already have 5 seeds; extend to 60) |
| E | Real OpenML contexts, `n_ctx = 100` | assumed | full |
| F | Degenerate: 10 duplicated rows; near-identity regime | holds | full, plus non-degeneracy |

Rung D sits in the **middle** of this ladder, not at the friendly end. Say so in the paper:
the existing audit family is neither in-family nor realistic.

### T1.3 The in-family-ness coordinate

Per context, compute the model's **own prequential log marginal likelihood**
`log q(y_{1:n} | X)`, chained through its PPD in a fixed random row order (average over 5 orders;
report the spread, since TabICL is not exactly exchangeable).

This is the natural x-axis because it estimates the same `log p(y)` whose gradient field the audit
tests — Tweedie's potential and the dose–response abscissa become the same object. Also record
held-out NLL as a second, more conventional coordinate.

### T1.4 Controls, on every rung

Existing: exact GP (wrapped + unwrapped), hierarchical GP, targeted imitator.
**Two new controls, both required:**

- **Noise-hyperprior GP.** Prior mixes over `σ²` (log-uniform, e.g. `[0.25, 4]`), lengthscale
  fixed or mixed. This supplies the **class-level floor for A1**, which the current paper does not
  have — its hierarchical GP mixes lengthscale only, at fixed `σ²`, where Brown applies directly
  and asymmetry is `~10⁻⁸`.
- **Leave-one-out / self-excluding smoother**, `J = D⁻¹(K − diag K)`. A new catalogue row, and a
  live mechanism candidate for A2 failure. It is row-scaled symmetric by construction
  (`d_i J_ij = K̃_ij = K̃_ji = d_j J_ji`), so it must profile cleanly in the Theorem 18 row system,
  whereas TabICL sits at residual 0.570.

**Registered predictions:**

| ID | Prediction | Consequence if it holds |
|---|---|---|
| P5 | Noise-hyperprior GP, **projected** `negeig < 10⁻¹⁰` on all 60 contexts, every rung | A2 is robust to noise-scale mixing; promotes Theorem 15 from a normaliser statement to a class-robustness statement; TabICL's 0.177 stands untouched |
| P6 | Noise-hyperprior GP, **projected** `asym` mean in `[0.02, 0.12]`, max ≤ 0.45; **ambient** `asym` 0.15–0.36 and **ambient** `negeig` 0.07–0.62 | the honest A1 floor is `~0.05–0.1`, not `10⁻³`. TabICL's 0.580 becomes a ~6–13× excess, not 570×. **Every symmetry ratio in the paper must be re-quoted against this floor.** Also: anyone running a raw ambient test on a PFN is measuring this artifact |
| P7 | LOO smoother: projected `negeig ≥ 0.15` (I measured mean 0.362, max 0.603 at n=60), and row-system residual `≤ 10⁻¹⁰` | if TabICL's negeig is in range but its row residual is not, the smoother row is excluded and the exclusion is sharper than "none of the above" |

### T1.5 Pre-register both readings — **before running T1.2**

Write both result paragraphs into `phase_3_results.md §Pre-registration`, then pick afterwards:

- **Reading 1 — fails in-family.** Rung A projected `asym ≥ 0.4` and `negeig ≥ 0.10`.
  Then the off-family escape route is closed and the verdict is unconditional.
- **Reading 2 — passes in-family, fails off-family.** Rung A `asym ≤ 3×` the P6 floor (≈ 0.15)
  and `negeig ≤ 0.02`, with rungs D/E failing. Then the audit is a **prior-support detector**:
  a parameter-free, label-only test, needing no held-out labels, that fires when you leave the
  model's prior support. This is the more useful paper — do not treat it as the disappointing branch.
- **Monotonicity, either way (P8).** Violation decreases in the T1.3 coordinate; Spearman
  `≤ −0.4` across all contexts pooled. A non-monotone or flat curve is itself the headline and
  must be reported as such, not smoothed.
- **P9.** TabICL's near-identity regime is an off-family artifact: Phase 2 found `trJ` reaching
  99.98–100.03 at `n = 100` on real data. Prediction: in-family `trJ/n ≤ 0.9`. If in-family also
  sits at `trJ/n ≈ 1`, the model interpolates its own prior's contexts and the non-degeneracy gate
  fails at home — a genuinely surprising result, and a reportable one.

### T1.6 Deliverables

- Dose–response figure: violation (circulation-based `asym`, and `negeig`) vs T1.3 coordinate,
  with the P5/P6 control floors drawn as horizontal bands. **This is the paper's headline figure.**
- Per-rung table with all three conditions, controls beside models, `mean ± sd` over contexts.

### T1.7 nano-PFN on the TabICL prior — *stretch, hard time box: 3 days*

Run **only** if T0.1 returns FAIL, or if T1+T2 land early. Precedent exists (nanoTabPFN paired
with open TabICL priors, per Balef et al.).

It buys four things nothing else can: a model whose prior is known exactly, so in-family holds by
construction; **two registered predictions** — trained at *fixed* `σ²`, projected `asym` should
decay toward the instrument floor, trained with *mixed* `σ²`, it should plateau near the P6 floor;
a training-compute sweep, which turns the audit from a verdict into a **progress measure** (does
structural violation shrink as the model approaches the PPD?); and closure of the
one-checkpoint-per-model limitation. If the time box expires, stop and log the partial result.

---

## 4. Tier 2 — downstream, with a manipulated cause

TabICL only. The point is no longer to measure the violation's correlates; it is to **dial it and
to remove it**.

### T2.1 The registered ordering

Greedy depends on the posterior mean alone; EI depends weakly on coherent updating;
**knowledge gradient depends on the tower property explicitly** — and KG was never run in Phase 2,
while the martingale drift result predicts it is the one that breaks.

**P10:** normalised-regret gap between TabICL and a refitted GP widens monotonically
`greedy < EI < KG`.
**P11:** the sign-violation rate is roughly *flat* across acquisitions (it is a property of
sequential updating, not of the acquisition), while the regret gap grows. If both move together,
the effect is acquisition-specific and P10 means less than it appears.

### T2.2 Arms as a ladder, each rung structurally explained

1. native predictive head
2. split conformal (Phase 2's winner; distribution-free, hence immune — keep it, it is the practical recommendation)
3. **martingale posterior (Nagler & Rügamer)** — non-negotiable. It is the field's own published
   fix for exactly this defect, motivated in print by PFN PPDs conflating epistemic and aleatoric
   uncertainty. Omitting it is the first thing a PFN reviewer will notice.
4. **potential-repaired field** (T2.3)
5. refitted GP (reference)

Keep the Jacobian-derived variance arm for completeness. It will lose again; that is fine now that
it is not carrying the section.

### T2.3 The repair — the causal clincher

Fit a local scalar potential `φ̂` by least squares against the measured field `(m(y) − y)/σ²` over
a neighbourhood, and run `m̃(y) = y + σ²∇φ̂(y)`. In the loop the context is `n ≈ 10–50`, so this is
cheap. Report **violation reduction and outcome change together.**

**P12:** the repaired field cuts the sign-violation rate by ≥ 50% relative to native and does not
worsen normalised regret. If it cuts the violation and regret is unchanged, that is a clean null and
it *bounds* the practical cost of the defect — report it that way rather than burying it.

### T2.4 The manipulation

Stratify every downstream metric by the T1.3 in-family-ness coordinate of the starting context.
**P13:** downstream cost tracks the coordinate for TabICL and is flat for the refitted GP.
This, not T2.1, is the experiment that answers "so what", because the dial is ours.

### T2.5 Power and budget

Target: 3 acquisitions × 4 arms × 8 objectives × 15 trajectories, `T = 40`.
KG is far more expensive (inner fantasies): run it on 5 objectives × 10 trajectories with 8
fantasies, and say so. Spend the *registered* comparisons first; put leftover budget into
trajectories, not into new objectives. Phase 2's `S = 5, T = 20` is the binding constraint on this
section — it is compute, not design.

Lead the write-up with the shrinkage result (6.0–7.8% sign violations vs 1.0% for a refitted GP;
median absorption 0.55 vs 0.99). It needs no Jacobian to state and no instrument to trust.

---

## 5. Tier 3 — theory that must land with the experiments

### T3.1 Unknown noise scale `(f, σ²)` — **blocking for every A1 number we quote**

Assumption 1 fixes `σ²` known. The prior these models were trained on does not. So the Bayesian
class a reviewer will invoke is posterior means under a joint prior on `(f, σ²)`, which is strictly
larger, and Brown's identity does not apply to it directly.

Derivation to write up (verified numerically at `n = 60–100`; label it derivation + numerics, and
prove it properly):

```
h(y) = E[σ⁻² f | y],     w(y) = E[σ⁻² | y]
Cov(σ⁻²(y − f) | y) = w(y) I + ∇² log p(y)   ⪰ 0,  symmetric
J_h = Cov(σ⁻²(y − f) | y) + y ∇w(y)ᵗ
```

The contamination is **rank-one along `y`**, and `span{1, u} = span{1, y}`, so the existing
projection `Q` annihilates it on the left, and tangent-sphere loops (T0.3) never move along it.
Numerically: for a `σ²`-mixing exact posterior mean with `‖J−Jᵗ‖_F/‖J‖_F = 0.287` ambient, the
skew component measured 2.2×10⁻³ in planes `⊥ span{1,y}` against 6.3×10⁻² in planes containing `y`
— a ~30× suppression. Prove the tangent-plane statement; it is what makes both the projection and
the loop test the right instruments for a second, independent reason.

Then state the consequence honestly: **the reported A1 floors are wrong by ~10 orders of magnitude
at the class level**, and re-quote every symmetry ratio against the P6 floor.

### T3.2 Circulation theorem

`I/(πr²) = aᵗ(J − Jᵗ)b`; wrapper-invariance on constant-`(ȳ, r)` loops; Stokes form for nonlinear
maps; Frobenius recovery from random planes; quantisation robustness; and the tangent-plane
blindness to noise-scale mixing from T3.1.

### T3.3 Housekeeping

Finish the Theorem 3 and Theorem 5 proofs. **Decide M0 now:** either run the radial–tangential
curvature gate with the geodesic correction and an amplitude sweep, or move Theorems 7/9/10 to an
appendix. Floating theory is worse than absent theory. Given the Phase 3 budget, my
recommendation is: move them, and say the hedging interpretation is untested.

Also build the Appendix A provenance table (paper section → source document section) as you go,
not at the end.

---

## 6. `phase_3_results.md` — write it *while* you work, not after

Create it at the start of T0 and append after **every** completed task. This is the main defence
against drift and confabulation: if a number is not in the file with a path beside it, it does not
exist.

Required structure:

```
§0  Provenance          model version, checkpoint hash, commit, env, prior sampler version
§1  Pre-registration    T1.5 and every P-prediction, timestamped BEFORE the runs
§2  Task log            one block per task, in completion order
§3  Decisions           each deviation from this document, with reason
§4  Surprises           anything unexpected, including things we will not chase
§5  Open threads        off-plan ideas, parked
§6  Retractions         quantities withdrawn, superseded conventions, corrected references
§7  Provenance table    paper section → script → output file
```

Every `§2` block carries: task ID; date; script path; exact command; output artifact path; the raw
numbers; the verdict against the stated acceptance criterion; and the sanity checks that were run.

**Tag every quantity** `MEASURED` / `DERIVED` / `ASSUMED`. Never write a derived number without the
expression that produced it. If a "known" Phase 1–2 value cannot be regenerated from the tracked
arrays, say so rather than reconstructing it — Phase 2 already has one such case (the rotary-embedding
paired result), and it must stay flagged.

---

## 7. Sanity checks and halt conditions

**Run these on every measurement path, and log them:**

- Controls read zero: exact GP wrapped/unwrapped through the identical pipeline, every rung.
- Transpose and permutation invariance of the estimator (these close the one bug class that would
  manufacture exactly the signature we report).
- Null perturbation returns exactly 0.
- Basis invariance across ≥ 3 `Q` seeds.
- Analytic-map round trip: finite-difference every control against its stated closed-form
  Jacobian (the Phase 2 regression test), and add the two new controls to it.
- Unit hygiene: `σ²` in normalised vs original target units; state which coordinate system each
  table is in. Phase 2's Table 12 reports an exact-GP slope of 0.25 where the law wants `σ² = 1`;
  do not repeat that.
- Leakage: features standardised on context rows only; no test statistics touching the fit.
- Loop-vs-FD agreement (P1) re-checked on each rung, not just once.
- Anything that looks too clean: state the bug hypothesis and the falsification you ran.

**Halt and write to `§4 Surprises`, then stop and wait:**

- **H1** T0.2 determinism fails (relative difference in `asym`/`negeig` > 1%).
- **H2** T0.3 P1 reconciliation is off by > 2× and neither estimator can be shown correct.
- **H3** Any control (exact GP, wrapped or not) manufactures `negeig > 10⁻⁶` on real or in-family
  contexts — the instrument would then be generating the signature we report.
- **H4** The in-family generator cannot produce contexts satisfying Assumption 1 at all.
- **H5** Compute overruns the §1 budget by more than 2× with Tier 2 incomplete.

Everything else: take the specified branch, log it, keep going. Do not stop to ask.

---

## 8. Ordering, with what runs in parallel

**Week 1**
- T0.1, T0.2, T0.4 concurrently (T0.4 is CPU).
- T0.3 implementation + analytic controls + P1–P4 reconciliation.
- T3.1 derivation — pure theory, starts immediately, blocks nothing.
- T1.1 generator, as soon as T0.1 resolves.
- T1.5 pre-registration written and timestamped **before** T1.2 runs.

**Week 2**
- T1.2 ladder (rungs A–F), T1.3 coordinate, T1.4 controls. This is the bulk of the GPU time.
- T1.6 headline figure.
- T3.2 write-up in parallel.

**Week 3**
- T2.1–T2.4, ordered: shrinkage/sign-violation first (cheapest, most interpretable), then the
  arm ladder, then the acquisition ordering, then the repair.
- T2.5 power spent on whatever budget remains.
- T3.3, provenance table, results file consolidation.

**T1.7 only** if T0.1 failed, or if the above lands early.

---

## 9. What Phase 3 does *not* do

- No TabPFN v2 or TabSwift re-audit. TabSwift moves to an appendix in the paper: it fails the
  wrapper form, so each of its three verdicts needs a separate caveat.
- No new mechanism-exclusion battery beyond adding the LOO-smoother row (T1.4). The calibrated
  exclusion (Phase 2's E2.6) is Phase 4.
- No classification, no non-diagonal noise structures, no decay-law measurement where A1/A2 fail.
- No claim that these models are poor predictors. Rung A–F value-channel numbers exist precisely
  so that the structural verdict cannot be confused with a performance verdict.