# Phase 3 — Results

**Live document. Written as the work happens, appended after every completed task.**
If a number is not in this file with a path beside it, it does not exist.

Plan: [`plan.md`](plan.md). Scope: **TabICL only** (plan §0). TabPFN v2 and TabSwift appear only
where a shared-context number falls out for free.

Every quantity is tagged `MEASURED` / `DERIVED` / `ASSUMED`.

---

## §0 Provenance

**Started** 2026-09-02. **Repo commit at start:** `40db70d`.

### Environment

| | |
|---|---|
| python | 3.10.12 |
| torch | 2.11.0+cu130 (CUDA 13.0) |
| numpy | 2.2.6 · scikit-learn 1.7.2 |
| **tabicl** | **2.1.1** |
| tabpfn | 8.0.8 (present; not audited in Phase 3) |
| GPU | NVIDIA RTX 6000 Ada Generation — **`CUDA_VISIBLE_DEVICES=0`** for all Phase 3 runs |

### The audited object, pinned  `MEASURED`

| | |
|---|---|
| HF repo | `jingang/TabICL` (`tabicl/_sklearn/regressor.py:298`) |
| checkpoint | `tabicl-regressor-v2-20260212.ckpt` |
| snapshot | `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` |
| **sha256** | `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a` |
| size | 114 324 594 bytes |

**Regression-head configuration**, from the checkpoint's own `config` block:
`max_classes 0` (the regression switch) · `num_quantiles 999` · `embed_dim 128` ·
`icl_num_blocks 12`, `icl_nhead 8` · `col_num_blocks 3`, `col_nhead 8`, `col_num_inds 128`,
`col_target_aware true` · `row_num_blocks 3`, `row_nhead 8`, `row_num_cls 4`,
`row_rope_base 100000` · `ff_factor 2` · `dropout 0.0` · `activation gelu` · `norm_first true` ·
`bias_free_ln true`.

**Name check (plan §1).** The release *does* call it v2 — `tabicl-regressor-v2-20260212.ckpt`, against
a `v1` dated `20250208`. **Phase 1–2's "TabICL v2" is correct. No rename.**

**Correction carried forward.** The head emits **999** quantiles, not 9999. Phase 1–2 requested a
9999-point `alphas` grid, which is a caller-supplied *interpolation* grid
(`_sklearn/regressor.py:480-503`), not the head width. The Phase 1 variance channel is unaffected —
its extractor was validated against the exact GP's closed form at 0.31% — but the paper must say
"999-quantile head, integrated on a 9999-point grid". Logged in §6.

### Prior sampler  `MEASURED`

Full analysis: [`provenance_prior.md`](provenance_prior.md). Verdict **PARTIAL** — see §2/T0.1.

---

## §1 Pre-registration

**Timestamped 2026-09-02, before any Tier 1 context was generated or any ladder number measured.**
T0.2 (determinism) and T0.1 (provenance) were already running or complete; neither touches Tier 1.

Predictions are targets. **Deviations get reported, not tuned toward.**

### T1.5 — both readings, written before the run

Whichever fires, the paragraph below is the one that goes in the paper. Picking afterwards is
allowed; rewriting afterwards is not.

> **Reading 1 — fails prior-family.** *Criterion: rung A projected `asym ≥ 0.4` **and**
> `negeig ≥ 0.10`.*
>
> TabICL's structural violations persist on contexts drawn from its own released prior. The
> off-family escape route is closed: the violation is not off-support approximation error, because it
> is present on support. The verdict from Phase 1 becomes unconditional, and the parameter-free
> conditions are a property of the model rather than of the context family it was audited on.

> **Reading 2 — passes prior-family, fails off-family.** *Criterion: rung A `asym ≤ 3×` the P6
> control floor (≈ 0.15) **and** `negeig ≤ 0.02`, with rungs D/E failing.*
>
> TabICL satisfies the structural conditions on its own prior's support and violates them off it.
> The audit is then a **prior-support detector**: a parameter-free, label-only diagnostic that needs
> no held-out labels and no ground truth, and that fires when a deployment context leaves the
> model's training distribution. **This is the more useful paper and is not to be written as the
> disappointing branch.**

**Neither criterion met** → report as *inconclusive between the two readings* and lead with the
dose–response curve (P8) instead of a binary verdict.

### Registered predictions

| ID | Prediction | Falsifier / consequence |
|---|---|---|
| **P1** | Loop-recovered `‖J−Jᵀ‖_F` on TabICL's audit contexts agrees with the finite-difference value within **20%** | disagreement > 2× → one estimator is wrong and identifying which becomes the task (**halt H2** if neither can be shown correct) |
| **P2** | Exact GP **behind the normaliser**, no projection: `\|I\|/(πr²) ≤ 1e-8` | any larger → the wrapper-invariance argument is wrong |
| **P3** | Loop estimate changes **≤ 5%** as an artificial output quantum sweeps `1e-5 → 1e-2`, where central differences at `t=1e-3` collapse to exactly 0 by `δ=1e-3` | loop degrades comparably to FD → the quantisation-immunity claim fails |
| **P4** | Frobenius recovery from **40 random planes** within 20% of the true centred `‖J−Jᵀ‖_F` on analytic maps | worse than 50% |
| **P5** | Noise-hyperprior GP, **projected** `negeig < 1e-10` on all 60 contexts, every rung | holds → A2 is robust to noise-scale mixing; promotes Theorem 15 from a normaliser statement to a class-robustness statement, and TabICL's 0.177 stands untouched |
| **P6** | Noise-hyperprior GP, **projected** `asym` mean in `[0.02, 0.12]`, max ≤ 0.45; **ambient** `asym` `0.15–0.36`, **ambient** `negeig` `0.07–0.62` | holds → the honest A1 floor is `~0.05–0.1`, not `1e-3`. TabICL's 0.580 becomes a **~6–13× excess, not 570×**, and **every symmetry ratio in the paper is re-quoted against this floor** |
| **P7** | LOO smoother: projected `negeig ≥ 0.15`, row-system residual `≤ 1e-10` | if TabICL's negeig is in range but its row residual is not, the smoother row is excluded and the exclusion is sharper than "none of the above" |
| **P8** | Violation decreases in the T1.3 in-family-ness coordinate; Spearman **`≤ −0.4`** pooled | a non-monotone or flat curve **is itself the headline** and is reported as such, not smoothed |
| **P9** | Prior-family `trJ/n ≤ 0.9` | if prior-family also sits at `trJ/n ≈ 1`, the model interpolates its own prior's contexts and the non-degeneracy gate fails at home — surprising and reportable |
| **P10** | Normalised-regret gap between TabICL and a refitted GP widens monotonically `greedy < EI < KG` | — |
| **P11** | Sign-violation rate roughly **flat** across acquisitions while the regret gap grows | if both move together, the effect is acquisition-specific and P10 means less than it appears |
| **P12** | The potential-repaired field cuts the sign-violation rate by **≥ 50%** vs native and does not worsen normalised regret | if it cuts the violation and regret is unchanged, that is a clean null that **bounds the practical cost** of the defect — reported that way, not buried |
| **P13** | Downstream cost tracks the T1.3 coordinate for TabICL and is **flat** for the refitted GP | this, not P10, is the experiment that answers "so what", because the dial is ours |

---

## §2 Task log

### T0.1 — Prior and checkpoint provenance · **PARTIAL** · 2026-09-02

**Deliverable:** [`provenance_prior.md`](provenance_prior.md).
**Method:** read installed source at `.venv/lib/python3.10/site-packages/tabicl/` (v2.1.1) and the
checkpoint on disk. No network, no memory.

**Findings** `MEASURED`:

1. **Continuous targets exist in the released prior components.** `Reg2Cls.__init__`
   (`prior/_reg2cls.py:281-296`) sets `class_assigner = None` iff `num_classes == 0`, and
   `forward` (`:322-327`) then only standard-scales `y`. Verified end to end:

   ```
   mlp_scm    X (100,10)  y unique = 100  mean -0.0000  std 1.0000   CONTINUOUS
   tree_scm   X (100,10)  y unique = 100  mean +0.0000  std 1.0000   CONTINUOUS
   ```

2. **The public sampler cannot reach that path.** `PriorDataset` re-samples `num_classes` at
   `prior/_dataset.py:650-654` *after* spreading `fixed_hp`, and constructing
   `PriorDataset(max_classes=0)` — the regression setting — **raises
   `ValueError: low >= high`** from `np.random.randint(2, 0+1)` at `_dataset.py:652`.
   There is **no `max_classes == 0` branch anywhere in that file** (all 17 occurrences checked).

3. **The checkpoint says `max_classes: 0`**, and the training driver passes one `max_classes` to
   *both* the model (`train/_run.py:172`) and the prior (`train/_run.py:236`). So the run that
   produced this checkpoint handed `0` to `PriorDataset` — **which the released `PriorDataset`
   cannot accept.** The training code path is not the shipped code path.

4. **No training metadata in the checkpoint.** Two top-level keys only, `state_dict` and `config`;
   `config` is model architecture, with no prior version, sampler hyperparameters, or commit.

5. **The noise mechanism is per-layer, not terminal, and is not controllable.**
   `prior/_mlp_scm.py:201-214` appends `GaussianNoise(noise_std)` **inside every SCM layer**;
   `_tree_scm.py:180-188` documents the same. The SCM's `y` is therefore *not* `f + ε` with a known
   `σ²`, and no exposed knob makes it so. Magnitude (`noise_std`) is exposed; mechanism is not.

**Verdict: PARTIAL.** Continuous targets: yes. Prior provably the training prior: **no**. Noise
mechanism controllable: **no**.

**Binding consequences, applied from here on:**
- The word **"in-family" is not used in Phase 3**. Rung A is **prior-family**.
- Every rung-A claim carries: *"contexts drawn from the released TabICL prior, which we could not
  verify is the exact pre-training prior for this checkpoint."*
- **A second caveat the plan did not anticipate:** the released `PriorDataset` cannot produce the
  regression setting at all, so rung A is assembled from the prior's **components**
  (`MLPSCM`/`TreeSCM` → `Reg2Cls(num_classes=0)`) rather than from its public sampler. That
  composition is our reconstruction of the documented pipeline, and is stated as such.
- **Rung A's terminal Gaussian noise is ours, not theirs.** Rung B exists to show what the native
  mechanism does.
- **T1.7 (nano-PFN) is not triggered** — that branch fires on FAIL. It stays a stretch item, and
  PARTIAL *raises* its value: a model whose prior is known exactly by construction is the only way
  to close the gap this finding opens.

**Sanity checks run:** both SCM types exercised, not just the default; `num_classes=0` verified to
produce >50 unique targets rather than assumed; the `PriorDataset(max_classes=0)` failure reproduced
rather than inferred from reading; all 17 `max_classes` occurrences enumerated before claiming no
branch exists.

---


### T0.2 — Determinism of the audit path · **PASS (bit-identical)** · 2026-09-03

**Script:** [`src/t0_2_determinism.py`](src/t0_2_determinism.py)
**Command:** `CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/phase_3/src/t0_2_determinism.py`
**Outputs:** `results/t0_2_determinism.json`, `results/t0_2_predictions.npz`, `logs/t0_2.log`
**Runtime:** ~13 min GPU (2 passes x 5 contexts x 196 fits).

Closes Phase 2's E0.1 gap. Two *independent model loads*, five audit contexts, `t=1e-3`, no dither,
`Q = get_Q(y, seed=0)`; differenced at `m(y)`, at `QᵀJQ`, and at the reported scalars.

`MEASURED` — all five seeds:

| seed | `max\|Δm\|` | `‖ΔJ‖_F/‖J‖_F` | `asym` rel diff | `negeig` rel diff | bit-identical |
|---|---|---|---|---|---|
| 42 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | **`m` and `J`** |
| 100 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | **`m` and `J`** |
| 200 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | **`m` and `J`** |
| 300 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | **`m` and `J`** |
| 400 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | **`m` and `J`** |

**Verdict: PASS**, and by a wide margin — the gate was `≤ 1%` relative and the result is **exactly
zero**. Not "within tolerance": the forward pass and the full 98×98 reduced Jacobian are
**bit-identical across independent model loads**. Halt H1 does not fire. `MEASURED`

**Sanity check, and it is a strong one.** Pass 1's five `asym` values are
`0.606662 / 0.532994 / 0.705353 / 0.470117 / 0.586529` and `negeig`
`0.241642 / 0.159460 / 0.220369 / 0.129146 / 0.132684`. These reproduce
`FINAL_NUMBERS.md` §3.1 and §4.1 **to every digit reported there**. So the audit path survives the
Phase 1→3 directory restructure unchanged, and TabICL's headline numbers regenerate from scratch on
demand. `MEASURED`

**What this licenses.** Every Phase 3 violation is signal, not jitter. It also retires a standing
Phase 1–2 caveat: E0.1 was listed as NOT MEASURED and is now measured, with the strongest possible
outcome.

---

### T0.3 part 1 — Circulation instrument, analytic controls · 2026-09-03

**Script:** [`src/circulation.py`](src/circulation.py) (the instrument),
[`src/t0_3_circulation_controls.py`](src/t0_3_circulation_controls.py) (the controls)
**Command:** `.venv/bin/python experiments/phase_3/src/t0_3_circulation_controls.py` — **CPU only**
**Output:** `results/t0_3_controls.json`

#### Identity and sign convention `MEASURED`

The loop recovers `aᵀ(J−Jᵀ)b` on an affine map to **max relative error `5.23e-15`** over 5 random
planes, and is already exact at `N = 16` quadrature points — the integrand of an affine map is a
degree-2 trigonometric polynomial, so the trapezoidal rule is exact, not merely spectrally accurate.

**Sign correction to the plan.** The plan states `I/(πr²) = aᵀ(J−Jᵀ)b`. Carrying out the integral
with `y(t) = ȳ1 + r(cos t·a + sin t·b)` gives `I = πr²(bᵀJa − aᵀJb) = −πr²·aᵀ(J−Jᵀ)b` — the
**negative**. This is an orientation convention, not a result. `circulation.py` pins the sign to
match `aᵀ(J−Jᵀ)b` and the identity check asserts it, so the convention cannot drift. `DERIVED`

#### P2 — wrapper invariance · **PASS by 8 orders of magnitude** `MEASURED`

Exact GP **behind the normaliser**, **no projection**, 8 random planes on each of 5 audit contexts:

```
seed 42  2.464e-16   seed 100 2.472e-16   seed 200 1.712e-16
seed 300 2.150e-16   seed 400 3.881e-16          worst 3.881e-16
```

Gate `≤ 1e-8`. **Measured `3.9e-16`.** The N1 confound — an `O(1)` asymmetric rank-one term that
forces every Jacobian-based test through a projection — **contributes exactly nothing to the loop**.
This is the result that lets the circulation instrument drop the projection, the dither, and the
per-model probe amplitude all at once.

#### P3 — quantisation immunity · **PASS** `MEASURED`

An affine map with known asymmetry, quantised at increasing output steps. Loop at `N=64` against
central differences at `t=1e-3`, both on the same plane, against the same truth:

| output quantum `δ` | loop rel err | **FD rel err** |
|---|---|---|
| none | `7.1e-16` | `2.7e-12` |
| `1e-5` | `2.8e-06` | `2.9e-02` |
| `1e-4` | `1.6e-05` | **`2.80`** |
| `1e-3` | `4.1e-04` | **`0.88`** |
| `1e-2` | `2.1e-03` | **`26.9`** |

Gate: loop changes `≤ 5%` across the sweep. **Measured `0.21%`.** Over three decades of output
quantum the loop moves by a fifth of a percent while the finite difference reaches **2690% error**.

*Deviation from the plan, reported not tuned:* the plan predicted FD would "collapse to exactly 0 by
`δ=1e-3`". It does not reach exactly zero — it reads `0.00175` against a truth of `0.01462`, an 88%
error. The substance (FD destroyed, loop intact) holds overwhelmingly; the specific wording does not.

#### P4 — Frobenius recovery · **target missed on 1 of 9; falsifier NOT fired** `MEASURED`

40 random planes, 3 analytic maps × 3 seeds, against the true centred `‖J−Jᵀ‖_F`:

| map | rel err by seed |
|---|---|
| wrapped exact GP (truth `0`) | `0.000`, `0.000`, `0.000` |
| Nadaraya–Watson | `0.239`, `0.187`, `0.059` |
| asymmetric affine | `0.085`, `0.001`, `0.041` |

Worst `0.239` against a `0.20` target. **The registered falsifier is "worse than 50%" and did not
fire.**

**Diagnosis, run before accepting the number** (`MEASURED`, 5 restarts per setting on the worst
case): the error is **pure Monte Carlo and the estimator is unbiased**. Predicted relative SE is
`1/(2√n_planes)`; observed:

| planes | predicted rel SE | observed rel err, 5 restarts |
|---|---|---|
| 40 | `0.112` | `0.084, 0.124, 0.010, 0.084, 0.093` |
| 80 | `0.079` | `0.012, 0.087, 0.042, 0.009, 0.074` |
| 160 | `0.056` | `0.042, 0.054, 0.067, 0.012, 0.039` |
| 320 | `0.040` | `0.003, 0.053, 0.021, 0.045, 0.034` |

Observed tracks theory across a factor of 8 in sample size. The `0.239` was a ~2σ draw from a
40-plane estimate. **The prediction is not changed and the gate is not moved**; the operating point
for model measurements is set with the variance now characterised (§3, D5).

**Correction to the plan's Frobenius constant.** The plan states
`E[(aᵀ(J−Jᵀ)b)²] = 2‖J−Jᵀ‖_F²/(m(m−1))`. That is a **factor of 2 too large**. For antisymmetric `A`
and random orthonormal `a, b` in an `m`-dimensional space, `aᵀAb = (OᵀAO)₁₂` for random orthogonal
`O`; `OᵀAO` is antisymmetric with the same Frobenius norm and its `m(m−1)` off-diagonal entries are
exchangeable, so `E[(aᵀAb)²] = ‖A‖_F²/(m(m−1))`. Measured empirically over 4000 planes at `n=60`:
**`c = 1.036`** against `1.0` predicted (Monte Carlo SE ≈ 0.022, so within 1.5σ) and `0.5` claimed.
`circulation.py` uses `c = 1`. `DERIVED` + `MEASURED`


### T1.1 — Prior-family context generator · **built and self-tested** · 2026-09-03

**Script:** [`src/prior_family.py`](src/prior_family.py) (module + self-test). CPU.

Drives `MLPSCM` / `TreeSCM` → `Reg2Cls(num_classes=0)` directly, which is the composition
`SCMPrior.generate_dataset` performs, because `PriorDataset` cannot be constructed in the regression
setting (T0.1). Returns `(X, y, f, meta)` with features standardised on context rows only.

- **rung A** — `f` := the SCM's continuous output; `y := f + ε`, `ε ~ N(0, σ²I)`, `σ = 1.0`.
  Assumption 1 **holds**, `σ²` recorded per context.
- **rung B** — `y :=` the SCM's native output, internal layer noise only. Assumption 1 **violated**.

`σ = 1.0` on a unit-variance latent is chosen so rung A matches the Phase 1 audit family's noise
level exactly, leaving **the prior over `f`** as the only difference between rungs A and D.

**Self-test** `MEASURED`, 40 draws:

```
prior_type mix   mlp_scm 29 / tree_scm 11   (72.5% / 27.5% against the prior's own 0.7 / 0.3)
d_kept           min 3, max 20, mean 11.0   (drawn by the prior's own sampler)
rung A SNR       0.91 - 1.15                (matches rung D's design SNR of 1)
f_std            0.995  (Reg2Cls standard-scales)      y_std 1.25 - 1.55
attempts         1 for every draw -- no rejection needed
```

Meta recorded per context: `prior_type`, `d_drawn`, `d_kept`, `σ²`, `num_layers`, `hidden_dim`,
`is_causal`, `scm_noise_std`, `sampling`, and an explicit `assumption1` flag.

---

### T0.3 part 2 — P1 reconciliation · *running* · 2026-09-03

**Script:** [`src/t0_3_reconcile_tabicl.py`](src/t0_3_reconcile_tabicl.py)
**Command:** `CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/phase_3/src/t0_3_reconcile_tabicl.py`
**Expected output:** `results/t0_3_reconcile_tabicl.json`

64 planes × 32 loop points = **2048 forward passes per context**, 5 contexts, plus a 4-point radius
sweep on seed 42. Planes drawn orthogonal to **both** `1` and the centred `y`, so the loop lives in
exactly the subspace `Q` spans and the loop and finite-difference numbers estimate the same object.
A bootstrap CI over the plane samples is reported beside each point estimate, because P1 compares two
*estimates* and a point-to-point comparison would be the wrong test.

The radius sweep is the check that matters for interpretation: TabICL is not affine, so by Stokes the
loop returns the **area-average** of the antisymmetric 2-form over the enclosed disc while the finite
difference returns a **point value**. If the loop value moves with `r`, the two estimators are
measuring genuinely different things and that must be said rather than averaged over.

Result appended on completion. Runtime is running longer than the ~45 min estimate.


### T0.4 — Literature artifact · **structural repair done; channel column outstanding** · 2026-09-03

**Script:** [`src/t0_4_repair_literature.py`](src/t0_4_repair_literature.py)
**Outputs:** `results/t0_4_literature_audit.json`,
`literature/literature_extraction_all_REPAIRED.md` (the original is **not** modified)

#### Two of the plan's three stated defects are wrong `MEASURED`

The plan describes: camp `a` claims 12 files and lists 10 while 20 `##` sections follow; camp `c` has
no manifest and no date; **camp `b` is absent entirely**.

| plan says | actually |
|---|---|
| camp `b` is absent entirely | **`b` is present, at line 772.** Its header is `Camp directory: b` — *plain text, not a `#` heading* — so a `^#` scan finds only 2 of the 3 camps. That same malformed header is why camp `a`'s region appeared to hold 20 sections: the `^#`-delimited block starting at line 1 runs straight through camp `b`'s papers. |
| camp `c` has no manifest and no date | **It has both**, lines 1617–1632. |
| camp `a` claims 12, lists 10 | **True**, and the true count is 11. |

**Ground truth is the directory listing:** `literature/{a,b,c}/*.md` — **11 files each, 33 total.**

#### The defects that are real `MEASURED`

| camp | manifest claims | manifest lists | sections in block | **true** |
|---|---|---|---|---|
| a | 12 | 10 | 10 | **11** |
| b | 11 | 10 | 10 | **11** |
| c | 11 | 11 | **13** | **11** |

**Two papers were filed under camp `c` that belong elsewhere:**

- line 2544 — *What Can Transformers Learn In-Context? A Case Study of Simple Function Classes* → camp **a**
- line 2617 — *A Mechanistic Study of Tabular Foundation Models* → camp **b**

Both camp-`a` shortfalls are explained by these: `10 + 1 = 11` for each of `a` and `b`, and
`13 − 2 = 11` for `c`. **The counts close exactly.**

Two sections also carry a *printed* title differing from their filename, so neither exact nor prefix
matching finds them; they are mapped by an explicit alias table rather than by loosening the matcher,
which would risk silent mis-assignment elsewhere. The script **refuses to write** if any section is
unassigned, rather than dropping it.

#### Repair verified `MEASURED`

```
camp headers   3   (all now `#` headings)
sections      33   against 33 files on disk        OK
unmatched      0
content       all 33 section bodies byte-identical to the original
```

#### The von Oswald question — **answered, and it does not weaken C-X3** `MEASURED`

The plan flags that von Oswald et al.'s extraction records *"comparison of the partial derivatives of
the transformer predictions against the analytical sensitivities of gradient descent steps"*, and asks
whether that is w.r.t. inputs/parameters (expected) or **labels** (which would weaken the gap claim).

Read from the **source**, `literature/a/Transformers learn in-context learning by gradient descent.md`:

- line 189 — the compared sensitivities are `∂ŷ_θ(x_τ,test)/∂x_test` and
  `∂ŷ_θGD(x_τ,test)/∂x_test`, i.e. **with respect to the test input**.
- line 265, Figure 5 — *"Norm of the partial derivatives of the output of the first self-attention
  layer **w.r.t. input tokens**."*
- line 273 — *"the mean of the norm of the partial derivative of the first layer's output **w.r.t. the
  input tokens**"*.

**It is inputs, not labels.** The sentence *"none states, derives, or measures the label-dependence of
the predictive map"* **stands**.

#### Outstanding

The **channel column** (value / internal-representation / derivative / discrete-perturbation, one per
paper) is not yet added. That is the C-X3 survey artifact proper and requires reading all 33
extractions; the structural repair above is its prerequisite and is complete.


### T3.1 — Unknown noise scale · **derivation proved, numerics confirm** · 2026-09-03

**Deliverable:** [`t3_1_noise_scale.md`](t3_1_noise_scale.md)
**Script:** [`src/t3_1_noise_scale.py`](src/t3_1_noise_scale.py) → `results/t3_1_noise_scale.json`. CPU.

Assumption 1 fixes `σ²` known; the prior these models were trained on does not. The class a reviewer
will invoke is posterior means under a **joint** prior on `(f, σ²)`, which is strictly larger, and
Brown's identity does not apply to it directly.

**Three identities, proved in the deliverable and verified here** at `n = 60`, exact `σ²`-mixing
posterior over a 9-point log-uniform grid on `[0.25, 4.0]`, 5 audit contexts:

| | identity | verification |
|---|---|---|
| (1) | `∇log p(y) = −g(y)`, `g = E[σ⁻²(y−f)\|y]` — generalised Tweedie | max rel err **`7.8e-10`** |
| (2) | `Cov(σ⁻²(y−f)\|y) = w(y)I + ∇²log p(y)` — symmetric, PSD | rel err `3.8e-6` (FD-Hessian limited); **symmetric to `3.1e-15`, min eig `−1.2e-15`** |
| (3) | `J_h = Cov(σ⁻²(y−f)\|y) + y∇w(y)ᵀ` — contamination **rank one along `y`** | `σ₂/σ₁ = **3.5e-10**`; top singular vector aligns with `y/‖y‖` at **`\|cos\| = 1.000000`** on every context |

**The consequence, which is the reason this was blocking.** Since `span{1,u} = span{1,y}`, the
projection gives `QᵀJ_hQ = Qᵀ Cov Q` **exactly** — the contamination is annihilated — and for loop
planes with `a,b ⊥ span{1,y}` the rank-one term contributes
`(aᵀy)(∇wᵀb) − (aᵀ∇w)(yᵀb) = 0`. **Both instruments are exactly blind to noise-scale mixing.** That
is a *second, independent* reason they are the right instruments; the first (N1, the normaliser) was
unrelated.

**And the honest floor** `MEASURED`:

| quantity | value |
|---|---|
| **ambient `asym` of `μ*(y)=E[f\|y]` under `σ²` mixing** | **mean `0.2830`**, range `[0.2418, 0.3380]` |
| mean `\|aᵀ(J−Jᵀ)b\|`, planes ⊥ `span{1,y}` | `3.2e-03` – `4.6e-03` |
| mean `\|aᵀ(J−Jᵀ)b\|`, planes containing `y` | `6.7e-02` – `9.6e-02` |
| **suppression** | **mean `22.2×`**, range `[14.8×, 28.0×]` |

*Deviation, reported not tuned:* the plan states `0.287` ambient and `~30×`. **Ambient reproduces
almost exactly** (`0.2830`); **suppression comes in lower at `22.2×`.** Direction and order of
magnitude hold; the specific factor does not.

**What this does to Phase 1's numbers.** TabICL's A1 of `0.580` is quoted against a
*quantisation-artifact* floor of `0.001–0.004` — `136×–570×`. That floor answers "what does the
instrument manufacture on a map with no asymmetry", **not** "what does a Bayesian in the class the
reviewer means produce". Against the class-level floor, the same `0.580` is a **~2× ambient excess**.
The **projected** comparison survives untouched, because the projection annihilates this
contamination exactly — but **the ambient ratios must not be quoted as if they were comparisons
against the right null.** This is P6 territory; the Tier 1 noise-hyperprior GP is the registered
measurement.

**A2 is unaffected** — the projected block is a compression of a symmetric PSD matrix regardless of
noise-scale mixing, so TabICL's `negeig = 0.177` stands. A1 needs a new floor and A2 does not, which
extends N3's "A2 is the more robust condition" from preprocessing to noise-scale mixing.

---

## §3 Decisions

| # | Decision | Reason |
|---|---|---|
| D1 | Rung A is built from `MLPSCM`/`TreeSCM` + `Reg2Cls(num_classes=0)` called directly, **not** from `PriorDataset` | `PriorDataset(max_classes=0)` raises (T0.1 finding 2). No alternative exists in the released package. |
| D2 | Rung A's `f` is the SCM output; terminal `ε ~ N(0, σ²I)` is **added** by us, `σ²` recorded per context | The prior's own noise is per-layer and nonlinearly propagated, so the SCM output is not `f + ε` with known `σ²` (T0.1 finding 5). Theorem 5 needs only that `P` be *some* prior on `ℝⁿ`; the SCM pushforward is one. |
| D3 | "prior-family" replaces "in-family" everywhere | T0.1 PARTIAL branch, plan §2. |
| D4 | Circulation sign convention pinned to `aᵀ(J−Jᵀ)b`; the plan's stated identity is the negative of what the integral gives | Orientation convention. Derived in `circulation.py`, asserted by the identity check so it cannot drift. |
| D5 | Frobenius constant is `‖A‖_F² = m(m−1)·E[(aᵀAb)²]`, i.e. `c = 1`, **not** the plan's `2‖A‖_F²/(m(m−1))` | Derived from exchangeability of the `m(m−1)` off-diagonal entries of `OᵀAO`; verified empirically at `c = 1.036 ± 0.022` over 4000 planes. The plan's constant is a factor of 2 too large. |
| D6 | Model-facing circulation runs use **64 planes**, not the 40 of P4 | P4's Monte Carlo SE is `1/(2√n_planes)`, measured and confirmed. 40 planes gives ~11% SE, enough to miss a 20% target on a 2σ draw. 64 gives ~9%, and a bootstrap CI over the plane samples is reported beside every point estimate so the comparison is never point-to-point. The registered P4 gate is **not** moved. |

---

## §4 Surprises

| # | What | Chasing? |
|---|---|---|
| S4 | **The class-level A1 floor is `0.283` ambient, not `10⁻³`.** A genuine `σ²`-mixing exact posterior mean — unimpeachably Bayesian — has ambient `asym` in `[0.24, 0.34]`. Phase 1's `570×` symmetry ratios are against a quantisation floor, which is the wrong null for the class a reviewer means. | Not a tangent — it is T3.1's registered consequence and it changes how every A1 ratio is quoted. The projected numbers survive; the ambient ones do not. |
| S1 | **The released `PriorDataset` cannot produce the setting its own regression checkpoint was trained in.** `max_classes=0` raises. This is stronger than "we cannot verify the prior" — the public sampler is provably *not* the one used, at least not at this version. | Not chasing further. Recorded, and it is the reason the PARTIAL caveat is worded as it is. It also strengthens the case for T1.7. |
| S3 | **The plan's description of the literature artifact is wrong in two of three particulars** — camp `b` is present (malformed header, not absent) and camp `c` does have a manifest. The single real structural fault is one plain-text header, which made a `^#` scan under-report. Worth noting because the same failure mode — a grep-shaped assumption about a file's structure — is how the "camp b is missing" belief formed in the first place. | Recorded. Repair done. |
| S2 | The regression head is **999** quantiles, not the 9999 Phase 1–2 implied. The 9999 was a caller-supplied interpolation grid. | Not a defect — the extractor was validated at 0.31% against closed form. Logged as a wording correction in §6. |

---

## §5 Open threads

| # | Idea | Status |
|---|---|---|
| O1 | The `col_target_aware: true` flag in the checkpoint config means the **column embedder sees the target**. That is a plausible mechanical origin for label-asymmetry upstream of the ICL stack, and Phase 1's E1.4 only zeroed *row* RoPE. Zeroing or ablating the target-aware column path is the natural next mechanistic probe. | Parked. Off-plan for Phase 3. |
| O2 | `mix_probs: (0.7, 0.3)` in `DEFAULT_FIXED_HP` sets the mlp/tree mechanism mix. Rung C ("mechanisms perturbed") could dial this directly rather than only deepening the DAG. | Fold into T1.2 rung C if cheap. |

---

## §6 Retractions and corrections

| # | Quantity | Correction |
|---|---|---|
| R1 | "TabICL's regression head emits 9999 quantiles" (implied by Phase 1–2's `alphas` grid) | The head emits **999** (`num_quantiles: 999`, checkpoint config). 9999 was the caller-supplied interpolation grid. No number changes; wording does. |

---

## §7 Provenance table

| paper section | script | output |
|---|---|---|
| §0 checkpoint pin | inline, `provenance_prior.md` §1 | `provenance_prior.md` |
| T0.1 prior provenance | inline source read | `provenance_prior.md` |
| T0.2 determinism | `src/t0_2_determinism.py` | `results/t0_2_determinism.json`, `results/t0_2_predictions.npz` |
