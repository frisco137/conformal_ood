# Deviations log

Companion to the frozen pre-registration [`experiments.md`](experiments.md). One row per experiment,
recording what actually happened against four outcomes:

| code | meaning |
|---|---|
| **RUN** | run as specified, thresholds as written |
| **RUN‑D** | run, with a deviation from the specification — the deviation is stated in the row |
| **NOT RUN** | no output exists |
| **FIRED** | the experiment's own pre-registered falsifier fired |

`experiments.md` is never edited. This file is. Last updated 2026-08-26.

Results: `PHASE1.md`, `PHASE2.md` (repository root). Phase 1 ledger: `FINAL_NUMBERS.md`.

---

## Tier 0 — Instrument validity

| ID | outcome | what happened |
|---|---|---|
| **E0.1** Determinism | **NOT RUN** | No determinism measurement exists on any model. Requires a GPU refit-twice pass on the audit path. The only related evidence (`results.json → 4_gaps`) was taken at `n=20, d=2` with fit-once/predict-twice, which is not the path the audit uses, and is now quarantined. |
| **E0.2** N5 analytic control battery | **NOT RUN** | `tier0_instrument/e0_2_analytic_control.py` exists and covers **6 of 9** N5 rows (exact GP, ridge, NW, 1-NN, hierarchical GP, imitator). **No saved output.** The GD family, Richardson+Jacobi and the prototype readout are not implemented. `experiments.md` calls this "the single highest-value Tier 0 item". |
| **E0.3** Step-size plateau | **RUN‑D** | Run per model, but **not on the `h ∈ {1e-4 … 1e-2}` grid specified**. Each model was swept on its own grid chosen from its measured output quantum: TabICL `1e-3/1e-2/1e-1` (drift `0.21%`), TabPFN `1e-2/3e-2/1e-1/3e-1` (drift `6.6%`), TabSwift `1e-2/1e-1/1e0` (drift `24%`, and `1e-3` is inside derivative collapse). The pass criterion "vary `< 20%` across ≥3 consecutive `h`" **holds for TabICL and TabPFN and does not hold for TabSwift**. `PHASE1.md` §3.2. |
| **E0.4** Autograd cross-check | **NOT RUN** | No code exists. |
| **E0.5** Smoothness scan | **RUN‑D** | A fine scan of `m(y + tq₁)` was run (501 points, seed 42) but **as a quantum measurement, not as a kink hunt**. It produced the output-quantum statistics in `PHASE1.md` §3.1. No kink analysis against bar-distribution bin borders was performed. `exp4_quanta_and_tabicl.py` → `exp4_results.json`. |
| **E0.6** Row-sum residual `r₁` | **RUN** | `PHASE1.md` §1.3. Floor `3.70e-12`; TabICL `0.0099`, TabPFN `0.5114`. **Halt condition `r₁ ≳ 1e-1` was met by TabPFN (`0.51`) and TabSwift (`0.91`)** and the audit did not halt. Justified for TabPFN because E1.1 confirms shift equivariance at `7.3e-7`, so `r₁` is instrument error rather than absence of the property — but the pre-registration did not license continuing, and this is a deviation. For TabSwift `r₁` is **not** an error bound and is not reported as one. |

## Tier 1 — Confound control

| ID | outcome | what happened |
|---|---|---|
| **E1.1** Preprocessing identification | **RUN** + **FIRED for TabSwift** | `e11_wrapper_form.py` → `e11_wrapper_form.json`. TabPFN and TabICL satisfy both identities at `7.3e-7` / `2.2e-6` against a `<1e-3` gate. **TabSwift satisfies neither** (`3.379`, `3.693`). Its target `StandardScaler` is commented out in source. The pre-registered falsifier is worded for a *rank/quantile* transform and did not literally fire, but the underlying condition — an affine wrapper — **is not met**, so N1(ii) is not licensed for TabSwift. `PHASE1.md` §1.5. |
| **E1.2** Projection validation, both directions | **RUN** | (a) wrapped exact GP `asym(J_red) = 3.14e-12`, gate `< 0.02`. (b) wrapped imitator survives at `negeig` mean `0.0845`, gate `≥ 10× A_floor`. Both pass. `PHASE1.md` §1.1–1.2. **See the correction note**: the imitator's *analytic reference* was wrong by a factor of two until 2026-08-26; the measured values never moved. |
| **E1.3** Ensemble-of-one | **RUN‑D** | Recorded only in `tier1_confounds/tier1_report.md` as "PASS (n_estimators=1)". **This is the only record in the tree**, it does not appear in `FINAL_NUMBERS.md`, and it was produced by a script using an **unseeded `get_Q`**. No comparison against default-`K` was reported. Should be re-run through the seeded instrument before the paper cites it. |
| **E1.4** Positional-encoding zeroing | **RUN‑D for TabICL, NOT RUN for TabPFN** | TabICL: paired analysis complete, mean `−0.002233`, 95% CI `[−0.028050, +0.023585]` — **contains zero, RoPE is not the source**. `PHASE1.md` §3.3. **Deviation:** the producing script was never saved, and this is the one orphaned result that **cannot be recomputed** from the tracked arrays (it needs forward passes through a *modified* model). TabSwift has no positional encodings. **TabPFN v2 NOT RUN** — needs source modification of the attention path. |
| **E1.5** M0 radial/tangential gate | **NOT RUN** | Only an Euler-defect ratio `4.1287e-04` exists, in `tier1_report.md`, from a script with an unseeded `Q`. **The great-circle tangential curvature with the geodesic correction was never constructed**, and the validation on the hierarchical GP the specification requires was never done. The pre-registered rule — *if curvature is mostly radial, the hedging interpretation dies* — **has never been evaluated.** |

## Tier 2 — The audit

| ID | outcome | what happened |
|---|---|---|
| **E2.1** Non-degeneracy | **RUN‑D** | Run for TabICL in full (`‖J−I‖_F/‖J‖_F` mean `0.8513`, min `0.5845`; `‖J‖_F` `6.99` vs `√98 = 9.899`). For TabPFN and TabSwift the check was applied at the chosen amplitude only. **TabSwift's `‖J‖_F` collapse `392.8 → 6.09` across the `h` sweep is recorded**, and `t=1e-3` is void for that model. `PHASE1.md` §2. |
| **E2.2** A2 positivity | **RUN** | All three, 5 seeds, against a measured artifact floor. **FAIL** for all three at `22 711×`, `91 381–3 795×`, `887 607–43 339×`. `PHASE1.md` §4. |
| **E2.3** A1 symmetry | **RUN** | All three, 5 seeds. **FAIL** for all three at `134.7×`, `264.5–56.7×`, `569.8–136.3×`. `PHASE1.md` §3. |
| **E2.4** Mechanism profiling | **RUN** | Both systems solved in log space with real residuals. **Neither admits a positive solution with small residual** — `0.476–0.616` against `1.5e-15` on the NW positive control. The pre-registered falsifier "neither → both classes excluded" **fired**, which the specification itself calls the stronger result. `PHASE1.md` §6. **Deviation:** run in *ambient* coordinates, not reduced — reduced-basis profiling was shown to have no power (NW profiles to `0.259940` from a raw `0.249240`). |
| **E2.5** A3 cross-channel | **RUN‑D** + see note | Run for TabICL and TabPFN with the required fixes: the integrated second moment over the full predictive distribution (not the `0.005/0.995` quantile spread), validated against the exact GP's closed form at `0.31%`. **FAIL**, `R² = 0.0199` and `0.1065` against a `≥0.90` gate. **TabSwift: unevaluable**, reported as such, never as failure. **Two deviations:** (i) the parameter-free `cv_s ≤ cv_J` check **holds** for both models and so returns no violation — it is reported as a secondary diagnostic only, because it is permutation-invariant; (ii) **the A3 artifact floor was not known when the threshold was set.** E5.1 later showed a map satisfying A3 *exactly* reads `R²` as low as `0.675` through the normaliser. The measured values are an order of magnitude below that, so the verdict stands, but the gate as pre-registered was not calibrated. `PHASE1.md` §5.0. |
| **E2.6** Mechanism surrogate fitting | **NOT RUN** | **The single most important gap.** The only artefact, `e2_6_mechanism.py`, is **quarantined**: its verdict is a **hardcoded 50% threshold** — precisely the invented tolerance E2.6 exists to avoid — and it is single-seed, single-model, and compares **ambient** Jacobians with no `Q` projection. `experiments.md` calls the resulting scatter "**the headline figure**". It does not exist. Any draft sentence resting on "TabICL's Jacobian differs by over 50% from all fitted surrogates" has **no valid support.** |

## Tier 3 — Robustness

| ID | outcome | what happened |
|---|---|---|
| **E3.1** Context distribution | **NOT RUN** | `experiments.md`: "**Unaddressed, this objection is fatal.**" One context family throughout Phase 1. Phase 2 §1.4 found TabICL's `tr J → 100` on real data against `‖J−I‖_F/‖J‖_F = 0.85` on the audit's contexts — a context-family effect of exactly the kind E3.1 exists to find. **A reason to run it, not a result about it.** |
| **E3.2** In-sample regime | **NOT RUN** | — |
| **E3.3** Context-design sweep | **NOT RUN** | Only the step-size axis was swept (E0.3). `d`, noise level, duplicate count, label distribution and `n` were never varied. |
| **E3.4** Query row vs context block | **NOT RUN** | — |
| **E3.5** Seed and checkpoint variation | **RUN‑D** | Five **seeds** were run everywhere. **No checkpoint variation** — one checkpoint per model throughout. The "one checkpoint" objection is not closed. |

## Tier 4 — The value channel

| ID | outcome | what happened |
|---|---|---|
| **E4.1** Value battery, same model, same contexts | **RUN‑D** | `tier4_value/e4_1_value_battery.py` → `e4_1_results.json`. `PHASE1.md` §8. MSE, NLL, calibration at four levels, coverage, interval score, rank correlation, on the exact audit contexts with five reference predictors. **Pre-registered falsifier did NOT fire**: the models do not perform badly, so the audit is **not** confounded with out-of-distribution failure. **Deviations:** (i) *sequential-consistency / martingale and prequential log loss were not run on the audit contexts* — they exist only in Phase 2, on real data; (ii) held-out queries had to be constructed (GP conditional `f_*\|f`), since the audit contexts ship no test set; (iii) the contexts are near-information-free by design (oracle `R² = 0.1425`), so raw MSE is uninterpretable and everything is reported as efficiency against the oracle. |
| **E4.2** Imitator on the value battery | **NOT RUN** | — |

## Tier 5 — Positive control

| ID | outcome | what happened |
|---|---|---|
| **E5.1** A predictor that passes, on real data | **RUN‑D** | `tier5_positive/e5_1_positive_control.py` → `e5_1_results.json`. `PHASE1.md` §7. **A1 and A2 pass 120/120** across both wrapped controls on 60 real OpenML contexts, `negeig` **exactly zero** throughout. **The blocking gate is cleared.** Hierarchical GP shows nonzero tangential curvature at `8.1e6×` the exact GP. **Deviations, three:** (i) the wrapped hierarchical GP **fails A3 on 3/60**; unwrapped it passes 60/60 at `R² = 1.000000`, locating the cause in the **normaliser**, not the pipeline — this is what established the A3 artifact floor now recorded in `PHASE1.md` §5.0; (ii) lengthscale is the **median heuristic** rather than a fixed `1.0`, chosen to avoid a *vacuous* pass at `d = 25/50` where a fixed kernel is near-diagonal — swept `×0.25–×4`, nothing moves; (iii) the curvature reported is the second difference along `Q`'s columns, **not** the M0 great-circle construction with the geodesic correction. |
| **E5.2** TabPFN v2 vs the Biloš prediction | **⚠ NOT RUN AS A REGISTERED TEST — falsifier fired on other numbers** | **Read this row carefully; it is the one most easily overclaimed.**<br><br>**What is true:** the prediction was **stated in advance** in `experiments.md` — TabPFN v2 and Mitra form an attention-weighted label vote at ~L9 (the `D⁻¹K` row of N5, row-scaled symmetric), so **TabPFN v2 should be materially closer to passing than TabICL v2, with row scaling recovered**. The pre-registered falsifier was: *TabPFN v2 fails identically to TabICL v2 → the readout distinction does not govern derivative structure.*<br><br>**What the numbers say:** TabPFN v2 is not closer to passing — it is **worse**, on both conditions. A1 `1.312` vs TabICL's `0.580`; A2 `0.404` vs `0.177`. Row scaling was **not** recovered: the row system's residual is `0.616` against `1.5e-15` on the NW positive control (`PHASE1.md` §6). **On every axis the prediction names, the result runs the other way.**<br><br>**What is NOT true:** **no registered E5.2 run exists.** These numbers came from the general audit — `exp3_reprobe.py` and `exp2_ambient_profiling.py` — not from a script written to execute E5.2, and `FINAL_NUMBERS.md` §8 lists "E5.2 as a registered test" as not run.<br><br>**The claim the paper may make:** *the audit's measurements are inconsistent with the Biloš readout prediction, which was stated in advance.* **The claim the paper may not make:** *a pre-registered test falsified the Biloš prediction.* The pre-registration is what separates a test from a fishing expedition, and it was not executed as one. |
| **E5.3** Adaptive-bandwidth Nadaraya–Watson | **NOT RUN** | A fixed-bandwidth NW is used as the profiling positive control (`PHASE1.md` §6), but the *adaptive-bandwidth* rebuttal — passes A2, fails A1 in the row-scaled way, zero curvature — was never constructed. |
| **E5.4** McCarter reproduction | **NOT RUN** | `experiments.md` calls it "cheap, high narrative value". No duplication-anomaly reproduction exists, and no alignment test between the leading negative eigenvector and the direction McCarter identifies. |

---

## Summary

| outcome | count | IDs |
|---|---|---|
| **RUN** | 5 | E0.6, E1.2, E2.2, E2.3, E2.4 |
| **RUN‑D** | 10 | E0.3, E0.5, E1.1, E1.3, E1.4 (TabICL), E2.1, E2.5, E3.5, E4.1, E5.1 |
| **NOT RUN** | 12 | E0.1, E0.2, E0.4, E1.4 (TabPFN), E1.5, **E2.6**, E3.1, E3.2, E3.3, E3.4, E4.2, E5.3, E5.4 |
| **FIRED** | 3 | E1.1 (TabSwift, condition not met), E2.4 (both mechanism classes excluded — the stronger result), **E5.2 (on unregistered numbers — see the row)** |

**The three gaps that most affect what the paper may claim:**
1. **E2.6** — the calibrated mechanism exclusion, called "the headline figure", never run correctly.
2. **E3.1** — the context-distribution objection, called "fatal if unaddressed", never run.
3. **E1.5 / M0** — the radial-vs-tangential gate, whose pre-registered rule would kill the hedging
   interpretation of curvature, never evaluated.
