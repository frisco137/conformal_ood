# Phase 3 — handoff

**Written 2026-09-03 at a machine migration.** Everything an agent picking this up needs, in the
order it needs it. If you read nothing else, read §1 and §3.

---

## 1. Read these, in this order

| # | file | what it is |
|---|---|---|
| 1 | [`plan.md`](plan.md) | the Phase 3 plan. **Treat its numerical predictions as targets to be checked, not facts** — see §6. |
| 2 | [`phase_3_results.md`](phase_3_results.md) | **the live record.** §0 provenance · §1 pre-registration · §2 task log · §3 decisions · §4 surprises · §5 open threads · §6 retractions · §7 script→artifact table. Append after every task. |
| 3 | [`provenance_prior.md`](provenance_prior.md) | T0.1. Why everything says "prior-family" and never "in-family". |
| 4 | [`t3_1_noise_scale.md`](t3_1_noise_scale.md) | the noise-scale derivation. Blocking for how every A1 number is quoted. |
| 5 | `../phase_1/phase_1_results.md`, `../phase_2/phase_2_results.md` | what Phase 3 is built on. |

**The rule that governs the results file:** if a number is not in it with a path beside it, it does
not exist. Tag every quantity `MEASURED` / `DERIVED` / `ASSUMED`.

---

## 2. Environment

Pinned in `phase_3_results.md` §0. Rebuild from the tracked `pyproject.toml` / `uv.lock`.

```
python 3.10.12 · torch 2.11.0+cu130 · numpy 2.2.6 · scikit-learn 1.7.2
tabicl 2.1.1 · tabpfn 8.0.8
```

**The audited object, and it must be verified before any measurement:**

```
checkpoint  tabicl-regressor-v2-20260212.ckpt
sha256      0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a
config      max_classes 0 (the regression switch) · num_quantiles 999 · icl_num_blocks 12
```

**Does not arrive with a git clone:** `intermediate/` (15 GB, gitignored legacy archive),
`.venv/`, and the HuggingFace cache. TabICL re-downloads; **`Prior-Labs/TabPFN-v2.5-clf` and
`v2.6-clf` are gated (HTTP 401)** — copy `~/.cache/huggingface/hub` if those are needed.
`models/checkpoints/tabswift/swift.ckpt` **is** tracked, because its upstream repo is gated too.

**Smoke test before trusting anything:**
```bash
.venv/bin/python -m pytest experiments/phase_1/core/test_controls.py -q     # 22 tests
.venv/bin/python experiments/phase_1/tier0_instrument/recompute_orphans.py  # 293 checks
.venv/bin/python experiments/phase_3/src/t1_analyse.py                      # ladder table
```

---

## 3. State at the cut

### Done and recorded

| task | outcome |
|---|---|
| **T0.1** prior provenance | **PARTIAL.** `PriorDataset(max_classes=0)` *raises*, so the shipped sampler cannot produce the regression setting its own checkpoint records. Everything is **prior-family**, never in-family. |
| **T0.2** determinism | **PASS, bit-identical** on all 5 seeds. Closes Phase 2's E0.1. |
| **T0.3** circulation | P2 passes by 8 orders, P3 shows FD at 2690% error where the loop holds at 0.21%. **P1's falsifier fired**; both estimators shown correct, so H2 did **not** fire. Verdict: the loop measures a **great-circle average**, not `asym`. Option **(b)** adopted — both kept as separate registered quantities (D7). |
| **T0.4** literature | structural repair done; **channel column still outstanding**. Von Oswald's derivatives are w.r.t. **inputs, not labels** — the C-X3 gap sentence stands. |
| **T3.1** noise scale | three identities proved and verified. Both instruments provably blind to `σ²` mixing. Class-level ambient A1 floor is `0.283`, not `10⁻³`. |
| **T1.1–T1.2** | ladder built (6 rungs × 60), **all six rungs audited**. |
| **T1.4** controls | **P5 PASS**, **P7 PASS**, **P6 ranges miss 3 of 4 but its consequence holds** (11.5×, inside the registered 6–13×). |
| **T1.7** nano-PFN | both arms trained and audited. `fixed` decays `1.408 → 0.585`; `mixed` decays identically, so its registered prediction **failed as mis-specified**, not falsified (R2). |
| **Block 1** | all four items. **A3 floors corrected an overstated round-1 claim** — A3 fails on D/E/F and is **UNEVALUABLE on A/B/C**. |

### The headline

**Reading 1 fires.** On the 43% of contexts from TabICL's own prior where it is non-degenerate,
`asym 0.6968`, `negeig 0.2336` — **higher than on the Phase 1 audit family** (`0.6686`/`0.2030` on 60
contexts). The off-family escape route is closed. Reading 2 and the prior-support-detector paper are
dead.

**The caveat that must travel with it:** 57% of rung A is degenerate (`trJ/n = 0.976`), passing
vacuously. **Never quote the ungated means.**

### Interrupted, no result

**Block 3.1 anti-shrinkage** — launched with P14/P15/P16 registered beforehand, stopped mid-run.
`phase_3_results.md` §2 has three partial points marked **"do not read this"**. They lean toward P15
failing, but that is 3 of 24 registered points. **Re-run to completion first.**

### Not started

Block 2 · Block 3.2 · Block 4 (T1.3 coordinate, circulation ladder, literature channel column) ·
Block 5 · **all of Tier 2** · T3.2 · T3.3.

`src/t1_coordinate.py` and `src/t1_circulation.py` are written, committed and never run.

---

## 4. The outstanding directive

**This is not in `plan.md`.** It came as a continuation directive after round 1 and is reproduced
here because otherwise it is lost. Blocks 1 is done; 2–5 are not.

### Block 2 — promote the degeneracy finding
The 57% is written as a filtering step and **it is a result.** Produce, per rung: full distributions
of `‖J−I‖_F/‖J‖_F` and `trJ/n`; the same for every control; and violation plotted against the
degeneracy ratio **within** rung. Add a **second degeneracy gate for the `J ≈ 0` direction** —
label-insensitivity — since the current gate catches *copy-the-labels* but not *ignore-the-labels*.
Apply it retrospectively to the nano-PFN checkpoints (which sit at `trJ/n ≈ 0.029`, i.e. exactly that
failure) **and** to all six rungs.

### Block 3 — two experiments, both already pre-registered (§1, round 2)
**3.1 anti-shrinkage, de-confounded.** Fix one context `(X, f)`, sweep `σ` over 6 values across two
decades with `ε` fixed, measure `trJ/n`, `‖J−I‖`, `asym`, `negeig`. Controls must move *opposite* to
TabICL. **P14** controls decrease (a theorem — if it fails the harness is wrong); **P15** TabICL
increases, else the round-1 finding is withdrawn as a feature-dimension/DAG-depth confound;
**P16** violation rises with `σ`. *If P14 and P15 both hold this leads the paper's empirical section:
a Bayes violation stated without any Jacobian machinery.*

**3.2 nano-PFN, exclude the magnitude confound.** `asym` is a ratio, the label response is 1–3% of a
posterior mean's at every checkpoint, and NLL moves only `0.04` nats over the run — so the decay is
consistent with `‖J‖_F` growing rather than the numerator shrinking. Plot violation against `‖J‖_F`;
**report numerator and denominator separately** (**P17**); run one arm with materially more capacity
or compute to test whether the `0.59` plateau moves (**P18**). *If the plateau is capacity-bound, say
so — the progress-measure claim needs it to move.*

### Block 4 — remaining registered items
Circulation column across the ladder at D8's reduced power (32 planes × 24 points, 20 contexts/rung);
the `C/asym_FD` ratio is a real registered measurement — keep it. **T1.3 coordinate: run it, but it is
no longer load-bearing** — the rung-level ordering already says the story is not monotone, and P8's
registration says a flat or non-monotone curve *is* the result. Finish the literature channel column;
it is the last thing licensing the C-X3 gap sentence.

### Block 5 — Tier 2 redesign *(a decision already taken, not a proposal)*
Tier 2 was built to stratify downstream cost by in-family-ness. **That variable probably carries no
signal, so the manipulation would manipulate nothing. Replace the dial with the degeneracy ratio.**
It moves the violation from identically zero to `0.7`, is measurable from the context **before any
decision is taken**, and gives the cleaner question: *does sequential cost appear exactly where the
label response is non-trivial and vanish where the model is copying labels?* Registered as **P19**.

**Unchanged:** the acquisition ordering `greedy < EI < KG`; the arm ladder (native → conformal →
martingale posterior → repaired field → refitted GP); the repair experiment as the causal handle; the
power targets. **Lead the write-up with the sign-violation and shrinkage numbers.**

---

## 5. Registered predictions — status

| | status |
|---|---|
| P1 | falsifier **fired**; H2 did **not** — both estimators shown correct |
| P2, P3 | **PASS** |
| P4 | target missed (0.239 vs 0.20); registered falsifier (>50%) **not** fired; diagnosed as pure Monte Carlo |
| P5, P7 | **PASS** |
| P6 | ranges **miss 3 of 4**; consequence **holds** |
| P8 | **unscored** — needs T1.3 |
| P9 | **PASS on the mean** (`0.844`), but 40/60 contexts individually exceed `0.9`; the intent is not satisfied |
| P10–P12 | Tier 2, not run |
| P13 | **superseded by P19** (Block 5) |
| P14–P16 | registered; B3.1 interrupted |
| P17–P19 | registered, not run |

---

## 6. How to work here

**Standing rules, and they have earned their place:**

- **The plan's numerical specifications have been wrong in detail four times** — the Frobenius
  constant (factor of 2), the loop/derivative premise, the noise-hyperprior floor ranges, and the D/F
  rung overlap. Each was caught with evidence and logged. **Check them; do not assume them.**
- **Distinguish a mis-specified prediction from a falsified one.** T1.7's mixed arm and P1 were both
  mis-specified — the prediction contradicted the project's own theory. This distinction has been the
  most useful thing in the results file.
- **A result that looks too good is a bug hypothesis first.** Log the falsification you ran.
- **Nothing untracked.** Every script gets committed whether or not it produced a number. Phase 2 lost
  ten result files to in-session code; §7 records that it nearly happened again here (B1.4).
- **Cut power, never a registered comparison** (plan §1 budget rule).
- Halt conditions **H1–H5** in `plan.md` §7 are unchanged.

**Operational, learned the hard way on the old machine:**

- Launch long GPU jobs with **`setsid`** and `< /dev/null`, or the shell reaping the wrapper kills
  them silently — no error, no output, just gone.
- **Check `ps -o pid,ppid` for duplicates before walking away.** A careless relaunch left two
  independent runs racing on the same log and output path.
- **Cap BLAS threads** (`OMP_NUM_THREADS=8`). These runs are **CPU-bound, not GPU-bound** — ~7% GPU
  utilisation while burning 30+ cores, because every probe point refits the model. Two uncapped runs
  once drove a 192-core machine to load average 209.
- Cost reference: a TabICL reduced Jacobian at `n=100` is ~196 fits ≈ 50 s; a full ladder rung
  (60 contexts, reduced + ambient) ≈ 1.8 h.

**Auth:** the old remote had a dead PAT embedded in its URL. Now clean, using `gh` as the credential
helper. On a new machine: `gh auth login && gh auth setup-git`. **Do not re-embed a token in the
remote URL.**

---

## 7. The three things most likely to trip you

1. **`asym` conventions.** Phase 1's canonical `asym = ‖J−Jᵀ‖_F/‖J‖_F` lives in
   `experiments/phase_1/core/metrics.py` and nowhere else. Anything half that value came from the
   quarantined `_archive/quarantine/tier2_audit/`.
2. **Gated vs ungated.** Rung A's ungated `asym` is `0.373`; its gated value is `0.697`. The gate
   excludes 57% of contexts. Quoting the wrong one inverts the finding.
3. **A3 is unevaluable on three rungs.** Do not write "A3 fails on every rung" — that was corrected in
   Block 1, and the binding control is the **noise-hyperprior** GP, not the hierarchical GP.
