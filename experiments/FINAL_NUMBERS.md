# FINAL NUMBERS

Frozen measurement record. Every number below is traceable to a file on disk written by this run;
Section 9 gives the file and key for each. Quantities that were not computed say `NOT MEASURED`.

No number is carried over from `final_audit_report.md`, `rebuttal_5_standardize.py`, or
`rebuttal_6_gaps.py`. Where a value here disagrees with a previously reported one, both are shown
side by side with the configuration difference named.

Contexts throughout: `generate_audit_context(n=100, d=5, sigma=1.0, seed=s)`,
seeds `[42, 100, 200, 300, 400]`. Reduced basis `get_Q(y, seed=0)`, shape `(100, 98)`.

**Revision 2.** Everything raised in `CLARIFICATIONS.md` and everything measured in
`E11_WRAPPER_FORM.md` is folded in here; this file supersedes both as the single record. New since
revision 1: Sections 2.7 (wrapper form), 2.8 (`r_1`), 2.9 (ambient floor), 5.8 (no-intercept A3
fits); corrected Sections 4.1 (TabICL A2 was marked not measured and is), 4.2 (lower bracket added),
4.3 (TabSwift added), 6.3 (ambient floor and ratios), 7.4 (`negfrac` at `t=1e-1`); Section 8 lost the
`r_1` row and gained the TabSwift N1(ii) row.

---

## Section 1 — Instrument provenance

### 1.1 Canonical formulas, printed from `core/metrics.py`

**`asym`** — `core/metrics.py:3`
```python
def asym(J):
    norm_denom = np.linalg.norm(J, ord='fro')
    if norm_denom < 1e-12:
        return 0.0
    return np.linalg.norm(J - J.T, ord='fro') / norm_denom
```

**`negeig`** — `core/metrics.py:12`
```python
def negeig(J):
    sym_J = (J + J.T) / 2
    vals = np.linalg.eigvalsh(sym_J)
    min_val = np.min(vals)
    if min_val >= 0:
        return 0.0
    norm_J = np.linalg.norm(J, ord=2)
    if norm_J < 1e-12:
        return 0.0
    return -min_val / norm_J
```
Range is `[0, 1]`: `|lambda_min(sym J)| <= ||sym J||_2 <= ||J||_2`.

**`negfrac`** — `core/metrics.py:107`
```python
def negfrac(J, rel_tol=1e-10):
    sym_J = (J + J.T) / 2
    eigs = np.linalg.eigvalsh(sym_J)
    scale = np.linalg.norm(J, ord=2)
    if scale < 1e-12:
        return 0.0
    return float(np.sum(eigs < -rel_tol * scale) / len(eigs))
```

**`second_difference_norm`** — `core/metrics.py:123`
```python
    denom = t**2 * np.linalg.norm(m_base)
    if denom < 1e-300:
        return 0.0
    return float(np.linalg.norm(m_plus - 2.0 * m_base + m_minus) / denom)
```

**Profile residual** — `core/metrics.py:250-261`
```python
    if weighted:
        sw = np.sqrt(w)                      # w = |J_ij * J_ji| per pair
        Af, bf = A * sw[:, None], b * sw
    ...
    residual_abs = float(np.linalg.norm(Af @ log_s - bf))
    norm_b = float(np.linalg.norm(bf))
    residual_rel = residual_abs / norm_b if norm_b > 0 else float("nan")
```
Computed explicitly, not read from `np.linalg.lstsq`. The design matrix has one `+1` and one `-1`
per row, so its columns sum to zero and its rank is `n-1`; numpy returns an **empty** residuals array
in that case, which is why `item3.py:95` emitted `0.00e+00` for every fit it ever reported.

**`get_Q` seeding** — `core/metrics.py:147`, `rng = np.random.RandomState(seed)`. **Seed used
throughout this run: `0`.** The 12 previous copies of this function in the tree were unseeded, so no
earlier measurement was reproducible.

### 1.2 Convention change

Every `asym` value under `tier2_audit/` was computed as `||(J - J^T)/2||_F / ||J||_F`, exactly half
the canonical definition fixed in `theory_and_claims.md`. **No number in this document is taken from
those files.** All model values here were **recomputed** from Jacobians measured in this run
(`exp12_ambient_jacobians.npz`, `exp3_reduced_jacobians.npz`). Where a historical value appears it is
labelled as such and doubled to the canonical convention, with the doubling stated.

### 1.3 Retracted quantities

| Quantity | Where it appeared | Reason |
|---|---|---|
| `negfrac`, all models | `results_batch1.json`, `results_tabpfn_5seed.json`, `results_tabswift.json`, `results_phase1.json`, `report.md` | Counts the rank deficiency of the context, not the model. The audit context duplicates 10 rows of `X` exactly, so `Q^T W Q` for the exact GP has 9 machine-zero eigenvalues and a hard gap to `2.2e-2`; any noise tips them negative. Control and TabICL both read `9/98 = 0.0918`. Section 4.4. |
| `negeig` as count fraction | `instrument_c_d.py:89` (fixed this run), propagated to `audit_e_results.json`, `instrument_c_d_results.json` | Returned `#{lambda < -1e-10}/dim` under the name `negeig`. Eight scripts consumed it. |
| Profile residual `0.00e+00` | `item3.py:95`, `item3_log.txt` | Rank-deficient design; numpy returns an empty residuals array. Verified reproduction: rank 7 of 8, `res = []`, reported `0.0`, true residual `4.786`. |
| Reduced-basis profiling | `tier2_report.md`, `RESULTS_REPORT.md` rev.1 §4.2 | No power. Section 6.1. |
| `cv_s <= cv_J` as a pass | `e2_5_report.md` | Permutation-invariant. Section 5.5. |

---

## Section 2 — Instrument validation

### 2.1 ExactGP, unwrapped — `t=1e-3`, no dither

```
asym    [2.848611e-12  2.650462e-12  2.848388e-12  2.671164e-12  3.366328e-12]  mean 2.876991e-12
negeig  [4.984192e-16  3.982686e-16  3.263680e-16  4.999470e-16  4.329369e-16]  mean 4.311879e-16
```

### 2.2 ExactGP, wrapped — the projection floor, `t=1e-3`

```
asym    [2.793328e-12  3.152690e-12  3.282476e-12  2.713298e-12  3.763195e-12]  mean 3.140997e-12
negeig  [5.892441e-16  4.132528e-16  5.210598e-16  5.974904e-16  3.813456e-16]  mean 5.004786e-16
```

### 2.3 Targeted imitator — measured vs analytic

Analytic reference is `Q^T G Q` with `G = W + M_anti - (c/2) v v^T`. `audit_phase2_imitator.py`'s
`get_analytic_jacobian` returns `... - c * outer(v,v)`, twice the rank-one term the map applies; the
corrected reference is used here.

```
seed   ||J_meas - Q^T G Q||_F / ||.||_F   negeig measured   negeig analytic
42               2.131e-12                    0.088194         0.088194
100              2.356e-12                    0.059000         0.059000
200              2.470e-12                    0.064744         0.064744
300              2.082e-12                    0.089827         0.089827
400              2.808e-12                    0.120723         0.120723

asym measured [6.005769e-01 6.016206e-01 6.001840e-01 6.010219e-01 6.004114e-01]
asym analytic [6.005769e-01 6.016206e-01 6.001840e-01 6.010219e-01 6.004114e-01]
```

### 2.4 Analytic curvature calibration, `m(y) = y + 0.1 y^2`

Second difference is exactly `0.2 t^2 q*q`, so true curvature `||0.2 q*q|| / ||m(y)||` is
`t`-independent. Max relative error over 98 probes x 5 seeds:

```
t=1e-03  measured 2.31980281e-03  analytic 2.31980281e-03  max rel err 2.222e-08
t=1e-02  measured 2.31980281e-03  analytic 2.31980281e-03  max rel err 1.862e-10
t=1e-01  measured 2.31980281e-03  analytic 2.31980281e-03  max rel err 2.765e-12
t=1e+00  measured 2.31980281e-03  analytic 2.31980281e-03  max rel err 2.513e-14
```

Jacobian on the same map, analytic `J = diag(1 + 0.2y)`, relative Frobenius error of `Q^T J Q`:
```
t=1e-03  [8.677e-13  8.374e-13  8.411e-13  8.001e-13  1.129e-12]
t=1e-02  [8.726e-14  8.562e-14  8.556e-14  8.036e-14  1.140e-13]
t=1e-01  [8.749e-15  8.620e-15  8.445e-15  7.722e-15  1.112e-14]
```

### 2.5 Seven stress tests

| # | Test | Number |
|---|---|---|
| 1 | estimator recovers a known asymmetric `J`, `\|\|J_meas − A\|\|_F/\|\|A\|\|_F` | `6.148e-13` |
| 2 | transpose / orientation, max deviation from `J[:,1]=1`, else `0` | `8.882e-16` |
| 3 | basis invariance across `Q` seeds `{0,1,2}`: asym spread / negeig spread | `0.000e+00` / `1.527e-16` |
| 4 | null perturbation, constant map, `\|\|J\|\|_F` | `0.000e+00` |
| 5 | determinism of the analytic pipeline, `max\|J1 − J2\|` | `0.000e+00` |
| 6 | reduced probe vs ambient-then-project, relative Frobenius | `2.051e-12` |
| 7 | permutation consistency, `\|\|J(Py) − P J P^T\|\|/\|\|J\|\|` | `0.000e+00` |

**Determinism on models: `NOT MEASURED`.** Requires a GPU refit-twice pass on the audit path. The
only existing evidence is `results.json → 4_gaps`, taken at `n=20, d=2` with fit-once/predict-twice
(`audit_4_gaps.py:14-19`), which is not the path the audit uses.

### 2.6 Quantised-ExactGP artifact table

Quantised wrapped exact GP, step `delta = 6.87e-4`. True `asym`, `negeig`, curvature all exactly `0`;
analytic `J = Q^T W Q`. Every nonzero entry is manufactured by output quantisation.

| `t` | dither | asym | negeig | `\|\|J\|\|_F` error | curvature |
|---|---|---|---|---|---|
| 1e-2 | off | `0.285744` | `0.020461` | `0.207544` | `4.515` |
| 1e-2 | `±delta/2`, N=10 | `0.133294` | `0.005028` | `0.095110` | `2.372` |
| 1e-2 | `±delta`, N=10 | `0.096875` | `0.001893` | `0.069192` | `2.599` |
| **1e-1** | **`±delta`, N=10** | **`0.009744`** | **`0.0000178`** | — | `2.614e-02` |
| — | unquantised, no dither | `3.174e-13` | `5.563e-16` | `2.25e-13` | `4.252e-12` |

Per-seed at `t=1e-1`: asym `[0.009832, 0.009680, 0.009678, 0.009736, 0.009796]`,
negeig `[0.0000180, 0.0000188, 0.0000153, 0.0000212, 0.0000161]`.

Dither-count sweep at `t=1e-2`, halfwidth `delta`: asym `0.096875 / 0.067359 / 0.053180 / 0.044510 /
0.039371` at `N = 10 / 25 / 50 / 100 / 200`; log-log slope vs `N` is `-0.3026` for asym and `-0.6153`
for negeig. Pure noise would give `-0.5`.

### 2.7 Wrapper form — E1.1, forward passes only

    shift:  m(y + c*1) =? m(y) + c*1    shift_err = ||m(y+c1) - m(y) - c1||_2 / ||m(y)||_2
    scale:  m(c*y)     =? c * m(y)      scale_err = ||m(c*y) - c*m(y)||_2 / ||c*m(y)||_2

No derivatives, no projection, **no dithering** — single forward passes, so dither has no role;
`e11_wrapper_form.json` records `_config.dither = false`. Worst value over five seeds:

| | shift `c=0.1` | shift `c=0.5` | shift `c=1.0` | scale `c=0.5` | scale `c=2.0` |
|---|---|---|---|---|---|
| wrapped ExactGP (control) | `1.518e-16` | `1.732e-16` | `1.854e-16` | `2.527e-16` | `3.124e-16` |
| **TabPFN v2** | `3.994e-07` | `4.899e-07` | `7.270e-07` | `0.000e+00` | `0.000e+00` |
| **TabICL v2** | `2.206e-06` | `2.095e-06` | `2.171e-06` | `0.000e+00` | `0.000e+00` |
| unwrapped ExactGP (control) | `8.122e-03` | `4.061e-02` | `8.122e-02` | `0.000e+00` | `0.000e+00` |
| **TabSwift** | `5.537e-01` | `2.672e+00` | `3.379e+00` | `3.693e+00` | `9.165e-01` |

Per-seed values in `e11_wrapper_form.json`. Against the `< 1e-3` relative criterion: TabPFN v2 and
TabICL v2 satisfy both identities; TabSwift satisfies neither. The unwrapped ExactGP satisfies scale
but not shift, which is expected for a linear map and shows the two identities are independent.

`err/c` for the shift identity distinguishes a fixed floor from a real violation:

```
                     c=0.1        c=0.5        c=1.0
exactgp_wrapped    1.0856e-15   2.7115e-16   1.6644e-16     falls as 1/c
tabicl_v2          1.6754e-05   3.3758e-06   1.6317e-06     falls as 1/c
tabpfn_v2          1.6411e-06   3.9476e-07   2.8088e-07     falls as 1/c
exactgp_unwrapped  7.4527e-02   7.4527e-02   7.4527e-02     constant
tabswift           3.0171e+00   2.7285e+00   1.7265e+00     constant to within 1.7x
```

**Target-preprocessing path, from source.** TabPFN v2: `tabpfn/regressor.py:877-880`,
`y = (y - self.y_train_mean_) / self.y_train_std_`, borders remapped at `:882`. TabICL v2:
`tabicl/_sklearn/regressor.py:412-413`, `StandardScaler` on the target, inverted at `:764`, `:769`.
TabSwift: `models/vendor/tabswift/regressor.py:319-324` and `:526` — the target `StandardScaler` is
**commented out in both directions**, and `self.scaler_ = None` at `:165`. No target transform is
applied.

**Consequence for N1(ii).** The identity `Q^T J Q = Q^T G Q` is derived only for
`m(y) = ybar*1 + s_y*g(u)`. It is licensed for TabPFN v2 and TabICL v2. It is **not** established for
TabSwift; its reduced-basis figures in Sections 3.3 and 4.1 carry that as a stated open assumption.
A2 is unaffected for all three: a compression `Q^T J Q` of a PSD matrix is PSD for any orthonormal
`Q`, so a positivity violation on a subspace is a violation of the full condition regardless of
whether the projection is exact.

### 2.8 Row-sum residual `r_1` — E0.6

`r_1 = ||J.1 - 1||_2 / ||1||_2`, from the ambient Jacobians in `exp12_ambient_jacobians.npz`. Exactly
zero for any shift-equivariant map.

```
exactgp_wrapped     [3.34e-12  4.12e-12  3.64e-12  3.38e-12  4.03e-12]   mean 3.70e-12
exactgp_unwrapped   [7.90e-02  8.51e-02  8.32e-02  8.10e-02  8.07e-02]   mean 8.18e-02
TabICL v2           [0.0093    0.0091    0.0117    0.0100    0.0094  ]   mean 0.0099
TabPFN v2 (dither)  [0.4845    0.2909    1.0345    0.5758    0.1712  ]   mean 0.5114
TabSwift            [0.9720    0.9846    0.9879    0.9064    0.6849  ]   mean 0.9071
```

`r_1` conflates instrument error with absence of shift equivariance; the two controls separate them.
Section 2.7 settles which applies:

- **TabICL v2** — shift equivariance holds at `2.21e-06`, so `r_1 = 0.0099` is instrument error:
  about `1%` on the ambient Jacobian at `h = 1e-3`.
- **TabPFN v2** — shift equivariance holds at `7.27e-07`, so `r_1 = 0.5114` is **entirely instrument
  error**: about `51%` on the ambient diagonal-probe Jacobian at `h = 1e-1` with `N = 10` dither.
- **TabSwift** — shift equivariance does not hold, so `r_1 = 0.9071` is not an instrument-error bound
  and is not reported as one.

### 2.9 Ambient floor

Ambient means the full `100x100` Jacobian from `central_jacobian(predict, y, h=1e-3)`, no projection.
N1(i) makes the ambient Jacobian of a normaliser-wrapped map asymmetric by construction, so the
ambient floor is not near zero and must be measured.

```
wrapped ExactGP (the ambient floor)
  asym   [1.230946e-02  1.241187e-02  1.272771e-02  1.266221e-02  1.167773e-02]  mean 1.235779e-02
  negeig [3.625256e-16  3.391797e-16  4.089907e-16  5.084879e-16  5.253392e-16]  mean 4.289046e-16
  ||J||_F mean 6.7356

unwrapped ExactGP (isolates the wrapper's contribution)
  asym   [8.823888e-13  8.733945e-13  8.656057e-13  7.916603e-13  1.148748e-12]  mean 9.123595e-13
  negeig [3.771840e-16  5.941268e-16  6.134814e-16  4.244303e-16  3.352268e-16]  mean 4.688899e-16
  ||J||_F mean 6.7277
```

The wrapper raises ambient `asym` from `9.124e-13` to `1.236e-02`, a factor of `1.4e10`, and leaves
ambient `negeig` at `~4e-16`. Used in Section 6.3.

---

## Section 3 — A1 asymmetry

All values canonical `||J - J^T||_F / ||J||_F`, reduced basis, measured this run.

### 3.1 TabICL v2 — `t=1e-3`, no dither, reduced basis, measured this run

```
seed      asym        negeig      ||J||_F   ||J-I||_F/||J||_F
42      0.606662    0.241642      7.6902        0.7142
100     0.532994    0.159460      7.8000        0.6485
200     0.705353    0.220369      5.0447        1.3761
300     0.470117    0.129146      8.0836        0.5845
400     0.586529    0.132684      6.3314        0.9329
mean    0.580331    0.176660      6.9900        0.8513
std     0.078422    0.046088
```

**Reproduction of the historical figures.** `results_batch1.json`, `t=1e-3`, no dither, with `asym`
doubled from the halved convention, gives `asym 0.5802 ± 0.0784` and `negeig 0.1769 ± 0.0464`. This
run gives `0.580331 ± 0.078422` and `0.176660 ± 0.046088`. They reproduce to four significant
figures on both quantities. The historical run used an **unseeded** `get_Q`, so exact agreement was
not expected; that it agrees this closely is consistent with the basis-invariance result in Section
2.5 (asym spread `0.000e+00`, negeig spread `1.527e-16` across `Q` seeds `{0,1,2}`).

**Amplitude sweep, 5 seeds at each `t`:**

```
t=1e-03   asym 0.580331 ± 0.078422   negeig 0.176660 ± 0.046088   ||J||_F 6.9900
t=1e-02   asym 0.580057 ± 0.078296   negeig 0.176656 ± 0.046107   ||J||_F 6.9886
t=1e-01   asym 0.579137 ± 0.078645   negeig 0.175941 ± 0.045373   ||J||_F 6.9861
```

Total drift across two decades: `0.21%` on `asym`, `0.41%` on `negeig`. For comparison, TabPFN drifts
`6.6%` over 1.5 decades (Section 3.2) and TabSwift `24%` from `1e-2` to `1e-1` (Section 3.3).

**Non-degeneracy at the headline amplitude.** `||J||_F` mean `6.9900` against `sqrt(98) = 9.899`, a
ratio of `0.706`; per-seed `[7.6902, 7.8000, 5.0447, 8.0836, 6.3314]`. `||J-I||_F/||J||_F` mean
`0.8513`, per-seed `[0.7142, 0.6485, 1.3761, 0.5845, 0.9329]`, minimum `0.5845`.

### 3.2 TabPFN v2 — `t=1e-1`, dither halfwidth `delta`, `N=10` (headline)

```
asym    [1.381972  1.348622  1.280438  1.402213  1.148824]  mean 1.312414  std 0.091648
```

Alongside, the previous amplitude, halved convention doubled: `t=1e-2`, dither halfwidth `delta`,
`N=10`, `results_tabpfn_5seed.json` gives `0.6655 ± 0.0221` halved = **`1.3309 ± 0.0443` canonical**.

Amplitude sweep, no dither, 5 seeds each, this run:
```
t=1e-02  asym 1.373350 ± 0.023903      t=1e-01  asym 1.311585 ± 0.090440
t=3e-02  asym 1.332991 ± 0.069292      t=3e-01  asym 1.275497 ± 0.121304
```
Total drift across 1.5 decades: `6.6%`. Dither at `t=1e-1` changes asym from `1.311585` to
`1.312414`.

### 3.3 TabSwift — `t=1e-1`, no dither

```
asym    [1.206691  1.270011  0.902896  0.905487  0.816161]  mean 1.020249  std 0.182063
```

Amplitude sweep, no dither: `t=1e-2` `1.339607 ± 0.032343`; `t=1e-1` `1.020249 ± 0.182063`;
`t=1e+00` `0.950777 ± 0.199566`. Drift `1e-2 → 1e-1` is `24%`.

**Disagreement flagged.** The prompt states TabSwift's `0.702` came from a copy-pasted `t=1e-3` and
that "the correct value is `0.514`". Two corrections, both citing lines:
- `0.514` is the halved convention. Canonical is `1.025`; this run independently measures
  `1.020249` at `t=1e-1`.
- `0.702` is not a `t=1e-3` copy-paste. It appears only in `final_audit_report.md`. The script that
  would have produced it, `batch4_lastrun.py`, cannot execute: line 16 imports
  `experiments.core.synthetic`, which does not exist. `batch3_tabswift.py:128` sets `t = 1e-1`
  correctly. There is no TabSwift `t=1e-3` run anywhere in the tree.

**Scope note.** TabSwift satisfies neither the shift nor the scale identity (Section 2.7), and its
target scaler is commented out in source. N1(ii)'s exactness guarantee for `Q^T J Q` is therefore not
established for this model, and the value above is reported as the asymmetry of the compressed
Jacobian rather than of a recovered inner map. Its ambient counterpart is in Section 6.3.

### 3.4 Control floor beside each model

Each floor is a quantised wrapped exact GP at that model's own measured output step, pushed
through that model's exact probe configuration. True `asym` is exactly zero in every row.

| | `t` | model asym | artifact floor | ratio |
|---|---|---|---|---|
| TabPFN v2 | 1e-1, dither `delta` N=10 | `1.312414` | `0.009744` | `134.7x` |
| TabPFN v2 | 1e-2, dither `delta` N=10 (historical) | `1.3309` | `0.096875` | `13.7x` |
| TabICL v2 | 1e-3, no dither | `0.580331` | `0.001018` – `0.004258` | `569.8x` – `136.3x` |
| TabSwift | 1e-1, no dither | `1.020249` | `0.003857` – `0.017996` | `264.5x` – `56.7x` |

**Measured output quanta** (fine scan of `m(y + t q_1)` across each model's probe range, 501 points,
seed 42; jumps are `|diff|` along the scan):

```
model       dt        frac zero jumps   median nz    p10 nz      p90 nz      min nz      max        dtype    mean |m|
tabicl_v2   4.000e-06     0.0259        9.8348e-07  2.3842e-07  2.9802e-06  1.8481e-09  1.3471e-05  float32  0.8822
tabswift    4.000e-04     0.0401        4.2725e-04  9.1553e-05  1.9531e-03  9.5367e-07  9.7656e-03  float16  0.1312
tabpfn_v2   (diagnostic_a_results.json)  6.8700e-04                                      8.2445e-03           —
```

**Why TabICL and TabSwift get a bracket, not a single floor.** Neither output is binned the way
TabPFN's is. TabICL's trace is continuous in float32: only `2.6%` of adjacent steps are exactly
equal, and the median nonzero jump `9.8348e-07` sits where the *smooth* increment would
(`dt × |dm/dt| ≈ 4e-6 × O(0.25)`), not at a fixed grid spacing. `np.spacing(float32(0.8822)) =
5.960e-08`. TabSwift's outputs are **float16** (`np.spacing(float16(0.1312)) = 1.2207e-04`), and its
jump structure sits on powers of two (`min nz 9.5367e-07 = 2^-20`, `p90 1.9531e-03 = 2^-9`), so it
mixes a genuine 16-bit grid with the local slope. The floors are therefore computed at two steps
each — `p10` of the nonzero jumps as the lower bracket, the median as the upper — and the ratio
range is reported rather than a point value.

**Correction to the previous text of this section.** It cited TabICL's quantum as `~3e-6`, inferred
from second-difference provenance in `results_batch1.json`. Direct measurement puts `3e-6` at the
`90th percentile` (`2.9802e-06`); the median is `9.8348e-07` and the smallest resolvable step is
`1.8481e-09`. The `~230x` ratio against TabPFN quoted there becomes `~700x` at the median.

### 3.5 Amplitude choice per model

| model | `t` | reason |
|---|---|---|
| TabPFN v2 | `1e-1` | artifact `1/t`; `0.096875 → 0.009744` from `1e-2`. asym drift over the sweep is `6.6%`. |
| TabSwift | `1e-1` | `1e-3` is inside the derivative-collapse regime; `‖J‖_F` at `1e-1` is `4.0598`. |
| TabICL v2 | `1e-3` | float32 output, median nonzero jump `9.8348e-07` (Section 3.4); measured sweep this run flat to `0.21%` across `t ∈ {1e-3, 1e-2, 1e-1}` (Section 3.1). |

### 3.6 Positional-encoding zeroing, TabICL v2 — paired

RoPE zeroed at `model_.row_interactor.tf_row.rope = None`, same contexts, `h=1e-3`, ambient `J`
projected through `Q(seed=0)`, canonical asym.

```
seed   raw        zeroed     paired diff   RMSE raw   RMSE zeroed
42     0.606412   0.622545   +0.016132     0.46541    0.50566
100    0.532348   0.524374   -0.007974     0.45649    0.42244
200    0.705529   0.707124   +0.001595     0.81533    0.78565
300    0.470630   0.435577   -0.035052     0.31354    0.29323
400    0.587082   0.601217   +0.014135     0.70064    0.71872

paired difference   mean -0.002233   sd 0.020796   se 0.009300
                    95% CI [-0.028050, +0.023585]   (t_{4,0.975} = 2.776)
paired RMSE diff    mean -0.005144   sd 0.032665
```

The CI contains zero. Accuracy change is `-0.005144 ± 0.032665` RMSE.

**Disagreement flagged.** `e1_4_report.md` reports `0.5801 → 0.5781` as two marginal means. This run
gives raw mean `0.580400`, zeroed mean `0.578167` — agreeing to `3 decimal places` — but the earlier
run used standardised targets `y_std` (`e1_4_positional.py:48`) and an unseeded `Q`, while this run
uses raw `y` and `Q(seed=0)`, and reports the paired statistic the plan requires.

**E1.4 for TabPFN v2: `NOT MEASURED`.** Requires source modification of the TabPFN attention path.

---

## Section 4 — A2 positivity

### 4.1 `negeig`, reduced basis, this run

```
TabPFN v2  t=1e-1 dither delta N=10
  [0.462003  0.408385  0.396210  0.512729  0.241219]  mean 0.404109  std 0.091393

TabSwift   t=1e-1 no dither
  [0.271481  0.355512  0.169250  0.132425  0.393546]  mean 0.264443  std 0.101490

TabICL v2  t=1e-3 no dither
  [0.241642  0.159460  0.220369  0.129146  0.132684]  mean 0.176660  std 0.046088
```

All three are measured this run. TabICL's figure is the same measurement reported in Section 3.1;
it reproduces the historical `0.1769 ± 0.0464` (`results_batch1.json`) to four significant figures,
and the measured value is the one quoted. `negeig` was never subject to the halved convention.

Amplitude sweeps, no dither unless stated:
```
TabPFN v2  0.438943 / 0.417079 / 0.402438 / 0.361321   at t = 1e-2 / 3e-2 / 1e-1 / 3e-1
TabSwift   0.509573 / 0.264443 / 0.243397              at t = 1e-2 / 1e-1 / 1e+00
TabICL v2  0.176660 / 0.176656 / 0.175941              at t = 1e-3 / 1e-2 / 1e-1
```

TabSwift's reduced-basis figure carries the scope note in Section 3.3.

### 4.2 Control floor beside each

| | model negeig | artifact floor | ratio |
|---|---|---|---|
| TabPFN v2, `t=1e-1` | `0.404109` | `0.0000178` | `22711.5x` |
| TabPFN v2, `t=1e-2` (historical) | `0.4003` | `0.00189260` | `211.5x` |
| TabICL v2, `t=1e-3` | `0.176660` | `1.9903e-07` – `4.0763e-06` | `887607.1x` – `43338.8x` |
| TabSwift, `t=1e-1` | `0.264443` | `2.8939e-06` – `6.9688e-05` | `91380.8x` – `3794.7x` |

Bracket construction as in Section 3.4. Per-seed floor values, **lower** bracket
(`p10` of the nonzero jumps):

```
TabICL v2  delta=2.384186e-07, t=1e-3
  asym   [1.073884e-03  9.735679e-04  1.018100e-03  1.009396e-03  1.017489e-03]  mean 1.018487e-03
  negeig [1.945368e-07  1.525739e-07  2.543381e-07  1.566100e-07  2.370887e-07]  mean 1.990295e-07

TabSwift   delta=9.155273e-05, t=1e-1
  asym   [4.030341e-03  3.825431e-03  3.926571e-03  3.961165e-03  3.540481e-03]  mean 3.856798e-03
  negeig [2.854633e-06  2.296345e-06  4.154820e-06  2.976005e-06  2.187480e-06]  mean 2.893857e-06
```

**Upper** bracket (median of the nonzero jumps):

```
TabICL v2  delta=9.8348e-07, t=1e-3
  asym   [4.369448e-03  4.299316e-03  4.196719e-03  4.240983e-03  4.182523e-03]  mean 4.257798e-03
  negeig [3.995556e-06  3.312172e-06  5.272950e-06  4.080489e-06  3.720104e-06]  mean 4.076254e-06
  ||J||_F rel err mean 3.023794e-03

TabSwift   delta=4.2725e-04, t=1e-1
  asym   [1.771571e-02  1.844219e-02  1.771207e-02  1.811571e-02  1.799218e-02]  mean 1.799557e-02
  negeig [7.407348e-05  6.399356e-05  7.849463e-05  7.909518e-05  5.278064e-05]  mean 6.968750e-05
  ||J||_F rel err mean 1.277130e-02
```

`negfrac` on both control floors is `0.091837` — the same `9/98` the context imposes (Section 4.4),
independent of the step used.

### 4.3 Where the leading negative eigenvector sits

Eigenvector of `lambda_min(sym Q^T J Q)`, mapped back to context points by `Q`. Participation ratio
`(sum w^2)^2 / sum w^4` on the squared loadings gives the effective number of context points carrying
the direction, out of 98.

```
TabPFN v2, t=1e-1 dither
  seed 42   lam_min -3.30369  top-5 points [25,76,75,26,80]  mass 55.2%  participation 12.3/98
  seed 100  lam_min -0.65786  top-5 points [69,78,21,24,70]  mass 35.3%  participation 23.9/98
  seed 200  lam_min -2.76867  top-5 points [95,91,50,96,52]  mass 43.2%  participation 17.8/98
  seed 300  lam_min -9.34969  top-5 points [66,95,16,54,59]  mass 28.9%  participation 32.8/98
  seed 400  lam_min -0.84400  top-5 points [8,86,30,36,43]   mass 49.1%  participation 14.8/98

TabSwift, t=1e-1
  seed 42   lam_min -0.43720  top-5 points [76,56,29,16,96]  mass 32.2%  participation 29.0/98
  seed 100  lam_min -1.43682  top-5 points [45,97,1,9,78]    mass 40.6%  participation 16.9/98
  seed 200  lam_min -0.26371  top-5 points [1,10,53,91,20]   mass 44.3%  participation 16.4/98
  seed 300  lam_min -0.28312  top-5 points [32,27,95,23,64]  mass 36.0%  participation 25.6/98
  seed 400  lam_min -0.55528  top-5 points [56,18,80,8,6]    mass 74.6%  participation 3.0/98
```

Ambient diagonal entries below zero, `#(J_ii < 0)` out of 100, from `exp12_ambient_jacobians.npz`:

```
TabPFN v2 (h=1e-1, dither):  seed 42: 4   100: 4   200: 2   300: 14   400: 0
TabPFN v2 (h=1e-1, no dith): seed 42: 4   100: 4   200: 2   300: 13   400: 1
TabSwift  (h=1e-1, no dith): seed 42: 7   100: 15   200: 3   300: 1   400: 3
TabICL v2 (h=1e-3):          0 on every seed
```

### 4.4 `negfrac` is not reported

The audit context duplicates the first 10 rows of `X` exactly (`core/context.py`), so `K` is
rank-deficient and `W = K(K+sigma^2 I)^-1` inherits a null space. The analytic reduced spectrum of
the exact GP, seed 42, fifteen smallest eigenvalues:

```
-3.38e-16 -2.90e-16 -1.76e-16 -9.55e-17 7.93e-17 1.99e-16 2.22e-16 4.18e-16 4.96e-16
 2.20e-02  9.55e-02  1.43e-01  1.76e-01  1.91e-01
```

Nine eigenvalues at machine zero, then a gap to `2.2e-2`. The quantised control returns
`negfrac = 9/98 = 0.0918367` in every seed and every dither configuration; TabICL v2 returns
`0.0918367` on every seed. `negeig` on the same control reads `0.001893` at `t=1e-2` and `0.0000178`
at `t=1e-1`, because it weights by magnitude.

---

## Section 5 — A3 cross-channel

> **Added 2026-08-26 — A3 now has an artifact floor, and it is not small.**
> Section 11 (E5.1) pushed a hierarchical GP — a map that satisfies A3 *exactly*, by
> construction — through this section's own wrapped-ambient A3 procedure on 60 real contexts.
> Unwrapped it returns `R² = 1.000000` on all 60. **Wrapped in the same normaliser the audited
> models wear, its `R²` falls as low as `0.675`,** and it fails the `R² ≥ 0.90` gate on 3 of 60.
> The exact GP does not show this, because its inner map is linear; the effect needs a *nonlinear*
> inner map, which all three audited models are.
>
> Consequence for this section: `R²` values below `~0.68` cannot be attributed to the model alone,
> because the wrapper can produce them on a map that satisfies the law exactly. The measured model
> values are TabICL `0.0199` and TabPFN `0.1065` — an order of magnitude below that floor, so the
> A3 verdict stands. But **the floor must now be quoted beside them**, and the bare statement "the
> models fail A3" is not licensed without it.


`s^2_i = a + b * (1 + J_ii)`, ambient diagonal, ordinary least squares across the 100 context
points. Specification, `tier0_instrument/exp1_a3_models.py:52-58`, called at `:137` as
`ols(1.0 + Jd, s2)`:

```python
def ols(x, y):
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2)); ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(coef[1]), float(coef[0]), (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
```

`R^2` is `1 - SS_res/SS_tot` from the model's own residuals, not a squared correlation. With an
intercept fitted, regressing on `(1 + J_ii)` is the same model as regressing on `J_ii`:
`a + b(1 + J_ii) = (a + b) + b*J_ii`, identical slope, intercept shifted by `b`. The `1 +` matters
only in the no-intercept form of Section 5.8. The law `s^2 = sigma^2 (1 + J_ii)` requires
`a = 0`, so the fitted intercept is itself a reported quantity.

### 5.1 Extractor validation on the exact GP

`extract_variance` against `diag(K - K(K+sigma^2 I)^-1 K) + sigma^2`. Max relative error over 5 seeds:

```
9999 levels, 1e-4..1-1e-4              0.0031     <- THE GRID USED for TabICL
9999 levels, 1e-5..1-1e-5              0.0003
 999 levels, 1e-4..1-1e-4              0.0040
 999 levels, 0.001..0.999              0.0292
   9 levels, 0.1..0.9 (TabICL default) 3.5229
```

The grid used is `np.linspace(1e-4, 1-1e-4, 9999)` (`exp12_ambient_jacobians.py:54`), worst-case
relative error **`0.0031`**, per seed `[0.0031, 0.0031, 0.0031, 0.0031, 0.0031]`. Gate is `< 5%`.

Correction to revision 2: that row was absent from the original table, which annotated the
`999 levels, 1e-4..1-1e-4` row as the one used and quoted `0.0003` — the value belonging to the
`[1e-5, 1-1e-5]` range — in the sentence beneath. Truncation range, not point count, dominates this
integral: widening `[1e-4, 1-1e-4]` to `[1e-5, 1-1e-5]` gains a factor of 10, while going from 999 to
9999 points on the same range gains only `0.0040 -> 0.0031`.

**Variance channel per model, stated explicitly:**
- **TabICL v2** — `output_type="quantiles"` with 9999 uniform `alphas`, integrated by
  `extract_variance`. **Not** `output_type="variance"`, which is `raw_quantiles.var(dim=-1)`
  (`tabicl/_model/tabicl.py:582`, docstring *"Variance of the predicted quantiles (fast, no tail
  modeling)"*) — a spread over the quantile grid. On seed 42 that estimator reads `0.2507` where the
  integrated second moment reads `0.5282`, a factor of `2.107`. `e2_5_cross_channel.py:26` used the
  former.
- **TabPFN v2** — `FullSupportBarDistribution.variance(logits) = mean_of_square(logits) −
  mean(logits)^2`, `tabpfn/architectures/base/bar_distribution.py:606` and `:412`, with half-normal
  tails on the outer bins. Bin probabilities are `softmax(logits)`, `5000` bins, `5001` borders.
  `crit.mean(logits)` reproduces `predict(output_type="mean")` exactly.
- **TabSwift** — `NOT COMPUTABLE`. `Linear(384,1)` point head, `has_predictive_distribution=False`
  in `models/registry.py:134`.

**Correction to the prompt.** Task 1.2 asks for TabPFN's variance as "bar-distribution bin
probabilities x centres". That formula gives `E[X]`, not the second moment; and applied to a
`FullSupportBarDistribution` it would omit the half-normal tails on the outer two bins. The library
method `mean_of_square` uses `(l^2 + r^2 + l*r)/3` per bucket, which is the exact second moment of a
uniform density on `[l, r]`, with the outer bins replaced by half-normals. That is what was used.

### 5.2 TabICL v2 — `h=1e-3`, ambient

```
seed         slope    intercept           R^2    sigma2_hat      s_y      cv_s      cv_J
42     -0.23715373     1.313420    0.03125857     -0.120889   1.4006   0.31372   0.33042
100    -0.21952813     1.432392    0.04055271     -0.104555   1.4490   0.19747   0.28158
200    -0.20387497     1.735116    0.01781061     -0.104656   1.3957   0.13527   0.32881
300     0.00484854     0.595611    0.00005461      0.002770   1.3229   0.21736   0.27665
400     0.15871847     1.390447    0.00959991      0.053614   1.7206   0.11889   0.22506
mean   -0.09939796                 0.01985528                          0.19654   0.28850
std     0.15610000                 0.01456000                          0.06921   0.03897
```

### 5.3 TabPFN v2 — `h=1e-1`, dither halfwidth `delta`, `N=10`, ambient

```
seed         slope    intercept           R^2    sigma2_hat      s_y      cv_s      cv_J
42      0.09682691     1.684646    0.02184280      0.049357   1.4006   0.04223   1.21014
100     0.17947987     1.807893    0.00778359      0.085481   1.4490   0.02458   0.56027
200     0.19434109     1.165380    0.03745496      0.099762   1.3957   0.10090   0.47289
300     0.09975854     1.471820    0.10576231      0.057002   1.3229   0.03088   1.29077
400     0.77149951     1.049749    0.35972481      0.260605   1.7206   0.05668   0.46508
mean    0.26838118                 0.10651369                          0.05105   0.79983
std     0.25470000                 0.13100000                          0.02721   0.37030
```

Without dither, same `h`: `R^2` mean `0.10970469`, slope mean `0.26555813`. Dither changes `R^2` by
`0.0032`.

### 5.4 TabSwift

`NOT COMPUTABLE` — no predictive distribution.

### 5.5 Reference row and the secondary diagnostic

From `chunk2_results.json`, identical pipeline, exact-GP mean map with the variance channel varied:

| head | slope | R² | cv_s | `cv_s <= cv_J` |
|---|---|---|---|---|
| exact GP (true) | `0.250000000` | `1.000000000000` | `0.0894008` | holds |
| hierarchical GP (true) | `0.250000001` | `1.000000000000` | `0.0817580` | holds |
| constant head | `0.000000000` | undefined, `SS_tot = 3.081e-31` | `0.0000000` | holds |
| scaled head | `0.360249836` | `0.047717761` | `0.5386459` | does not hold |
| permuted head | `0.015091401` | `0.010468752` | `0.0894008` | holds |
| wrapped exact GP | — | `0.99999115` | `0.0909900` | holds |

`cv_s` and `cv_J` are reported **as a secondary diagnostic only**. `cv_s <= cv_J` is a necessary
consequence of A3 and is **not sufficient**: permuting `s^2` across context points leaves `cv_s`
bit-identical to the correct value (`0.0894008` in both rows above) while `R^2` falls from
`1.000000000000` to `0.010468752`. It holds for both models here — TabICL `0.19654 <= 0.28850`,
TabPFN `0.05105 <= 0.79983`.

### 5.6 Leverage

Gate condition `std(J_ii)/|mean(J_ii)| > 0.05`, i.e. `cv_J`:

| model | `cv_J` per seed | `range(J_ii)` | `range(s^2_i)` |
|---|---|---|---|
| TabICL v2 | `0.33042, 0.28158, 0.32881, 0.27665, 0.22506` | `[0.00250, 1.11751]` | `[0.31296, 2.44115]` |
| TabPFN v2 | `1.21014, 0.56027, 0.47289, 1.29077, 0.46508` | `[-0.63814, 0.79610]` | `[1.18129, 2.31623]` |

### 5.7 `s_y` per context, for the `sigma^2 * s_y^2` conversion

For a normaliser-wrapped map the recovered slope is `sigma^2 * s_y^2`, not `sigma^2`
(`chunk2_results.json`, wrapped GP: slope rel err `2.7e-3` against `sigma^2 s_y^2`).

```
seed 42: s_y = 1.400625   seed 100: 1.449024   seed 200: 1.395749
seed 300: 1.322935        seed 400: 1.720577
```

`sigma2_hat = slope / s_y^2` is tabulated per seed in 5.2 and 5.3.

### 5.8 No-intercept fit, the form the law specifies

`s^2_i = b * (1 + J_ii)`, intercept constrained to zero. Two `R^2` conventions differ for a
no-intercept model and both are given: `R^2_unc = 1 - SS_res/sum(y^2)`,
`R^2_cen = 1 - SS_res/sum((y - mean y)^2)`, the latter negative when the constrained fit is worse
than the response mean.

```
                     b (no-intercept)   R^2_uncentred   R^2_centred    R^2 with intercept
exact GP control      0.250000000        1.000000000     +1.000000      1.000000000
hierarchical GP       0.250000000        1.000000000     +1.000000      1.000000000
TabICL v2   (mean)    0.723787           0.94370808      -0.418772      0.01985528
TabPFN v2   (mean)    1.528936           0.98969067      -4.955435      0.10651369
```

Per seed:
```
TabICL v2   b       [0.54443869  0.62024002  1.03302864  0.34588799  1.06033620]
            R^2_unc [0.88225804  0.94142742  0.97046813  0.94253646  0.98185036]
            R^2_cen [-0.314042  -0.560620  -0.643363  -0.273757  -0.302078]
TabPFN v2   b       [1.61788481  1.91188331  1.07922645  1.38388088  1.65180600]
            R^2_unc [0.98855572  0.99896312  0.98237788  0.98211483  0.99644181]
            R^2_cen [-5.428266  -0.717780  -0.748381  -17.771631  -0.111117]
```

The uncentred `R^2` of `0.94` and `0.99` is not a fit-quality statement: `s^2` is strictly positive
and far from zero (TabICL `[0.31, 2.44]`, TabPFN `[1.18, 2.32]`), so any `b` of roughly the right
scale explains most of `sum(y^2)`. The centred figure is the informative one and is negative for
both models. Both controls return `b = 0.250000000`, the true `sigma^2`, with `R^2 = 1.000000000`
under either convention.

**On the fitted intercepts in 5.2 and 5.3.** They absorb `mean(s^2)`. When the regressor carries no
information about the response, least squares drives the slope toward zero and the intercept toward
the response mean; TabICL's intercepts `[1.313, 1.432, 1.735, 0.596, 1.390]` sit near its per-seed
`mean(s^2)`. On the exact GP the law holds pointwise, so there is nothing for an intercept to
absorb and it returns `-1.97e-14` to `+2.75e-14`.

---

## Section 6 — Mechanism profiling, ambient

### 6.1 Why ambient

Reduced-basis profiling was abandoned. Nadaraya–Watson, `J = D^-1 K`, is row-scaled symmetric by
construction; in ambient coordinates it row-profiles to `9.503e-16` with residual `1.5e-15`, and in
the reduced basis it profiles to `0.259940` from a raw `0.249240` — the asymmetry rises
(`CONTROL_VALIDATION.md`, Chunk 1; `chunk1_profile_control.json`).

### 6.2 Controls

```
Nadaraya-Watson ambient (positive control), 5 seeds
  raw_asym   [2.835e-01 2.457e-01 2.801e-01 2.356e-01 2.849e-01]  mean 2.659e-01
  row_asym   [8.910e-16 8.573e-16 1.030e-15 9.791e-16 1.017e-15]  mean 9.503e-16
  row_resid  [1.460e-15 1.420e-15 1.569e-15 1.610e-15 1.685e-15]  mean 1.549e-15

wrapped ExactGP (must not be inflated)
  worst inflation factor, profiled/raw, weighted form: 0.9784
  Chunk 1 reference, UNWEIGHTED form on the same control: 3.1e-12 -> 5.3e-11, a factor of 17

targeted imitator (must survive; not of diagonal-scaling form)
  raw_asym   mean 5.974226e-01
  col_asym   mean 5.939819e-01   col_resid mean 9.560754e-01
  row_asym   mean 5.940019e-01   row_resid mean 9.560754e-01
```

Gate conditions: NW ambient row-profiled asym required `< 1e-12`, measured `9.503e-16`; wrapped
ExactGP inflation required `< 2`, measured `0.9784`.

### 6.3 Models, ambient, 5 seeds

```
TabICL v2   h=1e-3, no dither
  raw_asym   [7.768739e-01 7.718090e-01 9.759801e-01 5.752362e-01 7.637388e-01]  mean 7.727276e-01  std 1.268e-01
  col_asym   [4.993468e-01 4.624947e-01 6.382187e-01 3.749477e-01 5.257809e-01]  mean 5.001578e-01  std 8.581e-02
  col_resid  [5.221588e-01 5.379276e-01 5.542594e-01 6.355178e-01 5.997597e-01]  mean 5.699246e-01  std 4.182e-02
  row_asym   [5.909998e-01 4.965585e-01 8.264927e-01 4.877180e-01 5.873206e-01]  mean 5.978179e-01  std 1.223e-01
  row_resid  identical to col_resid

TabPFN v2   h=1e-1, dither halfwidth delta, N=10
  raw_asym   [1.359629e+00 1.242025e+00 1.244677e+00 1.387674e+00 1.087600e+00]  mean 1.264321e+00  std 1.062e-01
  col_asym   [1.157976e+00 1.052651e+00 1.175524e+00 1.298327e+00 1.015138e+00]  mean 1.139923e+00  std 9.987e-02
  col_resid  [4.453141e-01 5.641566e-01 7.754758e-01 5.100247e-01 7.853236e-01]  mean 6.160590e-01  std 1.394e-01
  row_asym   [1.279568e+00 1.195012e+00 1.232830e+00 1.370364e+00 1.040730e+00]  mean 1.223701e+00  std 1.086e-01

TabPFN v2   h=1e-1, no dither
  raw_asym   mean 1.269843e+00   col_asym mean 1.146172e+00   row_asym mean 1.227480e+00
  col_resid  mean 6.157567e-01

TabSwift    h=1e-1, no dither
  raw_asym   [1.353698e+00 1.354230e+00 1.261967e+00 1.221732e+00 1.036966e+00]  mean 1.245719e+00  std 1.165e-01
  col_asym   [1.081019e+00 1.045715e+00 9.318217e-01 6.930478e-01 8.027766e-01]  mean 9.108760e-01  std 1.461e-01
  col_resid  [4.414564e-01 3.943076e-01 4.690540e-01 5.251750e-01 5.503480e-01]  mean 4.760682e-01  std 5.632e-02
  row_asym   [1.253387e+00 1.307871e+00 1.151962e+00 1.142602e+00 7.241408e-01]  mean 1.115993e+00  std 2.055e-01
```

Ambient raw asym is a different object from the reduced-basis asym in Section 3: N1(i) puts an
asymmetric rank-one term in the ambient Jacobian of any normaliser-wrapped map. Against the ambient
floor measured in Section 2.9 (wrapped ExactGP, `1.235779e-02`):

| model | ambient raw asym | ambient floor | ratio | reduced-basis asym |
|---|---|---|---|---|
| TabICL v2 | `0.7727276` | `0.01235779` | `62.5x` | `0.580331` at `t=1e-3` |
| TabPFN v2 | `1.2643210` | `0.01235779` | `102.3x` | `1.312414` at `t=1e-1` |
| TabSwift | `1.2457190` | `0.01235779` | `100.8x` | `1.020249` at `t=1e-1` |

The ambient-to-reduced gap does not share a direction across the three, so it is not accounted for by
N1(i) alone. Two limits on the ambient floor: it is specific to this control, since N1(i)'s
antisymmetric contribution scales with how far `g` departs from parallel to `u` and a GP at
`sigma = 0.5` smooths mildly; and it applies only to normaliser-wrapped maps, which by Section 2.7
excludes TabSwift. The reduced-basis floor (`3.141e-12`, Section 2.2) is nine orders lower and does
not depend on the control's smoothing strength.

### 6.4 Recovered diagonal profiles

| | row profile range | col profile range | all positive |
|---|---|---|---|
| Nadaraya–Watson (control) | `[4.384e-01, 1.038e+00]` | `[9.638e-01, 2.281e+00]` | yes |
| targeted imitator (control) | `[7.449e-01, 1.291e+00]` | `[7.745e-01, 1.343e+00]` | yes |
| TabICL v2 | `[1.391e-01, 1.207e+01]` | `[8.282e-02, 7.191e+00]` | yes |
| TabPFN v2 | `[1.713e-01, 1.164e+01]` | `[8.593e-02, 5.838e+00]` | yes |
| TabSwift | `[7.721e-02, 1.305e+01]` | `[7.664e-02, 1.295e+01]` | yes |

The two controls recover profiles spanning a factor of `2.4` and `1.7`. The three models recover
profiles spanning factors of `87`, `68` and `169`.

### 6.5 What each null covers

- **Column system.** `J_ij sigma_j^2 = J_ji sigma_i^2`, the form `J = Cov(f|y) Sigma^-1` takes under
  heteroscedastic noise (N6). Residual `0.4761–0.6161` across the three models against `1.5e-15` on
  the NW control.
- **Row system.** `d_i J_ij = d_j J_ji`, the form `J = D^-1 K` takes for a kernel smoother or
  attention-weighted label vote (N5). Same residuals, by construction.
- **Residuals are identical between the two systems** because `b_row = -b_col`, so the least-squares
  solution is the negation and the norms coincide. Only post-profile `asym` separates them: column
  scaling gives a lower `asym` than row scaling for all three models
  (`0.5002 < 0.5978`, `1.1399 < 1.2237`, `0.9109 < 1.1160`).

---

## Section 7 — Theory verification

### 7.1 T2, `sigma^2 J = Cov(f|y)`

Exact GP, relative Frobenius error per seed, `chunk2_results.json → exactgp.t2_rel_err`:
```
[6.382e-13  6.311e-13  6.178e-13  5.636e-13  8.352e-13]
```

### 7.2 T4, `sigma^2 J = E_{theta|y}[C_theta] + Cov_{theta|y}(mu_theta)`

Hierarchical GP, relative Frobenius error per seed, `chunk2_results.json → hiergp.t4_rel_err`:
```
[1.206e-09  1.472e-09  4.170e-08  3.098e-09  7.536e-09]
```

### 7.3 N1(ii) projection exactness

`Q^T J_wrapped Q` vs `Q^T J_unwrapped Q` on the exact GP, relative Frobenius difference per seed:
```
[2.679942e-12  2.743502e-12  2.981320e-12  2.605865e-12  3.305024e-12]  mean 2.863131e-12
```

### 7.4 N3, `negeig` vs `negfrac` on the rank-deficient control

Quantised wrapped exact GP, true `negeig = 0`, true `negfrac = 0`:

| `t` | `negeig` | `negfrac` |
|---|---|---|
| 1e-2, dither `delta` | `0.001893` | `0.0918367` |
| 1e-1, dither `delta` | `0.0000178` | `0.0918367` |

`negfrac` is `0.0918367 = 9/98` at both amplitudes, at both bracket steps in Section 4.2, and on
every seed — it does not respond to the probe at all. `negeig` falls by a factor of `106` between the
two amplitudes over the same matrices.

---

## Section 8 — Explicitly not measured

| Quantity | Reason |
|---|---|
| Curvature on models | Measured at a single amplitude only. The artifact scales as `t^-2` (ExactGP floor `3.98e-10 → 3.97e-16` across `t = 1e-3 … 1`) while genuine curvature is flat in `t` (HierarchicalGP `2.868e-03 → 2.764e-03` over the same range), so one amplitude cannot separate them. Separately, `theory_and_claims.md` places the Bayesian reading of curvature out of scope wherever A1 or A2 fail, and the metric is not the M0 tangential great-circle curvature with geodesic correction. |
| A4 decay law | Same scope exclusion. |
| `negfrac`, all models | Context rank-deficiency artifact; Section 4.4. |
| `cv_s <= cv_J` as a pass | Permutation-invariant; a permuted variance head gives a bit-identical `cv_s` with `R^2 = 0.0105`. Section 5.5. |
| Reduced-basis profiling | NW control profiles to `0.259940` after projection from raw `0.249240`; Section 6.1. |
| Value-level battery, E4.2 only | E4.2 (imitator on the value battery) not run. **E4.1 is now measured — see Section 10**, added 2026-08-26. |
| E1.4 for TabPFN v2 | Requires source modification of the TabPFN attention path. |
| Determinism on models (E0.1) | Requires a GPU refit-twice pass on the audit path. |
| E0.2 N5 analytic battery | Script exists, no saved output, covers 6 of 9 rows. |
| E0.4 autograd cross-check | No code. |
| Tier 3 robustness, E3.1–E3.5 | Not run. One context family throughout: `n=100, d=5, sigma=1.0`. |
| E5.1 positive control on real data | **Run 2026-08-26 — see Section 11.** A1 and A2 pass 120/120 on real OpenML data through the identical pipeline. A3 passes for the exact GP 60/60 and for the hierarchical GP 60/60 unwrapped, but the *wrapped* hierarchical GP fails 3/60, which establishes an artifact floor for A3 — see Section 5's added note. |
| E5.2 as a registered test, E5.3, E5.4 | Not run. |
| A3 for TabSwift | No predictive distribution. |
| N1(ii) exactness for TabSwift | Not established: the map satisfies neither the shift nor the scale identity (Section 2.7). Its reduced-basis A1 is reported as the asymmetry of the compressed Jacobian. A2 is unaffected. |

---

## Section 9 — Data provenance

### 9.1 Number → file → key

| Section | Quantity | File | Key |
|---|---|---|---|
| 2.1–2.3 | control asym/negeig, imitator | `tier0_instrument/chunk1_results.json` | `1A.<name>.measured` / `.analytic` |
| 2.4 | curvature calibration | `tier0_instrument/chunk1_results.json` | `1B.calibration_quadratic`, `.calibration_quadratic_jacobian` |
| 2.5 | seven stress tests | `tier0_instrument/exp_stress_tests.json` | `1_…` through `7_permutation` |
| 2.6 | artifact table, `t=1e-2` | `tier0_instrument/chunk3_results.json` | `3.2 …`, `3.3 dither OFF`, `3.5_sweep` |
| 2.6 | artifact table, `t=1e-1` | `tier0_instrument/exp3_reprobe.json` | `artifact_t1e-01` |
| 3.2 | TabPFN asym headline | `tier0_instrument/exp3_reprobe.json` | `tabpfn_dither_t1e-01[].asym` |
| 3.2 | TabPFN amplitude sweep | `tier0_instrument/exp3_reprobe.json` | `tabpfn_nodither_t{1e-02,3e-02,1e-01,3e-01}` |
| 3.3 | TabSwift asym | `tier0_instrument/exp3_reprobe.json` | `tabswift_t{1e-02,1e-01,1e+00}` |
| 3.6 | E1.4 paired | `tier0_instrument/e14_paired.json` | `rows`, `paired_mean`, `ci95` |
| 4.1 | negeig, models | `tier0_instrument/exp3_reprobe.json` | same keys, `.negeig` |
| 4.3 | negative eigenvector | `tier0_instrument/exp_negeigvec.json` | `tabpfn_dither_t1e-01`, `tabswift_t1e-01` |
| 4.3 | `#(J_ii < 0)` | `tier0_instrument/exp1_a3_final.json` | `<tag>[].J_neg` |
| 4.4 | rank-deficiency spectrum | recomputed in-session from `core/surrogates.py` `ExactGP.jacobian` | — |
| 5.1 | extractor validation | `tier0_instrument/exp1_extractor_validation.json` | grid names |
| 5.2–5.3 | A3 regressions | `tier0_instrument/exp1_a3_final.json` | `tabicl_v2`, `tabpfn_v2_dither`, `tabpfn_v2_nodither` |
| 5.5 | reference heads | `tier0_instrument/chunk2_results.json` | `exactgp`, `hiergp`, `broken.*`, `wrapped` |
| 6.2–6.4 | profiling | `tier0_instrument/exp2_ambient_profiling.json` | `nw_ambient`, `exactgp_wrapped_ambient`, `targeted_imitator_ambient`, model tags |
| 7.1–7.2 | T2, T4 | `tier0_instrument/chunk2_results.json` | `exactgp.t2_rel_err`, `hiergp.t4_rel_err` |
| 7.3 | N1(ii) | `tier0_instrument/exp_n1ii.json` | `n1ii_wrapped_vs_unwrapped_rel` |
| 3.1 | TabICL reduced A1/A2, sweep, non-degeneracy | `tier0_instrument/exp4_results.json` | `tabicl_reduced.t{1e-03,1e-02,1e-01}` |
| 3.4 | measured output quanta | `tier0_instrument/exp4_results.json` | `quantum_tabicl`, `quantum_tabswift` |
| 3.4 / 4.2 | artifact floors, upper bracket | `tier0_instrument/exp4_results.json` | `tabicl_floor`, `tabswift_floor` |
| 3.4 / 4.2 | artifact floors, both brackets | `tier0_instrument/exp4_floor_brackets.json` | `TabICL v2`, `TabSwift` |
| — | raw TabICL reduced Jacobians | `tier0_instrument/exp4_tabicl_reduced.npz` | `tabicl_t<amp>__J__<seed>` |
| 2.7 | shift / scale identities, all models + controls | `tier0_instrument/e11_wrapper_form.json` | `<tag>[].shift.<c>`, `<tag>[].scale.<c>` |
| 2.7 | dither setting for E1.1 | `tier0_instrument/e11_wrapper_form.json` | `_config.dither` (`false`) |
| 2.8 | `r_1`, all models + controls | `tier0_instrument/clarif_item4_6.json` | `<tag>_r1` |
| 2.9 | ambient floor | `tier0_instrument/clarif_item4_6.json` | `exactgp_wrapped_ambient.*`, `exactgp_unwrapped_ambient.*` |
| 4.2 | artifact floors, lower bracket | `tier0_instrument/exp4_floor_brackets.json` | `<model>.lower (p10 jump).*` |
| 4.3 | TabSwift `#(J_ii<0)` | `tier0_instrument/clarif_item4_6.json` | `tabswift_diag_neg` |
| 5.8 | no-intercept fits | `tier0_instrument/clarif_item5.json` | `exactgp_unwrapped`, `hiergp_unwrapped`, `tabicl_v2`, `tabpfn_v2_dither` |
| 6.3 | model ambient raw asym | `tier0_instrument/exp2_ambient_profiling.json` | `<tag>.rows[].raw_asym` |
| 7.4 | `negfrac` on the floors | `tier0_instrument/exp3_reprobe.json` | `artifact_t{1e-02,1e-01}[].negfrac` |
| — | raw ambient Jacobians | `tier0_instrument/exp12_ambient_jacobians.npz` | `<tag>__J__<seed>`, `__y__`, `__s2__` |
| — | raw reduced Jacobians | `tier0_instrument/exp3_reduced_jacobians.npz` | `<tag>__J__<seed>` |
| — | run configs | `tier0_instrument/exp12_ambient_meta.json` | per tag |

### 9.2 Carry-over statement

No number in this document is taken from `final_audit_report.md`, `rebuttal_5_standardize.py`, or
`rebuttal_6_gaps.py`.

Two values are carried from an earlier run, labelled historical at each appearance, and shown only
beside a value measured this run: TabPFN `t=1e-2` `asym 1.3309 ± 0.0443` and `negeig 0.4003`
(`results_tabpfn_5seed.json`, `asym` doubled from the halved convention).

TabICL's reduced-basis `asym` and `negeig` are **no longer carried**: they were re-measured this run
(Section 3.1) and reproduce the historical `0.5802 ± 0.0784` / `0.1769 ± 0.0464` to four significant
figures.

### 9.3 Scripts written or modified in this run

| File | Status | Output |
|---|---|---|
| `core/metrics.py` | modified — added `negfrac`, `second_difference_norm`, seeded `get_Q`, `reduced_jacobian`, `profile_jacobian` | backup `core/metrics.py.chunk0_backup` |
| `core/surrogates.py` | modified — added `predictive_variance`, `posterior_covariance` | backup `core/surrogates.py.chunk1_backup` |
| `tier2_audit/instrument_c_d.py` | modified — `get_metrics` returned `negfrac` as `negeig` | backup `.chunk0_backup` |
| `tier0_instrument/chunk1_control_validation.py` | new | `chunk1_results.json`, `chunk1_profile_control.json` |
| `tier0_instrument/chunk2_a3_controls.py` | new | `chunk2_results.json` |
| `tier0_instrument/chunk3_dither_controls.py` | new | `chunk3_results.json`, `chunk3_control_matrix.json` |
| `tier0_instrument/exp1_extractor_validation.py` | new | `exp1_extractor_validation.json` |
| `tier0_instrument/exp12_ambient_jacobians.py` | new | `exp12_ambient_jacobians.npz`, `exp12_ambient_meta.json`, `exp12_ambient.log` |
| `tier0_instrument/exp2_ambient_profiling.py` | new | `exp2_ambient_profiling.json` |
| `tier0_instrument/exp3_reprobe.py` | new | `exp3_reprobe.json`, `exp3_reduced_jacobians.npz`, `exp3_reprobe.log` |
| `tier0_instrument/exp4_quanta_and_tabicl.py` | new | `exp4_results.json`, `exp4_tabicl_reduced.npz`, `exp4.log` |
| `tier0_instrument/e11_wrapper_form.py` | new | `e11_wrapper_form.json` |
| in-session scripts, **code recovered 2026-08-26** | `tier0_instrument/recompute_orphans.py` | `exp_stress_tests.json`, `exp_negeigvec.json`, `exp_n1ii.json`, `exp1_a3_final.json`, `exp1_tabicl_a3.json`, `e14_paired.json`, `exp4_floor_brackets.json`, `clarif_item4_6.json`, `clarif_item5.json` |

**Note on the row above.** These files were written in-session and their code was never saved. It has
since been reconstructed as `tier0_instrument/recompute_orphans.py`, which recomputes every affected
quantity from the tracked `.npz` Jacobians and compares it against the record. Result: **292 of 293
checks reproduce, and none differs** — 262 bit-identical, 25 agreeing to `≤ 1e-12` relative, 5 pairs
of instrument-error floors that are both at machine precision (true value exactly zero, so only
magnitude is meaningful). **No number in this document changed.** The single exception is Section
3.6, the E1.4 paired RoPE analysis, which needs fresh forward passes through a *modified* TabICL and
is recorded as NOT RECOMPUTABLE rather than approximated. Full per-quantity table in
`tier0_instrument/recompute_orphans.json`.
| `tier0_instrument/exp1_a3_models.py` | new, superseded by `exp12_…` | partial log only; not a source of any number here |

### 9.4 Gate conditions as measured

| Gate | Condition | Required | Measured |
|---|---|---|---|
| 1 | extractor vs exact GP | `< 5%` | `0.0003` |
| 1 | leverage `cv_J` | `> 0.05` | TabICL `0.22506–0.33042`; TabPFN `0.46508–1.29077` |
| 2 | NW ambient row-profiled asym | `< 1e-12` | `9.503e-16` |
| 2 | wrapped ExactGP inflation | `< 2` | `0.9784` |
| 3 | TabPFN signal-to-artifact, `t=1e-1` vs `t=1e-2` | ratio improves | asym `13.7x → 134.7x`; negeig `211.5x → 22711.5x` |
| E1.1 | shift and scale identities | `< 1e-3` relative | TabPFN `7.270e-07` / `0.000e+00`; TabICL `2.206e-06` / `0.000e+00`; TabSwift `3.379e+00` / `3.693e+00` |

---

## Section 10 — Value channel, E4.1

**Added 2026-08-26.** Produced by `tier4_value/e4_1_value_battery.py` →
`tier4_value/e4_1_results.json`. Supersedes the Section 8 row that read "Value-level battery (E4.1,
E4.2) — Not run."

Same contexts as the rest of this document: `generate_audit_context(n=100, d=5, sigma=1.0, seed=s)`,
seeds `[42, 100, 200, 300, 400]`. The contexts are **not modified**: held-out queries are drawn from
the GP conditional `f_* | f` at 200 fresh `X_* ~ N(0, I_5)` per seed, so `(X, y, f)` stay
bit-identical to what every other script sees.

### 10.1 Read this before judging any MSE

The RBF kernel has unit signal variance and the context noise is `sigma = 1.0`, so the
signal-to-noise ratio is **1 by design**. Worse, at `d = 5` with `lengthscale = 1.0` the kernel is
nearly diagonal — the median off-diagonal `K_ij` is `0.0158` (seed 42) and `0.0099` (seed 100), with
only `17–20%` of pairs above `0.1`. **The Bayes-optimal oracle itself reaches only `R² = 0.1425` on
held-out data.** A raw MSE is therefore uninterpretable here, and every number below is reported
against two fixed points:

    efficiency = (MSE_constant - MSE_model) / (MSE_constant - MSE_oracle)

`0` = predicting the context mean, `1` = the Bayes-optimal oracle GP (true kernel, true `sigma`).

### 10.2 Headline, mean over 5 seeds

`ctx_vs_f` is `MSE(m(y), f)` at the 100 context points — **the object the Jacobian audit
differentiates**. `test_vs_f` is `MSE(m(x_*), f_*)` at the 200 held-out queries. NLL is Gaussian
under each predictor's own mean and variance.

| | `ctx_vs_f` | /oracle | **eff_ctx** | `test_vs_f` | /oracle | **eff_test** | NLL | cov@90 |
|---|---|---|---|---|---|---|---|---|
| oracle GP (`sigma=1.0`, true model) | `0.3288` | `1.00` | `1.000` | `0.6827` | `1.00` | `1.000` | `1.6764` | `0.894` |
| hierarchical GP | `0.3514` | `1.07` | `0.968` | `0.7365` | `1.07` | `0.805` | `1.6963` | `0.906` |
| audit control GP (`sigma=0.5`) | `0.4293` | `1.32` | `0.858` | `0.7540` | `1.11` | `0.741` | `1.9203` | `0.722` |
| **TabICL v2** | `0.4936` | `1.51` | `0.768` | `0.8393` | `1.25` | `0.431` | `1.7554` | `0.896` |
| **TabSwift** | `0.5876` | `1.82` | `0.636` | `0.8571` | `1.26` | `0.367` | N/A | N/A |
| **TabPFN v2** | `0.6703` | `2.06` | `0.519` | `0.9180` | `1.37` | `0.145` | `1.7584` | `0.886` |
| constant `mean(y)` | `1.0387` | `3.37` | `0.000` | `0.9580` | `1.42` | `0.000` | `1.7666` | `0.914` |
| 1-NN | `0.9567` | `2.99` | `0.116` | `2.1456` | `3.19` | `-4.314` | N/A | N/A |

Per-seed values in `e4_1_results.json → per_seed`. `R²` against held-out `y`: oracle `0.1425`,
TabICL `0.0402`, TabSwift `0.0486`, **TabPFN `-0.0042`**, constant `-0.0019`.

TabSwift has no predictive distribution (`Linear(384,1)`, `registry.has_predictive_distribution =
False`), so NLL, coverage and interval score are **N/A, not zero and not synthesised**.

### 10.3 Verdict

**The models are not broken on these contexts, and the audit is not confounded with
out-of-distribution failure.** All three denoise substantially in the regime the audit measures:
`eff_ctx` of `0.77 / 0.64 / 0.52` against `0.12` for 1-NN and `0.00` for the constant predictor.
All three beat every trivial baseline on both regimes.

**They are, however, materially worse than a correctly-specified Bayesian**, at `1.5x`, `1.8x` and
`2.06x` the oracle's in-sample error. That gap is the honest headline and should not be softened.

**The uncertainty channel is the interesting part.** Coverage at 90% is `0.896` (TabICL) and `0.886`
(TabPFN) against the oracle's `0.894`, and NLL is `1.755` / `1.758` against the oracle's `1.676` —
both well calibrated, both close to optimal. Note that the **misspecified audit control GP is the
worst-calibrated map in the table** (cov@90 `0.722`), which is what an overconfident `sigma = 0.5`
on `sigma = 1.0` data should do, and confirms the coverage statistic has power here.

So on these contexts the two frozen models with a predictive distribution are **well calibrated and
near-oracle in NLL while failing A1 and A2 by factors of 57x–22711x over the artifact floor**
(Sections 3.4, 4.2). That is C-X2 — the value channel and the derivative channel disagreeing on the
same model on the same contexts — instantiated empirically rather than by construction.

**Scope, stated not hidden.** This is one context family, and a deliberately hard one. It does not
establish that the models perform at their *published* level, because published benchmarks are real
tabular data with exploitable structure, not a near-white `d=5` GP at SNR 1. It establishes the
narrower thing E4.1 was for: on the contexts where the violations were measured, the models work,
are calibrated, and beat every trivial baseline. E3.1 remains the experiment that would settle the
distributional question.

### 10.4 Provenance

| Quantity | File | Key |
|---|---|---|
| 10.2 all rows | `tier4_value/e4_1_results.json` | `aggregate.<name>` |
| per-seed | `tier4_value/e4_1_results.json` | `per_seed.<seed>[]` |
| config, query rule | `tier4_value/e4_1_results.json` | `_config` |

---

## Section 11 — Positive control on real data, E5.1

**Added 2026-08-26.** Produced by `tier5_positive/e5_1_positive_control.py` →
`tier5_positive/e5_1_results.json`. Supersedes the Section 8 row that read "E5.1 positive control on
real data — Not run."

`experiments.md` calls this **blocking**: "Nothing is reported before a real Bayesian model passes on
real data." Its falsifier is the one that ends the project — if a real Bayesian fails here, the
pipeline rather than the models is producing the violations.

### 11.1 Setup

Exact GP and hierarchical GP, wrapped in the same context-statistic normaliser the audited models
wear, through the **identical** pipeline: `core.metrics.reduced_jacobian`, `get_Q(y, seed=0)`,
`core.metrics.asym` / `negeig`, and the A3 ordinary least squares of `chunk2_a3_controls.py`.

Data: the Phase 2 OpenML selection read from the tracked `phase_2/arrays/chunk2_data.npz` — **12
datasets × 5 splits = 60 contexts per control**, `n_ctx = 100`, features standardised on context rows
only, targets in original units (sd `0.75` to `1083`).

**Probe step: relative**, `h = 6.860e-4 · std(y_ctx)` — the audit's own `1e-3` divided by its mean
context label sd `1.457782`, exactly as `phase_2/src/chunk2_estimators.py` defines it. These controls
are analytic float64 with no output quantisation, so an absolute step would not have broken them; the
relative step is used because E5.1's claim is "through the identical pipeline", and handing the
control an easier probe than the models got would void the comparison. A four-decade step sweep is
reported in `e5_1_results.json → sensitivity` and moves `asym` from `1.15e-9` to `1.18e-13` while
`negeig` stays exactly `0` and A3 `R²` stays `0.999982` throughout.

**Lengthscale: the median heuristic** per context, to avoid a *vacuous* pass. A GP is
Bayes-realisable for any hyperparameters so A1/A2 cannot be tuned into passing; the real risk is
degeneracy, since an RBF kernel with lengthscale `1.0` on standardised features at `d = 25` or `50` is
nearly diagonal and would land on N5's `J = I` row. A lengthscale sweep over `×0.25 … ×4` is
reported and moves nothing.

### 11.2 Result

| | A1 `asym` max | A2 `negeig` max | non-degenerate | A1 | A2 | A3 |
|---|---|---|---|---|---|---|
| exact GP, wrapped | `1.702e-11` | `0.000e+00` | `60/60` | **`60/60`** | **`60/60`** | **`60/60`** |
| hierarchical GP, wrapped | `2.643e-08` | `0.000e+00` | `53/60` | **`60/60`** | **`60/60`** | `57/60` |
| hierarchical GP, **unwrapped** | `2.642e-08` | `0.000e+00` | `53/60` | `60/60` | `60/60` | **`60/60`** |

Gates: A1 `≤ 0.05`, A2 `≤ 0.02`, A3 `R² ≥ 0.90` and slope within `±25%` of `σ²s_y²`, non-degeneracy
`‖J−I‖_F/‖J‖_F > 0.3`.

**A1 and A2 pass 120/120 across both wrapped controls.** `asym` clears its gate by nine orders of
magnitude on the exact GP and six on the hierarchical GP; `negeig` is **exactly zero** on all 120,
not merely small. A3 `R²` for the exact GP: min `0.999974`, mean `0.999990`, slope within `7.8e-3` of
`σ²s_y²`.

**Tangential curvature separates the two controls by `8.1e6`×** — hierarchical `3.122e-3` against
exact `3.848e-10` — which is the T3/T4 prediction. Caveat stated: this is the second difference along
`Q`'s columns, which preserves `ȳ` exactly and `‖y−ȳ‖` to second order. It is **not** the full M0
great-circle construction with the geodesic correction.

### 11.3 The one failure, and what it is

The wrapped hierarchical GP fails A3 on 3 of 60. Two diagnostic conditions, each changing exactly one
thing, locate the cause:

| condition | A1 | A2 | A3 | A3 `R²` min |
|---|---|---|---|---|
| wrapped, shipped lengthscale ladder (primary) | `60/60` | `60/60` | `57/60` | `0.675` |
| wrapped, ladder rescaled to the data | `60/60` | `60/60` | `55/60` | `0.843` |
| **unwrapped** | `60/60` | `60/60` | **`60/60`** | **`1.000000`** |

**It is the normaliser, not the pipeline, not real data, and not the lengthscale ladder.** Rescaling
the ladder does not fix it; removing the wrapper fixes it completely. The exact GP does not show the
effect because its inner map is linear — the effect requires a *nonlinear* inner map, which all three
audited models are.

### 11.4 Verdict

**The pipeline does not manufacture A1 or A2 violations on real data.** That is E5.1's load-bearing
claim and it holds without qualification: two real Bayesian predictors, 120 real contexts, identical
pipeline, `negeig` exactly zero throughout and `asym` at `1e-11`–`1e-8` against models measuring
`0.58`–`1.31`.

**A3 is a weaker instrument than this document previously implied**, and Section 5 now carries the
floor this measurement establishes.

### 11.5 Provenance

| Quantity | File | Key |
|---|---|---|
| 11.2 per-context rows | `tier5_positive/e5_1_results.json` | `rows[]` |
| 11.2 aggregates | `tier5_positive/e5_1_results.json` | `summary.<kind>` |
| 11.3 diagnostics | `tier5_positive/e5_1_results.json` | `diagnostic.{unwrapped,scaled_ladder}` |
| step-size and lengthscale sweeps | `tier5_positive/e5_1_results.json` | `sensitivity` |
| verdict booleans | `tier5_positive/e5_1_results.json` | `E5_1` |
