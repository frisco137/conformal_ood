# Phase 2 Results — Jacobian-Derived Predictive Variance

First real-data measurement in this project. 12 OpenML regression datasets, 5 splits each,
`n_ctx = 100`, 3 frozen models, 4 uncertainty estimators.

**Sections 1–8 are measurement. Section 9 is interpretation and is the only part that is not a
number.** Per-split values for everything are in `results/*.json`; the full chunk-by-chunk record
with gates and checklists is `UNCERTAINTY_EXPERIMENT.md`.

---

## 0. Headline table

Aggregated per dataset first, then across the 12 datasets. Interval score and NLL: lower is better.
`rho` is Spearman correlation between the estimator's variance and the realised squared residual.

| model | estimator | cov@90 | IS@90 | width@90 | width at matched cov | NLL | rho |
|---|---|---|---|---|---|---|---|
| **TabICL v2** | native | `0.9047` | **`515.8`** | `378.5` | `299.6` | **`2.238`** | **`0.300`** |
| | Jacobian | `0.7755` | `772.0` | `582.1` | `673.8` | `871.7` | `0.054` |
| | conformal | `0.8830` | `523.6` | **`352.0`** | `344.3` | `10.68` | n/a |
| | oracle GP | `0.9073` | `545.1` | `386.1` | `326.3` | `2.708` | `0.182` |
| **TabPFN v2** | native | `0.9057` | **`515.9`** | `375.5` | `301.0` | **`2.119`** | **`0.283`** |
| | Jacobian | `0.9030` | `547.4` | `416.1` | `336.5` | `40.948` | `0.124` |
| | conformal | `0.8957` | `519.9` | `356.6` | `312.1` | `14.075` | n/a |
| | oracle GP | `0.9073` | `545.1` | `386.1` | `326.3` | `2.708` | `0.182` |
| **TabSwift** | native | — | — | — | — | — | — |
| | Jacobian | `0.9337` | `1818.0` | `1714.0` | `1283.0` | `18.091` | `0.042` |
| | conformal | `0.9153` | **`1553.0`** | **`1370.0`** | `1269.0` | **`3.261`** | n/a |
| | oracle GP | `0.9073` | `545.1` | `386.1` | `326.3` | `2.708` | `0.182` |

TabSwift has no native head: `Linear(384, 1)` point output,
`models/registry.py: has_predictive_distribution=False`. No variance was synthesised for it.

The oracle GP row is identical across the three blocks by construction — it does not depend on the
model.

**Win/loss across the 12 datasets, interval score at 90%:**

| comparison | TabICL v2 | TabPFN v2 | TabSwift |
|---|---|---|---|
| Jacobian vs native | `1 – 11` | `0 – 12` | n/a |
| Jacobian vs conformal | `0 – 12` | `4 – 8` | `1 – 11` |
| Jacobian vs oracle GP | `2 – 10` | `8 – 4` | `5 – 7` |
| native vs conformal | `9 – 3` | `8 – 4` | n/a |
| native vs oracle GP | `10 – 2` | `9 – 3` | n/a |

---

## 1. The estimator form

The brief specified `s²(x_*) = σ̂²(1 + J_**)`, with `J_** = ∂m_*/∂y_*` measured after appending
`(x_*, y_*)` to the context. Measured against the exact GP's closed-form predictive variance, using
the true `σ²` so the form's error is isolated from `σ̂²`'s:

```
                    seed 42      seed 100     seed 200     seed 300     seed 400
sigma^2 (1+J_**)   median  4.170e-01    4.854e-01    4.822e-01    4.909e-01    5.196e-01
                   p90     6.107e-01    5.963e-01    6.224e-01    6.283e-01    5.991e-01
                   max     6.290e-01    6.255e-01    6.397e-01    6.388e-01    6.379e-01

sigma^2/(1-J_**)   median  2.664e-13    2.655e-13    9.789e-14    4.733e-14    3.292e-13
                   p90     7.050e-13    4.951e-13    2.795e-13    1.335e-13    8.340e-13
                   max     1.305e-12    8.924e-13    8.933e-13    4.275e-13    1.376e-12
```

Why the two differ: appending `(x_*, y_*)` makes the query a context point, so T2 applies to it and
gives `σ²J_** = Var(f_* | y, y_*)` — conditioned on the hypothetical label. The target quantity is
`Var(f_* | y) + σ²`, which conditions on the original context only. Writing `v = Var(f_*|y)`, the
Gaussian update gives `J_** = v/(v+σ²)`, which inverts to `s² = σ²/(1 - J_**)`. Since a Bayesian has
`J_** ∈ [0,1)`, the specified form is **bounded above by `2σ²`** however uncertain the query is,
while the target is unbounded.

`σ²/(1-J_**)` is used as the primary estimator throughout. `σ̂²(1 + J_**)` is computed and stored at
every query point as `s2_jac_specified`.

**Cost of the Gaussian assumption in the inversion.** On the hierarchical GP, whose posterior is a
mixture and therefore not Gaussian, the inverted form's p90 relative error per seed is
`[2.561e-02, 7.303e-02, 1.654e-01, 4.048e-02, 1.367e-02]`, max `1.135e+00`. On the exact GP it is
`~1e-13`.

**Measured `J_**` range**, exact GP, 50 queries x 5 seeds: `[0.3077, 0.7931]`, `[0.4009, 0.7909]`,
`[0.4346, 0.7998]`, `[0.4551, 0.7992]`, `[0.3155, 0.7987]`.

### 1.1 The noise scale `σ̂² = ‖m(y) − y‖²/(n − tr J)`

Exact GP, well-specified draw with true `σ² = 0.25`:

```
seed        tr J      n - tr J    sigma2_hat    sigma2_hat / sigma^2
42       59.9054      40.0946      0.277748           1.1110
100      62.0006      37.9994      0.320954           1.2838
200      60.0735      39.9265      0.240523           0.9621
300      59.4606      40.5394      0.218714           0.8749
400      62.4741      37.5259      0.309534           1.2381
```

Mean ratio `1.0940`, sample SD `0.1750`. The estimator's sampling SD in closed form is
`sqrt(2 tr(C^-2)) σ² / (n - tr J)` with `C = K + σ²I`, which evaluates to `0.1755` per seed and
`0.0785` for the five-seed average. Hierarchical GP: mean ratio `0.9638`, sample SD `0.1573`.

### 1.2 Choice of the hypothetical label `y_*`

Four candidates, scored against the closed form (inverted form, true `σ²`), mean over seeds of the
per-seed p90 relative error:

```
                            exactgp        hiergp
context mean               4.8941e-13     6.3633e-02
self-consistent m_*(y)     8.7066e-13     5.4676e-02
mean + SD                  1.3638e-12     4.2825e-01
mean - SD                  1.9566e-12     3.7403e-01
```

`y_* = m_*(y)` is used throughout. `J_**` is exactly `y_*`-independent on the exact GP (spread
`~1e-12`, a linear map) and strongly dependent on the hierarchical GP (relative spread up to `0.969`).

---

## 2. Probe step

The probe step is `h = h_frac · std(y_ctx)`, **relative to the target scale**. An absolute step was
tried first and discarded: every model renormalises the target internally and un-normalises its
output, so the output quantum in original target units scales with `std(y)`, and the accepted datasets
span target SDs from `0.7498` to `1083`.

Chosen fractions reproduce the audit's own amplitudes on the audit's own contexts, whose mean context
label SD is `1.457782`: TabICL v2 `6.8598e-04`, TabPFN v2 and TabSwift `6.8598e-02`.

Sweep at seed 200, `tr J` from the full `100x100` context Jacobian:

**TabPFN v2**

```
h_frac      did=583 (sd 0.923)        did=223 (sd 6.578)        did=509 (sd 1108)
            trJ    n-trJ  negJii      trJ    n-trJ  negJii      trJ     n-trJ   negJii
1.0e-04   123.834 -23.834    38     45.838   54.162    26    180.056  -80.056    37
1.0e-03    84.148  15.852    11     65.031   34.969     8     18.343   81.657    34
1.0e-02    70.738  29.262     0     65.736   34.264     0     15.758   84.242     5
3.0e-02    72.397  27.603     0     66.435   33.565     0     16.191   83.809     2
6.9e-02    72.453  27.547     0     66.138   33.862     0     16.515   83.485     2
2.0e-01    73.535  26.465     0     64.039   35.961     0     16.000   84.000     0
5.0e-01    71.533  28.467     0     61.931   38.069     0     16.177   83.823     0
```

**TabSwift** — returns `tr J` and `J_**` of exactly `0.000` at `h_frac <= 1e-3` on `did=223`: outputs
are float16 and the difference falls below one output quantum.

```
h_frac      did=583 (sd 0.923)        did=223 (sd 6.578)
            trJ    n-trJ  negJii      trJ    n-trJ  negJii
1.0e-04   -39.496 139.496    21      0.000  100.000     0
1.0e-03    40.290  59.710    30      0.000  100.000     0
1.0e-02    38.418  61.582     3     -1.528  101.528    53
3.0e-02    38.759  61.241     2     -1.356  101.356    65
6.9e-02    38.660  61.340     2     -1.119  101.119    71
2.0e-01    38.148  61.852     4     -1.140  101.140    73
5.0e-01    36.542  63.458     2     -1.005  101.005    74
```

**TabICL v2** — `tr J` flat at `99.98`–`100.03` against `n = 100` across four decades of step size.

```
h_frac      did=583 (sd 0.923)
            trJ    n-trJ  negJii   mean J_**
1.0e-04    99.990   0.010     0     +1.00655
1.0e-03    99.984   0.016     0     +1.00631
1.0e-02    99.982   0.018     0     +1.00632
3.0e-02    99.986   0.014     0     +1.00629
6.9e-02    99.997   0.003     0     +1.00619
2.0e-01   100.012  -0.012     0     +1.00634
5.0e-01   100.027  -0.027     0     +1.00484
```

---

## 3. Where the Jacobian estimator is defined

Clip floor `1e-3 · σ̂²`. `clip` = fraction of test queries raised to the floor; `J**>=1` = fraction
with `J_** >= 1`, where `σ̂²/(1-J_**)` is negative or infinite; `splits` = how many of 5 have
`n − tr J > 0` and therefore a defined `σ̂²`.

```
                            TabICL v2                     TabPFN v2                     TabSwift
did  name              splits  clip  J**>=1  n-trJ   splits  clip  J**>=1  n-trJ   splits  clip  J**>=1  n-trJ
223  stock                5/5 0.068  0.068    8.53      5/5 0.004  0.004   37.15      5/5 0.036  0.036  101.20
229  pwLinear             5/5 0.000  0.000   33.01      5/5 0.004  0.004   66.10      5/5 0.008  0.008   61.96
509  places               5/5 0.052  0.052    9.50      5/5 0.000  0.000   83.44      5/5 0.000  0.000  100.00
522  pm10                 5/5 0.180  0.180    6.30      5/5 0.000  0.000   69.81      5/5 0.000  0.000   69.63
540  mu284                5/5 0.000  0.000   43.42      5/5 0.000  0.000   82.92      5/5 0.000  0.000   92.21
547  no2                  5/5 0.148  0.148    8.45      5/5 0.000  0.000   71.24      5/5 0.000  0.000   66.64
549  strikes              5/5 0.060  0.060   17.96      5/5 0.016  0.016   64.33      5/5 0.000  0.000   99.10
560  bodyfat              5/5 0.072  0.072   16.44      5/5 0.004  0.004   78.95      5/5 0.000  0.000   89.53
579  fri_c0_250_5         5/5 0.132  0.132   10.14      5/5 0.004  0.004   39.53      5/5 0.012  0.012   27.91
581  fri_c3_500_25        2/5 0.184  0.628   -0.15      5/5 0.036  0.036   31.02      5/5 0.020  0.020   40.17
582  fri_c1_500_25        2/5 0.176  0.592   -0.23      5/5 0.028  0.028   29.61      5/5 0.036  0.036   45.27
583  fri_c1_1000_50       1/5 0.152  0.732   -0.62      5/5 0.068  0.068   28.26      5/5 0.032  0.032   50.78

totals                  50/60 0.102  0.222   12.73    60/60 0.0137 0.0137  56.86    60/60 0.0120 0.0120  70.37
```

Splits with `n − tr J <= 0` are excluded from every Jacobian aggregate rather than filled in or
clipped to a default. 10 of 60 TabICL splits are excluded on this basis.

**Negative diagonal entries of the context Jacobian**, out of 100, over all 60 splits per model:

```
model        mean     total   splits with >= 1
TabICL v2   0.1167       7          4
TabPFN v2   2.2333     134         22
TabSwift   16.65       999         39
```

---

## 4. Coverage

Signed deviation from nominal; positive is over-coverage.

```
TabICL v2          cov50    cov80    cov90    cov95     dev50    dev80    dev90    dev95
native            0.5643   0.8313   0.9047   0.9423   +0.0643  +0.0313  +0.0047  -0.0077
Jacobian          0.6038   0.7433   0.7755   0.7915   +0.1038  -0.0567  -0.1245  -0.1585
conformal         0.5010   0.7827   0.8830   0.9490   +0.0010  -0.0173  -0.0170  -0.0010
oracle GP         0.5597   0.8253   0.9073   0.9447   +0.0597  +0.0253  +0.0073  -0.0053

TabPFN v2          cov50    cov80    cov90    cov95     dev50    dev80    dev90    dev95
native            0.5560   0.8257   0.9057   0.9470   +0.0560  +0.0257  +0.0057  -0.0030
Jacobian          0.6090   0.8390   0.9030   0.9343   +0.1090  +0.0390  +0.0030  -0.0157
conformal         0.4883   0.8007   0.8957   0.9490   -0.0117  +0.0007  -0.0043  -0.0010
oracle GP         0.5597   0.8253   0.9073   0.9447   +0.0597  +0.0253  +0.0073  -0.0053

TabSwift           cov50    cov80    cov90    cov95     dev50    dev80    dev90    dev95
Jacobian          0.4993   0.8577   0.9337   0.9610   -0.0007  +0.0577  +0.0337  +0.0110
conformal         0.5013   0.8120   0.9153   0.9597   +0.0013  +0.0120  +0.0153  +0.0097
oracle GP         0.5597   0.8253   0.9073   0.9447   +0.0597  +0.0253  +0.0073  -0.0053
```

---

## 5. Interval score and sharpness

Interval score is the primary metric: it penalises width and miss jointly, so coverage cannot be
bought with wide intervals. `width at matched coverage` rescales each estimator's `s` by the
empirical `0.90` quantile of `|y − m|/s`, forcing achieved coverage to exactly `0.90`, so a
narrow-but-wrong estimator receives no sharpness credit.

```
                 TabICL v2              TabPFN v2              TabSwift
              IS@90 width recal      IS@90 width recal      IS@90 width recal
native        515.8 378.5 299.6      515.9 375.5 301.0        absent
Jacobian      772.0 582.1 673.8      547.4 416.1 336.5      1818.0 1714.0 1283.0
conformal     523.6 352.0 344.3      519.9 356.6 312.1      1553.0 1370.0 1269.0
oracle GP     545.1 386.1 326.3      545.1 386.1 326.3       545.1  386.1  326.3
```

TabICL's Jacobian estimator is the only cell whose matched-coverage width exceeds its raw width
(`673.8` against `582.1`), which follows from its under-coverage in Section 4.

### Per-dataset interval score at 90%

```
                        TabICL v2                        TabPFN v2                    TabSwift
did  name           native   jac  conf    gp        native   jac  conf    gp        jac  conf    gp
223  stock           5.038 6.796 5.236 4.753         4.829 4.872 5.122 4.753      256.4 103.4 4.753
229  pwLinear        7.842 7.683 7.336 7.986         8.157 8.485 8.100 7.986      12.11 11.71 7.986
509  places           3928  6047  3894  4070          3899  3950  3997  4070   1.831e4 1.533e4  4070
522  pm10            3.366 5.658 3.301 3.686         3.503 3.560 3.540 3.686      3.542 3.599 3.686
540  mu284           17.77 36.93 34.74 41.52         17.49 34.68 36.96 41.52      89.84 67.97 41.52
547  no2             2.185 3.405 2.195 2.382         2.239 2.289 2.310 2.382      2.343 2.311 2.382
549  strikes          2216  3116  2326  2392          2247  2555  2176  2392       3088  3068  2392
560  bodyfat         4.442 20.15 4.473 6.484         4.161 5.328 4.518 6.484      47.97  41.6 6.484
579  fri_c0_250_5    1.426 2.547 1.577 1.293         1.302 1.373 1.267 1.293      1.272 1.267 1.293
581  fri_c3_500_25    1.19 3.987 1.264 3.693        0.9869 1.205 0.9722 3.693      2.442  2.02 3.693
582  fri_c1_500_25   1.611 5.827 1.626 3.694        0.9807 1.185 1.037 3.694       2.820 2.374 3.694
583  fri_c1_1000_50  1.651 7.808 1.771 3.555         1.108 1.476 1.302 3.555      3.805 3.302 3.555
```

### Gaussian NLL, same mean

Conformal supplies no density; its variance is taken as `(qhat(0.90)/z(0.90))²`, a stated conversion.

```
                 TabICL v2    TabPFN v2     TabSwift
native               2.238        2.119       absent
Jacobian             871.7       40.948       18.091
conformal            10.68       14.075        3.261
oracle GP            2.708        2.708        2.708
```

---

## 6. Does the variance track where the model errs?

Spearman correlation between each estimator's variance and the realised squared residual, per split,
averaged over splits then datasets. Undefined for conformal: constant width has zero rank variance.

```
                 TabICL v2    TabPFN v2     TabSwift
native               0.300        0.283       absent
Jacobian             0.054        0.124        0.042
conformal        NOT MEASURED -- constant width by construction
oracle GP            0.182        0.182        0.182
```

Per dataset:

```
did   TabICL nat  TabICL jac    TabPFN nat  TabPFN jac    TabSwift jac
223       0.243       0.126         0.299       0.182           0.307
229       0.079       0.058         0.164       0.035           0.190
509       0.247      -0.005         0.130      -0.131          -0.059
522       0.067       0.030         0.118       0.026           0.050
540       0.679       0.103         0.721       0.446          -0.362
547       0.122       0.083         0.241       0.181           0.212
549       0.706       0.192         0.672       0.159           0.387
560       0.393       0.021         0.282       0.199          -0.461
579       0.188       0.193         0.052       0.115           0.081
581       0.304      -0.106         0.232       0.064           0.017
582       0.317      -0.023         0.284       0.156           0.038
583       0.250      -0.024         0.207       0.052           0.103
```

---

## 7. Paired comparisons

Differences paired on the same datasets and splits. Dataset level: 12 paired differences, percentile
bootstrap (10000 resamples) and Wilcoxon signed-rank. Split level: 60 paired differences. Interval
score at 90%; negative mean difference favours the first estimator.

```
model      comparison                mean diff   dataset-level 95% CI   Wilcoxon p   split-level 95% CI     p
TabICL     Jacobian - native          +256.12   [  +3.93,  +682.12]      0.00098     n/a (50 vs 60 splits)
TabICL     Jacobian - conformal       +248.36   [  +2.68,  +671.32]      0.00049     n/a (50 vs 60 splits)
TabICL     native   - conformal         -7.76   [ -28.93,    +5.76]      0.2661      [-30.65, +14.07]   0.233
TabICL     Jacobian - oracle GP       +226.85   [  +1.16,  +615.47]      0.02686     n/a (50 vs 60 splits)
TabICL     native   - oracle GP        -29.27   [ -67.81,    -1.17]      0.00488     [-60.62,  -0.77]   8.4e-06
TabPFN     Jacobian - native           +31.53   [  +0.33,   +85.63]      0.00049     [ +4.69, +67.58]   8.7e-08
TabPFN     Jacobian - conformal        +27.57   [ -11.86,   +94.82]      0.5186      [ -8.98, +78.47]   0.0353
TabPFN     native   - conformal          -3.97   [ -26.22,   +16.03]      0.1294      [-19.84, +13.50]   0.0690
TabPFN     Jacobian - oracle GP         +2.32   [ -30.87,   +39.74]      0.1763      [-33.44, +39.62]   0.0058
TabPFN     native   - oracle GP        -29.22   [ -67.82,    -1.39]      0.00928     [-70.12,  +7.38]   1.1e-05
TabSwift   Jacobian - conformal       +265.30   [  +3.04,  +764.94]      0.00244     [+74.17, +488.99]  4.3e-07
TabSwift   Jacobian - oracle GP      +1273.10   [ +14.67,  +3680.0]      0.09229     [+345.71, +2278.0] 4.1e-05
```

Split-level tests are omitted for the three TabICL rows involving the Jacobian estimator: it is
defined on 50 splits and its comparators on 60, so those differences are not paired.

NLL, dataset level:

```
model      comparison               mean diff   95% CI                 wins
TabICL     Jacobian - native         +869.51   [ +287.00, +1603.60]    1 - 11
TabPFN     Jacobian - native          +38.83   [   +5.12,   +75.75]    0 - 12
TabPFN     native   - conformal       -11.96   [  -35.75,    +0.05]   10 -  2
TabPFN     native   - oracle GP        -0.59   [   -1.00,    -0.22]   10 -  2
TabSwift   Jacobian - conformal       +14.83   [   +4.11,   +27.83]    3 -  9
```

---

## 8. Auxiliary measurements

### 8.1 SURE

`R_SURE = ‖m(y) − y‖² + 2σ² tr J − nσ²`, with `σ²` the conformal-calibration residual variance
`mean(r_cal²)`. `R_SURE` estimates in-sample risk on the context rows; the held-out comparator is
`mean((y_test − m_test)²) − σ²_cal`. These are not the same object — out-of-sample risk generally
exceeds in-sample risk — so the difference is reported per point as a measured quantity, not as an
error in either.

```
model       splits   tr J range        tr J mean   SURE gap mean   SURE gap sd
TabICL v2       60   [ 45.63, 101.99]      87.27        +138155        361747
TabPFN v2       60   [ 10.54,  79.14]      43.14         +40874        127240
TabSwift        60   [ -1.51,  80.53]      29.63         -39303        499676
```

Correlation between the SURE gap and the negative-diagonal count, over the 60 splits available per
model. **Stated as a correlation over the splits available; no causal relationship is claimed.**

```
model       Spearman   Pearson   n_splits
TabICL v2    -0.0587   -0.0924       60
TabPFN v2    -0.0379   +0.3004       60
TabSwift     +0.3972   -0.0788       60
```

Spearman and Pearson disagree in sign for TabPFN v2 and TabSwift. Both are reported; neither is
selected.

### 8.2 Directional monotonicity

`v` is the unit eigenvector of `λ_min(sym(QᵀJQ))`, `u = Qv` with `‖u‖ = 1`. For any posterior mean
`uᵀJu = vᵀ(QᵀJQ)v >= 0`. Measured as `D(t) = u·(m(y + tu) − m(y − tu))/(2t)`, with `t` a fraction of
`‖y − mean(y)‖`.

```
                  lambda_min                      D(t) mean, by fraction of ||y - mean(y)||
                 mean      min     neg/N      0.003     0.01      0.03      0.1       0.3
exact GP        -4.47e-16 -5.47e-16   5/5    -1.8e-30  -2.9e-31  -1.7e-31  +1.8e-32  -8.4e-32
imitator        -0.08442  -0.12057    5/5    -0.08442  -0.08442  -0.08442  -0.08442  -0.08442
TabICL v2       -0.11948  -2.19062   33/60   -0.10918  -0.06332  +0.09342  +0.32354  +0.47258
TabPFN v2       -1.65830  -7.02420   60/60   -1.58227  -0.87066  -0.22872  +0.20043  +0.29810
TabSwift        -0.47461  -3.09311   60/60   -0.41160  -0.38878  -0.28624  -0.05776  +0.08673

count of splits with D(t) < 0, out of 60:
TabICL v2         33        31        24        16         0
TabPFN v2         58        58        42         6         0
TabSwift          49        46        47        32        13
```

**Controls.** The exact GP's `λ_min` is at machine zero — the audit context duplicates 10 rows of
`X`, giving `QᵀWQ` nine machine-zero eigenvalues (`FINAL_NUMBERS.md` 4.4) — and its measured `D` is
`1e-30` to `1e-32`, so on that control only the magnitude is informative and the sign is not. The
targeted imitator's `D` reproduces its `λ_min` at all five amplitudes, and its `negeig` values
`0.08819, 0.05900, 0.06474, 0.08983, 0.12072` reproduce `FINAL_NUMBERS.md` 2.3's
`0.088194, 0.059000, 0.064744, 0.089827, 0.120723`.

**In interpretable units**, the most negative `D(0.003)` per model:

```
TabICL v2,  strikes,        seed 400,  lambda_min -2.1906
  labels raised  by 12.895   along u  ->  prediction moved -28.750   along u
  labels lowered by 12.895   along u  ->  prediction moved +26.958   along u

TabPFN v2,  strikes,        seed 200,  lambda_min -5.6117
  labels raised  by 10.282   along u  ->  prediction moved -61.043   along u
  labels lowered by 10.282   along u  ->  prediction moved +71.374   along u

TabSwift,   fri_c1_500_25,  seed 42,   lambda_min -3.0931
  labels raised  by 0.027433 along u  ->  prediction moved -0.10176  along u
  labels lowered by 0.027433 along u  ->  prediction moved +0.081488 along u
```

### 8.3 Harness checks

| check | required | measured |
|---|---|---|
| conformal coverage at nominal 90% | within `±0.03` | TabICL `0.8830`, TabPFN `0.8957`, TabSwift `0.9153` |
| oracle GP coverage at nominal 90% | within `±0.05` | `0.9073` |
| all four estimators produce finite intervals | — | all finite, 3 models x 4 levels x 60 splits |
| train/test GPs reproduce the audit's in-sample surrogates | — | worst deviation `9.659e-15` |

Conformal's per-split coverage SD is `0.0564`, `0.0494`, `0.0450` over 60 splits. That spread is
inherent: coverage conditional on the calibration draw is `Beta(k, n_cal+1-k)`, whose SD at
`level = 0.90, n_cal = 100` is `0.0297`, and `n_cal` here ranges 50 to 100.

---

## 9. Interpretation

Everything above is measurement. This section is not.

**The estimator as specified measures the wrong quantity, and the correction is exact.** The
appended-query route conditions on a label that was invented, so it returns the post-observation
variance rather than the predictive variance, and is capped at `2σ²`. The inversion recovers the
target exactly on the exact GP. On any genuinely non-Gaussian predictor the inversion carries a
third-cumulant residual, which the hierarchical GP puts at `1.4%`–`16.5%` at p90. That is the
ceiling on what this route can deliver even when everything else goes right.

**The Jacobian-derived variance does not beat the native head, on any model that has one, on any
dataset.** 0–12 for TabPFN v2 and 1–11 for TabICL v2 on interval score, with bootstrap CIs excluding
zero and Wilcoxon `p <= 0.001`; same direction on NLL. The audit's A3 finding is that the reported
variance does not satisfy `s² = σ²(1 + J_ii)`. This experiment says that building a variance out of
`J` instead makes calibration worse. Those are consistent — A3 is a structural identity, not a claim
about predictive quality — but the downstream cost this experiment was designed to find is not
present in this form. The A3 result should be stated as what it is, a structural violation, without
an implied practical penalty that has now been looked for and not found.

**Split conformal is the practical recommendation.** Lowest interval score on TabICL v2, within noise
of native on TabPFN v2 (`519.9` against `515.9`, CI `[-26.2, +16.0]`, `p = 0.13`), and 11–1 over the
Jacobian route on TabSwift. It is cheap, distribution-free, and — the part that matters for the audit
— indifferent to every structural violation the audit measured. "Wrap the model rather than trust its
head" is a useful finding, not a concession.

**The native heads carry real information about where the model errs.** Rank correlation with the
squared residual is `0.300` and `0.283`, against `0.054` and `0.124` for the Jacobian route and
`0.182` for a fitted GP, reaching `0.68`–`0.72` on `mu284` and `strikes`. A variance channel that
fails a structural identity is not thereby uninformative, and the paper should not imply that it is.

**TabICL v2 nearly interpolates its context on real data, and that is what breaks the estimator for
it.** `tr J` reaches `99.98`–`100.03` against `n = 100`, flat across four decades of step size, so
`n − tr J` goes negative and `σ̂²` is undefined on 10 of 60 splits, with `J_** >= 1` on up to `73%` of
queries. This is the `J = I` row of the N5 catalogue — the row that passes A1/A2 vacuously and the
reason E2.1's non-degeneracy check exists. It is dataset-dependent: `tr J` is `45.7` on `pwLinear`
against `87.3` on average. Worth flagging for the audit: the audit's synthetic contexts, with
`σ = 1.0` label noise and 10 duplicated rows, put TabICL far from `J = I`
(`‖J − I‖_F/‖J‖_F = 0.85`), while several real datasets put it close. That is a context-family effect
of exactly the kind E3.1 exists to find. It is a reason to run E3.1, not a result about it.

**The A2 violation is now visible as behaviour rather than as a matrix property.** On 58 of 60 splits
for TabPFN v2 and 49 of 60 for TabSwift, raising the context labels along one direction moves the
prediction the opposite way along that same direction, with both controls behaving as they must. The
effect is amplitude-dependent and vanishes by `t = 0.3‖y − ȳ‖` for TabPFN and TabICL, which is what a
derivative statement should do and is why five amplitudes are reported rather than one.

**The BO follow-on is not supported by this.** The stated dependency was that the Jacobian-derived
variance would have to improve calibration here to be worth trying as an acquisition signal. It does
not — worse than the native head on every model that has one, worse than conformal on all three. The
one place it is not dominated is TabSwift, which has no native head at all, and even there conformal
wins 11 of 12 datasets. A BO study built on this signal would start from a rank correlation of
`0.04`–`0.12` against realised error.

**What this does not settle.** All of it is at `n_ctx = 100`, on 12 datasets with `d <= 50` and
numeric features only. The append perturbation was never measured on the frozen models, where `n`
changes from 100 to 101 and behaviour is `n`-dependent in ways a GP's is not. The `σ̂²` route inherits
a `~18%` single-split sampling spread at this `n` that a larger context would shrink as
`1/sqrt(n − tr J)`. A variant taking `σ²` from the calibration split instead of from `tr J` would
remove that spread at the cost of no longer being derivable from the model alone; it was not run.
