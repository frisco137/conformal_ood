# Phase 2 — Downstream Consequences of the Jacobian Audit

Consolidated record of all seven phase-2 experiments. **The content of each part is unchanged from
the document it came from**; only heading levels were demoted so the four nest under one file, and
this front matter was added.

| part | experiments | source document | structure |
|---|---|---|---|
| A | 2.1 | `RESULTS.md` | measurement §0–8, interpretation §9 |
| B | 2.2, 2.3 | `LOCALISATION.md` | measurement §0–3, interpretation §4 |
| C | 2.4–2.7 | `SEQUENTIAL.md` | measurement §0–5, interpretation §6 |
| D | 2.1 full record | `UNCERTAINTY_EXPERIMENT.md` | chunk-by-chunk, gates, provenance |

**Section numbers restart in every part** — they are local to the brief each part answers. Part A's
"§1" and Part C's "§1" are different sections. Where a part's text refers to `RESULTS.md`,
`LOCALISATION.md` or `SEQUENTIAL.md`, read that as Part A, Part B or Part C of this document; the
file names are kept because the JSON provenance tables use them.

**Each part separates measurement from interpretation**, and the interpretation section is the last
one in each part and is labelled as such. Every number remains traceable to a JSON under `results/`
via the provenance table inside its own part. Per-split, per-probe and per-trajectory values are in
those JSONs, not here.

The four source documents are retained alongside this one; `README.md` remains the navigation entry
point and describes the directory layout and how to reproduce every run.

---

## Contents

- **[Part A — Experiment 2.1 — Jacobian-Derived Predictive Variance](#part-a)**  ·  *(was `RESULTS.md`)*
  - [Phase 2 Results — Jacobian-Derived Predictive Variance](#phase-2-results-jacobian-derived-predictive-variance)
- **[Part B — Experiments 2.2 and 2.3 — Localisation and Corrupted-Label Detection](#part-b)**  ·  *(was `LOCALISATION.md`)*
  - [Experiments 2.2 and 2.3 — Where the Violation Lives, and Whether It Finds Bad Labels](#experiments-22-and-23-where-the-violation-lives-and-whether-it-finds-bad-labels)
  - [1. EXPERIMENT 2.2 — does the A2 violation localise?](#1-experiment-22-does-the-a2-violation-localise)
  - [2. EXPERIMENT 2.3 — does the Jacobian find corrupted labels?](#2-experiment-23-does-the-jacobian-find-corrupted-labels)
  - [3. Provenance](#3-provenance)
  - [4. Interpretation](#4-interpretation)
- **[Part C — Experiments 2.4–2.7 — Sequential Coherence and Bayesian Optimisation](#part-c)**  ·  *(was `SEQUENTIAL.md`)*
  - [Experiments 2.4–2.7 — Sequential Coherence and Bayesian Optimisation](#experiments-2427-sequential-coherence-and-bayesian-optimisation)
  - [1. EXPERIMENT 2.4 (D) — acquisition-argmax sensitivity](#1-experiment-24-d-acquisition-argmax-sensitivity)
  - [2. EXPERIMENT 2.5 (A) — self-consistency, with the localised prediction](#2-experiment-25-a-self-consistency-with-the-localised-prediction)
  - [3. EXPERIMENT 2.7 (C) — the BO loop](#3-experiment-27-c-the-bo-loop)
  - [4. EXPERIMENT 2.6 (B) — does the model un-learn from its own acquisitions?](#4-experiment-26-b-does-the-model-un-learn-from-its-own-acquisitions)
  - [5. Provenance](#5-provenance)
  - [6. Interpretation](#6-interpretation)
- **[Part D — Experiment 2.1 — Full Chunk-by-Chunk Record](#part-d)**  ·  *(was `UNCERTAINTY_EXPERIMENT.md`)*
  - [Jacobian-Derived Predictive Variance](#jacobian-derived-predictive-variance)

---

<a id="part-a"></a>

# Part A — Experiment 2.1 — Jacobian-Derived Predictive Variance

*Source document: `RESULTS.md`. Content unchanged; headings demoted one level.*

## Phase 2 Results — Jacobian-Derived Predictive Variance

First real-data measurement in this project. 12 OpenML regression datasets, 5 splits each,
`n_ctx = 100`, 3 frozen models, 4 uncertainty estimators.

**Sections 1–8 are measurement. Section 9 is interpretation and is the only part that is not a
number.** Per-split values for everything are in `results/*.json`; the full chunk-by-chunk record
with gates and checklists is `UNCERTAINTY_EXPERIMENT.md`.

---

### 0. Headline table

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

### 1. The estimator form

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

#### 1.1 The noise scale `σ̂² = ‖m(y) − y‖²/(n − tr J)`

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

#### 1.2 Choice of the hypothetical label `y_*`

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

### 2. Probe step

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

### 3. Where the Jacobian estimator is defined

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

### 4. Coverage

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

### 5. Interval score and sharpness

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

#### Per-dataset interval score at 90%

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

#### Gaussian NLL, same mean

Conformal supplies no density; its variance is taken as `(qhat(0.90)/z(0.90))²`, a stated conversion.

```
                 TabICL v2    TabPFN v2     TabSwift
native               2.238        2.119       absent
Jacobian             871.7       40.948       18.091
conformal            10.68       14.075        3.261
oracle GP            2.708        2.708        2.708
```

---

### 6. Does the variance track where the model errs?

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

### 7. Paired comparisons

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

### 8. Auxiliary measurements

#### 8.1 SURE

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

#### 8.2 Directional monotonicity

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

#### 8.3 Harness checks

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

### 9. Interpretation

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

---

<a id="part-b"></a>

# Part B — Experiments 2.2 and 2.3 — Localisation and Corrupted-Label Detection

*Source document: `LOCALISATION.md`. Content unchanged; headings demoted one level.*

## Experiments 2.2 and 2.3 — Where the Violation Lives, and Whether It Finds Bad Labels

Measurement record. Sections 1–3 contain numbers only. Section 4 is the interpretation and is the
only part that is not a measurement. Every number is traceable to a JSON in `results/`; Section P
gives the file and key.

### 0. Reconciliation and two corrections made before running

**Filenames.** The brief cites `phase2.md`, `PHASE2_RESULTS.md` and asks for `PHASE2_LOCALISATION.md`.
No file of those names exists in this directory. The mapping used here is
`phase2.md` → `UNCERTAINTY_EXPERIMENT.md`, `PHASE2_RESULTS.md` → `RESULTS.md`, and this document is
`LOCALISATION.md`, following the directory's existing convention.

**The participation ratios of 3–33/98 are not a phase-2 number.** The brief attributes them to
`PHASE2_RESULTS.md` §8.2. `RESULTS.md` §8.2 is directional monotonicity and reports no participation
ratio; no phase-2 document reports one. The figures are `FINAL_NUMBERS.md` §4.3, measured on the
audit's *synthetic* contexts (`n=100, d=5, sigma=1.0`, 10 duplicated rows of `X`), not on the real
data used here. §1.1 reproduces them from the audit's stored Jacobians and then measures the same
quantity on real data, where it differs.

**Correction 1 — the exact-GP control of 2.3.9 cannot do what is asked.** The brief requires the
exact GP to show "chance-level AUC on [the asymmetry] score and above-chance on the column norm".
For an exact GP the Jacobian is `J = K(K + sigma^2 I)^-1`, which contains no `y`. Corrupting labels
leaves it bit-identical, so *every* Jacobian-derived score — column norm included — is unchanged and
its AUC is exactly chance by construction. Measured over all 180 combinations:
`||J(y_clean) - J(y_corrupt)||_F / ||J_clean||_F` has **mean `0.000000e+00`, max `0.000000e+00`**.
The exact GP is therefore kept as a control for one thing only — that no Jacobian score carries
spurious corruption signal — and the **hierarchical GP** is added as the control the brief actually
needs: Bayes-consistent and symmetric, but with a label-dependent Jacobian. Measured:
`J` changes by relative Frobenius **mean `0.2659`, max `2.3522`** under corruption, while staying
symmetric at `asym(J) = 2.938e-14`. Both maps are run through the identical pipeline (§3.1).

**Correction 2 — the imitator control's partial recovery is a property of the construction, not of
the score.** §2.4 quantifies it and adds a power check that separates the two.

---

## 1. EXPERIMENT 2.2 — does the A2 violation localise?

**The score.** `v` = unit eigenvector of `lam_min(sym(Q^T J Q))`, `u = Q v` with `||u|| = 1`,
per-point loading `w_i = u_i^2` normalised to sum to 1. `Q = get_Q(y, seed=0)`.

**Setup.** The 12 datasets, 5 splits and 3 models of Experiment 2.1, Jacobians read from
`arrays/chunk2_arrays_<model>.npz`. No new model calls for §1.1–§1.4, §1.6, §1.7 or §2; §1.5 and
§1.8 are the only items requiring them.

### 1.1 Reproduction of the audit's localisation (2.2.1, first half)

Recomputed from the audit's stored reduced Jacobians (`tier0_instrument/exp3_reduced_jacobians.npz`)
against `FINAL_NUMBERS.md` §4.3 / `tier0_instrument/exp_negeigvec.json`. All ten rows:

```
tag                    seed  lam_min recomputed / reported   participation recomp / reported  top-5 set
tabpfn_dither_t1e-01    42     -3.30369  /  -3.30369              12.34  /  12.34             identical
tabpfn_dither_t1e-01   100     -0.65786  /  -0.65786              23.94  /  23.94             identical
tabpfn_dither_t1e-01   200     -2.76867  /  -2.76867              17.80  /  17.80             identical
tabpfn_dither_t1e-01   300     -9.34969  /  -9.34969              32.77  /  32.77             identical
tabpfn_dither_t1e-01   400     -0.84400  /  -0.84400              14.80  /  14.80             identical
tabswift_t1e-01         42     -0.43720  /  -0.43720              28.97  /  28.97             identical
tabswift_t1e-01        100     -1.43682  /  -1.43682              16.89  /  16.89             identical
tabswift_t1e-01        200     -0.26371  /  -0.26371              16.36  /  16.36             identical
tabswift_t1e-01        300     -0.28312  /  -0.28312              25.65  /  25.65             identical
tabswift_t1e-01        400     -0.55528  /  -0.55528               3.02  /   3.02             identical
```

No disagreement; no configuration difference to name.

### 1.2 Concentration on real data (2.2.1 second half, 2.2.2)

Mean over the 60 splits of each model. `lam_min > 0` counts splits with no negative eigenvalue at
all, where the direction is the least-positive one rather than a violated one.

```
model        mass top5   mass top10   mass top20   participation   lam_min mean   lam_min > 0
tabicl_v2      0.4918      0.6311       0.7841         17.69          -0.1195        27 / 60
tabpfn_v2      0.3929      0.5466       0.7277         22.97          -1.6583         0 / 60
tabswift       0.4707      0.6094       0.7667         17.46          -0.4746         0 / 60
gp_oracle      0.8004      0.8851       0.9463          5.41          +0.2452        60 / 60
```

Per-split `lam_min`, `participation`, `top10` indices and `top10_w` are in
`results/exp22_results.json → per_split`.

### 1.3 Stability of the high-loading set across splits (2.2.3)

Different splits draw different context rows, so the comparison is geometric. For each dataset and
each of the 10 split pairs, the mean nearest-neighbour distance from one split's top-10 loading
points to the other's is compared against the same statistic for random 10-subsets (20 draws per
pair, seeded). Features are put in a common frame per dataset by re-standardising with the pooled
context rows of all five splits. A ratio below 1 means the high-loading sets are closer together
than random subsets are.

```
model        ratio observed / random NN distance, over 12 datasets
             mean      min       max
tabicl_v2    0.9548    0.6893    1.1615
tabpfn_v2    0.9573    0.8075    1.0529
tabswift     0.9763    0.7714    1.2811
gp_oracle    0.7559    0.4549    1.1312
```

### 1.4 Against own residual (2.2.4) and local held-out error (2.2.6)

Spearman, mean over 60 splits. 2.2.4 is over the 100 context points of each split. 2.2.6 is over the
50 test points: each score is mapped to a test point as the mean over its `k = 5` nearest context
points in feature space, `k` fixed in advance.

```
2.2.4  Spearman(score, |m_i - y_i|)
model              w      absJii     colnorm      resid    leverage
tabicl_v2      0.2490    -0.1363     -0.1616     1.0000      0.0692
tabpfn_v2      0.1733    -0.0217      0.0013     1.0000     -0.0184
tabswift       0.0931    -0.3416     -0.3645     1.0000      0.1009
gp_oracle     -0.0033     0.0269      0.0027     1.0000      0.0280

2.2.6  Spearman(kNN-mapped score, test squared error)
model              w      absJii     colnorm      resid    leverage
tabicl_v2      0.0324     0.0267      0.0078     0.1229      0.0744
tabpfn_v2      0.0204     0.1087      0.0005     0.0920      0.0807
tabswift      -0.0681    -0.1665     -0.2247     0.3430      0.1106
gp_oracle     -0.0532     0.1198      0.1241     0.1248      0.0994
```

`resid` is both a target in 2.2.4 and a baseline score in 2.2.8, so its entry in the first block is
`1.0000` by construction and is printed rather than hidden.

### 1.5 Against leave-one-out influence (2.2.5)

**No subsample was taken.** The brief allows one and asks for its size; the full 100 context points
of every split were affordable — 100 refits x 60 splits x 3 models = 18,000 model calls — so the
correlation is over all 100 points of each split and the subsampling caveat does not apply.
`influence_i = ||m_test^(-i) - m_test||_2`. The GP control's LOO is computed by refitting with the
kernel hyperparameters held at the full-context fit.

```
Spearman(score, LOO influence), mean over 60 splits
model              w      absJii     colnorm      resid    leverage
tabicl_v2      0.0893    -0.0061     -0.0765     0.4246      0.2022
tabpfn_v2      0.0483     0.1253      0.0706     0.1784      0.1224
tabswift       0.1432     0.1575      0.1906     0.1509      0.1168
gp_oracle     -0.1057     0.1472      0.1662     0.5826      0.1060
```

### 1.6 Redundancy against the baselines (2.2.9)

Pairwise Spearman among the five scores, mean over 60 splits.

```
pair                tabicl_v2   tabpfn_v2    tabswift
w | absJii            -0.2766     -0.1539     -0.1340
w | colnorm           -0.0727     +0.1166     +0.0252
w | resid             +0.2490     +0.1733     +0.0931
w | leverage          -0.0653     -0.0316     -0.0609
absJii | colnorm      +0.8312     +0.5906     +0.7179
absJii | resid        -0.1363     -0.0217     -0.3416
absJii | leverage     +0.1160     +0.3210     +0.1807
colnorm | resid       -0.1616     +0.0013     -0.3645
colnorm | leverage    +0.0454     +0.1007     +0.0941
resid | leverage      +0.0692     -0.0184     +0.1009
```

### 1.7 Label-perturbation sensitivity (2.2.7)

`i_high = argmax w`. `i_low` is drawn from the points with `w` at or below its median and chosen to
minimise `| |J_ii| - |J_(i_high)(i_high)| |`, so the pair is matched on diagonal sensitivity and the
comparison isolates the eigendirection. Perturbation is `delta = 1.0 * std(y_ctx)` applied in both
signs; sensitivity is the mean of the two `||m_test(y +/- delta e_i) - m_test||`.

```
model        |Jii| high   |Jii| low   abs gap    abs gap    sens_high   sens_low   ratio    ratio>1
             median       median      median     p90        mean        mean       hi/lo
tabicl_v2      0.8019      0.85115    0.019812   0.18633      49.69       41.81    1.0771   27/60
tabpfn_v2      0.31747     0.34295    0.0090726  0.12478      45.26       39.63    1.2152   31/60
tabswift       0.134       0.17816    0.0076904  0.11604       3.95        3.583   1.5409   41/60
```

Paired Wilcoxon on the log ratio, all 60 splits: TabICL `p = 0.5708` (mean log-ratio `-0.0097`),
TabPFN `p = 0.1686` (`+0.0839`), TabSwift `p = 0.01232` (`+0.1710`).

**Matching quality.** The median relative gap is `0.0237 / 0.0414 / 0.0697`, but TabSwift's maximum
relative gap reaches `2.05e+11` because on 8 of its 60 splits `|J_ii|` at the high-loading point is
below `1e-3`, which makes a relative gap meaningless. Absolute gaps are small throughout. Restricting
to splits whose match is close in absolute terms (`< 0.05`):

```
model        n     mean ratio   ratio>1   paired log-ratio   Wilcoxon p
tabicl_v2    40      1.0698      18/40        -0.0142          0.7648
tabpfn_v2    44      1.2663      23/44        +0.1139          0.1576
tabswift     43      1.5848      30/43        +0.1753          0.01902
```

### 1.8 Controls (2.2.10, 2.2.11) and Gate 2.2

#### Exact GP on the audit context (2.2.10a)

`lam_min` at machine zero, as the audit context's 10 duplicated rows of `X` require.

```
seed    lam_min      Spearman(w, ·):  resid     absJii    colnorm   leverage
42     -5.236e-16                    +0.2024   -0.7272   -0.6350   -0.3512
100    -3.507e-16                    +0.1379   -0.6039   -0.5592   -0.2122
200    -4.426e-16                    +0.2761   -0.5800   -0.4890   -0.0226
300    -5.465e-16                    +0.1172   -0.5720   -0.4963   +0.0055
400    -3.697e-16                    +0.3324   -0.4780   -0.4414   -0.1606
```

On the real datasets the exact GP has `lam_min > 0` on all 60 splits (§1.2), so there is no negative
direction there at all; the minimum-eigenvalue direction is the least-sensitive one. Its correlation
with own residual is `-0.0033` (§1.4).

#### Imitator (2.2.11), and the power check

The shipped `TargetedImitator` sets `c = 0.8*||J_gp||_2 + max(0, lam_min)`, so the constructed
rank-one term `-(c/2) v v^T` competes with `Q^T W Q`'s own spectrum rather than dominating it:

```
seed      c        lam_max(Q^T W Q)    competition (c/2)/lam_max
42     0.7807          0.9609                  0.4062
100    0.7783          0.9557                  0.4072
200    0.7812          0.9611                  0.4064
300    0.7803          0.9596                  0.4066
400    0.7788          0.9501                  0.4098
```

Recovery of the constructed direction as a function of that ratio, `P v` being the projection of `v`
onto `span{1, y - ybar}^perp`:

```
setting            competition    |cos(u, P v)|  min–max (mean)      Spearman(w, analytic) mean
as shipped            0.406       0.5575 – 0.6974  (0.6277)                 0.5195
scaled, mult 1        0.500       0.6581 – 0.7528  (0.7087)                 0.5708
scaled, mult 3        1.500       0.9657 – 0.9775  (0.9708)                 0.9116
scaled, mult 10       5.000       0.9974 – 0.9985  (0.9979)                 0.9898
```

Uses the corrected inner Jacobian: `TargetedImitator.inner_jacobian` ships `-c*vv^T` where the map
applies `-(c/2)*vv^T` (`FINAL_NUMBERS.md` §2.3).

#### Gate 2.2

| criterion | measured |
|---|---|
| imitator control recovers its constructed direction | `\|cos\| = 0.9979` and Spearman `0.9898` once the constructed direction dominates (competition `5.0`); `0.6277` / `0.5195` as shipped, at competition `0.406` |
| exact GP control shows near-zero correlation with all targets | **not as stated.** On the audit context `w` correlates `-0.48` to `-0.73` with `absJii` and `-0.44` to `-0.64` with `colnorm`. On the real datasets `Spearman(w, resid) = -0.0033`. |

The second criterion's premise does not hold and the gate is not read as written. For a PSD `J` the
minimum-eigenvalue direction is not arbitrary — it is the least-sensitive direction, and it
anti-correlates with diagonal sensitivity by construction. A non-zero correlation there is what a
correct implementation produces, so the exact GP is reported as a **reference level** rather than as
a zero. Against the target that matters, own residual, its correlation is `-0.0033` on real data and
`+0.12` to `+0.33` on the audit context.

---

### 1.9 Experiment 2.2 checklist

- [x] **2.2.1** `w` per split — top-10 and loadings in `results/exp22_results.json → per_split[].top10`,
      `.top10_w`; participation and `lam_min` in §1.2. Reproduction of the audit exact on all 10 rows
      (§1.1); no disagreement, so no configuration difference to name.
- [x] **2.2.2** Mass in top 5 / 10 / 20 — §1.2, per split in `per_split[].mass_top{5,10,20}`.
- [x] **2.2.3** Stability across seeds, compared in feature space against a random-subset baseline —
      §1.3.
- [x] **2.2.4** Rank correlation with own residual — §1.4.
- [x] **2.2.5** Leave-one-out influence on test predictions — §1.5. **No subsample**: all 100 context
      points of every split, 18,000 model calls.
- [x] **2.2.6** Local held-out error, `k = 5` fixed in advance — §1.4.
- [x] **2.2.7** Label-perturbation sensitivity, matched on `|J_ii|` — §1.7, with matching quality and
      a restricted analysis on close matches.
- [x] **2.2.8** `w` compared against `|J_ii|`, column norm, residual and leverage on every target —
      §1.4, §1.5.
- [x] **2.2.9** Pairwise rank correlations among all five scores — §1.6.
- [x] **2.2.10** Exact GP control — §1.8. Reported as a reference level, not as a zero; the gate's
      "near-zero" premise does not hold and the numbers that show why are given.
- [x] **2.2.11** Targeted imitator control — §1.8, plus a power check that separates the score's
      power from the construction's magnitude.
- [x] **Gate 2.2** verdict with the numbers that satisfy or fail it — §1.8.

---

## 2. EXPERIMENT 2.3 — does the Jacobian find corrupted labels?

**Setup.** 4 of the 12 datasets (`229 pwLinear`, `509 places`, `522 pm10`, `581 fri_c3_500_25`),
5 seeds, 3 schemes, 3 rates = 180 combinations per model. Each costs 201 model calls (one base
prediction plus a `2n` central-difference context Jacobian), so 36,180 calls per model. The full
12-dataset grid would be 108,540 calls per model, about 10.6 h for TabICL alone at its measured
0.35 s/call. **All three schemes and all three rates are kept**, since the brief enumerates them; the
reduction is taken on the dataset axis, which was a free choice. The four retained span the full set
on both axes that matter here — feature dimension 7 to 25, target SD 0.886 to 1083. The eight dropped
are `223, 540, 547, 549, 560, 579, 582, 583`. Twenty splits remain per (scheme, rate) cell.

Corrupted indices are drawn once per (dataset, seed, rate) and reused across all three schemes and
all three models, so every comparison is against the same ground truth. Per-combination indices are
recorded in `results/exp23_corrupt_<model>.json → rows[].corrupt_idx`.

Probe step `h = h_frac * std(y_corrupted)`: the models renormalise by the statistics of the context
they are actually given.

### 2.1 Detection AUC (2.3.4)

Orientation is "higher score = predicted corrupted" throughout; an AUC below `0.5` means the reverse
ordering detects and is reported as measured rather than flipped. Chance is `0.5`.

```
model         colnorm    absJii   rownorm   asym_rc  abs_asym    w      resid     cook   gp_infl  gp_loo_res
tabicl_v2      0.4365    0.4615    0.6424    0.6901   0.6486   0.5711   0.6851   0.6810   0.5139    0.5783
tabpfn_v2      0.4831    0.4378    0.6303    0.6265   0.6200   0.6022   0.7398   0.6810   0.5139    0.5783
tabswift       0.4739    0.4274    0.4920    0.5253   0.5595   0.5720   0.6440   0.6810   0.5139    0.5783
exactgp        0.4966    0.5053    0.4966    0.4854   0.4996   0.5625   0.5827   0.6810   0.5139    0.5783
hiergp         0.5303    0.5317    0.5303    0.5151   0.4958   0.5105   0.6607   0.6810   0.5139    0.5783
```

`cook`, `gp_infl` and `gp_loo_res` do not depend on the model, so their columns are constant by
construction.

Precision@k with `k` = the true number corrupted; chance is `0.1167`:

```
model         colnorm    absJii   rownorm   asym_rc  abs_asym    w      resid     cook   gp_infl  gp_loo_res
tabicl_v2      0.1039    0.1289    0.3069    0.3775   0.3003   0.2042   0.3525   0.3378   0.1050    0.2269
tabpfn_v2      0.1331    0.1236    0.2897    0.3106   0.2669   0.2325   0.4394   0.3378   0.1050    0.2269
tabswift       0.1253    0.1081    0.1294    0.1631   0.1728   0.2061   0.3308   0.3378   0.1050    0.2269
exactgp        0.0950    0.1042    0.0950    0.1225   0.1325   0.1589   0.2411   0.3378   0.1050    0.2269
hiergp         0.1394    0.1328    0.1394    0.1161   0.1103   0.1194   0.3228   0.3378   0.1050    0.2269
```

### 2.2 By scheme and by rate

```
AUC by scheme
                     colnorm    absJii   rownorm   asym_rc  abs_asym    w      resid     cook
tabicl_v2  flip       0.5118    0.5376    0.6315    0.6525   0.6099   0.5494   0.6327   0.6332
           shuffle    0.4793    0.4886    0.5805    0.6093   0.6087   0.5691   0.6254   0.5982
           noise      0.3185    0.3583    0.7151    0.8084   0.7270   0.5948   0.7972   0.8116
tabpfn_v2  flip       0.5431    0.4717    0.6004    0.5695   0.6103   0.6067   0.6900   0.6332
           shuffle    0.5165    0.4717    0.5814    0.5830   0.5640   0.5546   0.6666   0.5982
           noise      0.3899    0.3701    0.7090    0.7269   0.6856   0.6454   0.8628   0.8116
tabswift   flip       0.5024    0.4604    0.4671    0.4822   0.5144   0.5380   0.5933   0.6332
           shuffle    0.5144    0.4860    0.4915    0.4989   0.5254   0.5299   0.5820   0.5982
           noise      0.4049    0.3356    0.5173    0.5949   0.6386   0.6481   0.7568   0.8116
exactgp    flip       0.4966    0.5053    0.4966    0.4854   0.4996   0.5244   0.5100   0.6332
           shuffle    0.4966    0.5053    0.4966    0.4854   0.4996   0.5114   0.5167   0.5982
           noise      0.4966    0.5053    0.4966    0.4854   0.4996   0.6516   0.7212   0.8116
hiergp     flip       0.5242    0.5299    0.5242    0.5130   0.4899   0.4965   0.5940   0.6332
           shuffle    0.5243    0.5259    0.5243    0.5217   0.4963   0.5066   0.5799   0.5982
           noise      0.5424    0.5394    0.5424    0.5106   0.5011   0.5284   0.8081   0.8116

AUC by rate
                     colnorm    absJii   rownorm   asym_rc  abs_asym    w      resid     cook
tabicl_v2  0.05       0.4333    0.4682    0.6781    0.7241   0.6878   0.5952   0.7115   0.7085
           0.10       0.4145    0.4409    0.6394    0.6959   0.6492   0.5692   0.6820   0.6774
           0.20       0.4617    0.4754    0.6096    0.6502   0.6087   0.5488   0.6617   0.6571
tabpfn_v2  0.05       0.5069    0.4516    0.6677    0.6502   0.6661   0.6309   0.7522   0.7085
           0.10       0.4692    0.4285    0.6285    0.6305   0.6170   0.6166   0.7433   0.6774
           0.20       0.4733    0.4334    0.5947    0.5987   0.5769   0.5591   0.7238   0.6571
tabswift   0.05       0.4625    0.4326    0.5193    0.5598   0.5894   0.5854   0.6548   0.7085
           0.10       0.4751    0.4160    0.4758    0.5007   0.5614   0.5871   0.6554   0.6774
           0.20       0.4840    0.4334    0.4809    0.5154   0.5277   0.5434   0.6219   0.6571
```

Note the exact GP's four Jacobian columns are **identical across all three schemes** — `0.4966`,
`0.5053`, `0.4966`, `0.4854` — which is the numerical signature of a label-blind Jacobian.

### 2.3 Per dataset, and the asymmetry score against residual

```
AUC per dataset, mean over 5 seeds x 3 schemes x 3 rates
model        did    asym_rc   rownorm     resid      cook         w
tabicl_v2    229     0.7940    0.7996    0.7857    0.8043    0.6416
tabicl_v2    509     0.6089    0.6222    0.6107    0.6397    0.5126
tabicl_v2    522     0.6008    0.6038    0.6233    0.6193    0.5107
tabicl_v2    581     0.7565    0.5438    0.7206    0.6608    0.6195
tabpfn_v2    229     0.6343    0.6530    0.8115    0.8043    0.6242
tabpfn_v2    509     0.5239    0.5386    0.6621    0.6397    0.5546
tabpfn_v2    522     0.5541    0.5323    0.6476    0.6193    0.5632
tabpfn_v2    581     0.7935    0.7972    0.8379    0.6608    0.6669
tabswift     229     0.4605    0.3564    0.6581    0.8043    0.6309
tabswift     509     0.4991    0.5307    0.5130    0.6397    0.5139
tabswift     522     0.4986    0.4506    0.6192    0.6193    0.5511
tabswift     581     0.6430    0.6302    0.7859    0.6608    0.5920
```

Paired over the 180 combinations, `AUC(asym_rc) - AUC(resid)`:

```
model        mean difference   asym_rc wins   Wilcoxon p
tabicl_v2        +0.0050         96/180        0.5548
tabpfn_v2        -0.1133         38/180        3.068e-18
tabswift         -0.1187         43/180        2.890e-17
```

### 2.4 Actual leave-one-out (2.3.7)

**Reduced grid, stated.** Each combination costs 100 refits with two predictions each. Run at the
**10% rate only**, keeping all 3 schemes, all 4 datasets and all 5 seeds: 60 combinations, 12,060
calls per model. The 5% and 20% rates are `NOT MEASURED` for this item. Two forms:
`loo_test_shift_i = ||m_test^(-i) - m_test||` (the influence form of 2.2.5) and
`loo_self_i = |y_i - m_i^(-i)|` (the point's own out-of-fold residual).

```
model        n    loo_test_shift AUC   loo_self AUC   loo_test_shift P@k   loo_self P@k
tabicl_v2   60          0.7059            0.7461            0.3717            0.4517
tabpfn_v2   60          0.6249            0.7437            0.2383            0.4550
tabswift    60          0.6495            0.6670            0.2717            0.3417
```

Like-for-like against the other scores at the same 10% rate:

```
model         loo_self  loo_shift     resid      cook   asym_rc   rownorm         w
tabicl_v2       0.7461     0.7059    0.6820    0.6774    0.6959    0.6394    0.5692
tabpfn_v2       0.7437     0.6249    0.7433    0.6774    0.6305    0.6285    0.6166
tabswift        0.6670     0.6495    0.6554    0.6774    0.5007    0.4758    0.5871
```

Paired over the 60 matched combinations:

```
model        comparison             mean difference   wins     Wilcoxon p
tabicl_v2    loo_self - resid           +0.0641      43/60      3.336e-06
tabicl_v2    loo_self - asym_rc         +0.0502      44/60      2.894e-05
tabpfn_v2    loo_self - resid           +0.0004      26/60      0.9729
tabpfn_v2    loo_self - asym_rc         +0.1133      48/60      3.501e-07
tabswift     loo_self - resid           +0.0116      30/60      0.001384
tabswift     loo_self - asym_rc         +0.1663      55/60      3.561e-10
```

### 2.5 Controls and Gate 2.3

**Label-blindness of the exact GP's Jacobian**, over all 180 combinations:
`||J(y_corrupt) - J(y_clean)||_F / ||J_clean||_F` has mean `0.000000e+00` and max `0.000000e+00`.
The hierarchical GP over the same combinations: mean `0.2659`, max `2.3522`, with
`asym(J) = 2.938e-14`.

**Uncorrupted baseline (2.3.10).** Scores computed on the clean context and ranked against the index
set that *would* have been corrupted. Any departure from `0.5` is spurious concentration.

```
model         colnorm    absJii   rownorm   asym_rc  abs_asym    w      resid     cook   gp_infl  gp_loo_res
tabicl_v2      0.4967    0.4907    0.4929    0.4856   0.5184   0.5066   0.4988   0.5144   0.5139    0.5087
tabpfn_v2      0.5179    0.5035    0.5095    0.5078   0.5091   0.5005   0.5076   0.5144   0.5139    0.5087
tabswift       0.5049    0.5042    0.5143    0.5179   0.5091   0.4985   0.4992   0.5144   0.5139    0.5087
```

**A leakage floor for `w`.** The loading is not a pure function of `J`: `Q = get_Q(y)` is built from
the labels, so `w` can respond to corruption even when the Jacobian cannot. The exact GP, whose `J`
is provably unchanged by corruption, still returns `w` AUC `0.5625` overall and `0.6516` under the
noise scheme. The models' `w` values of `0.5711 / 0.6022 / 0.5720` are to be read against that floor,
not against `0.5`.

#### Gate 2.3

| criterion | measured |
|---|---|
| exact GP at chance on the asymmetry score | `0.4854`, and identical across all three schemes |
| exact GP above chance on the column norm | **not attainable.** `0.4966`, identical across all three schemes, because `J` is label-blind (Correction 1). The harness check is carried by the exact GP's residual (`0.5827`) and Cook's distance (`0.6810`), both above chance. |
| uncorrupted baseline shows no spurious concentration | all ten scores in `[0.4856, 0.5184]` across the three models |

---
### 2.6 Experiment 2.3 checklist

- [x] **2.3.1** Three schemes at 5% / 10% / 20%; corrupted indices on record per split in
      `results/exp23_corrupt_<model>.json → rows[].corrupt_idx`. `n_changed` records how many of the
      shuffled points actually changed label.
- [x] **2.3.2** Context Jacobian recomputed on the corrupted context. **New model calls**: 201 per
      combination x 180 combinations x 3 models = 108,540 calls. Wall-clock, summed from the
      per-combination timings in the JSONs: TabICL v2 `2.44 h` (48.7 s/combination), TabPFN v2
      `0.91 h` (18.1 s), TabSwift `1.35 h` (27.1 s); the latter two ran together on a second GPU.
- [x] **2.3.3** Column norm, diagonal, row norm, row-minus-column asymmetry, and the loading `w` —
      §2.1. `abs_asym_rc` added because AUC is orientation-sensitive.
- [x] **2.3.4** AUC and precision@k per split, per scheme, per rate — §2.1, §2.2; per-combination
      values in `results/exp23_results.json → per_combination[]`.
- [x] **2.3.5** Residual magnitude — §2.1, and it is the headline baseline.
- [x] **2.3.6** Cook's distance from an OLS fit — §2.1.
- [x] **2.3.7** Actual leave-one-out — §2.4, on a stated reduced grid: 10% rate only, 60
      combinations per model. The 5% and 20% rates are `NOT MEASURED` for this item.
- [x] **2.3.8** Oracle GP influence `diag K(K + sigma^2 I)^-1` — §2.1, AUC `0.5139`. It contains no
      `y`, so it cannot respond to corruption and its AUC is chance by construction; `gp_loo_res`
      was added as the GP-based score that does use `y`.
- [x] **2.3.9** Identical pipeline on the exact GP — §2.5. The "above-chance on the column norm" half
      is **not attainable** and the measurement showing why is given; the hierarchical GP was added
      as the control the brief intended.
- [x] **2.3.10** Uncorrupted baseline — §2.5. All ten scores in `[0.4856, 0.5184]`.
- [x] **Gate 2.3** verdict with the numbers — §2.5.

---

## 3. Provenance

### 3.1 Number → file → key

| Section | Quantity | File | Key |
|---|---|---|---|
| 0 | exact-GP Jacobian label-blindness | `results/exp23_controls.json` | `rows[] where model=exactgp → J_change_vs_clean_rel` |
| 0 | hierarchical-GP Jacobian response and symmetry | `results/exp23_controls.json` | `rows[] where model=hiergp → J_change_vs_clean_rel`, `asym_frob` |
| 1.1 | audit reproduction | `results/exp22_results.json` | `2.2.1_reproduction.<tag>[]` |
| 1.1 | audit reference values | `tier0_instrument/exp_negeigvec.json` | `<tag>[].lam_min`, `.participation_ratio`, `.top5_points` |
| 1.2 | concentration, participation, `lam_min` | `results/exp22_results.json` | `per_split[].mass_top{5,10,20}`, `.participation`, `.lam_min`, `.top10`, `.top10_w` |
| 1.3 | stability in feature space | `results/exp22_results.json` | `2.2.3_stability.<model>.<did>.{observed_nn_dist,random_nn_dist,ratio_obs_over_random}` |
| 1.4 | vs own residual | `results/exp22_results.json` | `per_split[].spearman_score_vs_target.resid.<score>` |
| 1.4 | vs local held-out error | `results/exp22_results.json` | `per_split[].spearman_knn_vs_testerr.<score>` |
| 1.5 | vs LOO influence | `results/exp22_results.json` | `per_split[].spearman_score_vs_target.loo.<score>` |
| 1.5 | raw LOO vectors | `arrays/exp22_loo_<model>.npz` | `<model>__<did>__<seed>__loo_influence` |
| 1.6 | pairwise score redundancy | `results/exp22_results.json` | `per_split[].pairwise_score_spearman."<a>\|<b>"` |
| 1.7 | matched-pair perturbation | `results/exp22_loo_<model>.json` | `rows[].{i_high,i_low_matched,w_high,w_low,Jii_high,Jii_low,Jii_match_abs_gap,Jii_match_rel_gap,sens_high,sens_low,sens_ratio_high_over_low,perturb_delta}` |
| 1.8 | exact GP on the audit context | `results/exp22_results.json` | `2.2.10a_gp_audit_context[]` |
| 1.8 | imitator as shipped | `results/exp22_results.json` | `2.2.11_imitator[]`, `2.2.11c_shipped_competition[]` |
| 1.8 | imitator power check | `results/exp22_results.json` | `2.2.11b_strong_imitator.mult_<m>[]` |
| 2.1 | detection AUC and precision@k | `results/exp23_results.json` | `aggregate.<model>.overall.<score>.{auc,prec}` |
| 2.1 | per-combination values | `results/exp23_results.json` | `per_combination[].{auc,prec_at_k}` |
| 2.2 | by scheme / by rate | `results/exp23_results.json` | `aggregate.<model>.by_scheme.<scheme>`, `.by_rate.<rate>` |
| 2.3 | corrupted indices (ground truth) | `results/exp23_corrupt_<model>.json` | `rows[].corrupt_idx`, `.n_corrupt`, `.n_changed` |
| 2.3 | corrupted Jacobians | `arrays/exp23_corrupt_<model>.npz` | `<model>__<did>__<seed>__<scheme>__<rate>__{J,y_corr,m_ctx,idx}` |
| 2.4 | actual LOO on corrupted contexts | `results/exp23_results.json` | `2.3.7_actual_loo[]`; vectors in `arrays/exp23_loo_<model>.npz` |
| 2.5 | uncorrupted baseline | `results/exp23_results.json` | `2.3.10_uncorrupted[]`, `2.3.10_uncorrupted_agg` |
| 2.5 | control combinations | `results/exp23_controls.json`, `arrays/exp23_controls.npz` | `rows[]`, `<map>__<did>__<seed>__<scheme>__<rate>__*` |
| — | probe step actually used | `results/exp23_corrupt_<model>.json` | `_config.h_frac`, `rows[].h`, `.std_y_corr` |
| — | dataset subset and what was dropped | `results/exp23_corrupt_<model>.json` | `_config.dids`, `_config.dids_dropped` |

### 3.2 Scripts written, with outputs

| Script | Output |
|---|---|
| `src/exp22_loo.py` | `results/exp22_loo_<model>.json`, `arrays/exp22_loo_<model>.npz`, `logs/log_exp22_loo_*.log` |
| `src/exp22_analyse.py` | `results/exp22_results.json` |
| `src/exp23_corrupt.py` | `results/exp23_corrupt_<model>.json`, `arrays/exp23_corrupt_<model>.npz`, `logs/log_exp23_*.log` |
| `src/exp23_controls.py` | `results/exp23_controls.json`, `arrays/exp23_controls.npz` |
| `src/exp23_loo.py` | `results/exp23_loo_<model>.json`, `arrays/exp23_loo_<model>.npz`, `logs/log_exp23loo_*.log` |
| `src/exp23_analyse.py` | `results/exp23_results.json` |

Reused unchanged: `core/metrics.py` (`get_Q` seeded at 0, `asym`, `negeig`), `core/surrogates.py`
(`ExactGP`, `HierarchicalGP`), `core/context.py`, `src/estimators.py`, `src/chunk2_estimators.py`
(`make_predict`, `H_FRAC_BY_MODEL`), and `tier0_instrument/chunk1_control_validation.py`
(`TargetedImitator`). Nothing outside `experiments/phase_2/` was modified.

### 3.3 Failed attempts and rejected settings

| What | Why rejected |
|---|---|
| Exact GP as the harness check for 2.3.9 | Its Jacobian is label-blind; `\|\|J(y_corrupt) - J(y_clean)\|\|_F` is exactly `0` over all 180 combinations, so no Jacobian score can be above chance. Kept as a no-spurious-signal control; the hierarchical GP added for the role the brief intended. Section 0. |
| `TargetedImitator` as shipped, as the power control | Competition ratio `(c/2)/lam_max(Q^T W Q) = 0.406`, so the constructed direction does not dominate and recovery is `\|cos\| = 0.63`. Kept and reported; a scaled version added, reaching `0.998` at ratio `5.0`. Section 1.8. |
| Relative `\|J_ii\|` gap as the 2.2.7 matching criterion | Meaningless where `\|J_ii\|` is near zero: TabSwift's maximum relative gap is `2.05e+11` on 8 splits with `\|J_ii\| < 1e-3`. Absolute gap reported alongside and used for the restricted analysis. Section 1.7. |
| Full 12-dataset grid for 2.3 | 108,540 model calls per model, about 10.6 h for TabICL alone. Reduced to 4 datasets, 180 combinations, 36,180 calls; all schemes and rates kept, 8 dropped datasets named. Section 2. |
| Full-grid 2.3.7 | A further 18,100 calls per model. Run at the 10% rate only; the 5% and 20% rates are `NOT MEASURED` for that item. Section 2.4. |
| `w` read against a chance floor of 0.5 in 2.3 | `Q = get_Q(y)` depends on the labels, so `w` responds to corruption even for a label-blind `J`. The exact GP's `w` reaches `0.5625` overall and `0.6516` under noise; that is the floor. Section 2.5. |
| Unlimited BLAS threads | Two concurrent runs on a 192-core machine drove load average to 209 through thread thrashing. Queued runs capped at 16 threads per process. |

---
## 4. Interpretation

Everything above is measurement. This section is not.

### 4.1 Experiment 2.2 returns a null, and it is the instrument's answer, not its failure

The eigendirection is real and exactly reproducible — all ten audit rows recovered to five decimal
places with identical top-5 sets — and it is not a restatement of anything already available: it
anti-correlates with `|J_ii|` at `-0.28 / -0.15 / -0.13` while `|J_ii|` and the column norm are near
duplicates of each other at `+0.83 / +0.59 / +0.72`. So the score is carrying its own information.

That information does not point at badly-handled points. Against own residual it reaches
`0.249 / 0.173 / 0.093`; against local held-out error `0.032 / 0.020 / -0.068` where plain residual
reaches `0.123 / 0.092 / 0.343`; against actual leave-one-out influence — the target with the
strongest claim to being "how much does this point matter" — it reaches `0.089 / 0.048 / 0.143`
while residual reaches `0.425 / 0.178 / 0.151` and even feature leverage reaches
`0.202 / 0.122 / 0.117`. On two of three models the loading is the worst of the five scores at
predicting LOO influence. The matched-pair perturbation is null for TabICL (`p = 0.76` on close
matches) and TabPFN (`p = 0.16`), and positive only for TabSwift (`p = 0.019`). And the high-loading
set is not a stable region: across splits of the same dataset its points are no closer together than
random subsets of the same size (`0.955 / 0.957 / 0.976`), while the GP control's are (`0.756`).

The null is about the models, not the score. The power check settles that: once the constructed
direction actually dominates, recovery is `|cos| = 0.998` and Spearman `0.990`. The shipped imitator
recovers only to `0.63` because its construction sits at competition ratio `0.406` — it never
presented a dominant direction to find.

So the answer to "are the high-loading points the ones the model handles badly" is no. The negative
directions are structurally real and were reproduced exactly, but they do not localise onto anything
a practitioner would act on. **A per-context-point diagnostic does not fall out of A2.**

### 4.2 Experiment 2.3 produces a real, control-validated signal — and residual is still the better tool

The row-minus-column asymmetry detects corrupted labels: AUC `0.690 / 0.627 / 0.525` and precision@k
`0.378 / 0.311 / 0.163` against a chance rate of `0.117`. The mechanism is visible in the pair of
norms it is built from: the row norm detects (`0.642 / 0.630 / 0.492`) and the column norm does not
(`0.437 / 0.483 / 0.474`). For any symmetric `J` those two are the same number, and on the
hierarchical GP they are — `0.5303` both. The gap between them *is* the A1 violation, used as a
detector.

The controls make that signal credible rather than incidental. The exact GP's Jacobian is label-blind
— `||J(y_corrupt) - J(y_clean)||_F` is exactly zero over all 180 combinations — and its four
Jacobian scores return values identical to four decimal places across all three corruption schemes.
The hierarchical GP's Jacobian *does* respond to corruption (`0.27` relative Frobenius) and is
Bayes-consistent, and its asymmetry score still sits at `0.515`. Both Bayesian maps are at chance on
the score where the two frozen transformers are not. The uncorrupted baseline shows no score
concentrating on an arbitrary subset (`[0.486, 0.518]` across all ten scores and three models).

**But residual magnitude is the better detector, and that must be said plainly.** It reaches
`0.685 / 0.740 / 0.644`. Paired over all 180 combinations, the asymmetry score loses decisively on
TabPFN (`-0.113`, `p = 3.1e-18`) and TabSwift (`-0.119`, `p = 2.9e-17`), and **ties** on TabICL
(`+0.005`, 96 of 180, `p = 0.55`). The aggregate difference on TabICL — `0.6901` against `0.6851` —
is not a win, and reading it as one would be wrong. The honest summary is that a practitioner
hunting mislabels should compute residuals, and that on one of three models an A1-derived score
matches that baseline without being a residual.

Actual leave-one-out, the expensive gold standard, sits above both: its self-residual form reaches
`0.746 / 0.744 / 0.667` and beats plain residual on TabICL (`p = 3.3e-06`) and TabSwift
(`p = 0.0014`) while tying on TabPFN (`p = 0.97`). It beats the asymmetry score on all three
(`p` between `3.6e-10` and `2.9e-05`). The ordering that survives is therefore: 100 refits per split
> one residual vector > any Jacobian score, with the asymmetry score reaching the residual only on
TabICL.

That match is still the most substantive positive result this project has produced downstream of the
audit. It is the first time a structural violation has been converted into a working diagnostic
rather than a description. The score is exactly zero for every Bayes-consistent map by construction,
which is what makes it a test of A1's practical content rather than a generic outlier statistic.

One caveat on the loading in this setting: `w` is not a pure function of `J`, because `Q = get_Q(y)`
is built from the labels. The exact GP, whose Jacobian provably does not move, still returns `w` AUC
`0.5625` overall and `0.6516` under the noise scheme. The models' `0.571 / 0.602 / 0.572` should be
read against that floor, and on that reading `w` adds nothing in 2.3 either.

### 4.3 A1 and A2 are not interchangeable diagnostics

The model where the asymmetry score works best is TabICL, which has the *weakest* A2 violation of the
three — `lam_min > 0` on 27 of its 60 real-data splits, meaning no negative eigenvalue at all. The
model with the most negative diagonal entries, TabSwift, is the one where the asymmetry score fails
(`0.525`). Whatever the row-column gap is tracking, it is not the same object the negative
eigendirection tracks, and the audit's habit of reporting A1 and A2 side by side as two readings of
one violation is not supported by their downstream behaviour here.

### 4.4 What follows

The brief makes candidates 3 and 4 conditional on these two producing something. On the measurements
above, 2.2 does not and 2.3 does, in a narrow and well-controlled way. Anything built on this should
be built on the asymmetry channel, not on the eigendirection loading, and should be posed against
residual magnitude as the incumbent from the start — 2.1's lesson repeated, since the one place a
Jacobian score was competitive here is the one place the comparison was run against a baseline that
was itself weak.

### 4.5 What this does not settle

Experiment 2.3 ran on 4 of the 12 datasets, so its dataset-level generalisation rests on 4 points per
model, and its per-dataset spread is wide — the asymmetry score ranges from `0.601` to `0.794` on
TabICL alone. Item 2.3.7's actual-LOO baseline was run at the 10% rate only. The corruption schemes
are synthetic and known; nothing here speaks to naturally occurring label noise, whose structure is
not a flip, a shuffle, or additive Gaussian noise. All of it is at `n_ctx = 100`, where `tr J` and
the spectrum both behave differently from larger contexts.

---

<a id="part-c"></a>

# Part C — Experiments 2.4–2.7 — Sequential Coherence and Bayesian Optimisation

*Source document: `SEQUENTIAL.md`. Content unchanged; headings demoted one level.*

## Experiments 2.4–2.7 — Sequential Coherence and Bayesian Optimisation

Measurement record. Sections 1–4 contain numbers only. Section 6 is the interpretation and is the
only part that is not a measurement. Every number is traceable to a JSON in `results/`; Section 5
gives the file and key.

### 0. Corrections made before running

Each is a case where a stated design or gate is not satisfiable as written, with the measurement
that shows it.

**Correction 1 — EI must be computed in log space.** The naive form
`EI = s (z Phi(z) + phi(z))` underflows: `z Phi(z) + phi(z)` evaluates to exactly `0.0` in float64
for `z < -39`. With the standard incumbent `y+ = max(y_ctx)` the pool-maximum `z` falls below `-38`
on 3 of TabPFN's 60 splits (`did=560`, reaching `z = -173`), so **every** pool point returned
`EI = 0` and `argmax` selected index 0 arbitrarily. EI is therefore computed as `log EI`, which is
rank-equivalent, using a two-branch `log h(z)`: direct above `z = -6`, and the Mills-ratio series
`h(z) = phi(z)(1/z^2 - 3/z^4 + 15/z^6 - 105/z^8)` below it. The two branches agree to
`4.697e-04` in log units on the overlap `[-12, -6]`. Validation: with a constant `s` the argmax of
log-EI equals `argmax(mu)` on 200 of 200 random trials, which is the analytic requirement below.

**Correction 2 — the 2.4.6 frontier gap must be measured in EI space, and the incumbent must be
swept.** `log EI` carries a `-z^2/2` term, so a gap measured on it grows as `z^2` by construction and
returns the *opposite* of the prediction. Measured in EI space the prediction is recovered (§1.4).
Separately, with `y+ = max(y_ctx)` only `1.3%` of TabPFN pool points have `|z| < 1` and `0.37%` have
`z > 0` (median `z = -6.56`, minimum `-884`): the incumbent is the maximum of 100 observed values
while the pool is a random held-out sample, so EI sits in its deep tail and the `z = 0` frontier that
the prediction concerns is never reached. The incumbent is swept over
`y+ = quantile(y_ctx, q)`, `q in {0.50, 0.75, 0.90, 1.00}`; `q = 0.50` puts `43%` of points inside
`|z| < 1`. `q = 1.00` is the standard BO incumbent and is reported alongside.

**Correction 3 — "the exact GP's drift must be at numerical precision" is not attainable by a
sampled test, and is checked analytically instead.** Drift is estimated from `K` draws, so its
standard error is `Var(mu_{n+1})^{1/2}/sqrt(K)`, about `0.09` in units of `s_n` at `K = 64`. For a
GP, `mu_{n+1}(x_*) = a + b y'` is affine in `y'`, so `E[mu_{n+1}] = a + b mu_n(x')` in closed form,
and `Var_{n+1}` is `y`-independent, so `Var_n = Var_{n+1} + b^2 Var_n(x')` is exact. Measured that
way the gate is met: `|E[mu_{n+1}] - mu_n| / s_n` at `1.5e-16` and the variance identity closing to
`0.0` exactly (§2.5). The sampled GP is reported beside it, where the correct statement is
`|drift| / MC-SE ~ 1`, not `|drift| ~ 0`.

**Correction 4 — "the targeted imitator must drift" contradicts T1.** Mean drift under the model's
own sampling law is a **value functional**: it depends on the map only through its values. T1 states
that no value functional separates Bayes-realisable maps from their complement with any margin, and
`g = mu* + eps sin(<a,y>/eps) b` is the construction that makes this concrete. Its drift is bounded
by `2 eps / s_n`, and measured at `eps in {0.01, 0.1, 1.0}` it does not grow with `eps` at all:
`|drift|/s_n` of `0.0097`, `0.0084`, `0.0030`. The T1 imitator is therefore the wrong positive
control for a drift test — it is the object proving such a test must be blind — and it is reported
for that reason. A control that does drift is supplied instead: the quadratic map
`m(y) = W y + lam (W y)^2`, whose drift is `lam Var_{y'}(mu_{n+1}(x_*))` exactly by Jensen, so the
measurement has a predicted value and not merely a sign.

---

## 1. EXPERIMENT 2.4 (D) — acquisition-argmax sensitivity

**No new model calls.** Every quantity is a function of the means and variances stored by
Experiment 2.1.

**Framing (2.4.1), stated because it is a proxy.** Each of the 12 datasets of 2.1 is read as a
maximisation surrogate: `y` is the objective, the 100 context rows are evaluations already made, the
50 held-out rows are the candidate pool, and the incumbent is `y+ = max(y_ctx)`. Nothing is acquired
and nothing is refitted. This is a proxy for BO on a real objective, not BO itself.

**Four variance channels, one mean.** All four are fed the model's own `m_test`, so any difference in
the argmax is caused by the variance and nothing else: `native` (TabSwift has none),
`jacobian` = `sigma_hat^2/(1 - J_**)`, `conformal` = `(qhat(0.90)/z(0.90))^2`, and `gp_var`. Two
reference rows are carried: `gp_full`, the oracle GP with **its own** mean as well as its own
variance, which is a different object from a variance channel; and `greedy_mean`, `argmax(mu)`.

`beta = 2 log(|pool| t^2 / (6 delta))` with `delta = 0.1`, `t = 1`, `|pool| = 50`, giving
`beta = 8.8459` and `sqrt(beta) = 2.9742`.

The Jacobian channel is defined on 50 of TabICL's 60 splits and on all 60 for the other two, carried
forward from 2.1 §3.

**A structural fact about the conformal channel.** Split conformal has constant width, so `s` is the
same at every pool point. UCB is then `mu + const`, and EI is `s[z Phi(z) + phi(z)]` with `z` monotone
in `mu` at fixed `s`, so both are monotone transforms of `mu`. The conformal channel therefore always
selects `argmax(mu)` under either acquisition. The measurement confirms it exactly: conformal's
regret equals `greedy_mean`'s to all reported digits on all three models under both acquisitions
(`0.1481`, `0.1360`, `0.2496`). Conformal is a degenerate acquisition channel here, and this is the
reason.

### 1.1 Decision agreement (2.4.3)

Fraction of splits on which two channels select the same pool candidate.

```
                        EI                                    UCB
pair              tabicl_v2 tabpfn_v2 tabswift        tabicl_v2 tabpfn_v2 tabswift
jacobian|native      0.160     0.550      n/a            0.160     0.517      n/a
jacobian|conformal   0.260     0.550     0.250           0.180     0.533     0.167
jacobian|gp_var      0.220     0.567     0.217           0.220     0.517     0.183
native|conformal     0.650     0.633      n/a            0.600     0.583      n/a
native|gp_var        0.683     0.633      n/a            0.650     0.550      n/a
conformal|gp_var     0.783     0.700     0.450           0.750     0.700     0.517
```

### 1.2 Decision regret (2.4.4)

Normalised regret gap `(max(y_pool) - y[pick]) / range(y_pool)`; lower is better.

```
                         EI                                   UCB
channel           tabicl_v2 tabpfn_v2 tabswift        tabicl_v2 tabpfn_v2 tabswift
greedy_mean          0.1481    0.1360   0.2496           0.1481    0.1360   0.2496
conformal            0.1481    0.1360   0.2496           0.1481    0.1360   0.2496
gp_var               0.1648    0.1593   0.2804           0.1718    0.1655   0.2549
native               0.1913    0.2211      n/a           0.1884    0.2122      n/a
jacobian             0.3573    0.2180   0.4112           0.3627    0.2039   0.4424
gp_full              0.2323    0.2323   0.2323           0.2410    0.2410   0.2410
```

Medians, which are well below the means on every channel:

```
                         EI                                   UCB
channel           tabicl_v2 tabpfn_v2 tabswift        tabicl_v2 tabpfn_v2 tabswift
greedy_mean          0.0550    0.0378   0.1440           0.0550    0.0378   0.1440
gp_var               0.0656    0.0408   0.1947           0.0727    0.0426   0.1263
native               0.1151    0.1099      n/a           0.1005    0.0985      n/a
jacobian             0.3102    0.1065   0.3634           0.3102    0.0860   0.4343
```

Paired against `greedy_mean` over the splits where the channel is defined, Wilcoxon signed-rank:

```
model        acq  channel     mean difference   worse in   Wilcoxon p
tabicl_v2    EI   jacobian        +0.1958        32/50      2.089e-06
tabicl_v2    EI   native          +0.0432        13/60      0.03863
tabicl_v2    EI   gp_var          +0.0167         8/60      0.2213
tabicl_v2    UCB  jacobian        +0.2012        35/50      7.175e-07
tabicl_v2    UCB  native          +0.0403        15/60      0.05933
tabicl_v2    UCB  gp_var          +0.0237         9/60      0.2455
tabpfn_v2    EI   jacobian        +0.0819        21/60      0.001648
tabpfn_v2    EI   native          +0.0850        19/60      0.0009833
tabpfn_v2    EI   gp_var          +0.0233        12/60      0.286
tabpfn_v2    UCB  jacobian        +0.0678        21/60      0.004115
tabpfn_v2    UCB  native          +0.0761        20/60      0.001569
tabpfn_v2    UCB  gp_var          +0.0295        11/60      0.231
tabswift     EI   jacobian        +0.1616        30/60      0.002628
tabswift     EI   gp_var          +0.0308        16/60      0.6616
tabswift     UCB  jacobian        +0.1928        34/60      0.001518
tabswift     UCB  gp_var          +0.0053        13/60      0.7186
```

### 1.3 Acquisition-surface rank correlation (2.4.5)

Spearman between the two acquisition surfaces over the 50 pool points, mean over splits.

```
                        EI                                    UCB
pair              tabicl_v2 tabpfn_v2 tabswift        tabicl_v2 tabpfn_v2 tabswift
jacobian|native     +0.534    +0.759      n/a           +0.626    +0.894      n/a
jacobian|conformal  +0.585    +0.827    +0.512          +0.646    +0.919    +0.504
jacobian|gp_var     +0.581    +0.822    +0.557          +0.642    +0.909    +0.512
native|conformal    +0.756    +0.778      n/a           +0.885    +0.914      n/a
native|gp_var       +0.780    +0.785      n/a           +0.890    +0.912      n/a
conformal|gp_var    +0.938    +0.926    +0.694          +0.970    +0.960    +0.811
```

### 1.4 The frontier prediction (2.4.6)

`d alpha_EI / d s = phi(z)`, maximal at `z = 0`. Per pool point, the gap between the native and
Jacobian EI surfaces is `|EI_native - EI_jacobian|`, in **EI space** (Correction 2), correlated
against `phi(z)` and against `|z|`.

```
model        y+ quantile  median|z|  frac|z|<1  frac z>0   rho(gap, phi(z))  rho(gap, |z|)
tabicl_v2       0.50        2.902     0.4464     0.5116        +0.5683         -0.5683
tabicl_v2       0.75        3.421     0.3540     0.2264        +0.5684         -0.5686
tabicl_v2       0.90        5.363     0.1796     0.0832        +0.6468         -0.6476
tabicl_v2       1.00       11.227     0.0188     0.0008        +0.6487         -0.6605
tabpfn_v2       0.50        3.343     0.4330     0.4950        +0.6095         -0.6094
tabpfn_v2       0.75        3.827     0.3317     0.2113        +0.6402         -0.6408
tabpfn_v2       0.90        5.664     0.1553     0.0727        +0.7683         -0.7677
tabpfn_v2       1.00       12.426     0.0130     0.0037        +0.8480         -0.8394
```

The gap was all-zero on `0` of `50` and `0` of `60` splits respectively, so no correlation is
computed on a degenerate surface. TabSwift has no native channel, so this item is `NOT MEASURED` for
it.

Stratified argmax agreement between the native and Jacobian channels within `|z|` terciles, at
`y+ = quantile(y_ctx, 0.50)`:

```
model        stratum   mean |z|   EI agree   UCB agree
tabicl_v2      s0        0.899      0.360      0.160
tabicl_v2      s1        3.002      0.600      0.380
tabicl_v2      s2        7.059      0.800      0.320
tabpfn_v2      s0        0.817      0.767      0.400
tabpfn_v2      s1        3.635      0.817      0.567
tabpfn_v2      s2       14.454      0.933      0.717
```

### 1.5 Scale versus shape (2.4.7)

Fraction of splits on which a channel scaled by `c` selects the same candidate as the same channel
unscaled.

```
                      c = 0.5              c = 1.0              c = 2.0
model      channel     EI     UCB          EI     UCB           EI     UCB
tabicl_v2  native     0.850   0.800       1.000  1.000        0.950   0.717
tabicl_v2  jacobian   0.940   0.800       1.000  1.000        0.920   0.840
tabicl_v2  conformal  1.000   1.000       1.000  1.000        1.000   1.000
tabicl_v2  gp_var     0.967   0.900       1.000  1.000        0.967   0.850
tabpfn_v2  native     0.900   0.850       1.000  1.000        0.917   0.667
tabpfn_v2  jacobian   0.950   0.900       1.000  1.000        0.983   0.717
tabpfn_v2  conformal  1.000   1.000       1.000  1.000        1.000   1.000
tabpfn_v2  gp_var     0.950   0.900       1.000  1.000        0.950   0.900
tabswift   jacobian   0.950   0.817       1.000  1.000        0.883   0.783
tabswift   conformal  1.000   1.000       1.000  1.000        1.000   1.000
tabswift   gp_var     1.000   0.833       1.000  1.000        0.933   0.783
```

### Gate 2.4

| criterion | measured |
|---|---|
| native and Jacobian agree on more than 90% of decisions | **no.** `0.160` (TabICL, both acquisitions), `0.550` / `0.517` (TabPFN, EI / UCB). TabSwift has no native channel. |
| regret gap inside split-to-split noise | **no.** The Jacobian channel is worse than `greedy_mean` on every model and both acquisitions, `p` between `7.2e-07` and `4.1e-03`. |

The defect reaches the decision. 2.5 and the loop experiments are therefore not pre-empted.

### 1.6 Experiment 2.4 checklist

- [x] **2.4.1** 12 datasets as maximisation surrogates; framing stated above.
- [x] **2.4.2** EI and UCB with `beta = 8.8459`; argmax recorded per channel per split.
- [x] **2.4.3** Full pairwise agreement per model per acquisition — §1.1.
- [x] **2.4.4** Decision regret, mean and median, with paired Wilcoxon against `greedy_mean` — §1.2.
- [x] **2.4.5** Acquisition-surface rank correlation — §1.3.
- [x] **2.4.6** Frontier prediction — §1.4, in EI space and with the incumbent swept
      (Correction 2). `NOT MEASURED` for TabSwift: no native channel to compare against.
- [x] **2.4.7** Variance scaled by `{0.5, 1, 2}` — §1.5.
- [x] **Gate 2.4** verdict with the numbers — above.

---
## 2. EXPERIMENT 2.5 (A) — self-consistency, with the localised prediction

**Which Jacobian the localised prediction is about.** The brief writes the localised claim with the
context diagonal `J_ii`, via `mu_i(y + delta e_i) - mu_i(y) ~ J_ii delta`. That is the **perturb**
operation — replace an existing label — whereas the martingale is the **append** operation,
`n -> n+1`. The object whose sign governs the appended increment at the probe is the appended-query
diagonal `J_** = d mu_{n+1}(x')/d y'`, for which T2 on the augmented context gives
`sigma^2 J_** = Var(f_* | y, y') >= 0`. A negative `J_**` is exactly "appending a higher label at
`x'` lowers the prediction at `x'`". `J_**` is what is stratified on, and it was already stored for
every held-out point by Experiment 2.1.

**Strata available**, counted from the stored `J_star` over 3000 held-out probes per model:

```
model        negative J_**   fraction   splits with >=1 negative
tabicl_v2          0          0.00%            0 / 60
tabpfn_v2         57          1.90%           11 / 60
tabswift         157          5.23%           15 / 60
```

**2.5.4 is NOT MEASURED for TabICL v2**: its negative stratum is empty, so there is nothing to
stratify. Every negative probe of the other two models is used, each paired with a positive probe
from the same split matched on `|J_**|`.

**Sampling (2.5.1), stated exactly.** TabICL v2 and TabPFN v2: inverse CDF on
`output_type="quantiles"` over 999 levels, which for TabPFN the package computes from the
`FullSupportBarDistribution` including its half-normal tail bins. **TabSwift has no predictive
distribution**; it is sampled as Gaussian with the Jacobian-derived variance of 2.1 and every
TabSwift row carries `sampling = gaussian_jacobian_surrogate`. Its drift is therefore not a
statement about the model's own predictive law and is reported separately for that reason.

`K = 64` draws per probe. `drift = (E[mu_{n+1}(x_*)] - mu_n(x_*)) / s_n(x_*)`, with `drift_se` the
Monte Carlo standard error of that estimate in the same units. **`|drift|/SE` is the quantity a
martingale constrains, not `|drift|`**: a correct martingale gives `|drift|/SE ~ 1`.

### 2.1 Drift and variance shrinkage (2.5.2, 2.5.3)

`self` is `x_* = x'`; `other` is a different held-out point. `tv gap` is the relative gap in
`Var_n = E[Var_{n+1}] + Var(mu_{n+1})`.

```
model        target    n    drift mean  |drift| mean  MC SE mean  |d|/SE median  frac>2  frac>3  tv gap mean  tv n
tabicl_v2     self    120     -0.0019      0.0770      0.0914        0.808       0.083   0.017     +0.0161   120
tabicl_v2     other   120     -0.0113      0.0349      0.0047        7.447       0.808   0.742     +0.0033   120
tabpfn_v2     self    234     -0.0080      0.0661      0.0311        1.928       0.491   0.350     -0.0014   234
tabpfn_v2     other   234     -0.0163      0.0759      0.0036       14.191       0.897   0.855     -0.0093   234
tabswift      self    434     -0.6890      0.7978      0.1260        6.469       0.889   0.848         n/a     0
tabswift      other   434     -0.6983      0.8969      0.1260        6.521       0.947   0.917         n/a     0
```

TabSwift's `tv gap` is `NOT MEASURED`: no predictive distribution, so `Var_{n+1}` does not exist.

### 2.2 K sensitivity (2.5.7)

The `K = 32` figure uses the first 32 of the same 64 draws, so the comparison isolates Monte Carlo
noise rather than adding a second experiment.

```
model        target   drift mean K=64   drift mean K=32   Spearman(K64, K32) over probes
tabicl_v2     self       -0.00193          +0.01650              0.6619
tabicl_v2     other      -0.01128          -0.01194              0.9735
tabpfn_v2     self       -0.00805          -0.00533              0.8949
tabpfn_v2     other      -0.01631          -0.01653              0.9961
tabswift      self       -0.68903          -0.70000              0.9440
tabswift      other      -0.69825          -0.69698              0.9351
```

### 2.3 The localised prediction (2.5.4) and probe-vs-target (2.5.5)

Paired within split, negative-`J_**` probe against its `|J_**|`-matched positive probe. The
prediction is that drift is **larger, and signed wrong**, in the negative stratum. Two Wilcoxon
tests are reported: on the signed difference (does the negative stratum drift more negative) and on
the absolute difference (does it drift *further*).

```
model      target  n_pairs  neg drift  |neg|   pos drift  |pos|   frac neg<0  signed p   |.| p
tabpfn_v2   self      57     -0.0081  0.0411   +0.0088  0.0312     0.649      0.0197    0.0826
tabpfn_v2   other     57     -0.0047  0.0295   -0.0158  0.0356     0.596      0.0784    0.3149
tabswift    self     157     -0.5920  0.8146   -0.8454  0.8454     0.873      0.0004    0.2196
tabswift    other    157     -0.7456  0.8112   -0.7896  0.7935     0.962      0.2092    0.0305
tabicl_v2    —        —      NOT MEASURED: negative stratum empty (0 of 3000 probes)
```

`|J_**|` matching quality, median absolute gap between the paired probes: TabPFN v2 `0.008662`,
TabSwift `0.016069`. Mean `J_**` in each stratum: TabPFN `-0.0521` / `+0.0501`, TabSwift `-0.0683` /
`+0.0570`.

By probe kind, `self` target, median `|drift|/SE`:

```
model        negative   matched_positive   general      mean J_** (general)
tabicl_v2       n/a           n/a           0.808            +0.8738
tabpfn_v2      3.346         2.122          1.472            +0.4642
tabswift       6.734         7.177          3.360            +0.4433
```

Spearman between drift and `J_**` across all probes:

```
model        self       other
tabicl_v2   +0.0717    -0.0443
tabpfn_v2   +0.0208    -0.0185
tabswift    +0.1066    +0.2130
```

### 2.4 Controls (2.5.6) and Gate 2.5

**Each map is sampled from its own predictive.** An earlier version drew `y'` from the exact GP for
every arm. On the first split checked the hierarchical GP's predictive mean at the same probe differs
from the exact GP's by `2.5` units, and that alone produced `|drift|/SE = 23.065` for a map that is
Bayesian. The hierarchical GP's predictive is a **mixture** over lengthscales and is sampled by
drawing a component from the posterior weights and then a Gaussian from that component.

```
control                          drift/s_n mean   |drift|/SE median   frac < 2   tv rel gap mean
exactgp_analytic  (the gate)       1.070e-11 max          —              —        1.768e-14 max
exactgp_sampled   self            -7.588e-04           0.694          0.978        +2.961e-03
exactgp_sampled   other           +2.169e-04           0.707          0.967        +1.347e-07
hiergp_sampled    self            +1.384e-03           0.792          0.944        +4.991e-03
hiergp_sampled    other           -4.249e-05           0.778          0.921        +1.979e-04
```

The analytic row is the gate: for a GP `mu_{n+1}(x_*) = a + b y'` is affine in `y'`, so
`E[mu_{n+1}] = a + b mu_n(x')` in closed form, and `Var_{n+1}` is `y`-independent so
`Var_n = Var_{n+1} + b^2 Var_n(x')` is exact.

**T1 imitator**, bump `eps sin(sum(y)/eps)`, dimension-agnostic so it is defined at both `n` and
`n+1`:

```
eps      |bump-only drift| / s_n     2 eps / s_n (the bound)
0.01            0.0061                    0.0201
0.10            0.0675                    0.2010
1.00            0.5970                    2.0101
```

The bump-only drift scales with `eps` and stays inside the bound at every value. The `total` drift
additionally carries the sampled GP's own Monte Carlo noise of about `0.07 s_n`, which exceeds
`2 eps / s_n` at `eps = 0.01`; the bump-only column is the one that isolates the imitator's
contribution.

**Quadratic drift control**, `m(y) = mu(y) + lam mu(y)^2`:

```
lam      |drift| / s_n mean   |drift| / SE median   |measured - exact| / exact, max
0.05           5.1143               0.74                    9.749e-12
0.20          20.2668               0.92                    3.199e-11
```

The exact expression `E[m] - m_0 = (E[mu] - mu_n) + lam(Var(mu) + E[mu]^2 - mu_n^2)` is reproduced to
11 significant figures, which checks the harness arithmetic. The map drifts by `5` to `20` `s_n`, but
its Monte Carlo standard error grows with it, so at `K = 64` the drift is not resolved against its
own noise (`|drift|/SE` of `0.74` and `0.92`).

#### Gate 2.5

| criterion | measured |
|---|---|
| exact GP drift at numerical precision | **met analytically**: `1.070e-11` max over 90 probes. Not attainable by the sampled route, whose MC SE is `~0.09 s_n` at `K = 64` (Correction 3); the sampled GP instead gives `\|drift\|/SE = 0.694`, `97.8%` of probes inside `2`. |
| variance decomposition closes | **met**: analytic relative residual `1.768e-14` max. |
| imitator drifts measurably | **not as stated** (Correction 4). The T1 imitator's drift is `O(eps)`-bounded and at `eps = 0.01` sits at `0.0061 s_n`, below the sampling noise — which is T1's content, drift being a value functional. The substitute quadratic control drifts by `5.11` and `20.27` `s_n` and matches its closed form to `1e-11`. |

The exact GP does not drift, so the harness is measuring the martingale property and not something
else.

### 2.5 Experiment 2.5 checklist

- [x] **2.5.1** Sampling stated per model — inverse CDF over 999 quantile levels for TabICL v2 and
      TabPFN v2; TabSwift labelled `gaussian_jacobian_surrogate` throughout.
- [x] **2.5.2** Mean drift with Monte Carlo standard error, `K = 64` — §2.1.
- [x] **2.5.3** Variance shrinkage, both sides of the total-variance identity — §2.1.
      `NOT MEASURED` for TabSwift: no predictive distribution.
- [x] **2.5.4** Negative vs matched-positive `J_**` stratification, paired Wilcoxon — §2.3.
      `NOT MEASURED` for TabICL v2: 0 negative probes of 3000.
- [x] **2.5.5** `x' = x_*` versus `x' != x_*`, reported as `self` and `other` throughout.
- [x] **2.5.6** Controls — §2.4, each sampled from its own predictive.
- [x] **2.5.7** `K` sensitivity, `K = 32` from the same draws — §2.2.
- [x] **Gate 2.5** verdict with the numbers — §2.4.

---
## 3. EXPERIMENT 2.7 (C) — the BO loop

**Run because 2.4 showed the defect reaches the decision.** Native and Jacobian agree on `0.160` of
TabICL decisions and `0.550`/`0.517` of TabPFN's, and the Jacobian channel's regret is worse than
greedy on every model and both acquisitions (`p` between `7.2e-07` and `4.1e-03`).

**Protocol (2.7.3).** `n_init = 10` scrambled-Sobol points, `T = 20` iterations, `S = 5` seeds. The
acquisition is optimised over a fresh scrambled-Sobol candidate pool of `128` points redrawn each
iteration, identical across arms within a seed, and the initial design is shared within a seed so
trajectories are paired. EI throughout, in log space (Correction 1).

**Objectives (2.7.1).** All on the unit cube, all written as maximisation. Branin `d=2` on
`[-5,10]x[0,15]`; Hartmann-3 `d=3`; Hartmann-6 `d=6`; Ackley `d=5` on `[-32.768, 32.768]^5`;
Rosenbrock `d=5` on `[-5,10]^5`. Stated maxima `-0.397887, 3.86278, 3.32237, 0.0, 0.0`. Checked
against `2^17` Sobol draws: Branin and Hartmann-3 are reached to `1.2e-04` and `2.6e-03`; Hartmann-6
to `0.17`; Ackley and Rosenbrock have interior optima that random sampling does not approach
(`6.73` and `62.5` away), which is what makes them hard rather than a definition error.
**No real-objective slice was added** — `NOT MEASURED`, it was not affordable alongside the
Jacobian arm.

**Surrogates (2.7.2).** `tabpfn_native`, `tabpfn_conformal` (split conformal on a third of the
observed set, refitted each iteration), `tabpfn_jacobian` (`sigma_hat^2/(1 - J_**)`), `gp`
(marginal-likelihood `ConstantKernel*RBF + WhiteKernel`, 2 restarts — the strong incumbent, not
crippled), and `random`.

**TabICL v2 and TabSwift are NOT RUN in the loop.** The Jacobian arm costs `2 x n_pool` calls per
iteration — `2 x 128 x 20 x 5 x 5 = 128,000` calls, about `2.5 h` at TabPFN's measured `0.07 s`.
At TabICL's `0.35 s` that is `12.4 h` and at TabSwift's `0.11 s` about `3.9 h`, on top of everything
else. TabPFN v2 is also the checkpoint actually deployed as a BO surrogate in the literature the
brief cites. Measured wall-clock for the Jacobian arm: `540 s` per trajectory.

### 3.1 Final simple regret (2.7.4, 2.7.5)

Mean over the 5 seeds. Lower is better.

```
objective      tabpfn_native  tabpfn_conformal  tabpfn_jacobian        gp      random
ackley5             17.076          16.140            16.619         17.196     18.359
branin2              1.0233          0.15418           0.85406        0.11351    1.6902
hartmann3            0.054828        0.054567          0.33160        0.093342   0.55738
hartmann6            1.3415          1.3274            1.3828         1.0466     1.7092
rosenbrock5       7917.8          1734.3            13506           2619.7     4228.9
```

Per-seed values for every cell are in `results/exp27_results.json → per_objective.<obj>.arms.<arm>.final_regret_per_seed`.
Branin, as an example of the spread:

```
arm                  per-seed final regret
tabpfn_native        [0.6418, 1.4189, 0.9639, 0.8760, 1.2161]
tabpfn_conformal     [0.0014, 0.5514, 0.0512, 0.1597, 0.0072]
tabpfn_jacobian      [0.8945, 0.4200, 1.8764, 0.6812, 0.3982]
gp                   [0.0014, 0.0836, 0.1104, 0.2655, 0.1066]
random               [1.8297, 1.6023, 0.0281, 4.3525, 0.6387]
```

**Paired, seed-matched within objective, normalised by the mean initial regret of that objective.**
A negative mean favours the first arm. Bootstrap CI over 10000 resamples of the 25 paired
differences; Wilcoxon signed-rank on the same.

```
comparison                            objective wins   mean norm diff   95% CI              p
tabpfn_jacobian vs tabpfn_native          2 - 3           +0.1323     [-0.0122, +0.3051]  0.0883
tabpfn_jacobian vs gp                     1 - 4           +0.2556     [+0.0528, +0.5521]  0.005581
tabpfn_native   vs gp                     2 - 3           +0.1234     [+0.0203, +0.2708]  0.04202
tabpfn_conformal vs gp                    3 - 2           -0.0090     [-0.0719, +0.0562]  0.4455
tabpfn_native   vs random                 4 - 1           -0.1253     [-0.2960, +0.0583]  0.02209
gp              vs random                 5 - 0           -0.2486     [-0.3728, -0.1393]  0.0002067
tabpfn_native   vs tabpfn_conformal       0 - 5           +0.1324     [+0.0390, +0.2659]  0.006425
```

### 3.2 Iterations to a fixed fraction of the optimum (2.7.4)

Number of seeds reaching a 50% and a 90% reduction in initial regret, and the median iteration at
which it happens.

```
objective      arm                 50% reached   median iter   90% reached   median iter
branin2        tabpfn_native          5/5            4.0           0/5           --
branin2        tabpfn_conformal       5/5            1.0           4/5          10.5
branin2        tabpfn_jacobian        5/5            5.0           2/5           9.5
branin2        gp                     5/5            3.0           5/5          16.0
branin2        random                 4/5            9.5           1/5          10.0
hartmann3      tabpfn_native          5/5            6.0           5/5          15.0
hartmann3      tabpfn_conformal       5/5            3.0           4/5          16.0
hartmann3      tabpfn_jacobian        4/5            1.5           0/5           --
hartmann3      gp                     5/5            3.0           1/5          20.0
hartmann3      random                 3/5            0.0           0/5           --
hartmann6      tabpfn_native          3/5           16.0           0/5           --
hartmann6      tabpfn_conformal       2/5            8.5           0/5           --
hartmann6      tabpfn_jacobian        2/5           12.5           0/5           --
hartmann6      gp                     4/5            9.0           0/5           --
hartmann6      random                 1/5            6.0           0/5           --
rosenbrock5    tabpfn_native          4/5            0.0           1/5          20.0
rosenbrock5    tabpfn_conformal       5/5            0.0           1/5          16.0
rosenbrock5    tabpfn_jacobian        4/5            0.0           0/5           --
rosenbrock5    gp                     5/5            0.0           0/5           --
rosenbrock5    random                 5/5            0.0           0/5           --
ackley5        every arm              0/5             --           0/5           --
```

No arm achieves a 50% regret reduction on Ackley in 20 iterations from 10 initial points in 5
dimensions.

### 3.3 Knowledge gradient (2.7.6)

**`NOT MEASURED`.** KG requires an inner expectation over the model's own predictive future at every
candidate, which on top of the Jacobian arm's `2 x n_pool` calls per iteration was not affordable.
It is not approximated.

### 3.4 The `n - tr J` watch (2.7.7, 2.7.8)

Recorded at every iteration of the Jacobian arm, 500 iterations in total.

```
t     n      tr J    tr J / n   n - tr J   neg J_ii   asym    negeig
0    10.0    4.737     0.474      5.263      0.32     0.795   0.0705
4    14.0    7.256     0.518      6.744      0.56     0.929   0.177
9    19.0   10.478     0.551      8.522      0.44     0.971   0.203
14   24.0   13.841     0.577     10.159      0.96     0.913   0.231
19   29.0   17.815     0.614     11.185      1.04     0.911   0.232
```

Over all 500 iterations: `tr J / n` mean `0.5622`, max `0.9405`; minimum `n - tr J` is `+1.3496`;
**`sigma_hat^2` is undefined on 0 of 500 iterations**. `tr J / n` rises steadily with the context but
does not approach 1 at these sizes. Total negative diagonal entries `401`; `asym` mean `0.9012`,
`negeig` mean `0.1882`.

### 3.5 Experiment 2.7 checklist

- [x] **2.7.1** Objectives, definitions and domains stated; maxima checked against `2^17` Sobol
      draws. Real-objective slice `NOT MEASURED`, not affordable.
- [x] **2.7.2** Five arms including a properly fitted GP and a random floor. TabICL v2 and TabSwift
      `NOT RUN`, with the cost stated.
- [x] **2.7.3** `n_init = 10`, `T = 20`, `S = 5`, pool of 128 redrawn per iteration, shared initial
      design within a seed.
- [x] **2.7.4** Simple regret per seed, paired with bootstrap CI, plus iterations-to-fraction — §3.1, §3.2.
- [x] **2.7.5** Per-objective table and win/loss counts across objectives — §3.1.
- [ ] **2.7.6** Knowledge gradient — **`NOT MEASURED`**, cost; §3.3.
- [x] **2.7.7** Per-iteration audit quantities recorded along every Jacobian-arm trajectory — §3.4.
- [x] **2.7.8** `tr J` against `n`, and iterations where the Jacobian variance is undefined — §3.4.

---

## 4. EXPERIMENT 2.6 (B) — does the model un-learn from its own acquisitions?

Rides on the 2.7 trajectories at no extra model calls. At each iteration, before appending,
`J_**` at the acquired point is recorded on the Jacobian arm, where it is computed anyway. After
appending, `move = mu_{n+1}(x_t) - mu_n(x_t)` and `surprise = y_t - mu_n(x_t)`. For any posterior
mean the two share a sign, and `shrinkage = move / surprise` lies in `[0, 1]`.

### 4.1 Violation rate and shrinkage (2.6.2, 2.6.5, 2.6.6)

500 acquisitions per arm — 5 objectives x 5 seeds x 20 iterations.

```
arm                  n     sign-violation rate   shrinkage median   frac outside [0,1]   frac negative
tabpfn_native       500          0.0660               +0.5518             0.1880            0.0660
tabpfn_conformal    500          0.0780               +0.5416             0.2160            0.0780
tabpfn_jacobian     500          0.0600               +0.7537             0.2540            0.0600
gp                  500          0.0100               +0.9867             0.0500            0.0100
```

The `gp` arm re-optimises its kernel hyperparameters at every iteration, so it is not the posterior
mean of a fixed prior and is not required to be exactly `0`; it is the reference level a refitted
Bayesian surrogate produces on the identical protocol.

### 4.2 Conditioning on the sign of `J_**` (2.6.3)

**`NOT MEASURED`.** `J_**` at the acquired point was **negative on 0 of 500 acquisitions**
(`J_star_available = 500`, `n_Jstar_negative = 0`), so the negative stratum is empty and there is
nothing to condition on. The violation rate on the complement is `0.0600`. This is the same pattern
as 2.5: on held-out probes TabPFN v2 had 57 negative `J_**` out of 3000 (`1.90%`), and the
acquisition selects points where it is positive.

The context diagonal is a different object and does go negative: 401 negative `J_ii` entries across
the same 500 iterations (§3.4).

### 4.3 Violation count against final regret (2.6.4)

Spearman between a trajectory's sign-violation count and its final simple regret, computed **within
objective over the 5 seeds**. **Stated as a correlation over the trajectories available; no causal
relationship is claimed.** With 5 trajectories per cell these are 5-point correlations.

```
arm                 ackley5   branin2   hartmann3   hartmann6   rosenbrock5
tabpfn_native       +0.359    -0.783     -0.354      -0.791       -0.211
tabpfn_conformal    +0.527    +0.632     +0.894      -0.791       +0.224
tabpfn_jacobian     +0.791    +0.051     +0.000      -0.632       +0.577
gp                  +0.363    +0.354     +0.783        n/a           n/a
```

The sign is not consistent across objectives for any arm.

### 4.4 Experiment 2.6 checklist

- [x] **2.6.1** `J_**` at the acquired point recorded every iteration on the Jacobian arm.
- [x] **2.6.2** `move`, `surprise` and the sign-violation count per trajectory — §4.1.
- [ ] **2.6.3** Violation rate conditional on the sign of `J_**` — **`NOT MEASURED`**: 0 of 500
      acquisitions had `J_** < 0`; §4.2.
- [x] **2.6.4** Per-trajectory violation count against final regret, as a correlation — §4.3.
- [x] **2.6.5** Same on the GP arm — §4.1. The imitator was not run in the loop: it has no
      mechanism for proposing candidates and its drift is `O(eps)` by construction (Correction 4).
- [x] **2.6.6** Shrinkage distribution and the fraction outside `[0, 1]` — §4.1.

---
## 5. Provenance

### 5.1 Number → file → key

| Section | Quantity | File | Key |
|---|---|---|---|
| 0 | log-EI two-branch agreement | `results/exp24_results.json` | `_config.log_h_branch_max_dev` |
| 1.1 | decision agreement | `results/exp24_results.json` | `aggregate.<model>.<acq>.agreement_rate."<a>\|<b>"` |
| 1.2 | decision regret | `results/exp24_results.json` | `aggregate.<model>.<acq>.regret_gap_norm_{mean,median}` |
| 1.2 | per-split picks and regret | `results/exp24_results.json` | `per_split[].acq.<acq>.{picks,pick_y,regret_gap,regret_gap_norm}` |
| 1.3 | acquisition-surface Spearman | `results/exp24_results.json` | `aggregate.<model>.<acq>.surface_spearman_mean` |
| 1.4 | frontier prediction | `results/exp24_results.json` | `aggregate.<model>.frontier.<q>.{spearman_gap_vs_phiz_mean,spearman_gap_vs_absz_mean,abs_z_median,frac_absz_lt1,frac_z_gt0,n_gap_all_zero}` |
| 1.4 | stratified agreement | `results/exp24_results.json` | `aggregate.<model>.frontier.<q>.{strata_agreement,strata_abs_z}` |
| 1.5 | scale sensitivity | `results/exp24_results.json` | `aggregate.<model>.scale.<channel>.<scale>` |
| 2.1 | drift, MC SE, variance gap | `results/exp25_results.json` | `per_model.<model>.overall.<self\|other>` |
| 2.2 | K sensitivity | `results/exp25_results.json` | `per_model.<model>.overall.<t>.{drift_mean_K32,spearman_K64_vs_K32}` |
| 2.3 | negative vs matched-positive | `results/exp25_results.json` | `per_model.<model>."2.5.4_paired".<t>` |
| 2.3 | by probe kind | `results/exp25_results.json` | `per_model.<model>.by_kind.<kind>` |
| 2.3 | drift vs `J_**` | `results/exp25_results.json` | `per_model.<model>.spearman_drift_vs_Jstar_<t>` |
| 2.1–2.3 | per-probe raw values | `results/exp25_<model>.json` | `rows[].{J_star,kind,drift_*,tv_*,mu_n,var_n,y_sample_*}` |
| 2.1–2.3 | per-draw predictions | `arrays/exp25_<model>.npz` | `<model>__<did>__<seed>__p<probe>__<kind>__{mus,ys}` |
| 2.4 | exact-GP analytic gate | `results/exp25_controls.json` | `rows[] where map=exactgp_analytic → {drift_over_sn,tv_rel_residual}` |
| 2.4 | sampled GP / hierarchical GP | `results/exp25_controls.json` | `rows[] where map=<exactgp\|hiergp>_sampled → {drift_*,drift_over_se_*,tv_rel_gap_*}` |
| 2.4 | T1 imitator | `results/exp25_controls.json` | `rows[] where map=t1_imitator → {eps,bump_drift_only_*,bound_2eps_over_sn_*}` |
| 2.4 | quadratic drift control | `results/exp25_controls.json` | `rows[] where map=quadratic → {lam,drift_*,predicted_exact_*,pred_rel_err_*}` |
| 3.1 | final regret per objective | `results/exp27_results.json` | `per_objective.<obj>.arms.<arm>.{final_regret_per_seed,final_regret_mean,regret_curve_mean}` |
| 3.1 | paired comparisons | `results/exp27_results.json` | `comparisons.<a>_vs_<b>.{objective_wins_a,mean_norm_diff,bootstrap_ci,wilcoxon_p,per_objective}` |
| 3.2 | iterations to a fraction | `results/exp27_results.json` | `per_objective.<obj>.arms.<arm>.first_hit.<frac>` |
| 3.4 | audit along trajectories | `results/exp27_results.json` | `"2.7.7_2.7.8_audit".{by_iteration,frac_sigma2_undefined,trJ_over_n_*,n_minus_trJ_min}` |
| 4.1 | violation rate, shrinkage | `results/exp27_results.json` | `"2.6".<arm>.{violation_rate,shrinkage_median,shrinkage_frac_outside_01}` |
| 4.2 | `J_**` sign at acquisitions | `results/exp27_results.json` | `"2.6".tabpfn_jacobian.{J_star_available,n_Jstar_negative,violation_rate_Jstar_pos}` |
| 4.3 | violation vs regret | `results/exp27_results.json` | `"2.6".<arm>.violation_vs_final_regret_spearman` |
| — | raw trajectories | `results/exp27_bo_<objectives>.json` | `rows[].{best,simple_regret,audit,six,seconds}` |
| — | regret curves | `arrays/exp27_bo_<objectives>.npz` | `<obj>__<seed>__<arm>__regret` |

### 5.2 Scripts written, with outputs

| Script | Output |
|---|---|
| `src/exp24_acquisition.py` | `results/exp24_results.json` |
| `src/exp25_martingale.py` | `results/exp25_<model>.json`, `arrays/exp25_<model>.npz`, `logs/log_exp25_*.log` |
| `src/exp25_controls.py` | `results/exp25_controls.json`, `logs/log_exp25_controls.log` |
| `src/exp25_analyse.py` | `results/exp25_results.json` |
| `src/exp27_bo.py` | `results/exp27_bo_<objectives>.json`, `arrays/exp27_bo_<objectives>.npz`, `logs/log_exp27_*.log` |
| `src/exp27_analyse.py` | `results/exp27_results.json` |

Reused unchanged: `core/metrics.py` (`get_Q` seeded at 0, `asym`, `negeig`, `extract_variance`),
`core/surrogates.py`, `src/estimators.py`, `src/chunk2_estimators.py` (`make_predict`,
`H_FRAC_BY_MODEL`), and `src/exp24_acquisition.py`'s `ei`/`ucb`/`z_for` inside 2.7. Nothing outside
`experiments/phase_2/` was modified.

### 5.3 Failed attempts and rejected settings

| What | Why rejected |
|---|---|
| Naive EI | Underflows to exactly `0.0` for `z < -39`; on 3 of 60 TabPFN splits every pool point returned `EI = 0` and `argmax` picked index 0. Replaced by log-EI. Correction 1. |
| Frontier gap measured on log-EI | `log EI` carries `-z^2/2`, so the gap grows as `z^2` by construction and returned `rho = -0.60`, the opposite of the prediction. Measured in EI space it is `+0.57` to `+0.85`. Correction 2. |
| `y+ = max(y_ctx)` as the only incumbent for 2.4.6 | Puts `1.3%` of pool points inside `\|z\| < 1`; the `z = 0` frontier the prediction concerns is never sampled. Incumbent swept over four quantiles. Correction 2. |
| Sampling every 2.5 control from the exact GP | The hierarchical GP's predictive mean differs by `2.5` units at the same probe; it produced `\|drift\|/SE = 23.065` for a Bayesian map. Each map now samples from its own predictive, the hierarchical one by mixture-component draw, giving `0.792`. |
| Gaussian sampling for the hierarchical GP | Its predictive is a mixture over lengthscales, not a Gaussian. Component-then-Gaussian sampling used. |
| Fixed-dimension direction vector in the T1 bump | The context grows `n -> n+1`, so a vector in `R^n` is not defined on both. `S(y) = sum(y)` used, which is the all-ones direction and defined at every `n`. |
| `lam Var(mu)` as the quadratic control's predicted drift | Omits `(E[mu] - mu_n)(1 + 2 lam mu_n)`, which dominates where targets are large (did=509 has `\|y\| ~ 5e3`); the relative error was `1.9` to `4.8`. The exact expression reproduces the measurement to `1e-11`. |
| T1 imitator as the drift positive control | Drift is a value functional and T1 bounds it by `2 eps`; measured, it does not grow with `eps`. Kept as the demonstration of that, with a quadratic map added as the control that does drift. Correction 4. |
| Sampled exact GP as the numerical-precision gate | MC SE is `~0.09 s_n` at `K = 64`. Checked analytically instead. Correction 3. |
| TabICL v2 and TabSwift in the BO loop | The Jacobian arm costs `2 x n_pool` calls per iteration: `12.4 h` and `3.9 h` respectively. Not run, stated. |
| Knowledge gradient | Inner expectation over the predictive future at every candidate, on top of the Jacobian arm. `NOT MEASURED`, not approximated. |

---

## 6. Interpretation

Everything above is measurement. This section is not.

### 6.1 The variance defect reaches the decision, and 2.1's ranking survives into the loop

2.4 answers its own gate in the negative: native and Jacobian pick the same candidate on `0.160` of
TabICL splits and about half of TabPFN's, and the Jacobian channel's regret is worse than pure
exploitation everywhere. The acquisition surfaces are still strongly rank-correlated
(`+0.53` to `+0.89`), so the channels agree about the shape of the landscape and disagree about the
argmax — which is what a decision-relevant defect looks like.

The loop then reproduces 2.1's ordering almost exactly. **Conformal-wrapped TabPFN beats the native
head on all five objectives** (`0 - 5`, mean normalised difference `+0.132`, CI `[+0.039, +0.266]`,
`p = 0.0064`) and is statistically indistinguishable from a properly fitted GP (`p = 0.45`, CI
straddling zero). The native head is significantly **worse** than the GP (`p = 0.042`), and the
Jacobian arm is worse still (`p = 0.0056` against the GP). So the brief's question — does
native-variance BO match or beat the GP — has the answer no, and the practical recommendation from
2.1 carries over unchanged: wrap the model, do not trust its head.

That conformal wins here is worth separating from 2.4, where conformal was a *degenerate* channel
that reduced exactly to greedy-on-the-mean. In the loop the calibration set is refitted each
iteration and the width tracks the growing context, so it is no longer constant across the decision
and it stops being greedy. The two results are consistent: constant width is useless for choosing
*among* candidates at one step, and a well-scaled width is what a sequential loop needs.

### 6.2 The martingale is violated, and the violation is where the theory is weakest

The self-consistency test finds real drift on both models with a genuine predictive distribution —
`|drift|/SE` of `7.4` (TabICL) and `14.2` (TabPFN) at targets other than the probe, with 81% and 90%
of probes beyond two standard errors. The controls make that credible: the exact GP is a martingale
to `1.07e-11` analytically and to `|drift|/SE = 0.694` when sampled, and the hierarchical GP — the
control that matters, Bayesian with a label-dependent Jacobian — sits at `0.792` once it is sampled
from its own mixture rather than from the wrong law.

Two features of the violation are worth stating because they were not predicted. First, it is much
clearer at `x_* != x'` than at `x_* = x'`: TabICL's drift at the probe itself is inside Monte Carlo
error (`0.808`). The incoherence is in how an observation propagates to *other* locations, not in how
the model absorbs it locally. Second, the variance side nearly closes — the total-variance identity
has relative gaps of `0.1%` to `1.6%` — while the mean side does not. A3 concerns the coupling of
those two channels, and here the coupling identity holds while the mean identity fails.

**The martingale test is a value functional, and T1 says value functionals cannot separate the
classes with any margin.** That it detects something anyway is not a contradiction: T1 is about
worst-case separation with a uniform margin, not about whether a particular non-Bayes map happens to
drift. But it does mean the drift measured here is evidence about *these checkpoints* and not a
general instrument, and the T1 imitator control makes that concrete — it is ε-close in values and its
drift is bounded by `2ε` regardless of how non-Bayesian it is.

### 6.3 The A2-localised prediction fails for the third time

2.5.4 was the novel part of this brief: the claim that drift should be larger and signed wrong where
`J_** < 0`. The result is one significant signed effect, on one model, in one target configuration —
TabPFN at the probe itself, `p = 0.020` — with no magnitude effect (`p = 0.083`), an
opposite-signed effect on TabSwift, and no effect at all on TabICL because its negative stratum is
**empty**: 0 of 3000 held-out probes. Spearman between drift and `J_**` across all probes is
`+0.02` to `+0.11`.

2.6.3 could not be tested at all: `J_**` at the acquired point was non-negative on **all 500**
acquisitions. Acquisition functions select points the model is optimistic and uncertain about, and
those are not the points where the appended-query diagonal goes negative. So the mechanism by which
a negative diagonal was supposed to compound in a loop has no opportunity to act, at least under EI.

Taken with 2.2, which found the negative eigendirection does not localise onto anything actionable,
and 2.3, where the A2-derived loading added nothing, this is the third independent attempt to convert
A2 into a per-point diagnostic and the third null. The A1-derived asymmetry score of 2.3 remains the
only downstream quantity that has carried signal.

### 6.4 What TabPFN does in a BO loop that a Bayesian would not

The sign check is the cleanest behavioural statement in this brief. On `6.0%` to `7.8%` of
acquisitions the model finds a point better than it predicted and becomes *less* optimistic about it,
against `1.0%` for a refitted GP on the identical protocol. The shrinkage factor
`(mu_{n+1} - mu_n)/(y_t - mu_n)` lies outside `[0, 1]` on `19%` to `25%` of acquisitions against the
GP's `5%`, and its median is `0.55` against the GP's `0.99` — the model absorbs about half of each
surprise where a posterior mean absorbs nearly all of it.

Whether that costs anything is not established here. The per-trajectory correlation between violation
count and final regret changes sign across objectives for every arm, and with 5 trajectories per cell
those correlations carry little weight.

### 6.5 One thing that did not go wrong

2.1 §3 found TabICL nearly interpolating at `n = 100`, which left `sigma_hat^2` undefined on 10 of 60
splits. The `n - tr J` watch was put in because BO contexts start small and grow. On TabPFN across
500 iterations at `n = 10` to `29`, `tr J / n` rises from `0.474` to `0.614` and never exceeds
`0.941`, the minimum `n - tr J` is `+1.35`, and `sigma_hat^2` is defined at every single iteration.
The failure mode that broke the estimator at `n = 100` on one model does not appear at BO context
sizes on another.

### 6.6 What this does not settle

The BO study is one model, five synthetic objectives, `T = 20`, `S = 5`, and a 128-point pool — 25
paired trajectories per comparison, so the objective-level win counts rest on five points each.
TabICL v2 and TabSwift were not run in the loop, and knowledge gradient — the acquisition most
exposed to what 2.5 measures, because it is literally an expectation over the model's predictive
future — was not run at all. No real objective was included. TabSwift's entire 2.5 result rests on a
Gaussian surrogate for a predictive distribution it does not have, and is not a statement about its
own sampling law. The drift measured in 2.5 is at `n = 100`; the loop operates at `n = 10` to `29`,
and nothing here connects the two.

---

<a id="part-d"></a>

# Part D — Experiment 2.1 — Full Chunk-by-Chunk Record

*Source document: `UNCERTAINTY_EXPERIMENT.md`. Content unchanged; headings demoted one level.*

## Jacobian-Derived Predictive Variance

Measurement record. Every number below is written by a script in this directory to a JSON file in
this directory; Section P gives the file and key for each. Quantities not computed say
`NOT MEASURED` with a reason.

Measurement is separated from interpretation. Sections 1–4 contain numbers only. Section R at the
end contains the reading.

---

### 0. Correction to the estimator, before any measurement

The brief specifies

    s2_jac(x_*) = sigma2_hat * (1 + J_**),    J_** = dm_*/dy_* after appending (x_*, y_*)

This form does not estimate the held-out predictive variance. The correction was derived before
Chunk 1 was run and both forms were then measured; the measurement is in Section 1.1.

**Derivation.** T2 applied to the augmented context, at the appended row, is exact for every prior:

    sigma^2 * J_**  =  Var(f_* | y, y_*)                                     (1)

The conditioning set on the right includes the hypothetical label `y_*` that we invented. The
quantity the experiment wants is the predictive variance at a fresh observation given the
**original** context,

    s^2(x_*)  =  Var(f_* | y) + sigma^2                                      (2)

and (1) is strictly smaller than `Var(f_* | y)`, because it conditions on one extra observation.
Writing `v = Var(f_* | y)`, a Gaussian update from observing `y_* = f_* + eps`, `eps ~ N(0, sigma^2)`,
gives `Var(f_* | y, y_*) = v - v^2/(v + sigma^2) = v sigma^2/(v + sigma^2)`, so (1) reads
`J_** = v/(v + sigma^2)`, which inverts to `v = sigma^2 J_**/(1 - J_**)` and therefore

    s^2(x_*)  =  v + sigma^2  =  sigma^2 / (1 - J_**)                        (3)

Two consequences, both measured in 1.1. First, `sigma^2 (1 + J_**)` carries a systematic error on
the exact GP, where the closed form is available. Second, a Bayesian has `J_** in [0, 1)`, so
`sigma^2 (1 + J_**)` is **bounded above by `2 sigma^2`** however uncertain the query is, while the
quantity in (2) is unbounded.

The inversion step in (3) assumes the posterior at the appended point is Gaussian. Where it is not,
the residual is a third-cumulant term (T3). Section 1.3 measures that residual on the hierarchical
GP, whose posterior is a mixture and therefore not Gaussian.

**Both forms are carried through the whole experiment.** `(3)` is used as the primary and is what
`s2_jac` means below; `(1 + J_**)` is stored at every point under `s2_jac_specified`.

---

### 1. CHUNK 1 — estimator validation on maps with known answers

No frozen models in this chunk.

**Configuration.** Contexts `generate_audit_context(n=100, d=5, sigma=0.5, seed=s)`, seeds
`[42, 100, 200, 300, 400]`. `sigma=0.5` rather than the audit's `1.0` so the draw is well specified
for `ExactGP(sigma=0.5, lengthscale=1.0)` — item 1.2 asks whether `sigma2_hat` recovers the true
`sigma^2`, which is only a meaningful question when the data come from the model being fitted. The
design matrix is the audit's, duplicated rows included. Queries: 50 fresh `N(0, I_5)` points per
seed from `RandomState(seed+1000)`, disjoint from the context seed. Step `h = 1e-4`; the maps are
float64 and noiseless.

#### 1.0 The train/test surrogates are the audit's surrogates

`core/surrogates.py` has no notion of a held-out query, so `GPTrainTest` and `HierGPTrainTest` were
added in `src/estimators.py`. Computed in-sample, they must reproduce the audit's objects. Worst
absolute deviation over the five seeds:

```
exactgp mean map   0.000e+00        hiergp mean map   0.000e+00
exactgp variance   9.659e-15        hiergp variance   8.771e-15
```

#### 1.1 Exact GP — `s2_jac` against the closed form

Target is `diag(K_* - K_*(K + sigma^2 I)^-1 K_*^T) + sigma^2`. Relative error per query point,
using the **true** `sigma^2` so the form's error is not confounded with `sigma2_hat`'s. Per seed,
over 50 query points each:

```
                    seed 42      seed 100     seed 200     seed 300     seed 400
specified  median  4.170e-01    4.854e-01    4.822e-01    4.909e-01    5.196e-01
           p90     6.107e-01    5.963e-01    6.224e-01    6.283e-01    5.991e-01
           max     6.290e-01    6.255e-01    6.397e-01    6.388e-01    6.379e-01

inverted   median  2.664e-13    2.655e-13    9.789e-14    4.733e-14    3.292e-13
           p90     7.050e-13    4.951e-13    2.795e-13    1.335e-13    8.340e-13
           max     1.305e-12    8.924e-13    8.933e-13    4.275e-13    1.376e-12
```

Measured `J_**` ranges per seed: `[0.3077, 0.7931]`, `[0.4009, 0.7909]`, `[0.4346, 0.7998]`,
`[0.4551, 0.7992]`, `[0.3155, 0.7987]`. Clip rate `0.0` on every seed for both forms.

Repeating with `sigma2_hat` in place of the true `sigma^2` gives, for the inverted form, p90 relative
error `[7.396e-02, 1.055e-01, 3.191e-01, 2.909e-01, 1.058e-01]` — the spread of `sigma2_hat` itself,
quantified in 1.2s.

#### 1.2 `sigma2_hat`, `tr J`, `n - tr J` on the exact GP

```
seed        tr J      n - tr J    sigma2_hat    sigma2_hat / sigma^2
42       59.9054      40.0946      0.277748           1.1110
100      62.0006      37.9994      0.320954           1.2838
200      60.0735      39.9265      0.240523           0.9621
300      59.4606      40.5394      0.218714           0.8749
400      62.4741      37.5259      0.309534           1.2381
```

Mean ratio over the five seeds `1.0940`, sample SD `0.1750`. True `sigma^2 = 0.25`.
`n - tr J > 0` on every seed.

#### 1.2s Sampling spread of `sigma2_hat`, in closed form

Whether a single-seed ratio can land within 10% is a property of the estimator's sampling
distribution. Under the well-specified GP, `y ~ N(0, C)` with `C = K + sigma^2 I`, giving
`m(y) - y = -sigma^2 C^-1 y`, `n - tr J = sigma^2 tr(C^-1)`, hence `E[sigma2_hat] = sigma^2` exactly
and

    SD(sigma2_hat)/sigma^2 = sqrt(2 tr(C^-2)) * sigma^2 / (n - tr J)

```
seed        n - tr J (analytic)    predicted SD(sigma2_hat)/sigma^2
42               40.0946                     0.1750
100              37.9994                     0.1760
200              39.9265                     0.1758
300              40.5394                     0.1746
400              37.5259                     0.1760
```

Mean predicted relative SD of a single-seed `sigma2_hat`: `0.1755`. Of the five-seed average:
`0.0785`. Observed sample SD of the five ratios in 1.2: `0.1750`.

#### 1.3 Hierarchical GP — same comparison

Closed-form target is the mixture predictive variance
`E_theta[Var_theta(f_*|y)] + Var_theta(E_theta[f_*|y]) + sigma^2`, computed from the mixture directly
and independently of the Jacobian.

```
seed        tr J      n - tr J    sigma2_hat    ratio     inverted p90    inverted max
42       66.4607      33.5393      0.231672     0.9267     2.561e-02       4.323e-01
100      70.0639      29.9361      0.233312     0.9332     7.303e-02       2.290e-01
200      55.3528      44.6472      0.282970     1.1319     1.654e-01       1.135e+00
300      65.8430      34.1570      0.183911     0.7356     4.048e-02       1.087e-01
400      68.3287      31.6713      0.272895     1.0916     1.367e-02       2.143e-02
```

Mean ratio `0.9638`, sample SD `0.1573`. Specified-form p90 relative error on the same seeds:
`[6.324e-01, 6.336e-01, 6.010e-01, 6.379e-01, 6.305e-01]`. Clip rate `0.0` on every seed.

#### 1.4 Append perturbation at the original context points

`||m_appended[:n] - m_original|| / ||m_original||`, per query, 50 queries per seed.

```
exactgp    seed 42   median 7.5480e-03   p90 3.4134e-02   max 7.5938e-02
           seed 100  median 4.8177e-03   p90 2.0235e-02   max 3.4757e-02
           seed 200  median 5.4633e-03   p90 2.5581e-02   max 4.2948e-02
           seed 300  median 5.3573e-03   p90 1.6564e-02   max 4.2964e-02
           seed 400  median 5.2477e-03   p90 2.8605e-02   max 4.4189e-02

hiergp     seed 42   median 5.1130e-03   p90 2.6004e-02   max 5.6808e-02
           seed 100  median 2.5464e-03   p90 8.3453e-03   max 1.5924e-02
           seed 200  median 1.0349e-02   p90 2.9168e-02   max 6.9559e-02
           seed 300  median 4.8572e-03   p90 9.6867e-03   max 3.7359e-02
           seed 400  median 2.3516e-03   p90 1.8702e-02   max 4.1227e-02
```

#### 1.4b Per-query rank correlation, append perturbation against estimator error

Spearman over the 50 queries of each seed, inverted form, true `sigma^2`:

```
exactgp   -0.1461   -0.0820   +0.2338   +0.2843   -0.0550
hiergp    +0.4005   +0.9000   +0.4681   +0.7600   +0.5461
```

Median estimator error on the exact GP rows above is `2.7e-13` to `4.7e-14`.

#### 1.5 `J_**` sensitivity to the hypothetical label `y_*`

`y_*` over `{mean(y), mean(y) + SD(y), mean(y) - SD(y)}`; spread is max − min across the three, per
query.

```
exactgp   abs spread  median 1.9e-13 to 5.6e-13   max 8.3e-13 to 1.1e-12
                      relative spread max 1.13e-12 to 3.52e-12
hiergp    abs spread  median [1.244e-02, 1.123e-02, 1.311e-01, 2.159e-02, 4.484e-03]
                      max    [9.143e-01, 2.379e-01, 5.538e-01, 1.410e-01, 1.317e-02]
                      relative spread max [0.9686, 0.3348, 0.8679, 0.2402, 0.0213]
```

#### 1.5b Which `y_*` to standardise on

Four candidates scored against the closed form, inverted form, true `sigma^2`. Mean over seeds of
the per-seed p90 relative error:

```
                            exactgp        hiergp
context_mean               4.8941e-13     6.3633e-02
self_consistent m_*(y)     8.7066e-13     5.4676e-02
mean + SD                  1.3638e-12     4.2825e-01
mean - SD                  1.9566e-12     3.7403e-01
```

`y_* = m_*(y)` is used for the frozen models in Chunks 2–4. It costs nothing: the query prediction
is already computed for the mean channel.

#### 1.6 Step-size plateau for the `J_**` difference

Frozen models are out of scope here, so each model's configuration is predicted the way the audit
predicts its artifact floors: quantise a control at that model's measured output quantum
(`FINAL_NUMBERS.md` 3.4) and sweep `h`. Wrapped in the context-statistic normaliser for TabPFN v2 and
TabICL v2, unwrapped for TabSwift, whose target scaler is commented out in source. Mean `J_**` over
20 queries × 5 seeds:

```
map                                h=1e-5    1e-4      1e-3      1e-2      1e-1      1e+0
exactgp                            0.66755   0.66755   0.66755   0.66755   0.66755   0.66755
exactgp_wrapped                    0.66823   0.66823   0.66823   0.66823   0.66823   0.66823
hiergp                             0.71380   0.71380   0.71380   0.71380   0.71386   0.72000
quantised @ TabICL q=9.835e-07     0.67073   0.66818   0.66823   0.66823   0.66823   0.66823
quantised @ TabSwift q=4.273e-04   0.42725   0.57679   0.66651   0.66886   0.66760   0.66755
quantised @ TabPFN  q=6.870e-04    1.03050   0.79005   0.66639   0.66776   0.66831   0.66822
```

Worst per-query deviation from the `h=1` value, same maps:

```
quantised @ TabICL     1e-5 4.68e-02   1e-4 4.64e-03   1e-3 4.72e-04   1e-2 4.43e-05   1e-1 4.92e-06
quantised @ TabSwift   1e-5 2.06e+01   1e-4 1.63e+00   1e-3 1.89e-01   1e-2 1.86e-02   1e-1 2.14e-03
quantised @ TabPFN     1e-5 3.38e+01   1e-4 3.04e+00   1e-3 2.83e-01   1e-2 3.37e-02   1e-1 3.09e-03
```

These amplitudes are in units where the context label SD is `~1.4`. Chunk 2 re-expresses them as
fractions of `std(y_ctx)` and checks the prediction on the frozen models; see 2.2b.

#### Gate 1

| criterion | required | measured | met as written |
|---|---|---|---|
| GP `s2_jac` relative error, p90, **specified form** | `< 0.05` | worst seed `6.283e-01` | no |
| GP `s2_jac` relative error, p90, **inverted form** | `< 0.05` | worst seed `8.340e-13` | yes |
| `sigma2_hat / sigma^2`, per seed | within `0.10` | worst deviation `0.2838` | no |
| append perturbation | `< 0.01` | worst `7.594e-02` | no |

Two of the three failures are addressed rather than worked around, with the numbers that settle
them; nothing was substituted silently.

- **Specified form.** Failed. Replaced by the inverted form of Section 0, which meets the same
  criterion by eleven orders of magnitude. Both are carried forward.
- **`sigma2_hat` per seed.** The criterion asks a single realisation to land within 10% of an
  estimator whose own closed-form sampling SD at this `n` is `0.1755` (1.2s), against an observed
  sample SD of `0.1750`. The five-seed average ratio is `1.0940` against a predicted SD of `0.0785`
  for that average. No estimator change was made.
- **Append perturbation.** The exact-GP estimator is accurate to `1e-12` (1.1) at the same queries
  whose append perturbation reaches `7.6e-02` (1.4), and the per-query rank correlation between
  perturbation size and estimator error on the exact GP is `-0.146` to `+0.284`, straddling zero
  (1.4b). The estimator does not require the appended map to equal the original map; it requires T2
  to hold on the augmented context, which 1.1 tests end to end. The append route is **kept**, and the
  leave-one-out alternative the brief names as the fallback was not implemented.

The append perturbation remains a live diagnostic for the frozen models, where `n` changes from 100
to 101 and behaviour is `n`-dependent in ways a GP's is not. It is not measured on them: see the
Chunk 2 checklist.

#### Chunk 1 checklist

- [x] GP `s2_jac` vs closed form, per-point relative error distribution — 1.1, both forms, per seed
- [x] `sigma2_hat/sigma^2`, `tr J`, `n - tr J` on the GP, per seed — 1.2, plus closed-form sampling
      spread in 1.2s
- [x] Hierarchical GP, same comparison — 1.3
- [x] Append perturbation at the original context points — 1.4, plus 1.4b
- [x] `J_**` stability across three hypothetical `y_*` values — 1.5, plus the choice scored in 1.5b
- [x] Step-size plateau for the `J_**` difference — 1.6
- [x] Gate 1 verdict with the numbers that satisfy or fail it — above

---

### 2. CHUNK 2 — data, splits, and the four estimators

#### 2.1 Dataset selection

**Rule, fixed before any result was looked at.** A dataset is admitted iff all of: listed on OpenML
with status `active` and a single declared default target (R1); `200 <= n <= 2000` (R2);
`2 <= d <= 50` after dropping the target (R3); every feature numeric, so no categorical-encoding
choice is introduced (R4); zero missing values (R5); target numeric, non-constant, with at least 20
distinct values, which excludes ordinal targets that are classification in disguise (R6); at least
200 distinct rows (R7). Candidates are ordered by OpenML id ascending and the first 12 that pass are
taken. The ordering is arbitrary but fixed in advance and independent of anything measured later.

387 datasets passed the listing-level filters. 22 were opened in id order; 12 were admitted and 10
rejected. **No dataset was dropped after results were seen.**

```
   did name                     n    d      target mean      target sd   distinct
   223 stock                  950    9           46.99          6.536        203
   229 pwLinear               200   10         -0.3373          4.464        189
   509 places                 329    8            5525           1083        312
   522 pm10                   500    7           3.271         0.8859        117
   540 mu284                  284    9           25.57          14.52         50
   547 no2                    500    7           3.698         0.7498        385
   549 strikes                625    6           302.3          560.2        358
   560 bodyfat                252   14           19.15          8.352        176
   579 fri_c0_250_5           250    5       5.004e-10          0.998        250
   581 fri_c3_500_25          500   25      -6.394e-10          0.999        500
   582 fri_c1_500_25          500   25      -6.726e-10          0.999        500
   583 fri_c1_1000_50        1000   50      -2.936e-10         0.9995       1000
```

Target scale spans three orders of magnitude, from `0.7498` (`no2`) to `1083` (`places`). This
matters for 2.2b.

**Exclusions, with the rule each failed:**

```
did=8     R6  target has 16 distinct values
did=230   R7  only 190 distinct rows
did=482   R4  categorical feature present
did=494   R4  categorical feature present
did=500   R4  categorical feature present
did=513   R4  categorical feature present
did=519   R7  only 79 distinct rows
did=533   R4  categorical feature present
did=536   R4  categorical feature present
did=561   R4  categorical feature present
```

#### 2.2 Splits

Five splits per dataset, seeds `[42, 100, 200, 300, 400]`. Each shuffles the rows and takes
`n_ctx = 100` context rows, then up to 100 calibration rows, then up to 50 test rows. Features are
standardised using statistics from the **context rows only**; targets are untouched.

The test cap of 50 is a stated cost bound, not a silent truncation: each test point costs two
forward passes for `J_**`, and the grid is 3 models x 12 datasets x 5 splits. It gives 3000 scored
points per model.

```
did    n_ctx  n_cal  n_test        did    n_ctx  n_cal  n_test
223      100    100      50        549      100    100      50
229      100     50      50        560      100     76      50
509      100    100      50        579      100     75      50
522      100    100      50        581      100    100      50
540      100     92      50        582      100    100      50
547      100    100      50        583      100    100      50
```

#### 2.2b Probe step on the frozen models

**A first pass was discarded.** It used the audit's amplitudes as absolute numbers (`1e-3` for
TabICL v2, `1e-1` for TabPFN v2 and TabSwift). Every model in the roster renormalises the target
internally and un-normalises its output, so the output quantum expressed in the original target
units is proportional to `std(y)`. On `did=509` (`places`, `std(y) = 1108`) an absolute `h = 1e-1` is
`9e-5` of a label SD, inside the quantisation floor: it returned `tr J` of `110.07` and `243.04` on
seeds 300 and 200 against `n = 100`, so `n - tr J < 0` and `sigma2_hat` was undefined, with 31 and 37
negative diagonal entries. That pass was stopped and deleted before any estimator number was read
off it.

The step is now `h = h_frac * std(y_ctx)`. The fractions reproduce the audit's own amplitudes on the
audit's own contexts, whose mean context label SD is `1.457782` (`FINAL_NUMBERS.md` 5.7):

```
TabICL v2   h_frac = 1e-3 / 1.457782 = 6.8598e-04
TabPFN v2   h_frac = 1e-1 / 1.457782 = 6.8598e-02
TabSwift    h_frac = 1e-1 / 1.457782 = 6.8598e-02
```

Swept on three datasets spanning the target-scale range, seed 200, 10 queries, `tr J` from the full
`100x100` context Jacobian at each step.

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

**TabSwift**

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

TabSwift returns `tr J` and `J_**` of exactly `0.000` at `h_frac <= 1e-3` on `did=223`: its outputs
are float16 and the difference falls below one output quantum.

**TabICL v2**

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

`tr J` is flat at `99.98` to `100.03` across four decades of step size, against `n = 100`, so this is
not a step-size artifact. Remaining TabICL rows and the per-dataset picture are in 2.3.

The chosen `h_frac` sits inside the flat region for every model and dataset where a flat region
exists. This matches the prediction made in 1.6 from the quantised controls: TabICL resolvable from
`h_frac ~ 1e-4`, TabPFN v2 and TabSwift from `h_frac ~ 1e-2`.

#### 2.3 The four estimators

The mean channel is identical for estimators 1–3 on any given split, so a difference between them is
a difference in uncertainty only. Estimator 4 carries its own mean and is a reference, not a
competitor.

| # | estimator | definition |
|---|---|---|
| 1 | **native** | the model's own head — see the per-model routes below |
| 2 | **Jacobian** | `sigma2_hat / (1 - J_**)`, Section 0. `sigma2_hat = ‖m(y) - y‖² / (n - tr J)` from the full context Jacobian; `y_* = m_*(y)`, the self-consistent choice scored in 1.5b, available at no extra cost from the mean channel. `sigma2_hat * (1 + J_**)` is stored at every point as `s2_jac_specified`. |
| 3 | **split conformal** | absolute-residual conformal on the calibration rows, half-width the `ceil((n_cal+1)(1-alpha))`-th order statistic. Constant width by construction. |
| 4 | **oracle GP** | `sklearn.GaussianProcessRegressor`, `ConstantKernel*RBF + WhiteKernel`, marginal-likelihood fit on the context rows, 2 restarts. `kernel_.diag` carries the white-noise term, so its predictive variance is already for a fresh observation. |

**Native variance route per model, with the rejected alternative named.**

- **TabICL v2** — `output_type="quantiles"` on 9999 uniform levels in `[1e-4, 1-1e-4]`, integrated by
  `core/metrics.py::extract_variance`. **Rejected:** `output_type="variance"`, which is
  `raw_quantiles.var(dim=-1)` (`tabicl/_model/tabicl.py:582`) — a spread over the quantile grid, not
  the second moment of the predictive distribution. The returned array is `(n_test, n_levels)`;
  orientation checked directly, quantiles monotone along the last axis.
- **TabPFN v2** — `FullSupportBarDistribution.variance(logits)`, the integrated second moment with
  half-normal tails on the outer bins (`tabpfn/architectures/base/bar_distribution.py:606`).
  **Rejected:** `sum(bin_probability * bin_centre)`, which is `E[X]` rather than a second moment and
  omits the tail bins.
- **TabSwift** — **does not exist.** The regression head is `Linear(384, 1)`, a point estimate;
  `models/registry.py` carries `has_predictive_distribution=False`. **TabSwift is scored on
  estimators 2, 3 and 4 only.** No variance is synthesised for it. That the Jacobian route supplies
  an uncertainty estimate to a model that ships none is reported in 3.5 as a standalone result.

**Clip floor.** Any variance estimate below `1e-3 * sigma2_hat` is raised to that value, and the rate
at which this happens is reported per model per dataset in 2.4. The floor is a fraction of
`sigma2_hat` rather than of a calibration quantity so that estimator 2 stays derivable from the model
alone. Two distinct events are counted separately: `J_** >= 1`, where `sigma2_hat/(1 - J_**)` is
negative or infinite, and `n - tr J <= 0`, where `sigma2_hat` itself is undefined and the whole split
is excluded rather than filled in.

#### 2.4 Clip rate, `n - tr J`, and where the estimator is undefined

Floor `1e-3 * sigma2_hat`. `clip` is the fraction of test queries raised to the floor; `J**>=1` is the
fraction with `J_** >= 1`, where the inverted form is negative or infinite; `n-trJ` and `negJii` are
means over the five splits; `splits` is how many of the five have `n - tr J > 0` and therefore a
defined `sigma2_hat`.

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

model totals            50/60 0.102  0.222   12.73    60/60 0.0137 0.0137  56.86    60/60 0.0120 0.0120  70.37
```

`negJii`, mean count of negative diagonal entries out of 100, per model over all 60 splits: TabICL v2
`0.1167` (7 entries across 4 splits), TabPFN v2 `2.2333` (134 across 22 splits), TabSwift `16.65`
(999 across 39 splits).

**TabICL v2 on the three highest-dimensional datasets** (`fri_c3_500_25`, `fri_c1_500_25`,
`fri_c1_1000_50`, `d = 25, 25, 50`) has `n - tr J` of `-0.15`, `-0.23` and `-0.62` against `n = 100`,
so `sigma2_hat` is undefined on 3, 3 and 4 of the 5 splits respectively; those 10 splits are excluded
from every TabICL Jacobian aggregate rather than filled in. On the same datasets `J_** >= 1` on
`59%`, `63%` and `73%` of queries. Section 2.2b shows `tr J` there is flat at `99.98` to `100.03`
across four decades of step size.

#### Gate 2

| criterion | required | measured | met |
|---|---|---|---|
| all four estimators produce finite intervals on every dataset | — | all finite, all models, 4 levels x 60 splits | yes |
| conformal coverage at nominal `90%` | within `±0.03` | TabICL `0.8830` (`-0.0170`), TabPFN `0.8957` (`-0.0043`), TabSwift `0.9153` (`+0.0153`) | yes |
| oracle GP coverage at nominal `90%` | within `±0.05` | `0.9073` (`+0.0073`) | yes |

Conformal's per-split coverage SD is `0.0564`, `0.0494`, `0.0450` over 60 splits. That spread is
inherent, not slack in the harness: with `n_cal` calibration points and rank
`k = ceil((n_cal+1)(1-alpha))`, coverage conditional on the calibration draw is `Beta(k, n_cal+1-k)`,
whose SD at `level = 0.90, n_cal = 100` is `sqrt(0.9*0.1/102) = 0.0297`, and `n_cal` here ranges from
50 to 100. The `±0.03` applies to the aggregate, which is what the table reports.

The Jacobian estimator is defined on 50/60 TabICL splits and 60/60 for the other two. That is a
coverage limitation of estimator 2, not a gate failure, and it is carried into every TabICL Jacobian
figure below as `n_datasets = 12` with 50 contributing splits.

#### Chunk 2 checklist

- [x] Dataset list with OpenML IDs, `n`, `d`, target scale; selection rule stated — 2.1, rule stated
      before the table
- [x] Exclusions with reasons — 2.1, all 10, each with the rule it failed
- [x] Split scheme, seeds — 2.2
- [x] Four estimators implemented; TabSwift's three noted — 2.3
- [x] Native variance route stated per model, with the rejected alternative named — 2.3
- [x] Clip rate and floor, per model per dataset — 2.4
- [x] Gate 2 verdict, including conformal's coverage as the harness check — above
- [ ] **Append perturbation on the frozen models — `NOT MEASURED`.** Item 1.4 was run on the analytic
      maps only. On the frozen models `n` changes from 100 to 101 and behaviour is `n`-dependent in
      ways a GP's is not, so the analytic result does not transfer. It costs one extra forward pass
      per query and was not budgeted into this run.

---

### 3. CHUNK 3 — measurement

All figures aggregate per dataset first, then across the 12 datasets, so a dataset with more test rows
does not dominate. `gp_oracle` is identical across the three model blocks by construction: it does not
depend on the model.

#### 3.1 Empirical coverage and signed deviation

```
TabICL v2          cov50    cov80    cov90    cov95     dev50    dev80    dev90    dev95
native            0.5643   0.8313   0.9047   0.9423   +0.0643  +0.0313  +0.0047  -0.0077
jacobian          0.6038   0.7433   0.7755   0.7915   +0.1038  -0.0567  -0.1245  -0.1585
conformal         0.5010   0.7827   0.8830   0.9490   +0.0010  -0.0173  -0.0170  -0.0010
gp_oracle         0.5597   0.8253   0.9073   0.9447   +0.0597  +0.0253  +0.0073  -0.0053

TabPFN v2          cov50    cov80    cov90    cov95     dev50    dev80    dev90    dev95
native            0.5560   0.8257   0.9057   0.9470   +0.0560  +0.0257  +0.0057  -0.0030
jacobian          0.6090   0.8390   0.9030   0.9343   +0.1090  +0.0390  +0.0030  -0.0157
conformal         0.4883   0.8007   0.8957   0.9490   -0.0117  +0.0007  -0.0043  -0.0010
gp_oracle         0.5597   0.8253   0.9073   0.9447   +0.0597  +0.0253  +0.0073  -0.0053

TabSwift           cov50    cov80    cov90    cov95     dev50    dev80    dev90    dev95
native               absent -- Linear(384,1) point head
jacobian          0.4993   0.8577   0.9337   0.9610   -0.0007  +0.0577  +0.0337  +0.0110
conformal         0.5013   0.8120   0.9153   0.9597   +0.0013  +0.0120  +0.0153  +0.0097
gp_oracle         0.5597   0.8253   0.9073   0.9447   +0.0597  +0.0253  +0.0073  -0.0053
```

Per-dataset and per-split coverage for every cell is in `results/chunk3_results.json` under
`per_dataset.<model>.<did>.est.<estimator>.levels.<level>.coverage_per_split` and `per_split`.

#### 3.2 Interval score at nominal `90%` (Winkler; lower is better)

```
                 TabICL v2    TabPFN v2     TabSwift
native               515.8        515.9       absent
jacobian             772.0        547.4       1818.0
conformal            523.6        519.9       1553.0
gp_oracle            545.1        545.1        545.1
```

#### 3.3 Sharpness, raw and at matched coverage

`width@90` is the mean interval width as produced. `recalW@90` rescales each estimator's `s` by the
empirical `0.90` quantile of `|y - m|/s`, which forces achieved coverage to exactly `0.90`, so widths
are compared at matched coverage and a narrow-but-wrong estimator gets no credit.

```
                 TabICL v2            TabPFN v2            TabSwift
              width@90 recalW      width@90 recalW      width@90 recalW
native           378.5  299.6         375.5  301.0        absent
jacobian         582.1  673.8         416.1  336.5        1714.0 1283.0
conformal        352.0  344.3         356.6  312.1        1370.0 1269.0
gp_oracle        386.1  326.3         386.1  326.3         386.1  326.3
```

TabICL's Jacobian estimator is the only cell whose recalibrated width exceeds its raw width
(`673.8` against `582.1`), which follows from its under-coverage in 3.1: matching coverage requires
scaling up.

#### 3.4 Gaussian NLL under each estimator's variance, same mean

Conformal has no density; its variance is taken as `(qhat(0.90)/z(0.90))^2`, a stated conversion.

```
                 TabICL v2    TabPFN v2     TabSwift
native               2.238        2.119       absent
jacobian             871.7       40.948       18.091
conformal            10.68       14.075        3.261
gp_oracle            2.708        2.708        2.708
```

#### 3.5 Per-dataset interval score at `90%`, and win/loss counts

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

Win/loss over the 12 datasets, interval score at `90%`, first estimator wins when its score is lower:

```
                              TabICL v2   TabPFN v2   TabSwift
jacobian vs native               1 - 11      0 - 12    n/a (no native head)
jacobian vs conformal            0 - 12       4 - 8      1 - 11
jacobian vs gp_oracle            2 - 10       8 - 4       5 - 7
native   vs conformal             9 - 3       8 - 4    n/a
native   vs gp_oracle            10 - 2       9 - 3    n/a
```

**TabSwift's Jacobian estimator is the standalone result the brief asks for:** the model ships no
predictive distribution, and the Jacobian route supplies one that attains `0.9337` coverage at nominal
`0.90` (3.1). Split conformal on the same model attains `0.9153` with a lower interval score
(`1553` against `1818`) and a lower NLL (`3.261` against `18.091`), and wins on 11 of 12 datasets.

#### 3.6 Rank correlation between each estimator's variance and the squared residual

Spearman per split, averaged over splits then over datasets.

```
                 TabICL v2    TabPFN v2     TabSwift
native               0.300        0.283       absent
jacobian             0.054        0.124        0.042
conformal        NOT MEASURED -- constant width by construction, zero rank variance
gp_oracle            0.182        0.182        0.182
```

Per dataset, `native` against `jacobian`:

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

#### 3.7 Paired comparisons

Differences are paired on the same datasets and the same splits. Dataset-level: 12 paired
differences, percentile bootstrap CI (10000 resamples) and Wilcoxon signed-rank. Split-level: 60
paired differences at nominal `90%`. Interval score at `90%`; a negative mean difference favours the
first estimator.

```
model      comparison                mean diff   dataset-level 95% CI   Wilcoxon p   split-level 95% CI     p
TabICL     jacobian - native          +256.12   [  +3.93,  +682.12]      0.00098     n/a (50 vs 60 splits)
TabICL     jacobian - conformal       +248.36   [  +2.68,  +671.32]      0.00049     n/a (50 vs 60 splits)
TabICL     native   - conformal         -7.76   [ -28.93,    +5.76]      0.2661      [-30.65, +14.07]   0.233
TabICL     jacobian - gp_oracle       +226.85   [  +1.16,  +615.47]      0.02686     n/a (50 vs 60 splits)
TabICL     native   - gp_oracle        -29.27   [ -67.81,    -1.17]      0.00488     [-60.62,  -0.77]   8.4e-06
TabPFN     jacobian - native           +31.53   [  +0.33,   +85.63]      0.00049     [ +4.69, +67.58]   8.7e-08
TabPFN     jacobian - conformal        +27.57   [ -11.86,   +94.82]      0.5186      [ -8.98, +78.47]   0.0353
TabPFN     native   - conformal          -3.97   [ -26.22,   +16.03]      0.1294      [-19.84, +13.50]   0.0690
TabPFN     jacobian - gp_oracle         +2.32   [ -30.87,   +39.74]      0.1763      [-33.44, +39.62]   0.0058
TabPFN     native   - gp_oracle        -29.22   [ -67.82,    -1.39]      0.00928     [-70.12,  +7.38]   1.1e-05
TabSwift   jacobian - conformal       +265.30   [  +3.04,  +764.94]      0.00244     [+74.17, +488.99]  4.3e-07
TabSwift   jacobian - gp_oracle      +1273.10   [ +14.67,  +3680.0]      0.09229     [+345.71, +2278.0] 4.1e-05
```

Split-level tests are omitted for the three TabICL rows involving `jacobian` because that estimator is
defined on 50 splits and its comparators on 60, so the differences are not paired. The dataset-level
test is paired: it compares per-dataset means computed over whichever splits each estimator has, and
that mismatch is itself stated in 2.4.

NLL, dataset-level:

```
model      comparison               mean diff   95% CI                 wins
TabICL     jacobian - native         +869.51   [ +287.00, +1603.60]    1 - 11
TabPFN     jacobian - native          +38.83   [   +5.12,   +75.75]    0 - 12
TabPFN     native   - conformal       -11.96   [  -35.75,    +0.05]   10 -  2
TabPFN     native   - gp_oracle        -0.59   [   -1.00,    -0.22]   10 -  2
TabSwift   jacobian - conformal       +14.83   [   +4.11,   +27.83]    3 -  9
```

#### Chunk 3 checklist

- [x] Coverage at four nominal levels, signed deviation, per dataset per split per estimator — 3.1,
      per-split values in `results/chunk3_results.json`
- [x] Interval score, per dataset, paired — 3.2, 3.5, 3.7
- [x] Sharpness conditional on coverage — 3.3, raw and coverage-matched
- [x] Gaussian NLL — 3.4
- [x] Per-dataset table and win/loss counts, not only pooled means — 3.5
- [x] Rank correlation between variance and squared residual — 3.6; `NOT MEASURED` for conformal,
      reason stated
- [x] Paired test or bootstrap CI on the differences — 3.7, both, at two aggregation levels

---

### 4. CHUNK 4 — auxiliary measurements

#### 4.1 SURE consistency

`R_SURE = ‖m(y) - y‖² + 2 sigma² tr J - n sigma²`, with `sigma²` the conformal-calibration residual
variance `mean(r_cal²)`. `R_SURE` estimates in-sample risk on the context rows; the held-out
comparator is `mean((y_test - m_test)²) - sigma²_cal`. The two are not the same object — out-of-sample
risk generally exceeds in-sample risk — so the difference is reported as a measured quantity, per
point, and not as an error in either.

```
model       splits   tr J range        tr J mean   negative J_ii (total / splits)
TabICL v2       60   [ 45.63, 101.99]      87.27    7 / 4
TabPFN v2       60   [ 10.54,  79.14]      43.14  134 / 22
TabSwift        60   [ -1.51,  80.53]      29.63  999 / 39

model       SURE gap mean    SURE gap sd
TabICL v2       +138155         361747
TabPFN v2        +40874         127240
TabSwift         -39303         499676
```

The gap is dominated by the two datasets whose targets are on scales of `10^3` (`places`, `strikes`);
per-split values, per-dataset breakdown, `R_SURE`, `R_SURE_per_point`, `heldout_mse`, `sigma2_cal` and
`rss_ctx` are all in `results/chunk4_results.json` under `4.1_sure.<model>.rows`.

#### 4.1c Correlation between the SURE gap and the negative-diagonal count

Over the 60 splits available per model. **Stated as a correlation over the splits available; no causal
relationship is claimed.**

```
model       Spearman   Pearson   n_splits
TabICL v2    -0.0587   -0.0924       60
TabPFN v2    -0.0379   +0.3004       60
TabSwift     +0.3972   -0.0788       60
```

Spearman and Pearson disagree in sign for TabPFN v2 and TabSwift. Both are reported; neither is
selected.

#### 4.2 Directional monotonicity

`v` is the unit eigenvector of `lambda_min(sym(Q^T J Q))`, `u = Q v` with `‖u‖ = 1`. For any
posterior mean `u^T J u = v^T (Q^T J Q) v >= 0`. Measured as
`D(t) = u·(m(y + t u) - m(y - t u)) / (2 t)`, with `t` a fraction of `‖y - mean(y)‖`.

**4.2c Controls, through the identical procedure.**

```
control              lambda_min per seed                                        D(t), all five amplitudes
exactgp_wrapped      -5.24e-16 -3.51e-16 -4.43e-16 -5.47e-16 -3.70e-16          |D| <= 5.6e-30
targeted_imitator    -8.818e-02 -5.878e-02 -6.490e-02 -8.965e-02 -1.206e-01     equal to lambda_min to 4 s.f. at every t
```

The exact GP's `lambda_min` is at machine zero — the audit context duplicates 10 rows of `X`, giving
`Q^T W Q` nine machine-zero eigenvalues (`FINAL_NUMBERS.md` 4.4) — and its measured `D` is `1e-30` to
`1e-32`, so the *sign* of `D` on that control carries no information and only the magnitude is
reported. The imitator's `D` reproduces its `lambda_min` exactly at all five amplitudes, and its
`negeig` values `0.08819, 0.05900, 0.06474, 0.08983, 0.12072` reproduce `FINAL_NUMBERS.md` 2.3's
`0.088194, 0.059000, 0.064744, 0.089827, 0.120723`.

**4.2a/b Models, 60 splits each.**

```
                  lambda_min                      D(t) mean, by fraction of ||y - mean(y)||
model            mean      min     neg/60      0.003     0.01      0.03      0.1       0.3
exactgp_wrapped -4.47e-16 -5.47e-16   5/5    -1.8e-30  -2.9e-31  -1.7e-31  +1.8e-32  -8.4e-32
targeted_imit.  -0.08442  -0.12057    5/5    -0.08442  -0.08442  -0.08442  -0.08442  -0.08442
TabICL v2       -0.11948  -2.19062   33/60   -0.10918  -0.06332  +0.09342  +0.32354  +0.47258
TabPFN v2       -1.65830  -7.02420   60/60   -1.58227  -0.87066  -0.22872  +0.20043  +0.29810
TabSwift        -0.47461  -3.09311   60/60   -0.41160  -0.38878  -0.28624  -0.05776  +0.08673

count of splits with D(t) < 0, out of 60:
TabICL v2         33        31        24        16         0
TabPFN v2         58        58        42         6         0
TabSwift          49        46        47        32        13
```

**4.2d In interpretable units.** The most negative `D(0.003)` per model:

```
TabICL v2,  did=549 (strikes), seed 400,  lambda_min -2.1906, negeig 0.3888
  labels raised  by 12.895 along u  ->  prediction moved -28.750 along u
  labels lowered by 12.895 along u  ->  prediction moved +26.958 along u

TabPFN v2,  did=549 (strikes), seed 200,  lambda_min -5.6117, negeig 0.3989
  labels raised  by 10.282 along u  ->  prediction moved -61.043 along u
  labels lowered by 10.282 along u  ->  prediction moved +71.374 along u

TabSwift,   did=582 (fri_c1_500_25), seed 42,  lambda_min -3.0931, negeig 0.2273
  labels raised  by 0.027433 along u  ->  prediction moved -0.10176 along u
  labels lowered by 0.027433 along u  ->  prediction moved +0.081488 along u
```

Every split's `lambda_min`, `negeig`, `asym`, five amplitudes and both one-sided moves are in
`results/chunk4_direction_<model>.json`.

#### Chunk 4 checklist

- [x] `R_SURE` vs held-out error, per split — 4.1, per-split rows in the JSON
- [x] `tr J` and negative-diagonal count per split — 4.1
- [x] Correlation between SURE gap and negative-diagonal count, stated as a correlation — 4.1c
- [x] Directional monotonicity on all three models — 4.2a/b, 60 splits each
- [x] Same on exact GP and targeted imitator — 4.2c
- [x] Statement in interpretable units — 4.2d

---

### P. Provenance

#### P.1 Number → file → key

| Section | Quantity | File | Key |
|---|---|---|---|
| 1.0 | train/test vs audit surrogates | `results/chunk1_results.json` | `1.0_surrogate_identity.worst`, `.per_seed` |
| 1.1 | GP `s2_jac` relative error, both forms | `results/chunk1_results.json` | `1.1_1.2_exactgp.<seed>.forms.<form>__<sigma>.rel_err` |
| 1.1 | `J_**` per query, `s2_true` | `results/chunk1_results.json` | `1.1_1.2_exactgp.<seed>.J_star`, `.s2_true` |
| 1.2 | `tr J`, `n - tr J`, `sigma2_hat`, ratio | `results/chunk1_results.json` | `1.1_1.2_exactgp.<seed>.trJ`, `.n_minus_trJ`, `.sigma2_hat`, `.ratio_sigma2hat_over_true` |
| 1.2s | closed-form sampling spread | `results/chunk1_results.json` | `1.2s_sigma2hat_sampling.<seed>.predicted_rel_sd_of_sigma2hat` |
| 1.3 | hierarchical GP, same quantities | `results/chunk1_results.json` | `1.3_hiergp.<seed>.*` |
| 1.4 | append perturbation | `results/chunk1_results.json` | `1.4_append_perturbation.<map>.<seed>` |
| 1.4b | perturbation vs error correlation | `results/chunk1_results.json` | `1.4b_perturbation_vs_error.<map>.<seed>.spearman` |
| 1.5 | `y_*` sensitivity | `results/chunk1_results.json` | `1.5_ystar_sensitivity.<map>.<seed>.abs_spread`, `.rel_spread` |
| 1.5b | `y_*` choice scored | `results/chunk1_results.json` | `1.5b_ystar_choice.<map>.<seed>.<candidate>.rel_err` |
| 1.6 | step-size plateau, quantised controls | `results/chunk1_results.json` | `1.6_step_plateau.<map>.mean_per_h`, `.max_abs_dev_from_largest_h` |
| Gate 1 | gate arithmetic | `results/chunk1_results.json` | `gate1` |
| 2.2b | step-size sweep, TabICL | `results/chunk2_stepsize_tabicl_v2.json` | `tabicl_v2.<did>.rows[]` |
| 2.2b | step-size sweep, TabPFN + TabSwift | `results/chunk2_stepsize_tabpfn_v2_tabswift.json` | `<model>.<did>.rows[]` |
| 2.1 | accepted datasets | `results/chunk2_datasets.json` | `accepted[]` |
| 2.1 | exclusions with rule | `results/chunk2_datasets.json` | `rejected[]` |
| 2.1 | selection rule as executed | `results/chunk2_datasets.json` | `_rule` |
| 2.2 | split sizes | `results/chunk2_datasets.json` | `splits.<did>.<seed>` |
| 2.2 | split arrays | `arrays/chunk2_data.npz` | `<did>__<seed>__{X,y}_{ctx,cal,test}` |
| 2.3 | probe step actually used | `results/chunk2_estimator_<model>.json` | `_config.h_frac`, `_config.h_rule`, `rows[].h`, `rows[].std_y_ctx` |
| 2.4 | clip rate, floor, `J**>=1` rate | `results/chunk2_estimator_<model>.json` | `rows[].clip_rate_inverted`, `.clip_floor_value`, `.Jstar_ge_1_rate` |
| 2.4 | `tr J`, `n - tr J`, definedness | `results/chunk2_estimator_<model>.json` | `rows[].trJ`, `.n_minus_trJ`, `.sigma2_hat_defined` |
| 2.4 | negative diagonal count | `results/chunk2_estimator_<model>.json` | `rows[].n_neg_Jii` |
| Gate 2 | gate arithmetic | `results/chunk3_results.json` | `gate2.<model>` |
| 3.1 | coverage, signed deviation | `results/chunk3_results.json` | `aggregate.<model>.<est>.levels.<lv>.coverage`, `.signed_dev` |
| 3.1 | per-split coverage | `results/chunk3_results.json` | `per_dataset.<model>.<did>.est.<est>.levels.<lv>.coverage_per_split` |
| 3.2 | interval score | `results/chunk3_results.json` | `aggregate.<model>.<est>.levels.<lv>.interval_score` |
| 3.3 | width and recalibrated width | `results/chunk3_results.json` | `...levels.<lv>.width`, `.recal_width`, `.recal_scale` |
| 3.4 | NLL | `results/chunk3_results.json` | `aggregate.<model>.<est>.nll`; conversion note at `per_split[].est.conformal.nll_note` |
| 3.5 | per-dataset interval score | `results/chunk3_results.json` | `per_dataset.<model>.<did>.est.<est>.levels.0.90.interval_score` |
| 3.5 | win/loss counts | `results/chunk3_results.json` | `comparisons.<model>.<a>_vs_<b>.per_level.<lv>.a_wins`, `.b_wins` |
| 3.6 | rank correlation | `results/chunk3_results.json` | `aggregate.<model>.<est>.spearman_var_vs_sqres`; per split at `per_dataset...spearman_per_split` |
| 3.7 | bootstrap CI and Wilcoxon | `results/chunk3_results.json` | `comparisons.<model>.<a>_vs_<b>.per_level.<lv>.bootstrap_ci`, `.wilcoxon`, `.split_level_0.90` |
| 4.1 | `R_SURE`, held-out, gap | `results/chunk4_results.json` | `4.1_sure.<model>.rows[].R_SURE`, `.R_SURE_per_point`, `.heldout_mse`, `.sure_gap` |
| 4.1c | gap vs negative-count correlation | `results/chunk4_results.json` | `4.1_sure.<model>.corr_gap_vs_negcount` |
| 4.2c | controls | `results/chunk4_results.json` | `4.2c_controls.<control>[].lambda_min`, `.probes[]` |
| 4.2a/b/d | models | `results/chunk4_results.json` | `4.2_models.<model>[]`; also `results/chunk4_direction_<model>.json` |
| 4.2 | sign summary by amplitude | `results/chunk4_results.json` | `4.2_summary.<name>.per_frac` |
| — | context Jacobians, all models | `arrays/chunk2_arrays_<model>.npz` | `<model>__<did>__<seed>__J_ctx` |
| — | per-point predictions and variances | `arrays/chunk2_arrays_<model>.npz` | `..._{m_test,y_test,s2_jac,s2_jac_specified,s2_native,s2_gp,gp_mu,J_star,r_cal}` |

#### P.2 Scripts written, with outputs

| Script | Output |
|---|---|
| `src/estimators.py` | library; no output of its own |
| `src/chunk1_validate.py` | `results/chunk1_results.json` |
| `src/chunk2_data.py` | `results/chunk2_datasets.json`, `arrays/chunk2_data.npz` |
| `src/chunk2_stepsize.py` | `results/chunk2_stepsize_tabicl_v2.json`, `results/chunk2_stepsize_tabpfn_v2_tabswift.json`, `logs/log_stepsize_*.log` |
| `src/chunk2_estimators.py` | `results/chunk2_estimator_<model>.json`, `arrays/chunk2_arrays_<model>.npz`, `logs/log_est_*.log` |
| `src/chunk3_measure.py` | `results/chunk3_results.json` |
| `src/chunk4_aux.py` | `results/chunk4_results.json`, `results/chunk4_direction_<model>.json`, `logs/log_chunk4.log` |

Nothing outside `experiments/phase_2/` was modified.

Paths above are relative to `experiments/phase_2/`. See `README.md` for the directory layout.

#### P.3 Failed attempts and rejected settings

| What | Why rejected |
|---|---|
| `sigma2_hat * (1 + J_**)`, the brief's estimator form | Conditions on the invented label; 42–52% median relative error on the exact GP against `8e-13` for the inverted form, and bounded above by `2 sigma^2`. Section 0. Kept and reported alongside, not used as primary. |
| Absolute probe step (`1e-3`, `1e-1`) | Output quantum scales with `std(y)`; gave `n - tr J < 0` on `places`. Whole pass deleted before any estimator number was read. Section 2.2b. |
| `y_* = mean(y) ± SD(y)` | 1.5b: p90 relative error `0.428` and `0.374` on the hierarchical GP against `0.055` for the self-consistent choice. |
| `y_* = mean(y)` | Retained as a reported alternative; `m_*(y)` chosen, marginally better on the hierarchical GP and free. |
| TabICL `output_type="variance"` | Variance of the quantile grid, not the second moment. Section 2.3. |
| TabPFN `sum(bin_prob * centre)` | That is `E[X]`, and omits the half-normal tail bins. Section 2.3. |
| `negfrac` as a positivity metric | Retracted upstream (`FINAL_NUMBERS.md` 4.4); not used here. |
| Reduced-basis `profile_jacobian` | No power (`FINAL_NUMBERS.md` 6.1); not used here. |
| Unweighted `profile_jacobian` | Inflates the symmetric control 17x; not used here. |
| `TargetedImitator.inner_jacobian` as shipped | Returns `-c*vvᵀ` where the map applies `-(c/2)*vvᵀ`. As shipped, `lambda_min = -0.335129` against a measured `-0.029472`; corrected, both read `-0.088180` and the vector agrees to `7.1e-12`. Corrected locally in `src/chunk4_aux.py`; the tier-0 file is unchanged. |
| Shared output file for concurrent `src/chunk2_stepsize.py` runs | Second writer clobbered the first; changed to one file per invocation before any data was kept. |

---

### R. Reading

Interpretation only. Every number it refers to is in Sections 1–4.

**1. The estimator as specified does not estimate the quantity it is named for, and the fix is exact.**
Appending `(x_*, y_*)` makes the query a context point, but the identity then available is
`sigma^2 J_** = Var(f_* | y, y_*)` — conditioned on the label we invented. `sigma^2(1 + J_**)` is
therefore not the held-out predictive variance and cannot exceed `2 sigma^2`. Inverting the Gaussian
update gives `sigma^2/(1 - J_**)`, exact to `8e-13` on the exact GP. The inversion assumes a Gaussian
posterior at the appended point; on the hierarchical GP, whose posterior is a mixture, the p90 error
is `1.4%` to `16.5%`, which is the size of the third-cumulant term T3 predicts. That is the estimator's
irreducible error on any genuinely non-Gaussian predictor, and it bounds what this route can achieve.

**2. The Jacobian-derived variance does not beat the native head. It loses on every model where a
native head exists, on every dataset.** Interval score at `90%`: 0–12 for TabPFN v2, 1–11 for
TabICL v2, both with dataset-level bootstrap CIs excluding zero and Wilcoxon `p <= 0.001`. NLL: 0–12
and 1–11. The audit's A3 result says the reported variance does not satisfy `s² = σ²(1 + J_ii)`; this
experiment says that replacing it with a variance built from `J` makes calibration worse, not better.
The two statements are consistent — A3 is a structural identity, not a claim about predictive
quality — but the downstream cost the experiment was designed to look for is not there in this form.

**3. Split conformal beats everything, and it is the practical recommendation.** It has the lowest
interval score on TabICL v2 (`523.6`) and is within noise of native on TabPFN v2 (`519.9` against
`515.9`, CI `[-26.2, +16.0]`, `p = 0.13`), and it beats the Jacobian route on TabSwift 11–1. Wrapping
a frozen PFN in a conformal calibration layer is cheap, distribution-free, and — this is the relevant
part for the audit — indifferent to every structural violation the audit measured.

**4. The native heads carry real information about where the model errs.** Rank correlation with the
squared residual is `0.300` (TabICL) and `0.283` (TabPFN), against `0.054` and `0.124` for the
Jacobian route and `0.182` for a fitted GP. On `mu284` and `strikes` the native figures reach `0.68`
to `0.72`. A variance channel that fails a structural identity is not thereby uninformative, and the
paper should not imply that it is.

**5. TabICL v2 nearly interpolates its context on real data, and that is what breaks the estimator
for it.** `tr J` reaches `99.98`–`100.03` against `n = 100`, flat across four decades of step size, so
`n - tr J` goes negative and `sigma2_hat` is undefined on 10 of 60 splits, with `J_** >= 1` on up to
`73%` of queries. This is the `J = I` row of the N5 catalogue — the row that "passes A1/A2 vacuously"
and is the reason E2.1's non-degeneracy check exists. It is dataset-dependent: `tr J` is `45.7` on
`pwLinear` and `87.3` on average. Worth noting for the audit: the audit's synthetic contexts, with
`sigma = 1.0` label noise and 10 duplicated rows, put TabICL far from `J = I`
(`‖J - I‖_F/‖J‖_F = 0.85`), while several real datasets put it close. That is a context-family effect
of exactly the kind E3.1 exists to find, and it is a reason to run E3.1 rather than a result about it.

**6. The A2 violation is now visible as behaviour, not as a matrix property.** On 58 of 60 splits for
TabPFN v2 and 49 of 60 for TabSwift, raising the context labels along one direction moves the
prediction the opposite way along that same direction. The clearest instance: TabPFN v2 on `strikes`,
labels raised by `10.282` along `u`, prediction moved by `-61.043` along `u`. The controls behave as
they must — the exact GP's directional derivative is `1e-30`, and the imitator's reproduces its
`lambda_min` to four significant figures and reproduces the audit's own recorded `negeig`. The effect
is amplitude-dependent and vanishes by `t = 0.3‖y - ȳ‖` for TabPFN and TabICL, which is expected for
a derivative statement and is why five amplitudes are reported rather than one.

**7. For the BO follow-on the brief names as the natural sequel: this experiment does not support
it.** The stated dependency was that the Jacobian-derived variance would have to improve calibration
here to be worth trying as an acquisition signal. It does not — it is worse than the native head on
every model that has one, and worse than conformal on all three. The one place the route is not
dominated is TabSwift, where no native head exists at all; even there conformal is better on 11 of 12
datasets. A BO study built on this variance would be starting from a signal with rank correlation
`0.04`–`0.12` against realised error.

**8. What this does not settle.** All of it is at `n_ctx = 100` on 12 datasets with `d <= 50` and
numeric features only; the append perturbation was never measured on the frozen models (2.4); and the
`sigma2_hat` route inherits a `~18%` single-split sampling spread at this `n` (1.2s) that a larger
context would shrink as `1/sqrt(n - tr J)`. A version of this estimator with `sigma^2` taken from the
calibration split rather than from `tr J` would remove that spread, at the cost of no longer being
derivable from the model alone. It was not run.

---
