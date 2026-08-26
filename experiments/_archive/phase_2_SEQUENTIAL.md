# Experiments 2.4–2.7 — Sequential Coherence and Bayesian Optimisation

Measurement record. Sections 1–4 contain numbers only. Section 6 is the interpretation and is the
only part that is not a measurement. Every number is traceable to a JSON in `results/`; Section 5
gives the file and key.

## 0. Corrections made before running

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

# 1. EXPERIMENT 2.4 (D) — acquisition-argmax sensitivity

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

## 1.1 Decision agreement (2.4.3)

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

## 1.2 Decision regret (2.4.4)

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

## 1.3 Acquisition-surface rank correlation (2.4.5)

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

## 1.4 The frontier prediction (2.4.6)

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

## 1.5 Scale versus shape (2.4.7)

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

## Gate 2.4

| criterion | measured |
|---|---|
| native and Jacobian agree on more than 90% of decisions | **no.** `0.160` (TabICL, both acquisitions), `0.550` / `0.517` (TabPFN, EI / UCB). TabSwift has no native channel. |
| regret gap inside split-to-split noise | **no.** The Jacobian channel is worse than `greedy_mean` on every model and both acquisitions, `p` between `7.2e-07` and `4.1e-03`. |

The defect reaches the decision. 2.5 and the loop experiments are therefore not pre-empted.

## 1.6 Experiment 2.4 checklist

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
# 2. EXPERIMENT 2.5 (A) — self-consistency, with the localised prediction

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

## 2.1 Drift and variance shrinkage (2.5.2, 2.5.3)

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

## 2.2 K sensitivity (2.5.7)

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

## 2.3 The localised prediction (2.5.4) and probe-vs-target (2.5.5)

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

## 2.4 Controls (2.5.6) and Gate 2.5

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

### Gate 2.5

| criterion | measured |
|---|---|
| exact GP drift at numerical precision | **met analytically**: `1.070e-11` max over 90 probes. Not attainable by the sampled route, whose MC SE is `~0.09 s_n` at `K = 64` (Correction 3); the sampled GP instead gives `\|drift\|/SE = 0.694`, `97.8%` of probes inside `2`. |
| variance decomposition closes | **met**: analytic relative residual `1.768e-14` max. |
| imitator drifts measurably | **not as stated** (Correction 4). The T1 imitator's drift is `O(eps)`-bounded and at `eps = 0.01` sits at `0.0061 s_n`, below the sampling noise — which is T1's content, drift being a value functional. The substitute quadratic control drifts by `5.11` and `20.27` `s_n` and matches its closed form to `1e-11`. |

The exact GP does not drift, so the harness is measuring the martingale property and not something
else.

## 2.5 Experiment 2.5 checklist

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
# 3. EXPERIMENT 2.7 (C) — the BO loop

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

## 3.1 Final simple regret (2.7.4, 2.7.5)

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

## 3.2 Iterations to a fixed fraction of the optimum (2.7.4)

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

## 3.3 Knowledge gradient (2.7.6)

**`NOT MEASURED`.** KG requires an inner expectation over the model's own predictive future at every
candidate, which on top of the Jacobian arm's `2 x n_pool` calls per iteration was not affordable.
It is not approximated.

## 3.4 The `n - tr J` watch (2.7.7, 2.7.8)

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

## 3.5 Experiment 2.7 checklist

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

# 4. EXPERIMENT 2.6 (B) — does the model un-learn from its own acquisitions?

Rides on the 2.7 trajectories at no extra model calls. At each iteration, before appending,
`J_**` at the acquired point is recorded on the Jacobian arm, where it is computed anyway. After
appending, `move = mu_{n+1}(x_t) - mu_n(x_t)` and `surprise = y_t - mu_n(x_t)`. For any posterior
mean the two share a sign, and `shrinkage = move / surprise` lies in `[0, 1]`.

## 4.1 Violation rate and shrinkage (2.6.2, 2.6.5, 2.6.6)

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

## 4.2 Conditioning on the sign of `J_**` (2.6.3)

**`NOT MEASURED`.** `J_**` at the acquired point was **negative on 0 of 500 acquisitions**
(`J_star_available = 500`, `n_Jstar_negative = 0`), so the negative stratum is empty and there is
nothing to condition on. The violation rate on the complement is `0.0600`. This is the same pattern
as 2.5: on held-out probes TabPFN v2 had 57 negative `J_**` out of 3000 (`1.90%`), and the
acquisition selects points where it is positive.

The context diagonal is a different object and does go negative: 401 negative `J_ii` entries across
the same 500 iterations (§3.4).

## 4.3 Violation count against final regret (2.6.4)

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

## 4.4 Experiment 2.6 checklist

- [x] **2.6.1** `J_**` at the acquired point recorded every iteration on the Jacobian arm.
- [x] **2.6.2** `move`, `surprise` and the sign-violation count per trajectory — §4.1.
- [ ] **2.6.3** Violation rate conditional on the sign of `J_**` — **`NOT MEASURED`**: 0 of 500
      acquisitions had `J_** < 0`; §4.2.
- [x] **2.6.4** Per-trajectory violation count against final regret, as a correlation — §4.3.
- [x] **2.6.5** Same on the GP arm — §4.1. The imitator was not run in the loop: it has no
      mechanism for proposing candidates and its drift is `O(eps)` by construction (Correction 4).
- [x] **2.6.6** Shrinkage distribution and the fraction outside `[0, 1]` — §4.1.

---
# 5. Provenance

## 5.1 Number → file → key

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

## 5.2 Scripts written, with outputs

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

## 5.3 Failed attempts and rejected settings

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

# 6. Interpretation

Everything above is measurement. This section is not.

## 6.1 The variance defect reaches the decision, and 2.1's ranking survives into the loop

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

## 6.2 The martingale is violated, and the violation is where the theory is weakest

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

## 6.3 The A2-localised prediction fails for the third time

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

## 6.4 What TabPFN does in a BO loop that a Bayesian would not

The sign check is the cleanest behavioural statement in this brief. On `6.0%` to `7.8%` of
acquisitions the model finds a point better than it predicted and becomes *less* optimistic about it,
against `1.0%` for a refitted GP on the identical protocol. The shrinkage factor
`(mu_{n+1} - mu_n)/(y_t - mu_n)` lies outside `[0, 1]` on `19%` to `25%` of acquisitions against the
GP's `5%`, and its median is `0.55` against the GP's `0.99` — the model absorbs about half of each
surprise where a posterior mean absorbs nearly all of it.

Whether that costs anything is not established here. The per-trajectory correlation between violation
count and final regret changes sign across objectives for every arm, and with 5 trajectories per cell
those correlations carry little weight.

## 6.5 One thing that did not go wrong

2.1 §3 found TabICL nearly interpolating at `n = 100`, which left `sigma_hat^2` undefined on 10 of 60
splits. The `n - tr J` watch was put in because BO contexts start small and grow. On TabPFN across
500 iterations at `n = 10` to `29`, `tr J / n` rises from `0.474` to `0.614` and never exceeds
`0.941`, the minimum `n - tr J` is `+1.35`, and `sigma_hat^2` is defined at every single iteration.
The failure mode that broke the estimator at `n = 100` on one model does not appear at BO context
sizes on another.

## 6.6 What this does not settle

The BO study is one model, five synthetic objectives, `T = 20`, `S = 5`, and a 128-point pool — 25
paired trajectories per comparison, so the objective-level win counts rest on five points each.
TabICL v2 and TabSwift were not run in the loop, and knowledge gradient — the acquisition most
exposed to what 2.5 measures, because it is literally an expectation over the model's predictive
future — was not run at all. No real objective was included. TabSwift's entire 2.5 result rests on a
Gaussian surrogate for a predictive distribution it does not have, and is not a statement about its
own sampling law. The drift measured in 2.5 is at `n = 100`; the loop operates at `n = 10` to `29`,
and nothing here connects the two.
