# Experiments 2.2 and 2.3 — Where the Violation Lives, and Whether It Finds Bad Labels

Measurement record. Sections 1–3 contain numbers only. Section 4 is the interpretation and is the
only part that is not a measurement. Every number is traceable to a JSON in `results/`; Section P
gives the file and key.

## 0. Reconciliation and two corrections made before running

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

# 1. EXPERIMENT 2.2 — does the A2 violation localise?

**The score.** `v` = unit eigenvector of `lam_min(sym(Q^T J Q))`, `u = Q v` with `||u|| = 1`,
per-point loading `w_i = u_i^2` normalised to sum to 1. `Q = get_Q(y, seed=0)`.

**Setup.** The 12 datasets, 5 splits and 3 models of Experiment 2.1, Jacobians read from
`arrays/chunk2_arrays_<model>.npz`. No new model calls for §1.1–§1.4, §1.6, §1.7 or §2; §1.5 and
§1.8 are the only items requiring them.

## 1.1 Reproduction of the audit's localisation (2.2.1, first half)

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

## 1.2 Concentration on real data (2.2.1 second half, 2.2.2)

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

## 1.3 Stability of the high-loading set across splits (2.2.3)

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

## 1.4 Against own residual (2.2.4) and local held-out error (2.2.6)

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

## 1.5 Against leave-one-out influence (2.2.5)

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

## 1.6 Redundancy against the baselines (2.2.9)

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

## 1.7 Label-perturbation sensitivity (2.2.7)

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

## 1.8 Controls (2.2.10, 2.2.11) and Gate 2.2

### Exact GP on the audit context (2.2.10a)

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

### Imitator (2.2.11), and the power check

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

### Gate 2.2

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

## 1.9 Experiment 2.2 checklist

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

# 2. EXPERIMENT 2.3 — does the Jacobian find corrupted labels?

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

## 2.1 Detection AUC (2.3.4)

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

## 2.2 By scheme and by rate

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

## 2.3 Per dataset, and the asymmetry score against residual

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

## 2.4 Actual leave-one-out (2.3.7)

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

## 2.5 Controls and Gate 2.3

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

### Gate 2.3

| criterion | measured |
|---|---|
| exact GP at chance on the asymmetry score | `0.4854`, and identical across all three schemes |
| exact GP above chance on the column norm | **not attainable.** `0.4966`, identical across all three schemes, because `J` is label-blind (Correction 1). The harness check is carried by the exact GP's residual (`0.5827`) and Cook's distance (`0.6810`), both above chance. |
| uncorrupted baseline shows no spurious concentration | all ten scores in `[0.4856, 0.5184]` across the three models |

---
## 2.6 Experiment 2.3 checklist

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

# 3. Provenance

## 3.1 Number → file → key

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

## 3.2 Scripts written, with outputs

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

## 3.3 Failed attempts and rejected settings

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
# 4. Interpretation

Everything above is measurement. This section is not.

## 4.1 Experiment 2.2 returns a null, and it is the instrument's answer, not its failure

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

## 4.2 Experiment 2.3 produces a real, control-validated signal — and residual is still the better tool

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

## 4.3 A1 and A2 are not interchangeable diagnostics

The model where the asymmetry score works best is TabICL, which has the *weakest* A2 violation of the
three — `lam_min > 0` on 27 of its 60 real-data splits, meaning no negative eigenvalue at all. The
model with the most negative diagonal entries, TabSwift, is the one where the asymmetry score fails
(`0.525`). Whatever the row-column gap is tracking, it is not the same object the negative
eigendirection tracks, and the audit's habit of reporting A1 and A2 side by side as two readings of
one violation is not supported by their downstream behaviour here.

## 4.4 What follows

The brief makes candidates 3 and 4 conditional on these two producing something. On the measurements
above, 2.2 does not and 2.3 does, in a narrow and well-controlled way. Anything built on this should
be built on the asymmetry channel, not on the eigendirection loading, and should be posed against
residual magnitude as the incumbent from the start — 2.1's lesson repeated, since the one place a
Jacobian score was competitive here is the one place the comparison was run against a baseline that
was itself weak.

## 4.5 What this does not settle

Experiment 2.3 ran on 4 of the 12 datasets, so its dataset-level generalisation rests on 4 points per
model, and its per-dataset spread is wide — the asymmetry score ranges from `0.601` to `0.794` on
TabICL alone. Item 2.3.7's actual-LOO baseline was run at the 10% rate only. The corruption schemes
are synthetic and known; nothing here speaks to naturally occurring label noise, whose structure is
not a flip, a shuffle, or additive Gaussian noise. All of it is at `n_ctx = 100`, where `tr J` and
the spectrum both behave differently from larger contexts.
