"""EXPERIMENT 2.5 controls (2.5.6), and two corrections to Gate 2.5.

CORRECTION A -- "the exact GP's drift must be at numerical precision" is not
attainable by a SAMPLED test. Drift is estimated from K draws, so its standard
error is Var(mu_{n+1})^{1/2}/sqrt(K), about 0.09 in units of s_n at K = 64. No
amount of correctness makes a Monte Carlo estimate land at 1e-16. The gate IS
attainable analytically and is checked that way: for a GP mu_{n+1}(x_*) = a + b y'
is affine in y', so E[mu_{n+1}] = a + b mu_n(x') in closed form, and Var_{n+1} is
y-independent, so Var_n = Var_{n+1} + b^2 Var_n(x') is exact. The sampled exact GP
is reported beside it, where the correct statement is |drift|/MC-SE ~ 1.

CORRECTION B -- "the targeted imitator must drift" contradicts T1. Mean drift
under the model's own sampling law is a VALUE functional, and T1 says no value
functional separates Bayes-realisable maps from their complement with any margin.
The T1 imitator is the construction that proves it, so it is the wrong positive
control for a drift test and is reported for that reason. A control that does
drift is supplied instead: the quadratic map m(y) = mu(y) + lam mu(y)^2, whose
drift has an exact closed form (below).

EACH MAP IS SAMPLED FROM ITS OWN PREDICTIVE. An earlier version drew y' from the
exact GP for every arm; the hierarchical GP's predictive mean at the same probe
differs from the exact GP's by 2.5 units on the first split checked, and that
alone produced |drift|/SE = 23 for a map that is Bayesian. The hierarchical GP's
predictive is a MIXTURE over lengthscales, so it is sampled by drawing a component
from the posterior weights and then a Gaussian from that component -- not by
matching moments.

THE T1 BUMP IS DIMENSION-AGNOSTIC. The context grows from n to n+1, so a fixed
direction vector a in R^n is not defined on both. The bump uses S(y) = sum(y),
i.e. a proportional to the all-ones direction, which is defined at every n and
keeps the T1 form. Then |E[bump] - bump_0| <= 2 eps holds by construction and is
reported against the measurement.

THE QUADRATIC PREDICTION IS EXACT, NOT LEADING-ORDER. For m = mu + lam mu^2,
    E[m] - m_0 = (E[mu] - mu_n) + lam (Var(mu) + E[mu]^2 - mu_n^2),
which reduces to lam Var(mu) only when the inner map is an exact martingale AND
mu_n is small. On did=509 the targets are of order 5e3, so the omitted term
dominates. The exact expression is used, making this a self-consistency check of
the harness as well as a drift control.

Writes results/exp25_controls.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC.parent.parent.parent))
warnings.filterwarnings("ignore")

from experiments.phase_2.src.paths import RESULTS, ARRAYS            # noqa: E402
from experiments.phase_2.src.estimators import (                     # noqa: E402
    GPTrainTest, HierGPTrainTest, rbf, jsonable,
)
from experiments.phase_2.src.exp25_martingale import K, K_SUB, SAMPLE_SEED, SEEDS  # noqa: E402

GP_SIGMA = 0.5
N_PROBES = 3
DIDS = [229, 509, 522, 549, 560, 581]
EPS_LIST = [0.01, 0.1, 1.0]
LAMBDAS = [0.05, 0.2]


def affine_coeffs(X, y, xp, targets):
    Xa = np.vstack([X, np.asarray(xp).reshape(1, -1)])
    Ka = rbf(Xa, Xa, 1.0)
    inv = np.linalg.inv(Ka + GP_SIGMA ** 2 * np.eye(len(Xa)))
    w = rbf(targets, Xa, 1.0) @ inv
    ga = GPTrainTest(Xa, np.append(y, 0.0), sigma=GP_SIGMA)
    return w[:, :-1] @ y, w[:, -1], ga.predictive_variance(targets)


def hier_components(hg, xq):
    """(weights, per-component predictive mean, per-component predictive var) at xq."""
    xq = np.asarray(xq, float).reshape(1, -1)
    mus, vs = [], []
    for i, ls in enumerate(hg.lengthscales):
        Ks = rbf(xq, hg.X, ls)
        kss = float(rbf(xq, xq, ls)[0, 0])
        mus.append(float(Ks @ hg.invs[i] @ hg.y))
        vs.append(kss - float(Ks @ hg.invs[i] @ Ks.T) + hg.sigma ** 2)
    return hg.w, np.asarray(mus), np.asarray(vs)


def main():
    data = np.load(ARRAYS / "chunk2_data.npz")
    rows = []
    for did in DIDS:
        for s in SEEDS:
            X = data[f"{did}__{s}__X_ctx"]; y = data[f"{did}__{s}__y_ctx"].astype(float)
            Xt = data[f"{did}__{s}__X_test"]
            rng = np.random.default_rng(SAMPLE_SEED + 1000 * s + did)
            g = GPTrainTest(X, y, sigma=GP_SIGMA)
            hg = HierGPTrainTest(X, y, sigma=GP_SIGMA)
            for pi in rng.choice(len(Xt), size=N_PROBES, replace=False):
                pi = int(pi); other = int((pi + 1) % len(Xt))
                targets = np.vstack([Xt[pi], Xt[other]])
                base = {"did": did, "seed": s, "probe": pi}
                Xa = np.vstack([X, Xt[pi]])

                # ---------------- exact GP, ANALYTIC (the gate)
                mu_n = g.predict(targets); var_n = g.predictive_variance(targets)
                mp = float(g.predict(Xt[pi:pi + 1])[0])
                vp = float(g.predictive_variance(Xt[pi:pi + 1])[0])
                a, b, var_next = affine_coeffs(X, y, Xt[pi], targets)
                rows.append({**base, "map": "exactgp_analytic",
                             "drift_over_sn": (np.abs(a + b * mp - mu_n)
                                               / np.sqrt(var_n)).tolist(),
                             "drift_rel": (np.abs((a + b * mp - mu_n) / mu_n)).tolist(),
                             "tv_rel_residual": ((var_n - (var_next + b ** 2 * vp))
                                                 / var_n).tolist()})

                # ---------------- exact GP, SAMPLED from its own predictive
                ys_gp = mp + np.sqrt(vp) * rng.standard_normal(K)
                mus_gp = np.empty((K, 2)); vrs_gp = np.empty((K, 2))
                for k in range(K):
                    gk = GPTrainTest(Xa, np.append(y, ys_gp[k]), sigma=GP_SIGMA)
                    mus_gp[k] = gk.predict(targets); vrs_gp[k] = gk.predictive_variance(targets)
                e = {**base, "map": "exactgp_sampled"}
                for tn, ti in [("self", 0), ("other", 1)]:
                    sn = float(np.sqrt(var_n[ti]))
                    for kk, KK in [("K64", K), ("K32", K_SUB)]:
                        mm = mus_gp[:KK, ti]
                        e[f"drift_{tn}_{kk}"] = float(mm.mean() - mu_n[ti]) / sn
                        e[f"drift_se_{tn}_{kk}"] = float(mm.std(ddof=1) / np.sqrt(KK)) / sn
                    se = e[f"drift_se_{tn}_K64"]
                    e[f"drift_over_se_{tn}"] = (abs(e[f"drift_{tn}_K64"]) / se
                                                if se > 0 else None)
                    lhs = float(var_n[ti])
                    rhs = float(vrs_gp[:, ti].mean() + mus_gp[:, ti].var(ddof=0))
                    e[f"tv_rel_gap_{tn}"] = (lhs - rhs) / lhs
                rows.append(e)

                # ---------------- hierarchical GP, SAMPLED FROM ITS OWN MIXTURE
                hmu_n = hg.predict(targets); hvar_n = hg.predictive_variance(targets)
                w_, cm, cv = hier_components(hg, Xt[pi])
                comp = rng.choice(len(w_), size=K, p=w_ / w_.sum())
                ys_h = cm[comp] + np.sqrt(np.maximum(cv[comp], 0.0)) * rng.standard_normal(K)
                mus_h = np.empty((K, 2)); vrs_h = np.empty((K, 2))
                for k in range(K):
                    hk = HierGPTrainTest(Xa, np.append(y, ys_h[k]), sigma=GP_SIGMA)
                    mus_h[k] = hk.predict(targets); vrs_h[k] = hk.predictive_variance(targets)
                e = {**base, "map": "hiergp_sampled",
                     "n_effective_components": float(1.0 / np.sum((w_ / w_.sum()) ** 2))}
                for tn, ti in [("self", 0), ("other", 1)]:
                    sn = float(np.sqrt(hvar_n[ti]))
                    for kk, KK in [("K64", K), ("K32", K_SUB)]:
                        mm = mus_h[:KK, ti]
                        e[f"drift_{tn}_{kk}"] = float(mm.mean() - hmu_n[ti]) / sn
                        e[f"drift_se_{tn}_{kk}"] = float(mm.std(ddof=1) / np.sqrt(KK)) / sn
                    se = e[f"drift_se_{tn}_K64"]
                    e[f"drift_over_se_{tn}"] = (abs(e[f"drift_{tn}_K64"]) / se
                                                if se > 0 else None)
                    lhs = float(hvar_n[ti])
                    rhs = float(vrs_h[:, ti].mean() + mus_h[:, ti].var(ddof=0))
                    e[f"tv_rel_gap_{tn}"] = (lhs - rhs) / lhs
                rows.append(e)

                # ---------------- T1 imitator, dimension-agnostic bump S(y) = sum(y)
                S0 = float(y.sum())
                for eps in EPS_LIST:
                    e = {**base, "map": "t1_imitator", "eps": eps}
                    bump_k = eps * np.sin((S0 + ys_gp) / eps)
                    bump_0 = eps * np.sin(S0 / eps)
                    for tn, ti in [("self", 0), ("other", 1)]:
                        sn = float(np.sqrt(var_n[ti]))
                        d = float((mus_gp[:, ti] + bump_k).mean() - (mu_n[ti] + bump_0))
                        e[f"drift_{tn}_K64"] = d / sn
                        e[f"bound_2eps_over_sn_{tn}"] = 2.0 * eps / sn
                        e[f"within_bound_{tn}"] = bool(abs(d) <= 2.0 * eps + 1e-12)
                        e[f"bump_drift_only_{tn}"] = float(bump_k.mean() - bump_0) / sn
                    rows.append(e)

                # ---------------- quadratic map, EXACT closed-form drift
                for lam in LAMBDAS:
                    e = {**base, "map": "quadratic", "lam": lam}
                    for tn, ti in [("self", 0), ("other", 1)]:
                        sn = float(np.sqrt(var_n[ti]))
                        mu = mus_gp[:, ti]
                        mq = mu + lam * mu ** 2
                        m0q = mu_n[ti] + lam * mu_n[ti] ** 2
                        d = float(mq.mean() - m0q)
                        exact = float((mu.mean() - mu_n[ti])
                                      + lam * (mu.var(ddof=0) + mu.mean() ** 2 - mu_n[ti] ** 2))
                        e[f"drift_{tn}_K64"] = d / sn
                        e[f"drift_raw_{tn}"] = d
                        e[f"predicted_exact_{tn}"] = exact
                        e[f"pred_rel_err_{tn}"] = (abs(d - exact) / abs(exact)
                                                   if exact != 0 else None)
                        se = float(mq.std(ddof=1) / np.sqrt(K)) / sn
                        e[f"drift_se_{tn}_K64"] = se
                        e[f"drift_over_se_{tn}"] = abs(d / sn) / se if se > 0 else None
                    rows.append(e)
        print(f"  did={did} done", flush=True)

    with open(RESULTS / "exp25_controls.json", "w") as f:
        json.dump(jsonable({"_config": {"K": K, "gp_sigma": GP_SIGMA, "dids": DIDS,
                                        "n_probes": N_PROBES, "eps_list": EPS_LIST,
                                        "lambdas": LAMBDAS,
                                        "sampling": "each map from its own predictive; "
                                                    "hiergp by mixture-component draw"},
                            "rows": rows}), f, indent=2)

    def sel(m, **kw):
        return [r for r in rows if r["map"] == m and all(r.get(k) == v for k, v in kw.items())]
    print("\n" + "=" * 82)
    an = sel("exactgp_analytic")
    print("exactgp ANALYTIC (the gate)")
    print(f"  |E[mu_n+1]-mu_n|/s_n      max {max(max(r['drift_over_sn']) for r in an):.3e}")
    print(f"  total-variance rel resid  max {max(max(abs(x) for x in r['tv_rel_residual']) for r in an):.3e}")
    for m in ["exactgp_sampled", "hiergp_sampled"]:
        for tn in ["self", "other"]:
            d = sel(m)
            v = np.array([r[f"drift_{tn}_K64"] for r in d])
            os_ = np.array([r[f"drift_over_se_{tn}"] for r in d
                            if r[f"drift_over_se_{tn}"] is not None])
            g_ = np.array([r[f"tv_rel_gap_{tn}"] for r in d])
            print(f"{m:<17}{tn:<6} drift/s_n mean {v.mean():+.3e}  |drift|/SE median "
                  f"{np.median(os_):.3f}  frac<2 {np.mean(os_ < 2):.3f}   tv rel gap mean {g_.mean():+.3e}")
    for eps in EPS_LIST:
        d = sel("t1_imitator", eps=eps)
        v = np.abs([r["drift_self_K64"] for r in d])
        bd = np.abs([r["bump_drift_only_self"] for r in d])
        wb = np.mean([r["within_bound_self"] for r in d])
        print(f"t1_imitator eps={eps:<5} |total drift|/s_n mean {v.mean():.4f}   "
              f"|bump-only drift|/s_n mean {bd.mean():.4f}   within 2eps bound {wb:.3f}")
    for lam in LAMBDAS:
        d = sel("quadratic", lam=lam)
        v = np.abs([r["drift_self_K64"] for r in d])
        os_ = np.array([r["drift_over_se_self"] for r in d if r["drift_over_se_self"] is not None])
        pe = np.array([r["pred_rel_err_self"] for r in d if r["pred_rel_err_self"] is not None])
        print(f"quadratic lam={lam:<5} |drift|/s_n mean {v.mean():.4f}  |drift|/SE median "
              f"{np.median(os_):.2f}  |measured-exact|/exact max {pe.max():.3e}")
    print(f"\nSaved: {RESULTS/'exp25_controls.json'}")


if __name__ == "__main__":
    main()
