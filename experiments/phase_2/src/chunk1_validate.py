"""CHUNK 1 -- validate the appended-query variance estimator on maps with known answers.

No frozen models in this chunk, per the brief.

Contexts: generate_audit_context(n=100, d=5, sigma=0.5, seed=s), seeds
[42, 100, 200, 300, 400]. sigma=0.5 rather than the audit's 1.0 so the draw is
WELL SPECIFIED for ExactGP(sigma=0.5, lengthscale=1.0) -- item 1.2 asks whether
sigma2_hat recovers the true sigma^2, which is only a meaningful question when
the data actually come from the model being fitted. The design matrix is the
audit's, duplicated rows included.

Queries: 50 fresh points per seed, drawn N(0, I_5), the same input distribution
the context is drawn from, with a disjoint seed.

Writes chunk1_results.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC.parent.parent.parent))                     # repo root

from experiments.phase_2.src.paths import RESULTS                     # noqa: E402
from experiments.core.context import generate_audit_context           # noqa: E402
from experiments.phase_2.src.estimators import (                      # noqa: E402
    GPTrainTest, HierGPTrainTest, check_matches_audit_surrogate,
    context_jacobian, jacobian_star_batch, s2_from_Jstar, sigma2_hat, jsonable,
)

SEEDS = [42, 100, 200, 300, 400]
N_CTX = 100
D = 5
GP_SIGMA = 0.5                 # noise SD; matches core.surrogates default use
N_QUERY = 50
H_DEFAULT = 1e-4               # analytic maps are float64 and noiseless

#: measured output quanta, FINAL_NUMBERS.md Section 3.4. Used in 1.6 to predict
#: the resolvable step for each frozen model WITHOUT loading one.
MODEL_QUANTA = {
    "tabicl_v2": 9.8348e-07,
    "tabswift": 4.2725e-04,
    "tabpfn_v2": 6.8700e-04,
}


def wrap_traintest(inner):
    """Affine context-statistic normaliser for a train/test map.

    Identical in form to tier0_instrument/chunk1_control_validation.py::wrap,
    lifted to the train/test interface: standardise y_train, run the inner map,
    undo the transform on the output.
    """
    def predict(X_train, y_train, X_test):
        y_train = np.asarray(y_train, float).ravel()
        mu, sd = np.mean(y_train), np.std(y_train) + 1e-8
        return mu + sd * np.asarray(inner(X_train, (y_train - mu) / sd, X_test)).ravel()
    return predict


def quantize_traintest(predict, delta):
    def q(X_train, y_train, X_test):
        return np.round(np.asarray(predict(X_train, y_train, X_test)).ravel() / delta) * delta
    return q


def gp_predict(X_train, y_train, X_test):
    return GPTrainTest(X_train, y_train, sigma=GP_SIGMA).predict(X_test)


def hiergp_predict(X_train, y_train, X_test):
    return HierGPTrainTest(X_train, y_train, sigma=GP_SIGMA).predict(X_test)


def make_case(seed):
    X, y, _ = generate_audit_context(n=N_CTX, d=D, sigma=GP_SIGMA, seed=seed)
    Xq = np.random.RandomState(seed + 1000).randn(N_QUERY, D)
    return X, y, Xq


def pct(a, q):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")


def describe(a):
    a = np.asarray(a, float)
    fin = a[np.isfinite(a)]
    return {"values": a.tolist(),
            "mean": float(fin.mean()) if fin.size else float("nan"),
            "median": float(np.median(fin)) if fin.size else float("nan"),
            "p90": pct(fin, 90), "p95": pct(fin, 95),
            "min": float(fin.min()) if fin.size else float("nan"),
            "max": float(fin.max()) if fin.size else float("nan"),
            "n_finite": int(fin.size), "n_total": int(a.size)}


# --------------------------------------------------------------------------- 1.0
def item_1_0():
    """The train/test surrogates must be the audit's surrogates, in-sample."""
    print("=" * 78)
    print("1.0  train/test surrogates vs core/surrogates.py, evaluated in-sample")
    print("=" * 78)
    out = {}
    for s in SEEDS:
        X, y, _ = make_case(s)
        out[s] = check_matches_audit_surrogate(X, y, sigma=GP_SIGMA, lengthscale=1.0)
    worst = {k: max(out[s][k] for s in SEEDS) for k in out[SEEDS[0]]}
    for k, v in worst.items():
        print(f"    worst |deviation|  {k:<16} {v:.6e}")
    return {"per_seed": out, "worst": worst}


# --------------------------------------------------------------------------- 1.1 / 1.2 / 1.3
def item_1_1_1_2_1_3(kind):
    """s2_jac vs closed form, and sigma2_hat, for one analytic map."""
    predict = gp_predict if kind == "exactgp" else hiergp_predict
    ctor = GPTrainTest if kind == "exactgp" else HierGPTrainTest
    print("\n" + "=" * 78)
    print(f"1.1/1.2 [{kind}]  s2_jac vs closed form; sigma2_hat vs true sigma^2"
          if kind == "exactgp" else f"1.3 [{kind}]  same comparison")
    print("=" * 78)

    per_seed = {}
    for s in SEEDS:
        t0 = time.time()
        X, y, Xq = make_case(s)
        gp = ctor(X, y, sigma=GP_SIGMA)

        # ---- closed form target: predictive variance at a FRESH observation
        s2_true = gp.predictive_variance(Xq)

        # ---- context Jacobian -> tr J -> sigma2_hat
        J_ctx = context_jacobian(predict, X, y, H_DEFAULT)
        trJ = float(np.trace(J_ctx))
        m_ctx = np.asarray(predict(X, y, X)).ravel()
        s2h, dof, defined = sigma2_hat(m_ctx, y, trJ)

        # ---- appended-query Jacobian diagonal
        J_star, y_star = jacobian_star_batch(predict, X, y, Xq,
                                             lambda yy: float(np.mean(yy)), H_DEFAULT)

        rows = {}
        for tag, sig2 in [("true_sigma2", GP_SIGMA ** 2), ("sigma2_hat", s2h)]:
            for form in ("specified", "inverted"):
                s2, clip, raw = s2_from_Jstar(sig2, J_star, form=form)
                rel = np.abs(s2 - s2_true) / s2_true
                rows[f"{form}__{tag}"] = {"rel_err": describe(rel),
                                          "clip_rate": float(np.mean(clip)),
                                          "s2": s2.tolist()}

        per_seed[s] = {
            "trJ": trJ, "n_minus_trJ": dof, "sigma2_hat": s2h,
            "sigma2_hat_defined": defined,
            "sigma2_true": GP_SIGMA ** 2,
            "ratio_sigma2hat_over_true": (s2h / GP_SIGMA ** 2) if defined else float("nan"),
            "residual_sq_norm": float(np.sum((m_ctx - y) ** 2)),
            "J_star": J_star.tolist(), "y_star": float(y_star[0]),
            "s2_true": s2_true.tolist(),
            "J_star_range": [float(J_star.min()), float(J_star.max())],
            "forms": rows,
            "seconds": time.time() - t0,
        }
        r = per_seed[s]
        print(f"  seed {s:<5} trJ {trJ:8.4f}  n-trJ {dof:8.4f}  "
              f"sigma2_hat {s2h:.6f}  ratio {r['ratio_sigma2hat_over_true']:.4f}  "
              f"J_** in [{J_star.min():+.4f}, {J_star.max():+.4f}]")
        for form in ("specified", "inverted"):
            d = rows[f"{form}__true_sigma2"]["rel_err"]
            print(f"          {form:<10} rel err (true sigma^2)  "
                  f"median {d['median']:.3e}  p90 {d['p90']:.3e}  max {d['max']:.3e}")
        for form in ("specified", "inverted"):
            d = rows[f"{form}__sigma2_hat"]["rel_err"]
            print(f"          {form:<10} rel err (sigma2_hat)    "
                  f"median {d['median']:.3e}  p90 {d['p90']:.3e}  max {d['max']:.3e}")
    return per_seed


# --------------------------------------------------------------------------- 1.4
def item_1_4():
    """Does appending a row move the predictions at the original context points?"""
    print("\n" + "=" * 78)
    print("1.4  append perturbation  ||m_appended[:n] - m_original|| / ||m_original||")
    print("=" * 78)
    out = {}
    for kind, predict in [("exactgp", gp_predict), ("hiergp", hiergp_predict)]:
        per_seed = {}
        for s in SEEDS:
            X, y, Xq = make_case(s)
            m0 = np.asarray(predict(X, y, X)).ravel()
            n0 = np.linalg.norm(m0)
            ys = float(np.mean(y))
            rels = []
            for q in range(len(Xq)):
                Xa = np.vstack([X, Xq[q]])
                ya = np.append(y, ys)
                ma = np.asarray(predict(Xa, ya, X)).ravel()
                rels.append(float(np.linalg.norm(ma - m0) / n0))
            per_seed[s] = describe(rels)
            print(f"  {kind:<8} seed {s:<5} median {per_seed[s]['median']:.6e}  "
                  f"p90 {per_seed[s]['p90']:.6e}  max {per_seed[s]['max']:.6e}")
        out[kind] = per_seed
    return out


# --------------------------------------------------------------------------- 1.5
def item_1_5():
    """Stability of J_** across three hypothetical labels y_*."""
    print("\n" + "=" * 78)
    print("1.5  J_** sensitivity to the hypothetical label y_*  (mean, mean +/- 1 SD)")
    print("=" * 78)
    out = {}
    for kind, predict in [("exactgp", gp_predict), ("hiergp", hiergp_predict)]:
        per_seed = {}
        for s in SEEDS:
            X, y, Xq = make_case(s)
            mu, sd = float(np.mean(y)), float(np.std(y))
            vals = {"mean": mu, "mean_plus_sd": mu + sd, "mean_minus_sd": mu - sd}
            Js = {}
            for name, ys in vals.items():
                J, _ = jacobian_star_batch(predict, X, y, Xq, ys, H_DEFAULT)
                Js[name] = J
            stack = np.vstack([Js[k] for k in vals])
            spread = stack.max(axis=0) - stack.min(axis=0)
            rel_spread = spread / np.maximum(np.abs(stack.mean(axis=0)), 1e-12)
            per_seed[s] = {"y_star_values": vals,
                           "J_star": {k: v.tolist() for k, v in Js.items()},
                           "abs_spread": describe(spread),
                           "rel_spread": describe(rel_spread)}
            print(f"  {kind:<8} seed {s:<5} abs spread median "
                  f"{per_seed[s]['abs_spread']['median']:.6e}  max "
                  f"{per_seed[s]['abs_spread']['max']:.6e}   rel spread max "
                  f"{per_seed[s]['rel_spread']['max']:.6e}")
        out[kind] = per_seed
    return out


# --------------------------------------------------------------------------- 1.6
def item_1_6():
    """Step-size plateau for the J_** difference.

    The frozen models are out of scope for this chunk, so the configuration each
    of them will use later is predicted the way the audit predicts its artifact
    floors: quantise a control at that model's MEASURED output quantum
    (FINAL_NUMBERS 3.4) and sweep h. Wrapped for TabPFN/TabICL, which normalise
    the target; unwrapped for TabSwift, whose target scaler is commented out.
    The prediction is checked against the real models in chunk 2.
    """
    print("\n" + "=" * 78)
    print("1.6  step-size plateau for J_**")
    print("=" * 78)
    hs = [1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    maps = [("exactgp", gp_predict), ("hiergp", hiergp_predict),
            ("exactgp_wrapped", wrap_traintest(gp_predict))]
    for mid, q in MODEL_QUANTA.items():
        base = wrap_traintest(gp_predict) if mid != "tabswift" else gp_predict
        maps.append((f"quantised_{mid}_q{q:.3e}", quantize_traintest(base, q)))

    out = {}
    for name, predict in maps:
        per_h = {}
        for h in hs:
            allJ = []
            for s in SEEDS:
                X, y, Xq = make_case(s)
                J, _ = jacobian_star_batch(predict, X, y, Xq[:20],
                                           lambda yy: float(np.mean(yy)), h)
                allJ.append(J)
            per_h[f"{h:.0e}"] = np.concatenate(allJ).tolist()
        # drift relative to the largest step, per h
        ref = np.array(per_h[f"{hs[-1]:.0e}"])
        out[name] = {"per_h": per_h,
                     "mean_per_h": {k: float(np.mean(v)) for k, v in per_h.items()},
                     "std_per_h": {k: float(np.std(v)) for k, v in per_h.items()},
                     "max_abs_dev_from_largest_h": {
                         k: float(np.max(np.abs(np.array(v) - ref))) for k, v in per_h.items()}}
        line = "  ".join(f"{h:.0e}:{out[name]['mean_per_h'][f'{h:.0e}']:+.5f}" for h in hs)
        print(f"  {name:<34} mean J_** by h   {line}")
    return out


# --------------------------------------------------------------------------- 1.2s
def item_1_2s():
    """Sampling distribution of sigma2_hat, in closed form, for the exact GP.

    Item 1.2's gate asks for sigma2_hat/sigma^2 within 10% per seed. Whether
    that is achievable is a property of the estimator's sampling distribution,
    not of the implementation, so it is computed rather than argued.

    Under the well-specified GP, y ~ N(0, C) with C = K + sigma^2 I, and
        m(y) - y = -sigma^2 C^-1 y
        ||m(y) - y||^2 = sigma^4 y' C^-2 y          a quadratic form
        n - tr J = sigma^2 tr(C^-1)
    so E||m-y||^2 = sigma^4 tr(C^-1) = sigma^2 (n - tr J), giving E[sigma2_hat]
    = sigma^2 exactly, and Var(y'Ay) = 2 tr((AC)^2) with A = sigma^4 C^-2 gives
        SD(sigma2_hat)/sigma^2 = sqrt(2 tr(C^-2)) * sigma^2 / (n - tr J).
    """
    print("\n" + "=" * 78)
    print("1.2s  closed-form sampling spread of sigma2_hat on the exact GP")
    print("=" * 78)
    out = {}
    for s in SEEDS:
        X, y, _ = make_case(s)
        C = rbf_C(X)
        Ci = np.linalg.inv(C)
        n_minus_trJ = GP_SIGMA ** 2 * float(np.trace(Ci))
        rel_sd = float(np.sqrt(2 * np.trace(Ci @ Ci)) * GP_SIGMA ** 2 / n_minus_trJ)
        out[s] = {"n_minus_trJ_analytic": n_minus_trJ,
                  "predicted_rel_sd_of_sigma2hat": rel_sd,
                  "predicted_rel_sd_of_mean_over_5_seeds": rel_sd / np.sqrt(len(SEEDS))}
        print(f"  seed {s:<5} n-trJ {n_minus_trJ:8.4f}   predicted SD(sigma2_hat)/sigma^2 "
              f"{rel_sd:.4f}")
    m = float(np.mean([out[s]["predicted_rel_sd_of_sigma2hat"] for s in SEEDS]))
    print(f"  mean predicted relative SD of a SINGLE-seed sigma2_hat: {m:.4f}")
    print(f"  mean predicted relative SD of the 5-seed average:       {m/np.sqrt(len(SEEDS)):.4f}")
    out["mean_predicted_rel_sd"] = m
    out["mean_predicted_rel_sd_of_5seed_mean"] = m / np.sqrt(len(SEEDS))
    return out


def rbf_C(X):
    from experiments.phase_2.src.estimators import rbf
    return rbf(X, X, 1.0) + GP_SIGMA ** 2 * np.eye(len(X))


# --------------------------------------------------------------------------- 1.5b
def item_1_5b():
    """Which hypothetical label y_* to standardise on.

    1.5 shows J_** is exactly y_*-independent for the exact GP (linear map) and
    strongly y_*-dependent for the hierarchical GP. Where it is dependent there
    is no a-priori-correct choice, so the four candidates are scored against the
    closed-form predictive variance directly, using the inverted form and the
    true sigma^2.

    The self-consistent candidate y_* = m_*(y), the model's own prediction at
    the query from the ORIGINAL context, is the only one that does not inject
    information the model did not already have.
    """
    print("\n" + "=" * 78)
    print("1.5b  y_* choice scored against the closed form (inverted form, true sigma^2)")
    print("=" * 78)
    out = {}
    for kind, predict, ctor in [("exactgp", gp_predict, GPTrainTest),
                                ("hiergp", hiergp_predict, HierGPTrainTest)]:
        per_seed = {}
        for s in SEEDS:
            X, y, Xq = make_case(s)
            s2_true = ctor(X, y, sigma=GP_SIGMA).predictive_variance(Xq)
            mu, sd = float(np.mean(y)), float(np.std(y))
            m_star = np.asarray(predict(X, y, Xq)).ravel()
            cands = {"context_mean": np.full(len(Xq), mu),
                     "self_consistent_m_star": m_star,
                     "mean_plus_sd": np.full(len(Xq), mu + sd),
                     "mean_minus_sd": np.full(len(Xq), mu - sd)}
            row = {}
            for name, ys in cands.items():
                J, _ = jacobian_star_batch(predict, X, y, Xq, ys, H_DEFAULT)
                s2, clip, _ = s2_from_Jstar(GP_SIGMA ** 2, J, form="inverted")
                rel = np.abs(s2 - s2_true) / s2_true
                row[name] = {"rel_err": describe(rel), "clip_rate": float(np.mean(clip))}
            per_seed[s] = row
            line = "  ".join(f"{k}:{v['rel_err']['p90']:.3e}" for k, v in row.items())
            print(f"  {kind:<8} seed {s:<5} p90 rel err by y_*   {line}")
        out[kind] = per_seed
        for name in ["context_mean", "self_consistent_m_star", "mean_plus_sd", "mean_minus_sd"]:
            v = [per_seed[s][name]["rel_err"]["p90"] for s in SEEDS]
            print(f"    {kind:<8} {name:<24} mean p90 over seeds {np.mean(v):.4e}")
    return out


# --------------------------------------------------------------------------- 1.4b
def item_1_4b(res):
    """Is the append perturbation predictive of estimator error, per query?

    1.4 gates on the size of the perturbation. The estimator's correctness does
    not depend on the appended map equalling the original one -- it depends on
    T2 holding on the AUGMENTED context, which 1.1 tests end to end. This item
    checks the two against each other query by query.
    """
    from experiments.phase_2.src.estimators import spearman
    print("\n" + "=" * 78)
    print("1.4b  per-query rank correlation: append perturbation vs estimator error")
    print("=" * 78)
    out = {}
    for kind, key in [("exactgp", "1.1_1.2_exactgp"), ("hiergp", "1.3_hiergp")]:
        rows = {}
        for s in SEEDS:
            pert = res["1.4_append_perturbation"][kind][str(s)]["values"]
            err = res[key][str(s)]["forms"]["inverted__true_sigma2"]["rel_err"]["values"]
            rows[s] = {"spearman": spearman(pert, err),
                       "pert_median": float(np.median(pert)),
                       "err_median": float(np.median(err))}
            print(f"  {kind:<8} seed {s:<5} spearman(perturbation, rel err) "
                  f"{rows[s]['spearman']:+.4f}   pert median {rows[s]['pert_median']:.3e}"
                  f"   err median {rows[s]['err_median']:.3e}")
        out[kind] = rows
    return out


# --------------------------------------------------------------------------- gate
def gate1(res):
    """GP rel err on s2_jac < 5% at p90; sigma2_hat/sigma^2 within 10%;
    append perturbation < 1%."""
    g = {}
    for form in ("specified", "inverted"):
        p90 = [res["1.1_1.2_exactgp"][str(s)]["forms"][f"{form}__true_sigma2"]["rel_err"]["p90"]
               for s in SEEDS]
        g[f"gp_relerr_p90__{form}"] = {"per_seed": p90, "worst": float(np.max(p90)),
                                       "required": 0.05,
                                       "met": bool(np.max(p90) < 0.05)}
    ratios = [res["1.1_1.2_exactgp"][str(s)]["ratio_sigma2hat_over_true"] for s in SEEDS]
    g["sigma2hat_ratio"] = {"per_seed": ratios,
                            "worst_abs_dev": float(np.max(np.abs(np.array(ratios) - 1.0))),
                            "required": 0.10,
                            "met": bool(np.max(np.abs(np.array(ratios) - 1.0)) < 0.10)}
    ap = [res["1.4_append_perturbation"]["exactgp"][str(s)]["max"] for s in SEEDS]
    g["append_perturbation"] = {"per_seed_max": ap, "worst": float(np.max(ap)),
                                "required": 0.01, "met": bool(np.max(ap) < 0.01)}
    return g


def main():
    res = {"_config": {"seeds": SEEDS, "n_ctx": N_CTX, "d": D, "gp_sigma": GP_SIGMA,
                       "n_query": N_QUERY, "h_default": H_DEFAULT,
                       "model_quanta": MODEL_QUANTA,
                       "context": "generate_audit_context(n=100, d=5, sigma=0.5, seed=s)",
                       "queries": "RandomState(seed+1000).randn(50, 5)"}}
    res["1.0_surrogate_identity"] = item_1_0()
    res["1.1_1.2_exactgp"] = item_1_1_1_2_1_3("exactgp")
    res["1.3_hiergp"] = item_1_1_1_2_1_3("hiergp")
    res["1.4_append_perturbation"] = item_1_4()
    res["1.5_ystar_sensitivity"] = item_1_5()
    res["1.6_step_plateau"] = item_1_6()
    res["1.2s_sigma2hat_sampling"] = item_1_2s()
    res["1.5b_ystar_choice"] = item_1_5b()

    res = jsonable(res)
    res["1.4b_perturbation_vs_error"] = jsonable(item_1_4b(res))
    res["gate1"] = jsonable(gate1(res))

    print("\n" + "=" * 78)
    print("GATE 1")
    print("=" * 78)
    for k, v in res["gate1"].items():
        print(f"  {k:<34} worst "
              f"{v.get('worst', v.get('worst_abs_dev')):.6e}  required < {v['required']}  "
              f"met={v['met']}")

    p = RESULTS / "chunk1_results.json"
    with open(p, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
