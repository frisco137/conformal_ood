"""E4.1 -- conventional predictive performance on the EXACT audit contexts.

The objection this closes: the audit contexts are adversarial by construction
(n=100, d=5, GP-drawn, sigma=1.0, ten exactly duplicated X rows, built to force
smoothing). If the models simply perform badly there, the structural violations
measured in FINAL_NUMBERS sections 3-6 are entangled with plain
out-of-distribution failure and the whole claim is confounded.

experiments.md E4.1: "a model that passes every published test and fails the
parameter-free structural conditions, on the same data."

WHAT IS MEASURED
----------------
Two regimes, because the audit measures one and the literature reports the other.

  (a) In-sample denoising, m(y) at the context points against the LATENT f.
      This is the object the Jacobian audit actually differentiates. The audit
      context hands the model y = f + eps with eps ~ N(0, 1); recovering f is
      the whole task. Reported as MSE(m(y), f) and as the shrinkage ratio
      MSE(m(y), f) / MSE(y, f) -- below 1 means the model denoised, at 1 means
      it returned its input, above 1 means it made things worse.

  (b) Held-out prediction at 200 fresh queries per seed.
      MSE, NLL, coverage and interval score at four nominal levels.

HOW THE HELD-OUT SET IS DRAWN, AND WHY IT DOES NOT PERTURB THE AUDIT CONTEXT
---------------------------------------------------------------------------
generate_audit_context draws f ~ N(0, K + 1e-4 I) over its own 100 rows. Fresh
queries must come from the SAME latent draw, so they are sampled from the GP
conditional f_* | f, which is exact and leaves (X, y, f) bit-identical to what
every other script in the tree sees. Sampling a joint (context + test) GP
instead would have changed the context.

    X_*      ~ N(0, I_d)                 fresh, seeded per audit seed
    f_* | f  ~ N(K_*c C^-1 f,  K_** - K_*c C^-1 K_c*),   C = K + 1e-4 I
    y_*       = f_* + sigma * eps

THE REFERENCE POINTS -- what makes the judgement checkable
----------------------------------------------------------
  oracle GP        the TRUE generative model: same RBF kernel, lengthscale 1.0,
                   sigma = 1.0. This is the Bayes-optimal predictor for these
                   contexts. Nothing can beat it in expectation, so it is the
                   floor, and every MSE is also reported as a ratio to it.
  audit control GP the map the audit actually pushes through its pipeline,
                   ExactGP(sigma=0.5). MISSPECIFIED against sigma=1.0 data --
                   see the note in the results; it is reported because it is the
                   control, not because it is the best GP.
  hierarchical GP  mixture over 8 lengthscales, the audit's other control.
  constant         predict mean(y_ctx). The do-nothing floor.
  1-NN             the N5 catalogue row that passes A1/A2 vacuously.

SNR WARNING, read before judging any MSE
----------------------------------------
The RBF kernel has unit signal variance and the context noise is sigma = 1.0, so
the signal-to-noise ratio is 1 BY DESIGN. Irreducible held-out MSE against y is
therefore ~1.0 + posterior variance, and an MSE near 1.0 is near-optimal, not
bad. This is why every number is reported beside the oracle rather than alone.

Writes e4_1_results.json. GPU, ~1 minute per model: one fit per seed, no
Jacobians.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np


class _MockAnalytics:
    def __getattr__(self, name):
        return lambda *a, **k: None


sys.modules["analytics"] = _MockAnalytics()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
warnings.filterwarnings("ignore")

import torch                                                          # noqa: E402
from models import load                                               # noqa: E402
from experiments.phase_1.core.context import generate_audit_context           # noqa: E402
from experiments.phase_1.core.metrics import extract_variance                 # noqa: E402
from experiments.phase_2.src.estimators import (                      # noqa: E402
    GPTrainTest, HierGPTrainTest, coverage, gaussian_interval, gaussian_nll,
    interval_score, jsonable, rbf, spearman,
)

SEEDS = [42, 100, 200, 300, 400]
MODELS = ["tabicl_v2", "tabpfn_v2", "tabswift"]
N_CTX, D, SIGMA, LENGTHSCALE, NUGGET = 100, 5, 1.0, 1.0, 1e-4
N_TEST = 200
LEVELS = [0.50, 0.80, 0.90, 0.95]
LEVELS_ICL = np.linspace(1e-4, 1 - 1e-4, 9999)
AUDIT_GP_SIGMA = 0.5          # the audit control's sigma, deliberately not 1.0


# --------------------------------------------------------------- the test set

def draw_queries(X_ctx, f_ctx, seed, n_test=N_TEST):
    """Fresh queries from the GP conditional f_* | f. Leaves the context alone."""
    rng = np.random.RandomState(10_000 + seed)
    X_s = rng.randn(n_test, D)
    C = rbf(X_ctx, X_ctx, LENGTHSCALE) + NUGGET * np.eye(len(X_ctx))
    K_sc = rbf(X_s, X_ctx, LENGTHSCALE)
    K_ss = rbf(X_s, X_s, LENGTHSCALE) + NUGGET * np.eye(n_test)
    A = np.linalg.solve(C, K_sc.T)
    mu = K_sc @ np.linalg.solve(C, f_ctx)
    cov = K_ss - K_sc @ A
    cov = (cov + cov.T) / 2.0
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(n_test))
    f_s = mu + L @ rng.randn(n_test)
    y_s = f_s + SIGMA * rng.randn(n_test)
    return X_s, y_s, f_s


# ------------------------------------------------------------------- scoring

def score(name, m_ctx, m_test, s2_test, y_ctx, f_ctx, y_test, f_test):
    """Every conventional number, in one place, for a model or a control."""
    m_ctx = np.asarray(m_ctx, float).ravel()
    m_test = np.asarray(m_test, float).ravel()

    mse_y_ctx = float(np.mean((m_ctx - y_ctx) ** 2))
    mse_f_ctx = float(np.mean((m_ctx - f_ctx) ** 2))
    mse_y_test = float(np.mean((m_test - y_test) ** 2))
    mse_f_test = float(np.mean((m_test - f_test) ** 2))
    var_y_test = float(np.var(y_test))

    out = {
        "name": name,
        # (a) in-sample denoising -- the object the audit differentiates
        "mse_ctx_vs_f": mse_f_ctx,
        "mse_ctx_vs_y": mse_y_ctx,
        "shrinkage_ratio": mse_f_ctx / float(np.mean((y_ctx - f_ctx) ** 2)),
        # (b) held-out
        "mse_test_vs_y": mse_y_test,
        "mse_test_vs_f": mse_f_test,
        "rmse_test_vs_y": float(np.sqrt(mse_y_test)),
        "r2_test": 1.0 - mse_y_test / var_y_test,
        "spearman_test": spearman(m_test, y_test),
        "pearson_test": float(np.corrcoef(m_test, y_test)[0, 1]),
    }

    if s2_test is None:
        out["nll_test"] = None
        out["has_predictive_distribution"] = False
        for lv in LEVELS:
            out[f"cov@{int(lv*100)}"] = None
            out[f"is@{int(lv*100)}"] = None
        return out

    s2 = np.maximum(np.asarray(s2_test, float).ravel(), 1e-12)
    out["has_predictive_distribution"] = True
    out["nll_test"] = float(np.mean(gaussian_nll(y_test, m_test, s2)))
    out["mean_s2"] = float(np.mean(s2))
    out["spearman_s2_vs_sqerr"] = spearman(s2, (y_test - m_test) ** 2)
    for lv in LEVELS:
        lo, hi = gaussian_interval(m_test, s2, lv)
        out[f"cov@{int(lv*100)}"] = coverage(y_test, lo, hi)
        out[f"is@{int(lv*100)}"] = float(np.mean(interval_score(y_test, lo, hi, 1 - lv)))
    return out


# ------------------------------------------------------------------ controls

def run_controls(X, y, f, X_s, y_s, f_s):
    rows = []

    gp_true = GPTrainTest(X, y, sigma=SIGMA, lengthscale=LENGTHSCALE)
    rows.append(score("oracle_gp_sigma1.0", gp_true.predict(X), gp_true.predict(X_s),
                      gp_true.predictive_variance(X_s), y, f, y_s, f_s))

    gp_audit = GPTrainTest(X, y, sigma=AUDIT_GP_SIGMA, lengthscale=LENGTHSCALE)
    rows.append(score("audit_control_gp_sigma0.5", gp_audit.predict(X), gp_audit.predict(X_s),
                      gp_audit.predictive_variance(X_s), y, f, y_s, f_s))

    hgp = HierGPTrainTest(X, y, sigma=SIGMA)
    rows.append(score("hierarchical_gp", hgp.predict(X), hgp.predict(X_s),
                      hgp.predictive_variance(X_s), y, f, y_s, f_s))

    c = float(np.mean(y))
    rows.append(score("constant_mean", np.full(len(y), c), np.full(len(y_s), c),
                      np.full(len(y_s), float(np.var(y))), y, f, y_s, f_s))

    nn_idx = np.argmin(((X_s[:, None] - X[None, :]) ** 2).sum(-1), axis=1)
    rows.append(score("one_nn", y.copy(), y[nn_idx], None, y, f, y_s, f_s))
    return rows


# -------------------------------------------------------------------- models

def model_channels(model, mid, X, y, X_s):
    est = model.estimator
    est.fit(X, y)
    if mid == "tabpfn_v2":
        m_ctx = np.asarray(est.predict(X, output_type="mean")).ravel()
        m_s = np.asarray(est.predict(X_s, output_type="mean")).ravel()
        full = est.predict(X_s, output_type="full")
        crit, lg = full["criterion"], full["logits"]
        lg = lg if torch.is_tensor(lg) else torch.as_tensor(lg)
        lg = lg.to(next(crit.buffers()).device).float()
        s2 = crit.variance(lg).detach().cpu().numpy().ravel()
        return m_ctx, m_s, s2
    if mid == "tabicl_v2":
        m_ctx = np.asarray(est.predict(X)).ravel()
        m_s = np.asarray(est.predict(X_s)).ravel()
        Qq = np.asarray(est.predict(X_s, output_type="quantiles",
                                    alphas=list(LEVELS_ICL)))
        if Qq.ndim == 2 and Qq.shape[0] == len(LEVELS_ICL):
            Qq = Qq.T
        return m_ctx, m_s, np.asarray(extract_variance(Qq, LEVELS_ICL)).ravel()
    # tabswift -- Linear(384,1) point head, registry has_predictive_distribution=False
    return (np.asarray(est.predict(X)).ravel(),
            np.asarray(est.predict(X_s)).ravel(), None)


def main():
    out = {"_config": {"n_ctx": N_CTX, "d": D, "sigma": SIGMA,
                       "lengthscale": LENGTHSCALE, "nugget": NUGGET,
                       "n_test": N_TEST, "seeds": SEEDS, "levels": LEVELS,
                       "query_rule": "GP conditional f_*|f; context untouched"},
           "per_seed": {}}

    data = {}
    for s in SEEDS:
        X, y, f = generate_audit_context(n=N_CTX, d=D, sigma=SIGMA, seed=s)
        data[s] = (X, y, f) + draw_queries(X, f, s)

    print("=" * 78)
    print("CONTROLS  (no GPU)")
    print("=" * 78)
    for s in SEEDS:
        out["per_seed"].setdefault(str(s), [])
        out["per_seed"][str(s)] += run_controls(*data[s])
    for nm in ["oracle_gp_sigma1.0", "audit_control_gp_sigma0.5",
               "hierarchical_gp", "constant_mean", "one_nn"]:
        v = [r for s in SEEDS for r in out["per_seed"][str(s)] if r["name"] == nm]
        print(f"  {nm:<26} mse_test_vs_y {np.mean([r['mse_test_vs_y'] for r in v]):.4f}"
              f"   shrinkage {np.mean([r['shrinkage_ratio'] for r in v]):.4f}")

    for mid in MODELS:
        print("\n" + "=" * 78)
        print(f"MODEL {mid}")
        print("=" * 78)
        model = load(mid, task="regression", device="cuda")
        for s in SEEDS:
            t0 = time.time()
            X, y, f, X_s, y_s, f_s = data[s]
            m_ctx, m_s, s2 = model_channels(model, mid, X, y, X_s)
            r = score(mid, m_ctx, m_s, s2, y, f, y_s, f_s)
            r["seconds"] = time.time() - t0
            out["per_seed"][str(s)].append(r)
            print(f"  seed {s:<5} {r['seconds']:5.1f}s  mse_test {r['mse_test_vs_y']:.4f}"
                  f"  R2 {r['r2_test']:+.4f}  shrinkage {r['shrinkage_ratio']:.4f}"
                  + (f"  NLL {r['nll_test']:.4f}" if r["nll_test"] is not None
                     else "  NLL n/a (point head)"))
        del model
        torch.cuda.empty_cache()

    # ---- aggregate, with the oracle ratio that makes the judgement checkable
    agg = {}
    orc = {s: [r for r in out["per_seed"][str(s)]
               if r["name"] == "oracle_gp_sigma1.0"][0] for s in SEEDS}
    names = [r["name"] for r in out["per_seed"][str(SEEDS[0])]]
    for nm in names:
        rows = [[r for r in out["per_seed"][str(s)] if r["name"] == nm][0] for s in SEEDS]
        e = {"name": nm, "n_seeds": len(rows)}
        for k in ["mse_test_vs_y", "mse_test_vs_f", "mse_ctx_vs_f", "shrinkage_ratio",
                  "r2_test", "nll_test", "spearman_test", "spearman_s2_vs_sqerr"] + \
                 [f"cov@{int(l*100)}" for l in LEVELS] + \
                 [f"is@{int(l*100)}" for l in LEVELS]:
            v = [r.get(k) for r in rows]
            if any(x is None for x in v):
                e[k] = None; e[k + "_sd"] = None
                continue
            e[k] = float(np.mean(v)); e[k + "_sd"] = float(np.std(v, ddof=1))
        e["mse_ratio_to_oracle"] = float(np.mean(
            [r["mse_test_vs_y"] / orc[s]["mse_test_vs_y"] for r, s in zip(rows, SEEDS)]))
        e["mse_f_ratio_to_oracle"] = float(np.mean(
            [r["mse_test_vs_f"] / orc[s]["mse_test_vs_f"] for r, s in zip(rows, SEEDS)]))
        e["ctx_ratio_to_oracle"] = float(np.mean(
            [r["mse_ctx_vs_f"] / orc[s]["mse_ctx_vs_f"] for r, s in zip(rows, SEEDS)]))
        agg[nm] = e

    # ---- efficiency: what FRACTION of the recoverable signal is captured.
    # A raw MSE cannot be judged when the oracle itself is weak, which it is
    # here -- see the SNR warning in the module docstring. This rescales so that
    # the do-nothing predictor sits at 0 and the Bayes-optimal oracle at 1:
    #
    #     eff = (MSE_constant - MSE_model) / (MSE_constant - MSE_oracle)
    #
    # Negative means worse than predicting the context mean.
    for regime, key in [("ctx", "mse_ctx_vs_f"), ("test", "mse_test_vs_f")]:
        base, top = agg["constant_mean"][key], agg["oracle_gp_sigma1.0"][key]
        span = base - top
        for nm in names:
            agg[nm][f"efficiency_{regime}"] = (
                float((base - agg[nm][key]) / span) if span > 0 else None)
    out["aggregate"] = agg

    print("\n" + "=" * 78)
    print("AGGREGATE over 5 seeds -- MSE ratio to the Bayes-optimal oracle GP")
    print("=" * 78)
    print("  efficiency: 0 = predicting the context mean, 1 = the Bayes-optimal oracle\n")
    print(f"  {'':<26}{'ctx_vs_f':>10}{'/orc':>7}{'eff_ctx':>9}"
          f"{'test_vs_f':>11}{'/orc':>7}{'eff_test':>10}{'NLL':>9}{'cov@90':>8}")
    for nm in names:
        e = agg[nm]
        nll = f"{e['nll_test']:.4f}" if e["nll_test"] is not None else "n/a"
        cov = f"{e['cov@90']:.3f}" if e["cov@90"] is not None else "n/a"
        print(f"  {nm:<26}{e['mse_ctx_vs_f']:>10.4f}{e['ctx_ratio_to_oracle']:>7.2f}"
              f"{e['efficiency_ctx']:>9.3f}{e['mse_test_vs_f']:>11.4f}"
              f"{e['mse_f_ratio_to_oracle']:>7.2f}{e['efficiency_test']:>10.3f}"
              f"{nll:>9}{cov:>8}")

    p = HERE / "e4_1_results.json"
    json.dump(jsonable(out), open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
