"""CHUNK 2, part 2 -- run the four estimators on every model x dataset x split.

THE FOUR ESTIMATORS. The mean channel is identical for 1, 2, 3; only the
uncertainty differs. Estimator 4 is a reference with its own mean.

  1 native      the model's own head.
                TabICL v2  -- output_type="quantiles" on 9999 uniform alphas in
                              [1e-4, 1-1e-4], integrated by
                              core.metrics.extract_variance. The REJECTED
                              alternative is output_type="variance", which is
                              raw_quantiles.var(dim=-1) (tabicl/_model/tabicl.py:582),
                              a spread over the quantile grid rather than the
                              second moment of the predictive distribution.
                TabPFN v2  -- FullSupportBarDistribution.variance(logits), the
                              integrated second moment with half-normal tails
                              (tabpfn/architectures/base/bar_distribution.py:606).
                              The REJECTED alternative is sum(bin_prob * centre),
                              which is E[X] and omits the tail bins.
                TabSwift   -- DOES NOT EXIST. Linear(384,1) point head,
                              registry.has_predictive_distribution=False.
                              Reported as absent, never synthesised.

  2 Jacobian    sigma2_hat / (1 - J_**), the form validated in chunk 1.
                sigma2_hat = ||m(y) - y||^2 / (n - tr J) from the context
                Jacobian. The brief's sigma2_hat * (1 + J_**) is computed and
                stored alongside under the key `s2_jac_specified`.
                y_* = m_*(y), the self-consistent choice scored best in 1.5b and
                already available from the mean channel at zero extra cost.

  3 conformal   split conformal on absolute residuals from the calibration rows.
                Constant width by construction -- consequences for 3.3 and 3.6
                are stated where they arise, not worked around.

  4 oracle GP   sklearn GaussianProcessRegressor, ConstantKernel*RBF+WhiteKernel,
                marginal-likelihood fit on the context rows. `kernel_.diag`
                includes the white-noise term, so its predictive variance is
                already for a fresh observation.

Probe step per model: h = H_FRAC * std(y_ctx), RELATIVE to the target scale.
See the H_FRAC_BY_MODEL comment below for why an absolute step was discarded and
what it did on did=509. No dither: the appended-query difference is a single
scalar per query, and chunk 1.6 shows the plateau is reached at these relative
steps for each model's measured output quantum.

Writes chunk2_estimator_<model>.json and chunk2_arrays_<model>.npz.
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
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC.parent.parent.parent))              # repo root
warnings.filterwarnings("ignore")

import torch                                                        # noqa: E402
from models import load                                             # noqa: E402
from experiments.phase_1.core.metrics import extract_variance               # noqa: E402
from experiments.phase_2.src.paths import RESULTS, ARRAYS           # noqa: E402
from experiments.phase_2.src.estimators import (                    # noqa: E402
    context_jacobian, jacobian_star_batch, s2_from_Jstar, sigma2_hat,
    conformal_halfwidth, jsonable, CLIP_FLOOR_FRAC,
)

SEEDS = [42, 100, 200, 300, 400]
LEVELS_ICL = np.linspace(1e-4, 1 - 1e-4, 9999)
NOMINAL = [0.50, 0.80, 0.90, 0.95]

#: THE PROBE STEP IS RELATIVE TO THE TARGET SCALE:  h = H_FRAC * std(y_ctx).
#:
#: A first pass used the audit's amplitudes as ABSOLUTE numbers (1e-3 / 1e-1)
#: and was discarded. Every model here renormalises the target internally and
#: un-normalises its output, so the output quantum in the ORIGINAL target units
#: scales with std(y). On OpenML did=509 (`places`, std(y) = 1108) an absolute
#: h = 1e-1 is 9e-5 of a label SD, deep inside the quantisation floor: it
#: returned tr J = 110 to 243 against n = 100, i.e. n - tr J < 0 and sigma2_hat
#: undefined, with 31-37 negative diagonal entries. The same split at
#: h = 0.0686 * std(y) returns tr J = 16.5 and 2 negative entries, stable to
#: within 5% from 0.01*std to 0.5*std. Recorded in chunk2_stepsize.json.
#:
#: The fractions reproduce the audit's own amplitudes on the audit's own
#: contexts: its mean context label SD is 1.457782 (FINAL_NUMBERS 5.7, mean of
#: [1.400625, 1.449024, 1.395749, 1.322935, 1.720577]), so 1e-1 -> 6.8598e-2
#: and 1e-3 -> 6.8598e-4.
AUDIT_MEAN_SY = 1.457782
H_FRAC_BY_MODEL = {"tabicl_v2": 1e-3 / AUDIT_MEAN_SY,
                   "tabpfn_v2": 1e-1 / AUDIT_MEAN_SY,
                   "tabswift": 1e-1 / AUDIT_MEAN_SY}


def make_predict(model, model_id):
    def predict(X_train, y_train, X_test):
        model.estimator.fit(np.asarray(X_train, float), np.asarray(y_train, float))
        if model_id == "tabpfn_v2":
            return np.asarray(model.estimator.predict(np.asarray(X_test, float),
                                                      output_type="mean")).ravel()
        return np.asarray(model.estimator.predict(np.asarray(X_test, float))).ravel()
    return predict


def native_variance(model, model_id, X_ctx, y_ctx, X_test):
    """Estimator 1. Returns None where the model has no predictive distribution."""
    if model_id == "tabswift":
        return None
    est = model.estimator
    est.fit(np.asarray(X_ctx, float), np.asarray(y_ctx, float))
    Xt = np.asarray(X_test, float)
    if model_id == "tabicl_v2":
        Q = np.asarray(est.predict(Xt, output_type="quantiles",
                                   alphas=list(LEVELS_ICL)))
        if Q.ndim == 2 and Q.shape[0] == len(LEVELS_ICL):
            Q = Q.T
        return np.asarray(extract_variance(Q, LEVELS_ICL)).ravel()
    full = est.predict(Xt, output_type="full")
    crit, lg = full["criterion"], full["logits"]
    lg = lg if torch.is_tensor(lg) else torch.as_tensor(lg)
    lg = lg.to(next(crit.buffers()).device).float()
    return crit.variance(lg).detach().cpu().numpy().ravel()


def oracle_gp(X_ctx, y_ctx, X_test):
    """Estimator 4. Returns (mean, variance-for-a-fresh-observation)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    k = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e3))
         + WhiteKernel(1.0, (1e-6, 1e3)))
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=2,
                                  random_state=0)
    gp.fit(np.asarray(X_ctx, float), np.asarray(y_ctx, float))
    out = gp.predict(np.asarray(X_test, float), return_std=True)
    mu, sd = out[0], out[1]
    return np.asarray(mu).ravel(), np.asarray(sd).ravel() ** 2, str(gp.kernel_)


def run_model(model_id, data, manifest):
    print("\n" + "=" * 78)
    print(f"MODEL {model_id}   h={H_FRAC_BY_MODEL[model_id]:.4e} x std(y_ctx), no dither")
    print("=" * 78)
    model = load(model_id, task="regression", device="cuda")
    predict = make_predict(model, model_id)
    h_frac = H_FRAC_BY_MODEL[model_id]

    rows, arrays = [], {}
    for rec in manifest["accepted"]:
        did = rec["did"]
        for s in SEEDS:
            t0 = time.time()
            g = lambda k: data[f"{did}__{s}__{k}"]                    # noqa: E731
            X_ctx, y_ctx = g("X_ctx"), g("y_ctx")
            X_cal, y_cal = g("X_cal"), g("y_cal")
            X_test, y_test = g("X_test"), g("y_test")
            h = h_frac * float(np.std(y_ctx))

            # ---- mean channel, shared by estimators 1-3
            m_ctx = predict(X_ctx, y_ctx, X_ctx)
            m_cal = predict(X_ctx, y_ctx, X_cal)
            m_test = predict(X_ctx, y_ctx, X_test)

            # ---- context Jacobian -> tr J -> sigma2_hat
            J_ctx = context_jacobian(predict, X_ctx, y_ctx, h)
            trJ = float(np.trace(J_ctx))
            s2h, dof, defined = sigma2_hat(m_ctx, y_ctx, trJ)

            # ---- estimator 2
            J_star, y_star = jacobian_star_batch(predict, X_ctx, y_ctx, X_test, m_test, h)
            if defined:
                s2_inv, clip_inv, raw_inv = s2_from_Jstar(s2h, J_star, form="inverted")
                s2_spec, clip_spec, _ = s2_from_Jstar(s2h, J_star, form="specified")
            else:
                nanv = np.full(len(J_star), np.nan)
                s2_inv = s2_spec = raw_inv = nanv
                clip_inv = clip_spec = np.zeros(len(J_star), bool)

            # ---- estimator 1
            s2_nat = native_variance(model, model_id, X_ctx, y_ctx, X_test)

            # ---- estimator 3
            r_cal = np.abs(np.asarray(y_cal, float) - m_cal)
            qhat = {f"{lv:.2f}": conformal_halfwidth(r_cal, 1.0 - lv) for lv in NOMINAL}

            # ---- estimator 4
            gp_mu, gp_s2, gp_kernel = oracle_gp(X_ctx, y_ctx, X_test)

            tag = f"{model_id}__{did}__{s}"
            arrays[f"{tag}__J_ctx"] = J_ctx.astype(np.float32)
            arrays[f"{tag}__J_star"] = J_star
            arrays[f"{tag}__m_test"] = m_test
            arrays[f"{tag}__y_test"] = np.asarray(y_test, float)
            arrays[f"{tag}__s2_jac"] = s2_inv
            arrays[f"{tag}__s2_jac_specified"] = s2_spec
            arrays[f"{tag}__s2_gp"] = gp_s2
            arrays[f"{tag}__gp_mu"] = gp_mu
            arrays[f"{tag}__m_ctx"] = m_ctx
            arrays[f"{tag}__y_ctx"] = np.asarray(y_ctx, float)
            arrays[f"{tag}__r_cal"] = r_cal
            if s2_nat is not None:
                arrays[f"{tag}__s2_native"] = s2_nat

            row = {
                "model": model_id, "did": did, "name": rec["name"], "seed": s,
                "n_ctx": int(len(y_ctx)), "n_cal": int(len(y_cal)),
                "n_test": int(len(y_test)), "d": rec["d"],
                "h": h, "h_frac": h_frac, "std_y_ctx": float(np.std(y_ctx)),
                "trJ": trJ, "n_minus_trJ": dof,
                "sigma2_hat": s2h, "sigma2_hat_defined": bool(defined),
                "n_neg_Jii": int(np.sum(np.diag(J_ctx) < 0)),
                "Jii_min": float(np.min(np.diag(J_ctx))),
                "Jii_max": float(np.max(np.diag(J_ctx))),
                "Jstar_min": float(np.min(J_star)), "Jstar_max": float(np.max(J_star)),
                "Jstar_ge_1_rate": float(np.mean(J_star >= 1.0)),
                "clip_rate_inverted": float(np.mean(clip_inv)),
                "clip_rate_specified": float(np.mean(clip_spec)),
                "clip_floor_frac": CLIP_FLOOR_FRAC,
                "clip_floor_value": float(CLIP_FLOOR_FRAC * s2h) if defined else float("nan"),
                "conformal_qhat": qhat,
                "native_available": s2_nat is not None,
                "gp_kernel": gp_kernel,
                "ctx_rmse": float(np.sqrt(np.mean((m_ctx - y_ctx) ** 2))),
                "test_rmse": float(np.sqrt(np.mean((m_test - np.asarray(y_test, float)) ** 2))),
                "gp_test_rmse": float(np.sqrt(np.mean((gp_mu - np.asarray(y_test, float)) ** 2))),
                "seconds": round(time.time() - t0, 1),
            }
            rows.append(row)
            print(f"  did={did:<6} seed {s:<5} {row['seconds']:6.1f}s  "
                  f"trJ {trJ:7.2f}  n-trJ {dof:7.2f}  sig2 {s2h:10.4g}  "
                  f"neg Jii {row['n_neg_Jii']:3d}  clip {row['clip_rate_inverted']:.3f}  "
                  f"rmse {row['test_rmse']:.4g}", flush=True)

            np.savez_compressed(ARRAYS / f"chunk2_arrays_{model_id}.npz", **arrays)
            with open(RESULTS / f"chunk2_estimator_{model_id}.json", "w") as f:
                json.dump(jsonable({"_config": {"h_frac": h_frac, "seeds": SEEDS,
                                                "audit_mean_sy": AUDIT_MEAN_SY,
                                                "h_rule": "h = h_frac * std(y_ctx)",
                                                "nominal": NOMINAL,
                                                "clip_floor_frac": CLIP_FLOOR_FRAC,
                                                "y_star": "self-consistent m_*(y)"},
                                    "rows": rows}), f, indent=2)
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--limit=")), None)
    which = args or ["tabicl_v2", "tabpfn_v2", "tabswift"]
    data = np.load(ARRAYS / "chunk2_data.npz")
    manifest = json.load(open(RESULTS / "chunk2_datasets.json"))
    if limit:
        manifest["accepted"] = manifest["accepted"][:limit]
    print(f"datasets: {[r['did'] for r in manifest['accepted']]}")
    for mid in which:
        run_model(mid, data, manifest)
    print("\nDone.")


if __name__ == "__main__":
    main()
