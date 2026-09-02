"""T1.3 -- the prior-family-ness coordinate. The x-axis of the headline figure.

Per context, the model's OWN prequential log marginal likelihood

    log q(y_1:n | X) = sum_k log q(y_k | y_<k, X)

chained through its predictive distribution in a fixed random row order, averaged
over several orders with the spread reported (TabICL is not exactly exchangeable,
so the spread is a real quantity, not noise to be hidden).

WHY THIS IS THE RIGHT X-AXIS, AND NOT AN ARBITRARY ONE
------------------------------------------------------
It estimates the same `log p(y)` whose gradient field the audit tests. Tweedie's
potential and the dose-response abscissa become the same object: the audit asks
whether `(m(y) - y)/sigma^2` is the gradient of a scalar potential, and this
measures the model's own estimate of that potential. A model far outside its
prior support should assign low log marginal likelihood AND, if the off-family
story is right, show a larger structural violation.

Held-out NLL is recorded alongside as a second, more conventional coordinate.

COST, AND THE POWER CUT
-----------------------
The chained likelihood needs one fit per prefix -- n fits per order. At n = 100
and 3 orders that is 300 fits per context, ~78 s. Over 360 contexts that is 7.8
GPU-hours on top of the ladder's 10.6.

Cut per plan section 1 (cut power, never a registered comparison): run on
N_SUBSET = 30 of the 60 contexts per rung. 180 points across six rungs is ample
for the pooled Spearman that P8 registers, and the per-rung means keep their
error bars. The FD ladder still runs on all 60.

Usage:  python t1_coordinate.py [rung ...]
Writes results/t1_coordinate_<rung>.json.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
warnings.filterwarnings("ignore")

import torch                                                            # noqa: E402
from models import load                                                 # noqa: E402
from experiments.phase_1.core.metrics import extract_variance           # noqa: E402

ORDER = ["A", "D", "B", "C", "E", "F"]
N_SUBSET = 30
N_ORDERS = 3
MIN_PREFIX = 5               # below this the model has too little to condition on
LEVELS = np.linspace(1e-4, 1 - 1e-4, 999)


def prequential(est, X, y, perm):
    """sum_k log q(y_k | y_<k, X), Gaussian at the model's own mean and variance.

    The quantile head is integrated to a mean and variance by the same
    `extract_variance` the audit uses, then scored as a Gaussian. Stated rather
    than implied: this is a Gaussian scoring of a non-Gaussian predictive, so it
    is a proper but not sharp proxy for the true chained log likelihood. It is
    used consistently across rungs, which is what a dose-response x-axis needs.
    """
    Xp, yp = X[perm], y[perm]
    n = len(yp)
    tot, terms = 0.0, []
    for k in range(MIN_PREFIX, n):
        est.fit(Xp[:k], yp[:k])
        xq = Xp[k:k + 1]
        mu = float(np.asarray(est.predict(xq)).ravel()[0])
        Qq = np.asarray(est.predict(xq, output_type="quantiles", alphas=list(LEVELS)))
        if Qq.ndim == 2 and Qq.shape[0] == len(LEVELS):
            Qq = Qq.T
        var = float(np.asarray(extract_variance(Qq, LEVELS)).ravel()[0])
        var = max(var, 1e-8)
        ll = -0.5 * (np.log(2 * np.pi * var) + (yp[k] - mu) ** 2 / var)
        tot += ll
        terms.append(float(ll))
    return float(tot), terms


def main():
    rungs = sys.argv[1:] or ORDER
    z = np.load(HERE.parent / "results" / "t1_contexts.npz")
    model = load("tabicl_v2", task="regression", device="cuda")
    est = model.estimator

    for rung in rungs:
        t0, rows = time.time(), []
        print("=" * 88, flush=True)
        print(f"T1.3  RUNG {rung}   {N_SUBSET} contexts x {N_ORDERS} orders", flush=True)
        print("=" * 88, flush=True)
        for k in range(N_SUBSET):
            key = f"{rung}__X__{k}"
            if key not in z:
                break
            X, y = z[key], z[f"{rung}__y__{k}"]
            n = len(y)
            rng = np.random.default_rng(9000 + k)
            tots = []
            for o in range(N_ORDERS):
                perm = rng.permutation(n)
                tot, _ = prequential(est, X, y, perm)
                tots.append(tot)
            tots = np.asarray(tots)
            # normalise per observation so rungs with different n stay comparable
            n_scored = n - MIN_PREFIX
            rows.append({"index": k, "logml": float(tots.mean()),
                         "logml_sd": float(tots.std(ddof=1)),
                         "logml_per_obs": float(tots.mean() / n_scored),
                         "n_scored": n_scored,
                         "orders": tots.tolist()})
            if k % 5 == 0:
                print(f"  ctx {k:>3}  logML/obs {rows[-1]['logml_per_obs']:+.4f}"
                      f"  spread over orders {rows[-1]['logml_sd']:.3f}"
                      f"  ({time.time()-t0:.0f}s)", flush=True)

        v = np.array([r["logml_per_obs"] for r in rows])
        out = {"_config": {"rung": rung, "n_subset": len(rows),
                           "n_orders": N_ORDERS, "min_prefix": MIN_PREFIX,
                           "model": "tabicl_v2"},
               "aggregate": {"logml_per_obs_mean": float(v.mean()),
                             "logml_per_obs_sd": float(v.std(ddof=1)),
                             "logml_per_obs_min": float(v.min()),
                             "logml_per_obs_max": float(v.max()),
                             "order_spread_mean": float(np.mean(
                                 [r["logml_sd"] for r in rows]))},
               "rows": rows}
        json.dump(out, open(HERE.parent / "results" / f"t1_coordinate_{rung}.json", "w"),
                  indent=2, default=float)
        print(f"\n  RUNG {rung}  logML/obs {v.mean():+.4f} +- {v.std(ddof=1):.4f}"
              f"   [{v.min():+.4f}, {v.max():+.4f}]   ({time.time()-t0:.0f}s)", flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
